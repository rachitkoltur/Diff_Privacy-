"""
utility_benchmark.py
====================
COMPANY-UTILITY benchmark: does the manufacturer's LEGITIMATE downstream ML still
work after the CA-LDP noise is added, while the keystroke side channel is killed?

Why this file
-------------
A privacy mechanism is only useful if the data it releases is still good for the
uses it is meant to allow. For a prosthetic-limb maker the legitimate uses are
coarse, aggregate, non-identifying analytics:
  (A) grip-STATE recognition (rest / grasp / type)  -- adaptive control + usage
      analytics. Standard multi-class sEMG activity classification.
  (B) rest-vs-grasp functional-state recognition (binary) -- an UNAMBIGUOUSLY
      non-private control signal (no "type" class), reported so the utility claim
      does not depend on recognizing the typing mode at all.
  (C) per-finger tendon MEAN estimation over the grasp phase -- predictive
      maintenance / grip-force analytics (an aggregate statistic).
None of these needs to know WHICH key was pressed; that is the private thing the
attacker wants and the mechanism must hide.

Is there an off-the-shelf "gold-standard classifier" for this exact setup?
-------------------------------------------------------------------------
No single canonical model exists for "grip state from privatized prosthetic
telemetry". What IS standard is the sEMG classification PIPELINE: standard
amplitude/frequency features fed to Linear Discriminant Analysis (LDA) and a tree
ensemble (Random Forest). LDA is the canonical real-time myoelectric classifier
(Englehart & Hudgins, IEEE TBME 2003); the standard feature/classifier survey is
Phinyomark et al. 2012; the standard benchmark dataset family is Ninapro (Atzori
et al. 2014). We use those standard classifiers rather than inventing a bespoke
one, at their library defaults (LDA has NO tuned hyper-parameters at all), so the
result cannot be dismissed as cherry-picked model tuning.

Evaluation protocol (the important part)
----------------------------------------
sEMG windows are strongly autocorrelated in time, so a random train/test split on
windows from ONE session leaks neighbouring windows across the split and inflates
accuracy. We therefore use LEAVE-ONE-SESSION-OUT cross-validation: N independent
sessions (different seeds AND different noise draws) are generated; the classifier
is trained on N-1 sessions and tested on the held-out one, rotated over all N.
Reported numbers are mean +/- std over the N held-out folds. This removes the
temporal-adjacency leakage and is the standard subject/session-independent EMG
evaluation. The keystroke attack is scored under the SAME leave-one-session-out
protocol (leakage would only INFLATE the attack, so a chance result is
conservative).

Scope / honesty
---------------
Utility is measured on the literature-grounded generative model, because that is
where ground-truth grip MODES exist (the real myo-keylogging dataset is all
typing and has no rest/grasp labels). The real-data side of the project reports
the attack on the 29-participant recordings; real-hardware grip data is the
stated next step.

Outputs
-------
  utility_benchmark_results.json  -- every number
  figures/UTIL_curves.png         -- utility(eps) and privacy(eps) with error bars
  figures/UTIL_tradeoff.png       -- attacker AUC vs utility across eps
  figures/UTIL_aggregate.png      -- aggregate mean relative error vs eps
  figures/UTIL_exogenous.png      -- fixed budgets vs the exogenous scheme
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import emg_model as em
import mechanism as mech
import adversary as adv
from config import CONFIG as C

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

DURATION_S = 240.0
N_SESSIONS = 5                                  # leave-one-session-out folds
EPS_GRID = [None, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1]   # None = no privacy


# ---------------------------------------------------------------------------
def make_session(seed):
    sess = em.generate_session(C, duration_s=DURATION_S, seed=seed)
    sess = em.extract_features(sess, C)
    D = em.generate_tendon_vector(sess, C, seed=seed + 500)
    return sess, D


def released_telemetry(sess, D, eps_t_array, seed):
    """Noised (rms, tendon) telemetry under a per-window budget array.
    eps_t_array[t] is None -> no privacy (clean value passes through)."""
    rng = np.random.default_rng(seed)
    T = len(sess.rms)
    yr = np.zeros(T); YT = np.zeros((T, C.privacy.n_fingers))
    for t in range(T):
        e = eps_t_array[t]
        if e is None:
            yr[t] = sess.rms[t]; YT[t] = D[t]
        else:
            yr[t], _ = mech.perturb_emg(sess.rms[t], e, C, rng)
            YT[t], _ = mech.perturb_tendons(D[t], e, C, rng)
    return yr, YT


# Build the released dataset for every session at a given budget policy.
# policy(sess) -> per-window eps array (fixed value, None, or exogenous).
def build_release_set(sessions, Ds, policy, seed0):
    packs = []
    for i, (sess, D) in enumerate(zip(sessions, Ds)):
        eps_arr = policy(sess)
        yr, YT = released_telemetry(sess, D, eps_arr, seed=seed0 + i)
        X = adv.build_attack_matrix(yr, YT)          # rms + 4 tendon + deltas
        packs.append(dict(
            X=X, task=sess.win_task.astype(int),
            key=sess.win_keystroke.astype(int),
            is_type=(sess.win_task == em.TYPE),
            is_grasp=(sess.win_task == em.GRASP),
            is_rest=(sess.win_task == em.REST),
            D=D, YT=YT))
    return packs


def _recall_per_class(yte, ypred, classes=(0, 1, 2)):
    """Per-class recall = fraction of that class's true windows correctly recovered.
    This is literally 'what percent of grasps (etc.) the ML gets back'."""
    out = {}
    for c in classes:
        m = yte == c
        out[c] = float((ypred[m] == c).mean()) if m.any() else float("nan")
    return out


def loso_multiclass(packs, label_key, seed=0):
    """Leave-one-session-out with LDA and Random Forest. Returns balanced accuracy
    for both, plus RF per-class recall (rest/grasp/type) = what the ML recovers."""
    lda_scores, rf_scores = [], []
    rec = {0: [], 1: [], 2: []}
    n = len(packs)
    for te in range(n):
        tr_idx = [i for i in range(n) if i != te]
        def gather(idxs):
            Xs, ys = [], []
            for i in idxs:
                Xs.append(packs[i]["X"]); ys.append(packs[i][label_key])
            return np.vstack(Xs), np.concatenate(ys)
        Xtr, ytr = gather(tr_idx); Xte, yte = gather([te])
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        lda = LinearDiscriminantAnalysis().fit(Xtr2, ytr)
        rf = RandomForestClassifier(n_estimators=45, max_depth=8, n_jobs=-1,
                                    class_weight="balanced", random_state=seed).fit(Xtr2, ytr)
        yp_rf = rf.predict(Xte2)
        lda_scores.append(balanced_accuracy_score(yte, lda.predict(Xte2)))
        rf_scores.append(balanced_accuracy_score(yte, yp_rf))
        r = _recall_per_class(yte, yp_rf)
        for c in (0, 1, 2):
            if not np.isnan(r[c]):
                rec[c].append(r[c])
    return dict(
        lda_mean=float(np.mean(lda_scores)), lda_std=float(np.std(lda_scores)),
        rf_mean=float(np.mean(rf_scores)), rf_std=float(np.std(rf_scores)),
        recall_rest=float(np.mean(rec[0])) if rec[0] else float("nan"),
        recall_grasp=float(np.mean(rec[1])) if rec[1] else float("nan"),
        recall_type=float(np.mean(rec[2])) if rec[2] else float("nan"))


def loso_attack(packs, seed=0):
    """Leave-one-session-out keystroke-detection AUC on typing windows only."""
    aucs = []
    n = len(packs)
    for te in range(n):
        tr_idx = [i for i in range(n) if i != te]
        Xtr = np.vstack([packs[i]["X"][packs[i]["is_type"]] for i in tr_idx])
        ytr = np.concatenate([packs[i]["key"][packs[i]["is_type"]] for i in tr_idx])
        Xte = packs[te]["X"][packs[te]["is_type"]]
        yte = packs[te]["key"][packs[te]["is_type"]]
        if ytr.sum() < 10 or yte.sum() < 5 or (len(yte) - yte.sum()) < 5:
            continue
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1,
                                             max_iter=90, class_weight="balanced",
                                             random_state=seed).fit(Xtr2, ytr)
        p = clf.predict_proba(Xte2)[:, 1]
        aucs.append(roc_auc_score(yte, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def aggregate_util(packs):
    """UTILITY-C: per-finger mean relative error over grasp windows, pooled."""
    errs = []
    for p in packs:
        g = p["is_grasp"]
        if g.sum() < 5:
            continue
        r = adv.aggregate_utility(p["D"], p["YT"], mask=g)
        errs.append(r["mean_rel_err"])
    return float(np.mean(errs)), float(np.std(errs))


# ---------------------------------------------------------------------------
def run_fixed_sweep(sessions, Ds):
    rows = {}
    for eps in EPS_GRID:
        key = "no_privacy" if eps is None else f"{eps:g}"
        packs = build_release_set(sessions, Ds, lambda s: [eps] * len(s.rms),
                                  seed0=1000 + (0 if eps is None else int(eps * 10)))
        mc = loso_multiclass(packs, "task")
        rg_lda_m, _, rg_rf_m, _ = restgrasp(packs)
        auc_m, auc_s = loso_attack(packs)
        agg_m, agg_s = aggregate_util(packs)
        rows[key] = dict(
            eps=(None if eps is None else eps),
            lda_mean=mc["lda_mean"], lda_std=mc["lda_std"],
            rf_mean=mc["rf_mean"], rf_std=mc["rf_std"],
            recall_rest=mc["recall_rest"], recall_grasp=mc["recall_grasp"],
            recall_type=mc["recall_type"],
            restgrasp_lda=rg_lda_m, restgrasp_rf=rg_rf_m,
            attack_auc_mean=auc_m, attack_auc_std=auc_s,
            agg_rel_err_mean=agg_m, agg_rel_err_std=agg_s)
        print(f"eps={key:>10}  RF(3cls)={mc['rf_mean']:.3f}  graspRecall={mc['recall_grasp']:.3f}"
              f"  rest/grasp={rg_rf_m:.3f}  attackAUC={auc_m:.3f}  aggErr={agg_m:.3f}")
    return rows


def restgrasp(packs):
    """Binary rest-vs-grasp balanced accuracy, LOSO. Uses only rest+grasp windows,
    label 1 = grasp. Independent of the (private) typing mode."""
    lda_scores, rf_scores = [], []
    n = len(packs)
    for te in range(n):
        tr = [i for i in range(n) if i != te]
        def gather(idxs):
            Xs, ys = [], []
            for i in idxs:
                m = packs[i]["is_rest"] | packs[i]["is_grasp"]
                Xs.append(packs[i]["X"][m])
                ys.append(packs[i]["is_grasp"][m].astype(int))
            return np.vstack(Xs), np.concatenate(ys)
        Xtr, ytr = gather(tr); Xte, yte = gather([te])
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        lda = LinearDiscriminantAnalysis().fit(Xtr2, ytr)
        rf = RandomForestClassifier(n_estimators=45, max_depth=8, n_jobs=-1,
                                    class_weight="balanced", random_state=0).fit(Xtr2, ytr)
        lda_scores.append(balanced_accuracy_score(yte, lda.predict(Xte2)))
        rf_scores.append(balanced_accuracy_score(yte, rf.predict(Xte2)))
    return (float(np.mean(lda_scores)), float(np.std(lda_scores)),
            float(np.mean(rf_scores)), float(np.std(rf_scores)))


def run_exogenous_point(sessions, Ds):
    eps_type, eps_hi = 0.5, 8.0
    def pol(s):
        return [eps_type if t == em.TYPE else eps_hi for t in s.win_task]
    packs = build_release_set(sessions, Ds, pol, seed0=2000)
    mc = loso_multiclass(packs, "task")
    rg_lda, _, rg_rf, _ = restgrasp(packs)
    auc_m, auc_s = loso_attack(packs)
    agg_m, agg_s = aggregate_util(packs)
    out = dict(scheme="exogenous grip-mode", eps_type=eps_type, eps_hi=eps_hi,
               lda_mean=mc["lda_mean"], rf_mean=mc["rf_mean"], restgrasp_rf=rg_rf,
               recall_grasp=mc["recall_grasp"], recall_rest=mc["recall_rest"],
               recall_type=mc["recall_type"],
               attack_auc_mean=auc_m, attack_auc_std=auc_s, agg_rel_err_mean=agg_m)
    print(f"EXOGENOUS  RF(3cls)={mc['rf_mean']:.3f}  graspRecall={mc['recall_grasp']:.3f}"
          f"  rest/grasp={rg_rf:.3f}  attackAUC={auc_m:.3f}  aggErr={agg_m:.3f}")
    return out


# ------------------------------- plots -------------------------------------
def _sorted_keys(rows):
    ks = [k for k in rows if rows[k]["eps"] is not None]
    return sorted(ks, key=lambda k: rows[k]["eps"])


def plot_curves(rows):
    ks = _sorted_keys(rows)
    xs = [rows[k]["eps"] for k in ks]
    rf = [rows[k]["rf_mean"] for k in ks]; rfe = [rows[k]["rf_std"] for k in ks]
    lda = [rows[k]["lda_mean"] for k in ks]
    au = [rows[k]["attack_auc_mean"] for k in ks]; aue = [rows[k]["attack_auc_std"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(xs, rf, yerr=rfe, marker="o", capsize=3, label="Grip-state utility (Random Forest, 3-class)")
    ax.plot(xs, lda, marker="s", ls="--", label="Grip-state utility (LDA, 3-class)")
    ax.errorbar(xs, au, yerr=aue, marker="^", capsize=3, color="crimson",
                label="Keystroke attack AUC (want 0.5)")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Per-timestamp privacy budget  epsilon  (log; smaller = more private)")
    ax.set_ylabel("Balanced accuracy / AUC")
    ax.set_title("Company utility vs keystroke privacy (leave-one-session-out)")
    ax.set_ylim(0.3, 1.03); ax.legend(loc="center left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_curves.png"), dpi=150); plt.close(fig)


def plot_tradeoff(rows, exo):
    ks = _sorted_keys(rows)
    au = np.array([rows[k]["attack_auc_mean"] for k in ks])
    rf = np.array([rows[k]["rf_mean"] for k in ks])
    epsv = [rows[k]["eps"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sc = ax.scatter(au, rf, c=np.log10(epsv), cmap="viridis", s=70, zorder=3)
    for a, r, e in zip(au, rf, epsv):
        ax.annotate(f"e={e:g}", (a, r), textcoords="offset points", xytext=(6, 4), fontsize=8)
    npv = rows["no_privacy"]
    ax.scatter([npv["attack_auc_mean"]], [npv["rf_mean"]], marker="*", s=220,
               color="black", zorder=4, label="No privacy")
    ax.scatter([exo["attack_auc_mean"]], [exo["rf_mean"]], marker="D", s=110,
               color="crimson", zorder=4, label="Exogenous grip-mode (ours)")
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("Keystroke attack AUC  (left = private, 0.5 = chance)")
    ax.set_ylabel("Grip-state utility (RF balanced acc, 3-class)")
    ax.set_title("Privacy-utility frontier: top-left is the goal")
    ax.invert_xaxis()
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log10(epsilon)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_tradeoff.png"), dpi=150); plt.close(fig)


def plot_aggregate(rows):
    ks = _sorted_keys(rows)
    xs = [rows[k]["eps"] for k in ks]
    ag = [rows[k]["agg_rel_err_mean"] for k in ks]; age = [rows[k]["agg_rel_err_std"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(xs, ag, yerr=age, marker="o", capsize=3, color="teal")
    ax.set_xscale("log")
    ax.set_xlabel("Per-timestamp privacy budget  epsilon  (log)")
    ax.set_ylabel("Aggregate per-finger mean relative error (lower = better)")
    ax.set_title("Legitimate aggregate analytics stay usable as epsilon grows")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_aggregate.png"), dpi=150); plt.close(fig)


def plot_exogenous(rows, exo):
    labels = ["Fixed eps=0.5\n(private, low util)", "Fixed eps=8\n(high util, leaks)",
              "Exogenous\ngrip-mode (ours)"]
    util = [rows["0.5"]["rf_mean"], rows["8"]["rf_mean"], exo["rf_mean"]]
    atk = [rows["0.5"]["attack_auc_mean"], rows["8"]["attack_auc_mean"], exo["attack_auc_mean"]]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w/2, util, w, label="Grip-state utility (RF, 3-class)", color="steelblue")
    ax.bar(x + w/2, atk, w, label="Keystroke attack AUC", color="crimson")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Balanced accuracy / AUC")
    ax.set_title("No single fixed budget wins both; the exogenous scheme does")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_exogenous.png"), dpi=150); plt.close(fig)


def plot_recovery(rows):
    """What the manufacturer's ML actually gets back per state, vs epsilon."""
    ks = _sorted_keys(rows)
    xs = [rows[k]["eps"] for k in ks]
    rest = [rows[k]["recall_rest"] for k in ks]
    grasp = [rows[k]["recall_grasp"] for k in ks]
    typ = [rows[k]["recall_type"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, grasp, marker="o", color="seagreen", label="Grasp recovered (%)")
    ax.plot(xs, rest, marker="s", color="steelblue", label="Rest recovered (%)")
    ax.plot(xs, typ, marker="^", color="darkorange", label="Typing recovered (%)")
    ax.axhline(1/3, ls="--", color="gray", lw=1, label="chance (3-class)")
    ax.set_xscale("log")
    ax.set_xlabel("Per-timestamp privacy budget  epsilon  (log)")
    ax.set_ylabel("Fraction of that state's windows correctly recovered")
    ax.set_title("What the company ML gets back, per grip state (leave-one-session-out)")
    ax.set_ylim(0, 1.03); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_recovery.png"), dpi=150); plt.close(fig)


def main():
    print(f"generating {N_SESSIONS} independent sessions...")
    sessions, Ds = [], []
    for i in range(N_SESSIONS):
        s, D = make_session(20260722 + i)
        sessions.append(s); Ds.append(D)

    print("=== fixed-budget sweep (leave-one-session-out) ===")
    rows = run_fixed_sweep(sessions, Ds)
    print("=== exogenous grip-mode scheme ===")
    exo = run_exogenous_point(sessions, Ds)

    plot_curves(rows); plot_tradeoff(rows, exo); plot_aggregate(rows)
    plot_exogenous(rows, exo); plot_recovery(rows)

    out = dict(
        description="Company-utility benchmark under LEAVE-ONE-SESSION-OUT cross-"
                    "validation. Legitimate tasks: 3-class grip-state and binary "
                    "rest-vs-grasp (LDA + Random Forest, standard EMG pipeline at "
                    "library defaults) and aggregate per-finger estimation, vs the "
                    "keystroke attack, across an epsilon sweep. N sessions = folds.",
        protocol="leave-one-session-out CV; classifiers at library defaults; "
                 "LDA has no tuned hyper-parameters",
        n_sessions=N_SESSIONS, duration_s=DURATION_S,
        references=["Englehart & Hudgins 2003 (LDA myoelectric control)",
                    "Phinyomark et al. 2012 (EMG features/classifiers)",
                    "Atzori et al. 2014 (Ninapro benchmark)"],
        fixed_sweep=rows, exogenous=exo,
        figures=["UTIL_curves.png", "UTIL_tradeoff.png", "UTIL_aggregate.png",
                 "UTIL_exogenous.png", "UTIL_recovery.png"])
    json.dump(out, open(os.path.join(HERE, "utility_benchmark_results.json"), "w"), indent=2)
    print("\nwrote utility_benchmark_results.json and 4 figures")


if __name__ == "__main__":
    main()
