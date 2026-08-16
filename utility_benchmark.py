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
      analytics. This is the standard sEMG activity-classification task.
  (B) per-finger tendon MEAN estimation over the grasp phase -- predictive
      maintenance / grip-force analytics (an aggregate statistic).
Neither of these needs to know WHICH key was pressed; that is the private thing
the attacker wants and the mechanism must hide.

Is there an off-the-shelf "gold-standard classifier" for this exact setup?
-------------------------------------------------------------------------
No single canonical model exists for "grip state from privatized prosthetic
telemetry". What IS standard is the sEMG classification PIPELINE: standard
amplitude/frequency features fed to Linear Discriminant Analysis (LDA) and a
tree ensemble (Random Forest) -- the workhorse classifiers in the EMG gesture-
recognition literature (Phinyomark et al. 2012; the Ninapro benchmark family,
Atzori et al. 2014). So we use those standard, defensible classifiers rather
than inventing a bespoke one, and we sweep the whole privacy-utility frontier.

What it measures, over a sweep of epsilon and many seeds
--------------------------------------------------------
For each per-timestamp budget eps in {no-privacy, 8, 4, 2, 1, 0.5, 0.25, 0.1}:
  * UTILITY-A: grip-state balanced accuracy of LDA and Random Forest on the
    NOISED telemetry (3-class rest/grasp/type), mean +/- std over seeds.
  * UTILITY-B: aggregate per-finger mean relative error (want LOW).
  * PRIVACY:   keystroke-detection attack AUC on typing windows (want ->0.5).
Then it adds the EXOGENOUS grip-mode adaptive point to show it beats any single
fixed budget: attack at chance AND grip-state/aggregate utility near the top.

Outputs
-------
  utility_benchmark_results.json  -- every number
  figures/UTIL_tradeoff.png       -- attacker AUC vs utility across eps
  figures/UTIL_curves.png         -- utility(eps) and privacy(eps) with error bars
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

import emg_model as em
import mechanism as mech
import adversary as adv
from config import CONFIG as C

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

DURATION_S = 300.0
SEEDS = list(range(8))                       # 8 independent sessions/noise draws
EPS_GRID = [None, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1]   # None = no privacy


# ---------------------------------------------------------------------------
def make_session(seed):
    sess = em.generate_session(C, duration_s=DURATION_S, seed=seed)
    sess = em.extract_features(sess, C)
    D = em.generate_tendon_vector(sess, C, seed=seed + 500)
    return sess, D


def released_telemetry(sess, D, eps_t_array, seed):
    """Return noised (rms, tendon) telemetry under a per-window budget array.
    eps_t_array[t] = None means no privacy (pass the clean value through)."""
    rng = np.random.default_rng(seed)
    T = len(sess.rms)
    yr = np.zeros(T)
    YT = np.zeros((T, C.privacy.n_fingers))
    for t in range(T):
        e = eps_t_array[t]
        if e is None:
            yr[t] = sess.rms[t]
            YT[t] = D[t]
        else:
            yr[t], _ = mech.perturb_emg(sess.rms[t], e, C, rng)
            YT[t], _ = mech.perturb_tendons(D[t], e, C, rng)
    return yr, YT


def grip_state_utility(yr, YT, labels_task, seed):
    """UTILITY-A: 3-class grip-state (rest/grasp/type) balanced accuracy from the
    noised telemetry, using the standard EMG classifiers LDA and Random Forest."""
    X = adv.build_attack_matrix(yr, YT)          # rms + 4 tendon + deltas (10 feats)
    y = labels_task.astype(int)
    # need every class present on both sides of the split
    if len(np.unique(y)) < 2:
        return dict(lda=float("nan"), rf=float("nan"))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight="balanced", random_state=seed).fit(Xtr, ytr)
    return dict(
        lda=float(balanced_accuracy_score(yte, lda.predict(Xte))),
        rf=float(balanced_accuracy_score(yte, rf.predict(Xte))),
    )


def keystroke_attack(yr, YT, sess, seed):
    """PRIVACY: keystroke-detection AUC on typing windows only."""
    typ = sess.win_task == em.TYPE
    X = adv.build_attack_matrix(yr, YT)
    res = adv.run_keystroke_attack(X[typ], sess.win_keystroke[typ], seed=seed, model="gb")
    return res["auc"]


def aggregate_util(D, YT, sess):
    """UTILITY-B: per-finger mean relative error over the grasp phase."""
    grasp = sess.win_task == em.GRASP
    if grasp.sum() < 5:
        return float("nan")
    r = adv.aggregate_utility(D, YT, mask=grasp)
    return r["mean_rel_err"]


# ---------------------------------------------------------------------------
def run_fixed_sweep():
    rows = {}
    for eps in EPS_GRID:
        key = "no_privacy" if eps is None else f"{eps:g}"
        lda_s, rf_s, auc_s, agg_s = [], [], [], []
        for sd in SEEDS:
            sess, D = make_session(20260722 + sd)
            eps_arr = [eps] * len(sess.rms)
            yr, YT = released_telemetry(sess, D, eps_arr, seed=1000 + sd)
            gs = grip_state_utility(yr, YT, sess.win_task, seed=sd)
            lda_s.append(gs["lda"]); rf_s.append(gs["rf"])
            auc_s.append(keystroke_attack(yr, YT, sess, seed=sd))
            agg_s.append(aggregate_util(D, YT, sess))
        rows[key] = dict(
            eps=(None if eps is None else eps),
            lda_mean=float(np.nanmean(lda_s)), lda_std=float(np.nanstd(lda_s)),
            rf_mean=float(np.nanmean(rf_s)), rf_std=float(np.nanstd(rf_s)),
            attack_auc_mean=float(np.nanmean(auc_s)), attack_auc_std=float(np.nanstd(auc_s)),
            agg_rel_err_mean=float(np.nanmean(agg_s)), agg_rel_err_std=float(np.nanstd(agg_s)),
        )
        print(f"eps={key:>10}  LDA={rows[key]['lda_mean']:.3f}  RF={rows[key]['rf_mean']:.3f}"
              f"  attackAUC={rows[key]['attack_auc_mean']:.3f}"
              f"  aggRelErr={rows[key]['agg_rel_err_mean']:.3f}")
    return rows


def run_exogenous_point():
    """The grip-mode adaptive scheme: eps_type in typing, eps_hi otherwise.
    Should give chance-level attack AND high utility at once."""
    eps_type, eps_hi = 0.5, 8.0
    lda_s, rf_s, auc_s, agg_s = [], [], [], []
    for sd in SEEDS:
        sess, D = make_session(20260722 + sd)
        eps_arr = [eps_type if t == em.TYPE else eps_hi for t in sess.win_task]
        yr, YT = released_telemetry(sess, D, eps_arr, seed=2000 + sd)
        gs = grip_state_utility(yr, YT, sess.win_task, seed=sd)
        lda_s.append(gs["lda"]); rf_s.append(gs["rf"])
        auc_s.append(keystroke_attack(yr, YT, sess, seed=sd))
        agg_s.append(aggregate_util(D, YT, sess))
    out = dict(scheme="exogenous grip-mode", eps_type=eps_type, eps_hi=eps_hi,
               lda_mean=float(np.nanmean(lda_s)), rf_mean=float(np.nanmean(rf_s)),
               attack_auc_mean=float(np.nanmean(auc_s)),
               agg_rel_err_mean=float(np.nanmean(agg_s)))
    print(f"EXOGENOUS  LDA={out['lda_mean']:.3f}  RF={out['rf_mean']:.3f}"
          f"  attackAUC={out['attack_auc_mean']:.3f}  aggRelErr={out['agg_rel_err_mean']:.3f}")
    return out


# ---------------------------------------------------------------------------
def epsvals(rows):
    return [rows[k]["eps"] for k in rows if rows[k]["eps"] is not None]


def plot_curves(rows, exo):
    ks = [k for k in rows if rows[k]["eps"] is not None]
    ks = sorted(ks, key=lambda k: rows[k]["eps"])
    xs = [rows[k]["eps"] for k in ks]
    rf = [rows[k]["rf_mean"] for k in ks]; rfe = [rows[k]["rf_std"] for k in ks]
    lda = [rows[k]["lda_mean"] for k in ks]; ldae = [rows[k]["lda_std"] for k in ks]
    au = [rows[k]["attack_auc_mean"] for k in ks]; aue = [rows[k]["attack_auc_std"] for k in ks]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(xs, rf, yerr=rfe, marker="o", capsize=3, label="Grip-state utility (Random Forest)")
    ax.errorbar(xs, lda, yerr=ldae, marker="s", capsize=3, label="Grip-state utility (LDA)")
    ax.errorbar(xs, au, yerr=aue, marker="^", capsize=3, color="crimson",
                label="Keystroke attack AUC (want 0.5)")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Per-timestamp privacy budget  epsilon  (log scale; smaller = more private)")
    ax.set_ylabel("Balanced accuracy / AUC")
    ax.set_title("Company utility vs keystroke privacy across the budget sweep")
    ax.set_ylim(0.35, 1.03)
    ax.legend(loc="center left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_curves.png"), dpi=150)
    plt.close(fig)


def plot_tradeoff(rows, exo):
    ks = [k for k in rows if rows[k]["eps"] is not None]
    au = np.array([rows[k]["attack_auc_mean"] for k in ks])
    rf = np.array([rows[k]["rf_mean"] for k in ks])
    epsv = [rows[k]["eps"] for k in ks]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sc = ax.scatter(au, rf, c=np.log10(epsv), cmap="viridis", s=70, zorder=3)
    for a, r, e in zip(au, rf, epsv):
        ax.annotate(f"e={e:g}", (a, r), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    # no-privacy reference
    npv = rows["no_privacy"]
    ax.scatter([npv["attack_auc_mean"]], [npv["rf_mean"]], marker="*", s=220,
               color="black", zorder=4, label="No privacy")
    # exogenous scheme
    ax.scatter([exo["attack_auc_mean"]], [exo["rf_mean"]], marker="D", s=110,
               color="crimson", zorder=4, label="Exogenous grip-mode (ours)")
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("Keystroke attack AUC  (left = private, 0.5 = chance)")
    ax.set_ylabel("Grip-state utility (Random Forest balanced acc)")
    ax.set_title("Privacy-utility frontier: top-left is the goal")
    ax.invert_xaxis()
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log10(epsilon)")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_tradeoff.png"), dpi=150)
    plt.close(fig)


def plot_aggregate(rows):
    ks = [k for k in rows if rows[k]["eps"] is not None]
    ks = sorted(ks, key=lambda k: rows[k]["eps"])
    xs = [rows[k]["eps"] for k in ks]
    ag = [rows[k]["agg_rel_err_mean"] for k in ks]
    age = [rows[k]["agg_rel_err_std"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(xs, ag, yerr=age, marker="o", capsize=3, color="teal")
    ax.set_xscale("log")
    ax.set_xlabel("Per-timestamp privacy budget  epsilon  (log scale)")
    ax.set_ylabel("Aggregate per-finger mean relative error (lower = better)")
    ax.set_title("Legitimate aggregate analytics stay usable as epsilon grows")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_aggregate.png"), dpi=150)
    plt.close(fig)


def plot_exogenous(rows, exo):
    labels = ["Fixed eps=0.5\n(private, low util)", "Fixed eps=8\n(high util, leaks)",
              "Exogenous\ngrip-mode (ours)"]
    util = [rows["0.5"]["rf_mean"], rows["8"]["rf_mean"], exo["rf_mean"]]
    atk = [rows["0.5"]["attack_auc_mean"], rows["8"]["attack_auc_mean"], exo["attack_auc_mean"]]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w/2, util, w, label="Grip-state utility (RF)", color="steelblue")
    ax.bar(x + w/2, atk, w, label="Keystroke attack AUC", color="crimson")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Balanced accuracy / AUC")
    ax.set_title("No single fixed budget wins both; the exogenous scheme does")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "UTIL_exogenous.png"), dpi=150)
    plt.close(fig)


def main():
    print("=== fixed-budget sweep (utility vs privacy) ===")
    rows = run_fixed_sweep()
    print("=== exogenous grip-mode scheme ===")
    exo = run_exogenous_point()

    plot_curves(rows, exo)
    plot_tradeoff(rows, exo)
    plot_aggregate(rows)
    plot_exogenous(rows, exo)

    out = dict(
        description="Company-utility benchmark: legitimate grip-state classifier "
                    "(LDA + Random Forest, standard EMG pipeline) and aggregate "
                    "estimation vs the keystroke attack, across an epsilon sweep, "
                    "8 seeds each.",
        seeds=len(SEEDS), duration_s=DURATION_S,
        fixed_sweep=rows, exogenous=exo,
        figures=["UTIL_curves.png", "UTIL_tradeoff.png",
                 "UTIL_aggregate.png", "UTIL_exogenous.png"],
    )
    json.dump(out, open(os.path.join(HERE, "utility_benchmark_results.json"), "w"), indent=2)
    print("\nwrote utility_benchmark_results.json and 4 figures")


if __name__ == "__main__":
    main()
