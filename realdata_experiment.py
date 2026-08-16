"""
realdata_experiment.py
======================
Run CA-LDP on the REAL myo-keylogging EMG dataset (Gazzari et al., 2021) and
reproduce the privacy-utility result of Figure 3 on real muscle recordings.

This script needs two things this sandbox did not have:
  1. the raw dataset (CSV recordings) from Zenodo, and
  2. the authors' `preprocess` package (from their repository) on the path.

Setup (run on a machine with internet):
    git clone https://github.com/seemoo-lab/myo-keylogging.git
    cd myo-keylogging
    wget https://zenodo.org/record/5594651/files/myo-keylogging-dataset.zip
    unzip myo-keylogging-dataset.zip           # creates train-data/ and test-data/
    pip install -e preprocess                   # installs their loader
    pip install numpy scipy scikit-learn matplotlib
    # then copy this file next to the sim package (config.py, mechanism.py, ...)
    python realdata_experiment.py --data test-data

What it does:
  * loads each recording with the authors' loader (binary press labels),
  * rescales the LFA/HFV bands to the recording's true sample rate (the Myo
    streams EMG at ~200 Hz, so the high band must sit below the 100 Hz Nyquist),
  * builds a per-200 ms telemetry feature and applies fixed / naive / sound
    CA-LDP at a sweep of budgets,
  * trains a keystroke-detection attacker with leave-one-participant-out
    grouping and reports balanced accuracy vs budget,
  * writes realdata_results.json and figures/RD_privacy.png.

The analysis half (features_from_emg, ca_ldp_sweep, run_attack) is unit-tested
by --selftest on synthetic arrays shaped like the loader output, so the code is
known to run before any download.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import CONFIG as C
import mechanism as mech

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (uses the authors' preprocess package; not exercised in selftest)
# ---------------------------------------------------------------------------
def load_real(data_path, max_tasks_per_pid=3, freq=200):
    """Directly read the myo-keylogging CSV recordings (no external package).

    Each task folder entry has record-N.tM.{left,right}.emg.csv (8 channels each,
    ~200 Hz, 8-bit) and record-N.tM.key.csv (keystroke press/release with
    timestamps in the same clock). We stack the two Myos into 16 channels and
    label each EMG sample 1 if a key PRESS occurs at it.

    Returns (recordings, freq); each recording is dict(emg=(T,16), key=(T,), pid).
    pid = participant (record) number, for leave-one-participant-out grouping."""
    import re, glob, os
    import pandas as pd
    files = os.listdir(data_path)
    tasks = sorted(set(m.group(1) for m in (re.match(r"(record-\d+\.t\d+)\.", f) for f in files) if m))
    per_pid = {}
    recs = []
    for task in tasks:
        pid = int(re.match(r"record-(\d+)\.", task).group(1))
        if per_pid.get(pid, 0) >= max_tasks_per_pid:
            continue
        try:
            R = pd.read_csv(os.path.join(data_path, task + ".right.emg.csv"))
            L = pd.read_csv(os.path.join(data_path, task + ".left.emg.csv"))
            K = pd.read_csv(os.path.join(data_path, task + ".key.csv"))
        except Exception:
            continue
        cols = [f"emg{i}" for i in range(8)]
        n = min(len(R), len(L))
        emg = np.hstack([R[cols].values[:n], L[cols].values[:n]]).astype(float)
        t = R["time"].values[:n]
        key = np.zeros(n, dtype=int)
        presses = K.loc[K["event"] == "press", "time"].values
        for pt in presses:
            j = int(np.searchsorted(t, pt))
            if 0 <= j < n:
                key[j] = 1
        recs.append(dict(emg=emg, key=key, pid=pid))
        per_pid[pid] = per_pid.get(pid, 0) + 1
    if not recs:
        raise RuntimeError("No recordings loaded from %s" % data_path)
    return recs, freq


# ---------------------------------------------------------------------------
# Feature extraction (bands rescaled to the true sample rate)
# ---------------------------------------------------------------------------
def _bands_for_fs(fs):
    nyq = fs / 2.0
    lo = (20.0, min(80.0, 0.5 * nyq))          # low-frequency amplitude band
    hi = (min(0.55 * nyq, nyq - 15), nyq - 5)  # high-frequency variance band
    return lo, hi


def features_from_emg(emg, fs, win_ms=200):
    """Per-window RMS telemetry feature, LFA and HFV, plus per-window keystroke
    label alignment indices. emg is (T, n_ch). Returns dict of arrays."""
    T, n_ch = emg.shape
    W = max(4, int(fs * win_ms / 1000))
    n_win = T // W
    lo_band, hi_band = _bands_for_fs(fs)
    sos_lo = butter(4, lo_band, btype="band", fs=fs, output="sos")
    sos_hi = butter(4, hi_band, btype="band", fs=fs, output="sos")
    # rectify + normalize per channel, then band-filter the ensemble mean
    x = np.abs(emg - emg.mean(0, keepdims=True))
    x = x / (x.std(0, keepdims=True) + 1e-9)
    ens = x.mean(1)  # ensemble EMG envelope
    lo = sosfiltfilt(sos_lo, ens)
    hi = sosfiltfilt(sos_hi, ens)
    rms = np.zeros(n_win); lfa = np.zeros(n_win); hfv = np.zeros(n_win)
    for i in range(n_win):
        a, b = i * W, (i + 1) * W
        rms[i] = np.sqrt(np.mean(ens[a:b] ** 2))
        lfa[i] = np.sqrt(np.mean(lo[a:b] ** 2))
        hfv[i] = np.var(hi[a:b])
    def norm(v):
        hiv = np.percentile(v, 99) + 1e-12
        return np.clip(v / hiv, 0, 1)
    return dict(rms=norm(rms), lfa=norm(lfa), hfv=norm(hfv), W=W, n_win=n_win)


def window_keystrokes(key, W, n_win):
    lab = np.zeros(n_win, dtype=int)
    for i in range(n_win):
        if key[i * W:(i + 1) * W].any():
            lab[i] = 1
    return lab


# ---------------------------------------------------------------------------
# CA-LDP budget schedules on real features
# ---------------------------------------------------------------------------
def epsilon_schedules(lfa, hfv):
    R = mech.risk_score(hfv, lfa, C)
    eps_naive = mech.epsilon_of_risk(R, C)
    eps_sound = mech.epsilon_of_risk(mech.trailing_mean(R, 15), C)
    return eps_naive, eps_sound


def scale_to_mean(eps, target):
    e = eps * (target / (eps.mean() + 1e-12))
    return np.clip(e, C.privacy.eps_min, C.privacy.eps_max)


def privatize(rms, eps, rng):
    b = C.privacy.delta_f_event / (C.privacy.frac_emg * eps)
    return rms + mech.laplace_inverse_transform(b, rng, size=len(rms))


# ---------------------------------------------------------------------------
# Attack (leave-one-participant-out)
# ---------------------------------------------------------------------------
def run_attack(X, labels, groups, seed=0, return_oof=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    if labels.sum() < 20 or (len(labels) - labels.sum()) < 20:
        return dict(bal_acc=0.5, auc=0.5)
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, max(2, n_groups)))
    accs, aucs = [], []
    oof_y, oof_p = [], []
    for tr, te in gkf.split(X, labels, groups):
        if labels[tr].sum() < 5 or labels[te].sum() < 5:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                             class_weight="balanced", random_state=seed)
        clf.fit(X[tr], labels[tr])
        p = clf.predict_proba(X[te])[:, 1]
        accs.append(balanced_accuracy_score(labels[te], (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(labels[te], p))
        oof_y.append(labels[te]); oof_p.append(p)
    if not accs:
        return dict(bal_acc=0.5, auc=0.5)
    out = dict(bal_acc=float(np.mean(accs)), auc=float(np.mean(aucs)))
    if return_oof and oof_y:
        out["oof_y"] = np.concatenate(oof_y)
        out["oof_p"] = np.concatenate(oof_p)
    return out


def per_participant_loo(feats, seed=0):
    """Leave-one-participant-out: hold each participant out, train the no-privacy
    attacker on the rest, report that participant's AUC. Gives the inter-subject
    spread reviewers ask for (does the channel exist for everyone, or one person?)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([build_matrix(f["rms"]) for f in feats])
    y = np.concatenate([f["label"] for f in feats])
    g = np.concatenate([np.full(len(f["label"]), f["pid"]) for f in feats])
    per = {}
    for pid in np.unique(g):
        tr, te = (g != pid), (g == pid)
        if y[te].sum() < 5 or y[tr].sum() < 5:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                             class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        per[int(pid)] = float(roc_auc_score(y[te], p))
    vals = np.array(list(per.values()))
    return dict(per_pid=per, mean=float(vals.mean()), sd=float(vals.std()),
                lo=float(vals.min()), hi=float(vals.max()), n=int(len(vals)))


def roc_curve_points(base, n_pts=60):
    """Downsample the pooled out-of-fold ROC to n_pts for a compact figure/JSON."""
    from sklearn.metrics import roc_curve
    if "oof_y" not in base:
        return None
    fpr, tpr, _ = roc_curve(base["oof_y"], base["oof_p"])
    idx = np.linspace(0, len(fpr) - 1, min(n_pts, len(fpr))).astype(int)
    return dict(fpr=[float(x) for x in fpr[idx]], tpr=[float(x) for x in tpr[idx]])


def build_matrix(rms):
    d = np.diff(rms, prepend=rms[0])
    return np.column_stack([rms, d])


def ca_ldp_sweep(feats_per_rec, budgets=(0.5, 1.0, 2.0, 4.0), seed=20260722):
    """feats_per_rec: list of dicts with keys rms,lfa,hfv,label,pid."""
    rng = np.random.default_rng(seed)
    schemes = {"fixed": None, "naive": None, "sound": None}
    results = {k: [] for k in schemes}
    # no-privacy baseline
    Xr = np.concatenate([build_matrix(f["rms"]) for f in feats_per_rec])
    yr = np.concatenate([f["label"] for f in feats_per_rec])
    gr = np.concatenate([np.full(len(f["label"]), f["pid"]) for f in feats_per_rec])
    base = run_attack(Xr, yr, gr, return_oof=True)
    for m in budgets:
        for scheme in schemes:
            Xs = []
            for f in feats_per_rec:
                if scheme == "fixed":
                    eps = np.full(len(f["rms"]), m)
                else:
                    en, es = epsilon_schedules(f["lfa"], f["hfv"])
                    eps = scale_to_mean(en if scheme == "naive" else es, m)
                Xs.append(build_matrix(privatize(f["rms"], eps, rng)))
            Xs = np.concatenate(Xs)
            r = run_attack(Xs, yr, gr)
            results[scheme].append(dict(mean_eps=m, **r))
    return base, results


# ---------------------------------------------------------------------------
def run_real(data_path):
    recs, fs = load_real(data_path)
    feats = []
    for rc in recs:
        fe = features_from_emg(rc["emg"], fs)
        lab = window_keystrokes(rc["key"], fe["W"], fe["n_win"])
        feats.append(dict(rms=fe["rms"], lfa=fe["lfa"], hfv=fe["hfv"],
                          label=lab, pid=rc["pid"]))
    base, results = ca_ldp_sweep(feats)
    roc = roc_curve_points(base)
    inter = per_participant_loo(feats)
    base_clean = {k: v for k, v in base.items() if k not in ("oof_y", "oof_p")}
    n_pid = len(set(f["pid"] for f in feats))
    out = dict(fs=fs, n_recordings=len(recs), n_participants=n_pid,
               no_privacy=base_clean, inter_subject=inter, roc=roc, sweep=results)
    json.dump(out, open(os.path.join(HERE, "realdata_results.json"), "w"), indent=1, default=float)
    _plot(base_clean, results, roc, inter)
    print("no-privacy attack AUC %.3f" % base["auc"])
    print("inter-subject AUC %.3f +/- %.3f (range %.3f-%.3f, n=%d)"
          % (inter["mean"], inter["sd"], inter["lo"], inter["hi"], inter["n"]))
    print("wrote realdata_results.json and figures/RD_privacy.png + RD_roc.png")
    return out


def _plot(base, results, roc=None, inter=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"fixed": "#57606a", "naive": "#d1242f", "sound": "#1a7f37"}

    # Figure 1: privacy-utility sweep on real data
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for s in results:
        m = [p["mean_eps"] for p in results[s]]
        a = [p["auc"] for p in results[s]]
        ax.plot(m, a, "-o", color=col[s], label=s + (" (ours)" if s == "sound" else ""))
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.axhline(base["auc"], color="#1f6feb", ls=":", lw=1.0, label="no privacy %.2f" % base["auc"])
    ax.set_xlabel("mean budget E[epsilon]"); ax.set_ylabel("keystroke attack AUC (real data)")
    ax.set_title("CA-LDP on the myo-keylogging dataset"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "RD_privacy.png"), dpi=160)
    plt.close(fig)

    # Figure 2: ROC of the no-privacy attack + per-participant spread
    if roc is not None:
        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.9))
        a1.plot(roc["fpr"], roc["tpr"], color="#1f6feb", lw=1.8,
                label="no-privacy attack (AUC %.2f)" % base["auc"])
        a1.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8, label="chance")
        a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
        a1.set_title("Keystroke attack ROC (real EMG)"); a1.legend(fontsize=8, loc="lower right")
        if inter is not None:
            vals = sorted(inter["per_pid"].values())
            a2.hist(vals, bins=8, color="#8250df", edgecolor="white")
            a2.axvline(inter["mean"], color="#1a7f37", lw=1.5,
                       label="mean %.2f +/- %.2f" % (inter["mean"], inter["sd"]))
            a2.axvline(0.5, color="gray", ls="--", lw=0.8, label="chance")
            a2.set_xlabel("held-out participant AUC"); a2.set_ylabel("participants")
            a2.set_title("Inter-subject spread (leave-one-out)"); a2.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, "RD_roc.png"), dpi=160)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Self-test: exercise the analysis on synthetic arrays shaped like the loader
# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    fs = 200
    feats = []
    for pid in range(4):
        T = fs * 60
        emg = rng.standard_normal((T, 16))
        key = np.zeros(T, dtype=int)
        onsets = rng.integers(0, T, size=300)
        for o in onsets:
            emg[o:o + 8] += rng.standard_normal((min(8, T - o), 16)) * 3  # keystroke burst
            key[o] = 1
        fe = features_from_emg(emg, fs)
        lab = window_keystrokes(key, fe["W"], fe["n_win"])
        feats.append(dict(rms=fe["rms"], lfa=fe["lfa"], hfv=fe["hfv"], label=lab, pid=pid))
    base, results = ca_ldp_sweep(feats, budgets=(0.5, 1.0, 2.0))
    assert 0.4 <= base["auc"] <= 1.0
    for s in results:
        for p in results[s]:
            assert 0.3 <= p["auc"] <= 1.0
    print("selftest OK. no-privacy AUC %.3f; fixed@1.0 AUC %.3f; sound@1.0 AUC %.3f"
          % (base["auc"],
             [p for p in results["fixed"] if p["mean_eps"] == 1.0][0]["auc"],
             [p for p in results["sound"] if p["mean_eps"] == 1.0][0]["auc"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to test-data folder (Zenodo dataset)")
    ap.add_argument("--selftest", action="store_true", help="run the analysis self-test only")
    args = ap.parse_args()
    if args.selftest or not args.data:
        selftest()
    else:
        run_real(args.data)
