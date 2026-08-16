"""
adversary.py
============
The honest-but-curious MANUFACTURER. Sees only the noisy BLE telemetry stream
and tries to infer private keystrokes (the Gazzari et al. 2021 threat model).
We also implement the LEGITIMATE cloud analytics the manufacturer is entitled to
(coarse aggregate estimation) and a leakage test for the adaptive budget.

Attack model
------------
Given a short sliding window of telemetry features (noisy EMG RMS + noisy tendon
vector + simple deltas), predict whether a keystroke occurred in that timestep.
We use a gradient-boosted / logistic classifier and report balanced accuracy and
ROC-AUC vs. ground truth. AUC -> 0.5 means the side channel is destroyed.

Utility model
-------------
The manufacturer's *legitimate* use (predictive maintenance, grip optimization)
needs COARSE AGGREGATES, e.g. the mean tendon displacement per finger over a
session. Because Laplace noise is zero-mean, the sample mean of noisy telemetry
is an UNBIASED estimator of the true aggregate; we measure its relative error.
This is the formal content of the draft's claim that "statistical properties
remain intact" for ML while individual keystrokes are hidden.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def build_attack_matrix(y_emg, Y_tendon, context=1):
    """Feature matrix for the adversary: each row = telemetry at t plus local
    temporal context (deltas to neighbours), which is what a real attacker uses."""
    T = len(y_emg)
    feats = [y_emg.reshape(-1, 1), Y_tendon]
    # first differences (movement onsets are the keystroke signature)
    d_emg = np.diff(y_emg, prepend=y_emg[0]).reshape(-1, 1)
    d_ten = np.diff(Y_tendon, axis=0, prepend=Y_tendon[:1])
    feats += [d_emg, d_ten]
    X = np.hstack(feats)
    return X


def run_keystroke_attack(X, labels, seed=0, model="gb"):
    """Train/test the keystroke-detection attack. Returns balanced acc + AUC."""
    if labels.sum() < 10 or (len(labels) - labels.sum()) < 10:
        return dict(bal_acc=0.5, auc=0.5, n_pos=int(labels.sum()))
    Xtr, Xte, ytr, yte = train_test_split(
        X, labels, test_size=0.4, random_state=seed, stratify=labels)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if model == "gb":
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.1, max_iter=200,
            class_weight="balanced", random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return dict(
        bal_acc=float(balanced_accuracy_score(yte, pred)),
        auc=float(roc_auc_score(yte, proba)),
        n_pos=int(labels.sum()),
    )


def aggregate_utility(D_true, Y_tendon, mask=None):
    """Manufacturer's legitimate coarse analytics: per-finger mean displacement,
    estimated over the GRASP phase (the predictive-maintenance / grip signal).
    Returns bias and relative error of the noisy-mean estimator vs. truth."""
    if mask is not None:
        D_true, Y_tendon = D_true[mask], Y_tendon[mask]
    true_mean = D_true.mean(axis=0)
    est_mean = Y_tendon.mean(axis=0)
    bias = est_mean - true_mean
    rel_err = np.abs(bias) / (np.abs(true_mean) + 1e-9)
    # per-window signal-reconstruction fidelity (predictive-maintenance signal):
    # this does NOT average away, so it reflects the noise the manufacturer's
    # per-grasp ML model actually sees.
    recon_rmse = float(np.sqrt(np.mean((Y_tendon - D_true) ** 2)))
    return dict(
        true_mean=true_mean, est_mean=est_mean,
        bias=bias, mean_abs_bias=float(np.mean(np.abs(bias))),
        mean_rel_err=float(np.mean(rel_err)),
        recon_rmse=recon_rmse,
    )


def empirical_epsilon(ya, yb, edges, min_count=200):
    """Estimate the max log density-ratio between two output sample sets ya, yb
    (the empirical epsilon of the mechanism for that adjacent input pair).

    We evaluate the ratio only in bins where BOTH samples have at least
    min_count observations, so that sparsely-populated tail bins (whose ratio is
    dominated by sampling noise, not by the mechanism) do not inflate the
    estimate. Densities are counts normalized by total * bin width."""
    ca, _ = np.histogram(ya, edges)
    cb, _ = np.histogram(yb, edges)
    w = np.diff(edges)
    da = ca / (ca.sum() * w)
    db = cb / (cb.sum() * w)
    m = (ca >= min_count) & (cb >= min_count)
    if not m.any():
        return 0.0, da, db
    return float(np.max(np.abs(np.log(da[m] / db[m])))), da, db


def budget_leakage_mi(eps_series, private_bit):
    """Mutual information (bits) between the released budget eps(t) and the
    private 'is-typing' bit. If eps(t) is computed from RAW current features it
    leaks; if computed from already-released (public) features it should not.
    Estimated by discretizing eps into quantile bins."""
    from sklearn.metrics import mutual_info_score
    eps = np.asarray(eps_series)
    bins = np.quantile(eps, np.linspace(0, 1, 9))
    bins[0] -= 1e-9
    eps_disc = np.digitize(eps, bins)
    mi_nats = mutual_info_score(private_bit, eps_disc)
    return float(mi_nats / np.log(2))  # convert nats -> bits
