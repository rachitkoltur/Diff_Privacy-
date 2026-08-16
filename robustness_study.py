"""
robustness_study.py
===================
Gold-standard consistency test. Every headline claim is re-tested across many
independent simulated sessions (different seeds = different EMG realisations and
keystroke patterns) and across several privacy budgets, and reported as a
SUCCESS RATE: the fraction of runs in which the claim holds. A claim that only
holds on one lucky seed is not a result; a claim that holds on almost every seed
is.

Claims tested per run:
  C1  side-channel exists          : no-privacy typing attack AUC > 0.75
  C2  endogenous adaptivity FAILS  : DP-released context detection AUC < 0.55
  C3  fixed low-budget defends     : fixed eps=0.5 typing attack AUC < 0.58
  C4  exogenous-adaptive private   : exo-adaptive typing attack AUC < 0.58
  C5  exogenous-adaptive useful    : exo-adaptive grasp RMSE <= 1.5x fixed-high RMSE
  C6  exogenous-adaptive DOMINATES : private like fixed-low AND useful like fixed-high

Output: robustness_results.json with per-condition means, sds, and success rates.
"""
from __future__ import annotations
import json, os, math
import numpy as np
from sklearn.metrics import roc_auc_score

import emg_model as em
import mechanism as mech
import adversary as adv
from config import CONFIG as C

HERE = os.path.dirname(os.path.abspath(__file__))
F_TEL = 2


def one_session(seed, duration_s):
    sess = em.generate_session(C, duration_s=duration_s, seed=seed)
    sess = em.extract_features(sess, C)
    D = em.generate_tendon_vector(sess, C, seed=seed + 500)
    return sess, D


def telem(sess, D, eps_arr, seed):
    rng = np.random.default_rng(seed)
    T = len(eps_arr); yr = np.zeros(T); YT = np.zeros((T, C.privacy.n_fingers))
    for t in range(T):
        yr[t], _ = mech.perturb_emg(sess.rms[t], eps_arr[t], C, rng)
        YT[t], _ = mech.perturb_tendons(D[t], eps_arr[t], C, rng)
    return yr, YT


def typing_attack_auc(sess, yr, YT):
    typ = sess.win_task == em.TYPE
    X = adv.build_attack_matrix(yr, YT)
    return adv.run_keystroke_attack(X[typ], sess.win_keystroke[typ], seed=0)["auc"]


def grasp_rmse(sess, D, YT):
    gr = sess.win_task == em.GRASP
    return float(np.sqrt(np.mean((YT[gr] - D[gr]) ** 2)))


def endogenous_context_auc(sess):
    R = mech.risk_score(sess.hfv, sess.lfa, C)
    typ = (sess.win_task == em.TYPE).astype(int)
    Rsm = mech.trailing_mean(R, C.privacy.ctx_smooth)
    b_ctx = (F_TEL * (C.privacy.alpha + C.privacy.beta) / C.privacy.ctx_smooth) / C.privacy.eps_ctx
    aucs = []
    for s in range(5):
        rng = np.random.default_rng(9000 + s)
        out = np.empty_like(Rsm); last = 0.0
        for t in range(len(Rsm)):
            if t % C.privacy.ctx_smooth == 0:
                u = rng.uniform(-0.5, 0.5)
                last = Rsm[t] - b_ctx * np.sign(u) * np.log1p(-2 * abs(u))
            out[t] = last
        aucs.append(roc_auc_score(typ, out))
    return float(np.mean(aucs))


def run(n_seeds=15, duration_s=250.0, eps_type=0.5, eps_hi=8.0):
    rows = []
    for i in range(n_seeds):
        seed = C.seed + 1000 * (i + 1)
        sess, D = one_session(seed, duration_s)
        mode = sess.win_task
        eps_low = np.full(len(mode), eps_type)
        eps_high = np.full(len(mode), eps_hi)
        eps_exo = np.where(mode == em.TYPE, eps_type, eps_hi).astype(float)

        nopriv = typing_attack_auc(sess, sess.rms.astype(float),
                                   D.astype(float))  # raw telemetry attack
        ctx_auc = endogenous_context_auc(sess)
        yl, YL = telem(sess, D, eps_low, seed + 1)
        yh, YH = telem(sess, D, eps_high, seed + 2)
        ye, YE = telem(sess, D, eps_exo, seed + 3)
        low_auc = typing_attack_auc(sess, yl, YL)
        high_auc = typing_attack_auc(sess, yh, YH)
        exo_auc = typing_attack_auc(sess, ye, YE)
        high_rmse = grasp_rmse(sess, D, YH)
        exo_rmse = grasp_rmse(sess, D, YE)
        low_rmse = grasp_rmse(sess, D, YL)
        rows.append(dict(seed=seed, nopriv_auc=nopriv, ctx_auc=ctx_auc,
                         low_auc=low_auc, high_auc=high_auc, exo_auc=exo_auc,
                         low_rmse=low_rmse, high_rmse=high_rmse, exo_rmse=exo_rmse))

    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "seed"}
    def frac(mask): return round(100.0 * float(np.mean(mask)), 1)
    checks = {
        "C1 side-channel exists (nopriv AUC>0.75)": frac(A["nopriv_auc"] > 0.75),
        "C2 endogenous adaptivity fails (ctx AUC<0.55)": frac(A["ctx_auc"] < 0.55),
        "C3 fixed-low defends (AUC<0.58)": frac(A["low_auc"] < 0.58),
        "C4 exo-adaptive private (AUC<0.58)": frac(A["exo_auc"] < 0.58),
        "C5 exo-adaptive useful (RMSE<=1.5x fixed-high)": frac(A["exo_rmse"] <= 1.5 * A["high_rmse"]),
        "C6 exo-adaptive dominates (private like low AND useful like high)":
            frac((A["exo_auc"] <= A["low_auc"] + 0.05) & (A["exo_rmse"] <= 1.5 * A["high_rmse"])),
    }
    summary = {k: dict(mean=round(float(A[k].mean()), 3), sd=round(float(A[k].std()), 3),
                       lo=round(float(A[k].min()), 3), hi=round(float(A[k].max()), 3))
               for k in A}
    out = dict(n_seeds=n_seeds, duration_s=duration_s, eps_type=eps_type, eps_hi=eps_hi,
               per_metric=summary, success_rates_pct=checks, per_seed=rows)
    return out


def main():
    all_out = {"conditions": []}
    # main condition + budget-variation conditions (robustness across eps)
    for et, eh in [(0.5, 8.0), (1.0, 8.0), (0.5, 4.0)]:
        cond = run(n_seeds=15, duration_s=250.0, eps_type=et, eps_hi=eh)
        all_out["conditions"].append(cond)
        print(f"\n=== eps_type={et}, eps_hi={eh}  (n=15 seeds) ===")
        for k, v in cond["success_rates_pct"].items():
            print(f"  {v:5.1f}%  {k}")
    with open(os.path.join(HERE, "robustness_results.json"), "w") as f:
        json.dump(all_out, f, indent=2, default=float)
    print("\n-> robustness_results.json")


if __name__ == "__main__":
    main()
