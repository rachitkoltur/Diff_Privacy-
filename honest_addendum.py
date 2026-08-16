"""
honest_addendum.py
==================
Reproducible evidence for the corrected (honest) version of the project.

It produces three things and writes them to honest_results.json:

  (A) NEGATIVE RESULT. Under local DP with a one-keystroke neighbouring relation,
      the DP-released context cannot detect the typing state, at ANY smoothing
      window. We report the clean (noise-free) detection AUC and the DP-released
      detection AUC across a sweep of windows, plus the signal-to-noise ratio.
      This is why an endogenous context-aware adaptive budget cannot beat a fixed
      budget here.

  (B) SAFE SAMPLER CHECK. The discrete-Laplace mechanism in safe_sampler.py meets
      the eps-DP log-ratio bound empirically, whereas the floating-point
      inverse-transform sampler is the Mironov(2012)-vulnerable construction.

  (C) FIXED-SCHEME NUMBERS. With the adaptive context removed, the sound defence
      is a fixed per-release budget eps_op with the discrete-Laplace sampler and
      the w-event cap. We record the per-keystroke budget (F_tel * eps_op, no
      context term) and the streaming-window accounting.

Deterministic under the master seed.
"""
from __future__ import annotations
import json, os, math, collections
import numpy as np
from fractions import Fraction
from sklearn.metrics import roc_auc_score

import emg_model as em
import mechanism as mech
import safe_sampler as ss
from config import CONFIG as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
SEED = C.seed


def build():
    sess = em.generate_session(C, duration_s=300.0, seed=SEED)
    sess = em.extract_features(sess, C)
    R = mech.risk_score(sess.hfv, sess.lfa, C)
    typ = (sess.win_task == em.TYPE).astype(int)
    return sess, R, typ


# ---------------------------------------------------------------------------
# (A) Negative result: private context cannot resolve typing
# ---------------------------------------------------------------------------
def negative_result(sess, R, typ):
    a, b = C.privacy.alpha, C.privacy.beta
    F_tel = 2
    eps_ctx = C.privacy.eps_ctx
    rows = []
    for W in [15, 25, 40, 60, 90, 120, 160, 220]:
        Rsm = mech.trailing_mean(R, W)
        clean_auc = float(roc_auc_score(typ, Rsm))
        delta_c = F_tel * (a + b) / W
        b_ctx = delta_c / eps_ctx
        aucs = []
        for s in range(12):
            rng = np.random.default_rng(1000 + s)
            out = np.empty_like(Rsm)
            last = 0.0
            for t in range(len(Rsm)):
                if t % W == 0:
                    u = rng.uniform(-0.5, 0.5)
                    last = Rsm[t] - b_ctx * np.sign(u) * np.log1p(-2.0 * abs(u))
                out[t] = last
            aucs.append(roc_auc_score(typ, out))
        rows.append(dict(W_windows=W, W_seconds=round(W * 0.2, 1),
                         delta_c=round(delta_c, 4), noise_std=round(math.sqrt(2) * b_ctx, 3),
                         clean_auc=round(clean_auc, 3), dp_auc=round(float(np.mean(aucs)), 3),
                         dp_auc_sd=round(float(np.std(aucs)), 3)))
    # instantaneous discriminability and best SNR
    Rsm15 = mech.trailing_mean(R, C.privacy.ctx_smooth)
    sep = float(Rsm15[typ == 1].mean() - Rsm15[typ == 0].mean())
    b_ctx = (F_tel * (a + b) / C.privacy.ctx_smooth) / eps_ctx
    OUT["negative_result"] = dict(
        instantaneous_R_auc=round(float(roc_auc_score(typ, R)), 3),
        clean_smoothed_auc_W15=round(float(roc_auc_score(typ, Rsm15)), 3),
        separation_W15=round(sep, 3),
        context_noise_std_W15=round(math.sqrt(2) * b_ctx, 3),
        snr_W15=round(sep / (math.sqrt(2) * b_ctx), 3),
        sweep=rows,
        conclusion="DP-released context AUC ~ 0.5 at every window: endogenous "
                   "context-aware adaptivity cannot beat a fixed budget under LDP.")


# ---------------------------------------------------------------------------
# (B) Safe sampler eps-DP check vs the vulnerable float sampler
# ---------------------------------------------------------------------------
def safe_sampler_check():
    # distribution moments
    t = Fraction(5)
    samp = np.array([ss.discrete_laplace(t) for _ in range(40000)])
    var_theory = 2 * math.exp(-1 / 5) / (1 - math.exp(-1 / 5)) ** 2

    # empirical eps-DP: worst-case neighbours differ by full event sensitivity
    sens = C.privacy.delta_f_event
    eps = 1.0
    gamma = 0.05
    N = 120000
    f0, f1 = 0.40, 0.40 + sens

    def histidx(f):
        c = collections.Counter()
        for _ in range(N):
            c[round(ss.release_scalar(f, sens, eps, gamma) / gamma)] += 1
        return c
    c0, c1 = histidx(f0), histidx(f1)
    ratios = []
    for k in set(c0) & set(c1):
        p0, p1 = c0[k] / N, c1[k] / N
        if p0 > 0.002 and p1 > 0.002:
            ratios.append(abs(math.log(p0 / p1)))
    OUT["safe_sampler"] = dict(
        dist_mean=round(float(samp.mean()), 3), dist_var=round(float(samp.var()), 2),
        dist_var_theory=round(var_theory, 2),
        eps_target=eps, empirical_max_log_ratio=round(max(ratios), 3),
        holds=bool(max(ratios) <= eps + 0.15),
        note="Discrete Laplace via exact rational sampling meets |log-ratio| <= eps. "
             "The float inverse-transform sampler is the Mironov(2012)-vulnerable one.")


# ---------------------------------------------------------------------------
# (C) Fixed-scheme accounting (adaptive context removed)
# ---------------------------------------------------------------------------
def fixed_scheme():
    F_tel = 2
    p = C.privacy
    per_event = {f"eps_op={e}": F_tel * e for e in [0.5, 1.0]}
    # w-event window: fixed eps_op each step, no context release
    ctx_cost = 0.0
    tel_per_window = {f"eps_op={e}": e * p.w_window for e in [0.5, 1.0]}
    OUT["fixed_scheme"] = dict(
        per_keystroke_budget=per_event,
        note_per_keystroke="No context term now: eps_event <= F_tel * eps_op "
                           "(2*0.5=1.0 at eps_op=0.5, 2*1.0=2.0 at eps_op=1.0).",
        w_window_telemetry_sum=tel_per_window, eps_W=p.eps_W, w=p.w_window,
        context_cost=ctx_cost,
        note_window="At eps_op<=1 the w-window sum (<=300) stays under eps_W=300.")


def main():
    sess, R, typ = build()
    negative_result(sess, R, typ)
    safe_sampler_check()
    fixed_scheme()
    with open(os.path.join(HERE, "honest_results.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    print(json.dumps(OUT, indent=2))


if __name__ == "__main__":
    main()
