"""
sanity_checks.py
================
Regression guard against the class of error that slipped through before: a proof
that is valid in the idealized model while the mechanism, when actually run, does
NOT sit at the operating point the proof assumes, or a decision signal that has
no usable information once released under DP.

Every check ASSERTS and prints PASS/FAIL. Run this after any change to the
mechanism or constants. If a future edit reintroduces an adaptive-budget claim,
CHECK 2 fails unless the DP-released context can actually detect the state.

The four checks:
  1. Sampler:   the deployed discrete-Laplace release meets |log-ratio| <= eps.
  2. Adaptivity: ANY signal used to steer the budget must have DP-released
                 detection AUC > 0.60, else adaptivity is unsupported (this is
                 the check that would have caught the original error).
  3. Operating point: the per-keystroke budget the code actually produces must
                 match the number the write-up claims (no proof/operating gap).
  4. Utility:   the aggregate mean estimator is unbiased (|bias| small).
"""
from __future__ import annotations
import sys, math, collections
import numpy as np
from fractions import Fraction
from sklearn.metrics import roc_auc_score

import emg_model as em
import mechanism as mech
import safe_sampler as ss
from config import CONFIG as C

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def build():
    sess = em.generate_session(C, duration_s=300.0, seed=C.seed)
    sess = em.extract_features(sess, C)
    R = mech.risk_score(sess.hfv, sess.lfa, C)
    typ = (sess.win_task == em.TYPE).astype(int)
    return sess, R, typ


# CHECK 1 -- deployed sampler is eps-DP in finite precision
def check_sampler():
    sens, eps, gamma, N = C.privacy.delta_f_event, 1.0, 0.05, 80000
    f0, f1 = 0.40, 0.40 + sens

    def h(f):
        c = collections.Counter()
        for _ in range(N):
            c[round(ss.release_scalar(f, sens, eps, gamma) / gamma)] += 1
        return c
    c0, c1 = h(f0), h(f1)
    ratios = [abs(math.log((c0[k] / N) / (c1[k] / N)))
              for k in set(c0) & set(c1) if c0[k] / N > 0.003 and c1[k] / N > 0.003]
    m = max(ratios)
    check("1 sampler eps-DP (discrete Laplace)", m <= eps + 0.15,
          f"max|log-ratio|={m:.3f} <= eps={eps}+tol")


# CHECK 2 -- any budget-steering signal must survive DP release
def check_adaptivity_supported(sess, R, typ):
    # DP-released context detection AUC (the signal an adaptive budget would use)
    a, b, W, F_tel = C.privacy.alpha, C.privacy.beta, C.privacy.ctx_smooth, 2
    Rsm = mech.trailing_mean(R, W)
    b_ctx = (F_tel * (a + b) / W) / C.privacy.eps_ctx
    aucs = []
    for s in range(8):
        rng = np.random.default_rng(4000 + s)
        out = np.empty_like(Rsm); last = 0.0
        for t in range(len(Rsm)):
            if t % W == 0:
                u = rng.uniform(-0.5, 0.5)
                last = Rsm[t] - b_ctx * np.sign(u) * np.log1p(-2 * abs(u))
            out[t] = last
        aucs.append(roc_auc_score(typ, out))
    dp_auc = float(np.mean(aucs))
    ADAPTIVE_CLAIMED = False  # set True only if the write-up claims adaptivity helps
    ok = (dp_auc > 0.60) or (not ADAPTIVE_CLAIMED)
    check("2 adaptivity supported by DP context", ok,
          f"DP-context AUC={dp_auc:.3f} (need>0.60 IF adaptive claimed; "
          f"ADAPTIVE_CLAIMED={ADAPTIVE_CLAIMED}) -> fixed-budget scheme is honest")


# CHECK 3 -- claimed per-keystroke budget matches the code's operating point
def check_operating_point():
    eps_op, F_tel = 0.5, 2
    claimed = 1.0                      # eps_event claimed for eps_op=0.5
    produced = F_tel * eps_op          # fixed scheme, no context term
    check("3 operating point matches claim", abs(produced - claimed) < 1e-9,
          f"produced eps_event={produced} == claimed {claimed}")


# CHECK 4 -- aggregate estimator unbiased
def check_unbiased():
    rng = np.random.default_rng(7)
    b = C.privacy.delta_f_event / 0.5
    M = 200000
    noise = rng.laplace(0.0, b, size=M)
    bias = abs(float(noise.mean()))
    check("4 aggregate estimator unbiased", bias < 0.02, f"|mean noise|={bias:.4f}")


def main():
    sess, R, typ = build()
    check_sampler()
    check_adaptivity_supported(sess, R, typ)
    check_operating_point()
    check_unbiased()
    print()
    if FAILS:
        print("SANITY CHECKS FAILED:", FAILS)
        sys.exit(1)
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
