"""
mechanism.py
============
The Context-Aware Local Differential Privacy (CA-LDP) mechanism:

  1. Continuous Privacy Risk Score   R(t) = alpha*HFV - beta*LFA
  2. Adaptive per-timestamp budget    eps(t) = clip(eps_max * exp(-k R), eps_min, eps_max)
  3. Laplace perturbation of the EMG RMS feature (Delta f = 0.1)
  4. Laplace perturbation of the 4-D tendon vector (Delta d = 1.0, budget split)
  5. w-event streaming composition (bounded loss over any w-window)
  6. SOUND adaptive budget: eps(t) derived from ALREADY-RELEASED features so the
     budget itself does not leak (fixes the data-dependent-eps problem).

SAMPLER NOTE (corrected). The float inverse-transform sampler below,
x = -b*sgn(u)*ln(1-2|u|) with u a float uniform, is the construction Mironov
(CCS 2012) proved is NOT eps-DP in finite precision (extended to Gaussian/timing
by Jin, McMurtry, Rubinstein & Ohrimenko, IEEE S&P 2022). It is kept ONLY as the
fast Monte-Carlo approximation for the utility/attack experiments and as the
"vulnerable baseline" in the sampler experiment. The DEPLOYED, privacy-critical
release uses the discrete-Laplace mechanism in safe_sampler.py, which meets the
eps-DP log-ratio bound exactly (verified in honest_addendum.py).
"""
from __future__ import annotations
import numpy as np
from config import Config
import safe_sampler as ss


# ---------------------------------------------------------------------------
# Laplace sampling via inverse transform  (FAST but finite-precision UNSAFE)
# ---------------------------------------------------------------------------
def laplace_inverse_transform(scale_b, rng, size=None):
    """Draw Lap(0, b) via inverse CDF: x = -b*sgn(u)*ln(1-2|u|), u~Uniform(-0.5,0.5).

    WARNING: float inverse transform is the Mironov(2012)-vulnerable construction;
    do NOT use on-device. Use safe_release_scalar for the deployed mechanism."""
    u = rng.uniform(-0.5, 0.5, size=size)
    return -scale_b * np.sign(u) * np.log1p(-2.0 * np.abs(u))


def safe_release_scalar(value, sensitivity, eps, gamma=0.01):
    """Deployed release: discrete-Laplace mechanism (exact rational sampling).
    eps-DP holds in finite precision. gamma is the output grid resolution."""
    return ss.release_scalar(float(value), float(sensitivity), float(eps), float(gamma))


# ---------------------------------------------------------------------------
# Risk score and adaptive budget
# ---------------------------------------------------------------------------
def risk_score(hfv, lfa, cfg: Config):
    p = cfg.privacy
    return p.alpha * hfv - p.beta * lfa


def trailing_mean(x, win):
    """Causal trailing moving average (uses only past+current samples).
    Used to build the SOUND budget: a bout-level risk estimate that tracks
    activity MODE (typing vs grasping) but is too smooth to resolve individual
    keystrokes, so the resulting noise level cannot pinpoint a keystroke."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        a = max(0, i - win + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def release_context(true_risk, cfg: Config, rng):
    """Release a coarse, smoothed activity context under DP so that the adaptive
    budget can depend on it without leaking the current sample.

    We first low-pass the risk (trailing mean over ctx_smooth windows), then add
    Laplace noise calibrated to the smoothed risk's event-sensitivity delta_ctx
    with budget eps_ctx. The RETURNED value is a DP output; any budget derived
    from it is post-processing and therefore sound (Theorem 1 in the paper)."""
    p = cfg.privacy
    ctx = trailing_mean(true_risk, p.ctx_smooth)
    b = p.delta_ctx / p.eps_ctx
    # release once per ctx_release_period steps and HOLD between releases, so the
    # slowly-varying context is not re-released (and re-paid for) at 5 Hz.
    out = np.empty_like(ctx)
    last = ctx[0] + laplace_inverse_transform(b, rng)
    for t in range(len(ctx)):
        if t % p.ctx_release_period == 0:
            last = ctx[t] + laplace_inverse_transform(b, rng)
        out[t] = last
    return out


def epsilon_of_risk(R, cfg: Config):
    """eps(t) = clip(eps_max * exp(-k R), eps_min, eps_max).

    Clamp to eps_max (draft) prevents infinite bounds when R<0 (strong grasp);
    the eps_min floor (our refinement) prevents b = Delta f / eps -> infinity
    (unbounded noise variance) when R is large (intense typing)."""
    p = cfg.privacy
    eps = p.eps_max * np.exp(-p.k * np.asarray(R))
    return np.clip(eps, p.eps_min, p.eps_max)


# ---------------------------------------------------------------------------
# RMS sensitivity (verified empirically in experiments.py)
# ---------------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x)))


# ---------------------------------------------------------------------------
# Per-timestamp perturbation
# ---------------------------------------------------------------------------
def perturb_emg(x_rms, eps_t, cfg: Config, rng):
    """y = x_rms + Lap(Delta f / eps_emg),  eps_emg = frac_emg * eps(t)."""
    p = cfg.privacy
    eps_emg = p.frac_emg * eps_t
    b = p.delta_f_event / eps_emg   # event-level (keystroke) sensitivity
    return x_rms + laplace_inverse_transform(b, rng), b


def perturb_tendons(D_vec, eps_t, cfg: Config, rng, clamp=False):
    """Perturb the 4-D tendon vector.

    Sequential composition: the tendon sub-vector receives frac_tendon_vector*eps(t),
    split uniformly across the 4 fingers -> each finger eps_i = frac*eps(t)/4.
    Scale b_i = Delta d / eps_i.  (With frac=0.5 this is b_i = 1.0/(eps/8)=8/eps.)

    clamp=False  -> no-clamp transmission (raw uncapped floats).  Clamping the
    NOISY output is post-processing and preserves eps-DP either way; we keep the
    raw values only to avoid estimator BIAS (see experiments)."""
    p = cfg.privacy
    eps_vec = p.frac_tendon_vector * eps_t
    eps_i = eps_vec / p.n_fingers
    b_i = p.delta_d / eps_i
    y = D_vec + laplace_inverse_transform(b_i, rng, size=len(D_vec))
    if clamp:
        y = np.clip(y, 0.0, 1.0)
    return y, b_i


# ---------------------------------------------------------------------------
# w-event streaming budget manager (Kellaris et al., 2014)
# ---------------------------------------------------------------------------
class WEventBudget:
    """Enforces sum of per-timestamp budgets over any sliding window of length w
    to be <= eps_W. Two allocation schemes:

      * 'uniform' : eps_i = eps_W / w                (Kellaris baseline)
      * 'adaptive': request eps(t) from the risk engine, but PROJECT it onto the
                    feasible set so the running window sum never exceeds eps_W.
                    Budget saved during low-risk (grasp/rest) windows is what
                    lets us spend very little (add lots of noise) during typing.
    """

    def __init__(self, cfg: Config):
        self.w = cfg.privacy.w_window
        self.eps_W = cfg.privacy.eps_W
        self.eps_min = cfg.privacy.eps_min
        self.history = []  # spent budgets

    def window_sum(self):
        return float(np.sum(self.history[-self.w:]))

    def spend_uniform(self):
        e = self.eps_W / self.w
        self.history.append(e)
        return e

    def spend_adaptive(self, eps_requested):
        # budget already spent in the trailing (w-1) window that overlaps this step
        used = float(np.sum(self.history[-(self.w - 1):])) if self.history else 0.0
        remaining = max(0.0, self.eps_W - used)
        e = min(eps_requested, remaining)  # if remaining==0 -> suppress (e=0)
        self.history.append(e)
        return e

    def suppressed_fraction(self):
        h = np.array(self.history)
        return float(np.mean(h <= 0.0)) if len(h) else 0.0
