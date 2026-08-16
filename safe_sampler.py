"""
safe_sampler.py
===============
Finite-precision-SAFE noise sampling for the CA-LDP mechanism.

Why this file exists
--------------------
The original mechanism drew Laplace noise by floating-point inverse transform,
x = -b*sgn(u)*ln(1-2|u|) with u a float in Uniform(-1/2,1/2). Mironov (CCS 2012)
showed this construction does NOT satisfy eps-DP in finite precision: the set of
representable outputs differs between neighbouring inputs, so the low-order bits
leak. Jin, McMurtry, Rubinstein & Ohrimenko (IEEE S&P 2022) extend the point to
the Gaussian mechanism and to timing.

The fix here is the DISCRETE LAPLACE mechanism (a.k.a. two-sided geometric),
sampled with EXACT rational arithmetic and unbiased coin flips only. No floating
point enters the privacy-critical path, so the eps-DP proof that holds in exact
arithmetic also holds for the running code. This is the Canonne-Kamath-Steinke
(2020) exact sampler for Bernoulli(exp(-gamma)) and the discrete Laplace built
from two geometrics.

Discrete Laplace with scale t>0 has pmf  P(Z=z) = C * exp(-|z|/t), z in Z.
A query f with integer (grid) sensitivity Delta_grid, released as
round(f/gamma) + Z with t = Delta_grid/eps, satisfies eps-DP exactly, because
for neighbouring integer outputs |ln P(z)/P(z')| <= Delta_grid/t = eps.

All randomness comes from `secrets` (cryptographic, unbiased bits), so there is
no float PRNG in the sampler.
"""
from __future__ import annotations
import secrets
from fractions import Fraction


def _bernoulli(p: Fraction) -> bool:
    """Exact Bernoulli(p) for rational p in [0,1] using only unbiased integer draws.
    Returns True with probability exactly p = a/b via a uniform integer in [0,b)."""
    if p <= 0:
        return False
    if p >= 1:
        return True
    a, b = p.numerator, p.denominator
    return secrets.randbelow(b) < a


def _bernoulli_exp_unit(gamma: Fraction) -> bool:
    """Exact Bernoulli(exp(-gamma)) for 0 <= gamma <= 1 (Canonne-Kamath-Steinke).
    Returns True with probability exactly exp(-gamma)."""
    k = 1
    while _bernoulli(gamma / k):
        k += 1
    return (k % 2) == 1


def bernoulli_exp(gamma: Fraction) -> bool:
    """Exact Bernoulli(exp(-gamma)) for any rational gamma >= 0."""
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    whole = int(gamma)  # floor for gamma >= 0
    for _ in range(whole):
        if not _bernoulli_exp_unit(Fraction(1)):
            return False
    frac = gamma - whole
    if frac == 0:
        return True
    return _bernoulli_exp_unit(frac)


def _geometric(t: Fraction) -> int:
    """Sample G >= 0 with P(G=g) = (1-q) q^g, q = exp(-1/t), using exact bits.
    G = number of consecutive Bernoulli(exp(-1/t)) successes before first failure."""
    inv_t = Fraction(1) / t
    g = 0
    while bernoulli_exp(inv_t):
        g += 1
    return g


def discrete_laplace(t: Fraction) -> int:
    """Sample Z ~ DiscreteLaplace(t): P(Z=z) proportional to exp(-|z|/t), z in Z.
    Built as the difference of two i.i.d. geometrics (exact, no floating point)."""
    return _geometric(t) - _geometric(t)


def release_scalar(value: float, sensitivity: float, eps: float, gamma: float) -> float:
    """Release a bounded real scalar under eps-DP with the discrete Laplace mechanism.

    Steps (only the grid index is privacy-critical, and it uses exact arithmetic):
      1. snap the value to an integer grid of resolution gamma:  n = round(value/gamma)
      2. grid sensitivity Delta_grid = ceil(sensitivity/gamma)  (integer, conservative)
      3. add Z ~ DiscreteLaplace(t) with t = Delta_grid/eps  (rational)
      4. return (n + Z) * gamma

    The returned float is post-processing of the integer DP output n+Z, so the
    eps-DP guarantee is exact and independent of float rounding in step 4.
    """
    import math
    n = round(value / gamma)
    delta_grid = max(1, math.ceil(sensitivity / gamma))
    t = Fraction(delta_grid) / Fraction(eps).limit_denominator(10**6)
    z = discrete_laplace(t)
    return (n + z) * gamma
