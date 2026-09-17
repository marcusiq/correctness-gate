"""The statistical core of the gate.
Why paired: baseline and candidate score the SAME items, so the evidence is
per-item flips. McNemar's test conditions on the discordant items (flipped
in either direction) and asks how surprising the observed split is if flips
were direction-neutral coin tosses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_Z = NormalDist().inv_cdf


def binomial_se(p: float, n: int) -> float:
    """SE of a proportion: sqrt(p*(1-p)/n). At p=0.6, n=100 this is 0.049,
    which is why a fixed 2-point threshold on 100 items fires on noise."""
    return math.sqrt(p * (1.0 - p) / n)


@dataclass(frozen=True)
class PairedCounts:
    n: int
    both_right: int
    both_wrong: int
    b: int  # baseline right, candidate wrong: regressions
    c: int  # baseline wrong, candidate right: improvements


def paired_counts(base: list[bool], cand: list[bool]) -> PairedCounts:
    if len(base) != len(cand):
        raise ValueError("paired test requires identical item sets")
    b = sum(x and not y for x, y in zip(base, cand))
    c = sum(y and not x for x, y in zip(base, cand))
    both_right = sum(x and y for x, y in zip(base, cand))
    return PairedCounts(len(base), both_right,
                        len(base) - both_right - b - c, b, c)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value.

    Under 'no real change', each discordant item flips either way with
    probability 1/2. Condition on m = b + c discordant items; the observed
    split's surprise is a Binomial(m, 1/2) tail. Doubling the smaller tail
    equals the minimum-likelihood two-sided method here because the null
    distribution is symmetric at p = 1/2.
    """
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2.0 ** m
    return min(1.0, 2.0 * tail)


def paired_delta_ci(base, cand, n_boot: int = 10_000, seed: int = 1234,
                    level: float = 0.95):
    """Percentile bootstrap confidence interval for
    (candidate accuracy - baseline accuracy).

    Resamples ITEMS with replacement and recomputes the delta on each
    resample, so item difficulty stays paired inside every resample.
    Returns (delta, low, high).
    """
    base = np.asarray(base, dtype=float)
    cand = np.asarray(cand, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=(n_boot, len(base)))
    deltas = cand[idx].mean(axis=1) - base[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(cand.mean() - base.mean()), float(lo), float(hi)


def required_n_paired(p_disc: float, delta: float, alpha: float = 0.05,
                      power: float = 0.80) -> int:
    """Items needed for McNemar to detect a net accuracy change `delta`
    at the given alpha and power, when a fraction `p_disc` of items are
    discordant. Connor's (1987) approximation.

    Read it as: sensitivity is bought with discordant items. If only 10%
    of items ever flip, detecting a 3-point true drop needs
    required_n_paired(0.10, 0.03) items, which is ~950, not 100.
    """
    if not 0.0 < delta < math.sqrt(p_disc):
        raise ValueError("need 0 < delta < sqrt(p_disc)")
    za, zb = _Z(1 - alpha / 2), _Z(power)
    n = (za * math.sqrt(p_disc)
         + zb * math.sqrt(p_disc - delta ** 2)) ** 2 / delta ** 2
    return math.ceil(n)


def required_n_unpaired(p: float, delta: float, alpha: float = 0.05,
                        power: float = 0.80) -> int:
    """Two-proportion (unpaired) sample size per arm, the TensorRT-LLM-style
    design. Kept for the README's paired-vs-unpaired comparison; assumes
    both configurations sit near accuracy p."""
    za, zb = _Z(1 - alpha / 2), _Z(power)
    return math.ceil((za + zb) ** 2 * 2.0 * p * (1.0 - p) / delta ** 2)


def mde_at_n(n: int, p_disc: float, alpha: float = 0.05,
             power: float = 0.80) -> float:
    """Invert required_n_paired: the smallest true drop detectable with n
    items. The gate PRINTS this on every run so a pass can never silently
    mean 'underpowered'. Bisection, because the closed form inverts
    awkwardly."""
    lo, hi = 1e-4, math.sqrt(p_disc) - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_n_paired(p_disc, mid, alpha, power) > n:
            lo = mid
        else:
            hi = mid
    return hi


def power_by_simulation(n: int, p01: float, p10: float, alpha: float = 0.05,
                        n_sim: int = 5000, seed: int = 1234) -> float:
    """Cross-check the closed form by direct simulation: each of n items
    regresses with probability p01, improves with probability p10; run the
    exact test; count rejections. If simulation and formula disagree,
    trust the simulation."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        u = rng.random(n)
        b = int((u < p01).sum())
        c = int(((u >= p01) & (u < p01 + p10)).sum())
        if mcnemar_exact(b, c) < alpha:
            hits += 1
    return hits / n_sim
