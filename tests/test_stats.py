import math

import pytest
from scipy.stats import binomtest

from correctness_gate import stats


@pytest.mark.parametrize("b,c", [(3, 1), (10, 2), (7, 7), (0, 5), (25, 5)])
def test_mcnemar_matches_scipy(b, c):
    ours = stats.mcnemar_exact(b, c)
    ref = binomtest(min(b, c), b + c, 0.5).pvalue
    assert math.isclose(ours, ref, rel_tol=1e-12)


def test_mcnemar_no_discordant_is_1():
    assert stats.mcnemar_exact(0, 0) == 1.0


def test_binomial_se_textbook_value():
    assert math.isclose(stats.binomial_se(0.6, 100), 0.04899, abs_tol=1e-4)


def test_paired_counts():
    base = [True, True, False, False, True]
    cand = [True, False, True, False, True]
    pc = stats.paired_counts(base, cand)
    assert (pc.b, pc.c, pc.both_right, pc.both_wrong) == (1, 1, 2, 1)


def test_required_n_agrees_with_simulation():
    # 10% of items regress, 5% improve: net drop 0.05, discordant rate 0.15
    p01, p10 = 0.10, 0.05
    n = stats.required_n_paired(p01 + p10, p01 - p10)
    power = stats.power_by_simulation(n, p01, p10, n_sim=2000)
    assert 0.70 < power < 0.90  # target is 0.80


def test_mde_roundtrip():
    mde = stats.mde_at_n(200, 0.15)
    assert stats.required_n_paired(0.15, mde) <= 201


def test_bootstrap_ci_brackets_delta():
    base = [True] * 80 + [False] * 120
    cand = [True] * 70 + [False] * 130
    delta, lo, hi = stats.paired_delta_ci(base, cand, n_boot=2000)
    assert lo <= delta <= hi
    assert math.isclose(delta, -0.05, abs_tol=1e-9)
