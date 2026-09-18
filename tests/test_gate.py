import pytest

from correctness_gate.config import GateConfig
from correctness_gate.gate import GateError, compare


def mk_run(correct, model="m", fingerprint="f", config_fingerprint="c"):
    return {
        "model": model,
        "fingerprint": fingerprint,
        "acc": sum(correct) / len(correct),
        "acc_norm": sum(correct) / len(correct),
        "config": {"backend": "test", "device": "cpu"},
        "config_fingerprint": config_fingerprint,
        "items": [{"qid": str(i), "correct": c, "correct_norm": c}
                  for i, c in enumerate(correct)],
    }


CFG = GateConfig(n_boot=500)


def test_fingerprint_mismatch_fails_closed():
    with pytest.raises(GateError):
        compare(mk_run([True] * 10), mk_run([True] * 10, fingerprint="g"), CFG)


def test_missing_key_fails_closed():
    bad = mk_run([True] * 10)
    del bad["acc"]
    with pytest.raises(GateError):
        compare(bad, mk_run([True] * 10), CFG)


def test_identical_runs_pass():
    r = mk_run([True] * 150 + [False] * 50)
    v = compare(r, r, CFG)
    assert v.status == "PASS" and v.p_value == 1.0


def test_clear_regression_fails():
    base = [True] * 150 + [False] * 50
    cand = [True] * 125 + [False] * 25 + [False] * 50  # 25 broke, 0 fixed
    v = compare(mk_run(base), mk_run(cand), CFG)
    assert v.status == "FAIL" and v.counts.b == 25


def test_catastrophic_hits_hard_floor():
    v = compare(mk_run([True] * 100), mk_run([False] * 100), CFG)
    assert v.status == "FAIL"
    assert any("catastrophic" in r for r in v.reasons)


def test_small_run_reports_underpowered():
    r = mk_run([True] * 30 + [False] * 20)
    v = compare(r, r, CFG)
    assert any("underpowered" in x for x in v.reasons)


def mk_run2(correct, correct_norm, model="m", fingerprint="f",
            config_fingerprint="c"):
    """A run whose two metrics disagree, which is the normal case on real data:
    43 of 200 items disagreed on the 0.5B reference run."""
    return {
        "model": model,
        "fingerprint": fingerprint,
        "acc": sum(correct) / len(correct),
        "acc_norm": sum(correct_norm) / len(correct_norm),
        "config": {"backend": "test", "device": "cpu"},
        "config_fingerprint": config_fingerprint,
        "items": [{"qid": str(i), "correct": c, "correct_norm": cn}
                  for i, (c, cn) in enumerate(zip(correct, correct_norm))],
    }


def test_regression_on_acc_only_still_fails():
    """The whole point of gating on both: acc_norm is untouched, acc collapses."""
    stable = [True] * 150 + [False] * 50
    base = mk_run2(stable, stable)
    broken_acc = [True] * 125 + [False] * 75
    cand = mk_run2(broken_acc, stable)
    v = compare(base, cand, CFG)
    assert v.status == "FAIL"
    assert v.per_metric["acc"].status == "FAIL"
    assert v.per_metric["acc_norm"].status == "PASS"
    assert any(r.startswith("acc:") for r in v.reasons)


def test_regression_on_acc_norm_only_still_fails():
    stable = [True] * 150 + [False] * 50
    broken_norm = [True] * 125 + [False] * 75
    v = compare(mk_run2(stable, stable), mk_run2(stable, broken_norm), CFG)
    assert v.status == "FAIL"
    assert v.per_metric["acc_norm"].status == "FAIL"
    assert v.per_metric["acc"].status == "PASS"


def test_alpha_is_split_across_metrics():
    r = mk_run([True] * 150 + [False] * 50)
    v = compare(r, r, CFG)
    assert v.alpha_per_metric == pytest.approx(CFG.alpha / len(CFG.metrics))
    assert all(mv.alpha == v.alpha_per_metric for mv in v.per_metric.values())


def test_unknown_metric_fails_closed():
    cfg = GateConfig(n_boot=500, metrics=["acc", "perplexity"])
    with pytest.raises(GateError):
        compare(mk_run([True] * 10), mk_run([True] * 10), cfg)


def test_no_metrics_fails_closed():
    cfg = GateConfig(n_boot=500, metrics=[])
    with pytest.raises(GateError):
        compare(mk_run([True] * 10), mk_run([True] * 10), cfg)


def test_missing_config_fingerprint_fails_closed():
    """A legacy run record cannot say what produced it, so it cannot be gated."""
    legacy = mk_run([True] * 10)
    del legacy["config_fingerprint"]
    with pytest.raises(GateError, match="config_fingerprint"):
        compare(legacy, mk_run([True] * 10), CFG)


def test_config_mismatch_fails_closed():
    """CPU vs CUDA on the same weights moved acc_norm by 0.015 on this machine,
    so a config change and a quality change are not separable."""
    base = mk_run([True] * 10)
    cand = mk_run([True] * 10, config_fingerprint="d")
    cand["config"] = {"backend": "test", "device": "cuda"}
    with pytest.raises(GateError, match="execution configs differ"):
        compare(base, cand, CFG)
