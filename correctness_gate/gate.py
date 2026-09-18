"""Verdict logic: two per-item run files in, PASS or FAIL out.

Failure-closed by design: anything that prevents a valid comparison raises
GateError, and callers treat that as FAIL.

Every metric in cfg.metrics is tested and all of them must pass. acc and
acc_norm disagree on a meaningful fraction of items (43 of 200 on the 0.5B
reference run), so gating on one alone leaves the other free to regress
unwatched. Testing two hypotheses at alpha each would inflate the family-wise
false-positive rate to about 1 - (1 - alpha)^2, so alpha is divided across the
metrics (Bonferroni). The cost is a slightly larger minimum detectable effect
per metric, which every report prints, so the tradeoff stays visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import stats
from .config import GateConfig

# Metric name -> the per-item boolean field written by mcq.score_item.
METRIC_KEYS = {"acc": "correct", "acc_norm": "correct_norm"}


class GateError(Exception):
    """The runs cannot be compared. Callers must treat this as FAIL."""


@dataclass
class MetricVerdict:
    """The full statistical picture for one metric."""
    metric: str
    status: str            # "PASS" or "FAIL"
    reasons: list[str]
    delta: float           # candidate - baseline on this metric
    ci_low: float          # bootstrap confidence interval bounds
    ci_high: float
    p_value: float
    counts: stats.PairedCounts
    mde: float             # smallest drop this run could detect at target power
    alpha: float           # the per-metric alpha actually used
    broke: list[str] = field(default_factory=list)   # qids: right -> wrong
    fixed: list[str] = field(default_factory=list)   # qids: wrong -> right


@dataclass
class Verdict:
    status: str                            # FAIL if any metric failed
    reasons: list[str]                     # each prefixed with its metric
    per_metric: dict[str, MetricVerdict]
    alpha_per_metric: float

    @property
    def primary(self) -> MetricVerdict:
        """The first configured metric. Convenience for callers that print one
        number; anything making a decision should read per_metric."""
        return next(iter(self.per_metric.values()))

    # Delegating shims so a caller that wants a single headline number does not
    # have to reach into per_metric. They always report the primary metric.
    @property
    def delta(self) -> float:
        return self.primary.delta

    @property
    def p_value(self) -> float:
        return self.primary.p_value

    @property
    def counts(self) -> stats.PairedCounts:
        return self.primary.counts

    @property
    def mde(self) -> float:
        return self.primary.mde


def _compare_one(baseline: dict, candidate: dict, cfg: GateConfig,
                 metric: str, alpha: float) -> MetricVerdict:
    key = METRIC_KEYS[metric]
    base = [bool(it[key]) for it in baseline["items"]]
    cand = [bool(it[key]) for it in candidate["items"]]

    counts = stats.paired_counts(base, cand)
    p = stats.mcnemar_exact(counts.b, counts.c)
    delta, lo, hi = stats.paired_delta_ci(base, cand, cfg.n_boot, cfg.seed)
    p_disc = max((counts.b + counts.c) / counts.n, 1.0 / counts.n)
    mde = stats.mde_at_n(counts.n, p_disc, alpha, cfg.power)

    qids = [it.get("qid", str(i)) for i, it in enumerate(baseline["items"])]
    broke = [q for q, x, y in zip(qids, base, cand) if x and not y]
    fixed = [q for q, x, y in zip(qids, base, cand) if y and not x]

    status, reasons = "PASS", []
    if delta <= -cfg.hard_floor:
        status = "FAIL"
        reasons.append(f"catastrophic drop: {delta:+.3f} is past the hard "
                       f"floor of -{cfg.hard_floor}")
    elif p < alpha and delta < 0:
        status = "FAIL"
        reasons.append(f"significant regression: {counts.b} items broke vs "
                       f"{counts.c} fixed (p={p:.4f} < alpha={alpha:.4f})")
    elif p < alpha and delta > 0:
        reasons.append(f"significant improvement ({counts.c} fixed vs "
                       f"{counts.b} broke); consider promoting a new baseline "
                       "(a human decision, never automatic)")
    if mde > cfg.mde:
        reasons.append(
            f"underpowered: at n={counts.n} this run only detects drops >= "
            f"{mde:.3f}; the configured target is {cfg.mde:.3f} (needs ~"
            f"{stats.required_n_paired(p_disc, cfg.mde, alpha, cfg.power)}"
            " items at the observed discordant rate)")
    return MetricVerdict(metric, status, reasons, delta, lo, hi, p, counts,
                         mde, alpha, broke, fixed)


def compare(baseline: dict, candidate: dict, cfg: GateConfig) -> Verdict:
    for run, tag in ((baseline, "baseline"), (candidate, "candidate")):
        for key in ("items", "fingerprint", "acc", "acc_norm"):
            if key not in run:
                raise GateError(f"{tag} run is missing {key!r}; refusing to pass")
    if baseline["fingerprint"] != candidate["fingerprint"]:
        raise GateError(
            f"item-set fingerprints differ ({baseline['fingerprint']} vs "
            f"{candidate['fingerprint']}); runs on different item sets are "
            "not comparable")
    bcfg, ccfg = (baseline.get("config_fingerprint"),
                  candidate.get("config_fingerprint"))
    if bcfg is None or ccfg is None:
        raise GateError(
            "a run record has no config_fingerprint, so there is no way to "
            "tell what produced it (device, threads, batch size, dtype). "
            "Re-run it rather than comparing blind")
    if bcfg != ccfg:
        diff = {k: (baseline.get("config", {}).get(k),
                    candidate.get("config", {}).get(k))
                for k in set(baseline.get("config", {}))
                | set(candidate.get("config", {}))
                if baseline.get("config", {}).get(k)
                != candidate.get("config", {}).get(k)}
        raise GateError(
            f"execution configs differ: {diff}. A quality change and a "
            "configuration change are not separable from these two runs; "
            "re-measure the baseline under the candidate's configuration")
    if not cfg.metrics:
        raise GateError("no metrics configured; a gate that tests nothing "
                        "cannot pass")
    unknown = [m for m in cfg.metrics if m not in METRIC_KEYS]
    if unknown:
        raise GateError(f"unknown metric(s) {unknown}; known metrics are "
                        f"{sorted(METRIC_KEYS)}")

    alpha = cfg.alpha / len(cfg.metrics)   # Bonferroni, see module docstring
    per_metric = {m: _compare_one(baseline, candidate, cfg, m, alpha)
                  for m in cfg.metrics}

    status = "FAIL" if any(v.status == "FAIL" for v in per_metric.values()) else "PASS"
    reasons = [f"{m}: {r}" for m, v in per_metric.items() for r in v.reasons]
    return Verdict(status, reasons, per_metric, alpha)
