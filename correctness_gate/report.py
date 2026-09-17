"""Render a Verdict as the markdown that becomes the PR comment."""
from __future__ import annotations

from .config import GateConfig
from .gate import MetricVerdict, Verdict


def _metric_block(mv: MetricVerdict) -> list[str]:
    icon = "✅" if mv.status == "PASS" else "❌"
    lines = [
        f"### `{mv.metric}`: {icon} {mv.status}",
        "",
        f"Delta: **{mv.delta:+.3f}** "
        f"(95% confidence interval [{mv.ci_low:+.3f}, {mv.ci_high:+.3f}], "
        f"McNemar p = {mv.p_value:.4f} against alpha = {mv.alpha:.4f})",
        "",
        f"Flips: {mv.counts.b} broke, {mv.counts.c} fixed "
        f"({mv.counts.b + mv.counts.c} discordant of {mv.counts.n})",
        "",
        f"Power: smallest drop detectable at this n is {mv.mde:.3f}",
    ]
    if mv.broke:
        shown = ", ".join(mv.broke[:10])
        more = f" (+{len(mv.broke) - 10} more)" if len(mv.broke) > 10 else ""
        lines += ["", f"Items that broke: {shown}{more}"]
    if mv.reasons:
        lines += [""] + [f"- {r}" for r in mv.reasons]
    return lines


def render(v: Verdict, baseline: dict, candidate: dict,
           cfg: GateConfig) -> str:
    icon = "✅" if v.status == "PASS" else "❌"
    metrics = ", ".join(f"`{m}`" for m in v.per_metric)
    n = v.primary.counts.n
    lines = [
        f"## correctness-gate: {icon} {v.status}",
        "",
        f"Metrics {metrics} on {n} frozen items "
        f"(fingerprint `{candidate['fingerprint']}`). All metrics must pass; "
        f"alpha {cfg.alpha:.3f} is split across {len(v.per_metric)} "
        f"({v.alpha_per_metric:.4f} each) so the family-wise false-positive "
        "rate stays at the configured value.",
        "",
        "| | baseline | candidate |",
        "|---|---|---|",
        f"| model | {baseline.get('model', '?')} | {candidate.get('model', '?')} |",
        f"| acc | {baseline['acc']:.3f} | {candidate['acc']:.3f} |",
        f"| acc_norm | {baseline['acc_norm']:.3f} | {candidate['acc_norm']:.3f} |",
        "",
    ]
    for mv in v.per_metric.values():
        lines += _metric_block(mv) + [""]
    return "\n".join(lines).rstrip() + "\n"
