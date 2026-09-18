"""Run the five known cases through the full pipeline and check verdicts.

Each model is scored once and cached under results/validation/, so the six
distinct models cost six eval runs regardless of how many cases reuse them.
"""
import argparse
import gc
import json
from pathlib import Path

import yaml

from correctness_gate.backends import make_backend
from correctness_gate.config import load_gate, load_models
from correctness_gate.gate import GateError, compare
from correctness_gate.items import fingerprint, hash_json, load_items
from correctness_gate.mcq import evaluate


def run_model(name, specs, items, cache: Path) -> dict:
    out = cache / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    print(f"scoring {name} ...")
    be = make_backend(specs[name])
    res = evaluate(be, items)
    cfg_desc = be.describe()
    res |= {"model": name, "backend": specs[name].backend,
            "fingerprint": fingerprint(items),
            "config": cfg_desc, "config_fingerprint": hash_json(cfg_desc)}
    out.write_text(json.dumps(res, indent=1))
    del be
    gc.collect()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="validation/cases.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--gate", default="configs/gate.yaml")
    ap.add_argument("--cache", default="results/validation",
                help="per-model score cache. Scores depend on the execution "
                        "config, so a run on different hardware or a different "
                        "device needs its own cache directory.")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.cases).read_text())
    specs, _ = load_models(a.models)
    cfg = load_gate(a.gate)
    items = load_items(spec["items"])
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    failures = 0
    # One column per metric: the gate fails if any of them regresses, so a
    # table showing only the headline metric would hide half the decision.
    print(f"\n| case | expect | verdict | "
          + " | ".join(f"{m} delta (p)" for m in cfg.metrics) + " | note |")
    print("|---|---|---" + "|---" * len(cfg.metrics) + "|---|")
    for case in spec["cases"]:
        base = run_model(case["baseline"], specs, items, cache)
        cand = run_model(case["candidate"], specs, items, cache)
        try:
            v = compare(base, cand, cfg)
            verdict = v.status
            cols = [f"{v.per_metric[m].delta:+.3f} ({v.per_metric[m].p_value:.4f})"
                    if v.per_metric[m].status == "PASS"
                    else f"**{v.per_metric[m].delta:+.3f}** "
                         f"({v.per_metric[m].p_value:.4f}) FAIL"
                    for m in cfg.metrics]
            extra = "; ".join(v.reasons)
        except GateError as e:
            verdict, cols, extra = "FAIL", ["n/a"] * len(cfg.metrics), str(e)
        ok = case["expect"] in ("REPORT", verdict)
        if not ok:
            failures += 1
        mark = "" if ok else "  <-- MISMATCH"
        print(f"| {case['name']} | {case['expect']} | {verdict}{mark} | "
              + " | ".join(cols) + f" | {extra[:60]} |")

    print(f"\n{failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
