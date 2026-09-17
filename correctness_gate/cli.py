"""The cgate command. Thin by design: all logic lives in the modules, so
every command is scriptable from Python without the CLI."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cgate")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score a model on the frozen items")
    r.add_argument("--model", required=True)
    r.add_argument("--models", default="configs/models.yaml")
    r.add_argument("--items", default="data/arc_easy_200.json")
    r.add_argument("--limit", type=int, default=None,
                   help="score only the first N items (fingerprint changes, "
                        "so limited runs never compare against full baselines)")
    r.add_argument("--out", required=True)

    c = sub.add_parser("compare", help="gate a candidate run against a baseline")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--gate", default="configs/gate.yaml")
    c.add_argument("--report", default=None, help="also write markdown here")

    pr = sub.add_parser("promote", help="promote a run to baselines/ (manual, deliberate)")
    pr.add_argument("run")
    pr.add_argument("--dir", default="baselines")

    s = sub.add_parser("signals", help="cheap funnel: candidate vs reference")
    s.add_argument("--reference", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--items", default="data/arc_easy_200.json")
    s.add_argument("--k", type=int, default=20)

    i = sub.add_parser("invariance", help="batch-geometry invariance check")
    i.add_argument("--model", required=True)
    i.add_argument("--models", default="configs/models.yaml")
    i.add_argument("--items", default="data/arc_easy_200.json")
    i.add_argument("--k", type=int, default=8)

    a = p.parse_args(argv)
    return {"run": _run, "compare": _compare, "promote": _promote,
            "signals": _signals, "invariance": _invariance}[a.cmd](a)


def _run(a) -> int:
    from .backends import make_backend
    from .config import load_models
    from .items import fingerprint, load_items
    from .mcq import evaluate

    specs, _ = load_models(a.models)
    items = load_items(a.items)
    if a.limit:
        items = items[: a.limit]
    res = evaluate(make_backend(specs[a.model]), items)
    res |= {"model": a.model, "backend": specs[a.model].backend,
            "fingerprint": fingerprint(items)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"acc={res['acc']:.3f} acc_norm={res['acc_norm']:.3f} "
          f"n={res['n']} -> {out}")
    return 0


def _compare(a) -> int:
    from .config import load_gate
    from .gate import GateError, compare
    from .report import render

    cfg = load_gate(a.gate)
    base = json.loads(Path(a.baseline).read_text())
    cand = json.loads(Path(a.candidate).read_text())
    try:
        v = compare(base, cand, cfg)
    except GateError as e:
        md = f"## correctness-gate: ❌ FAIL\n\nGate error: {e}\n"
        if a.report:
            Path(a.report).write_text(md)
        print(md)
        return 1
    md = render(v, base, cand, cfg)
    if a.report:
        Path(a.report).write_text(md)
    print(md)
    return 0 if v.status == "PASS" else 1


def _promote(a) -> int:
    run = json.loads(Path(a.run).read_text())
    dest = Path(a.dir) / f"{run['model']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.run, dest)
    print(f"promoted {a.run} -> {dest} (commit this deliberately)")
    return 0


def _signals(a) -> int:
    from .config import load_models
    from .items import load_items
    from .signals import compare_configs

    specs, _ = load_models(a.models)
    out = compare_configs(specs[a.reference], specs[a.candidate],
                          load_items(a.items), a.k)
    print(json.dumps(out, indent=2))
    return 0


def _invariance(a) -> int:
    from .config import load_models
    from .items import load_items
    from .signals import (batch_invariance_hf, batch_invariance_llamacpp)

    specs, _ = load_models(a.models)
    spec = specs[a.model]
    items = load_items(a.items)
    fn = (batch_invariance_llamacpp if spec.backend == "llamacpp"
          else batch_invariance_hf)
    print(json.dumps(fn(spec, items, a.k), indent=2))
    return 0
