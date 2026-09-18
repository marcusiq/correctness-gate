"""Diff two score caches item for item.

Accuracy alone cannot tell you whether two runs computed the same thing: two
runs can agree on every headline number and disagree on many items. This walks
the per-item log-probabilities and reports how far they moved, how many
decisions changed, and whether the two runs are bit-identical.

Distribution, not just the max: one pathological item can set a maximum, and a
continuation the model considers nearly impossible has a log-probability steep
enough to move a lot for numerical reasons alone. The median and the 95th
percentile are what describe the typical case.

    python scripts/diff_runs.py results/validation-cuda-run1 results/validation-cpu
    python scripts/diff_runs.py results/validation-cuda-run1 results/validation-cpu --worst 3
"""
import argparse
import json
from pathlib import Path

import numpy as np


def diff_one(a: dict, b: dict) -> dict:
    ia, ib = a["items"], b["items"]
    if len(ia) != len(ib):
        raise SystemExit("runs have different item counts; not comparable")
    if a.get("fingerprint") != b.get("fingerprint"):
        raise SystemExit("runs are on different item sets; not comparable")

    deltas, where = [], []
    for p, q in zip(ia, ib):
        for choice, (x, y) in enumerate(zip(p["logprobs"], q["logprobs"])):
            deltas.append(abs(x - y))
            where.append((p.get("qid", "?"), choice, x, y))
    d = np.asarray(deltas)
    order = np.argsort(d)[::-1]
    return {
        "acc": (a["acc"], b["acc"]),
        "acc_norm": (a["acc_norm"], b["acc_norm"]),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "pick_diffs": sum(p["pick"] != q["pick"] for p, q in zip(ia, ib)),
        "pick_norm_diffs": sum(p["pick_norm"] != q["pick_norm"]
                               for p, q in zip(ia, ib)),
        "bitwise_identical": bool((d == 0).all()),
        "n": len(ia),
        "worst": [where[i] for i in order],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="directory of run records")
    ap.add_argument("right", help="directory of run records to compare against")
    ap.add_argument("--worst", type=int, default=0,
                    help="also list the N largest per-choice divergences, with "
                         "the log-probabilities behind them")
    a = ap.parse_args()

    left, right = Path(a.left), Path(a.right)
    names = sorted(p.name for p in left.glob("*.json"))
    if not names:
        raise SystemExit(f"no run records in {left}")

    print(f"| model | acc {left.name} -> {right.name} | acc_norm | median "
          f"|dlogp| | p95 | max | pick diffs | pick_norm diffs | bitwise |")
    print("|---|---|---|---|---|---|---|---|---|")
    details, total_picks = [], 0
    for name in names:
        r = right / name
        if not r.exists():
            print(f"| {name[:-5]} | missing in {right.name} | | | | | | | |")
            continue
        la, lb = json.loads((left / name).read_text()), json.loads(r.read_text())
        d = diff_one(la, lb)
        details.append((name[:-5], la, lb, d))
        total_picks += d["pick_diffs"] + d["pick_norm_diffs"]
        print(f"| {name[:-5]} | {d['acc'][0]:.3f} -> {d['acc'][1]:.3f} "
              f"| {d['acc_norm'][0]:.3f} -> {d['acc_norm'][1]:.3f} "
              f"| {d['median']:.3g} | {d['p95']:.3g} | {d['max']:.3g} "
              f"| {d['pick_diffs']}/{d['n']} | {d['pick_norm_diffs']}/{d['n']} "
              f"| {d['bitwise_identical']} |")
    print(f"\ntotal decisions changed: {total_picks}")

    for name, la, lb, d in details:
        ca, cb = la.get("config"), lb.get("config")
        if ca and cb and ca != cb:
            keys = sorted(set(ca) | set(cb))
            shown = {k: (ca.get(k), cb.get(k)) for k in keys if ca.get(k) != cb.get(k)}
            print(f"\n{name} config differs: {shown}")
        elif not ca or not cb:
            print(f"\n{name}: at least one run predates config recording; "
                  "what produced it is not knowable from the record")

    if a.worst:
        for name, _, _, d in details:
            print(f"\nlargest divergences, {name}:")
            print("| qid | choice | left logprob | right logprob | abs delta |")
            print("|---|---|---|---|---|")
            for qid, choice, x, y in d["worst"][: a.worst]:
                print(f"| {qid} | {choice} | {x:.4f} | {y:.4f} | {abs(x - y):.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
