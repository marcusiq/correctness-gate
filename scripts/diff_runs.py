"""Diff two score caches item for item.

Accuracy alone cannot tell you whether two runs computed the same thing: two
runs can agree on every headline number and disagree on many items. This walks
the per-item log-probabilities and reports the largest divergence, how many
picks moved, and whether the two runs are bit-identical.

Use it to compare the same models across devices, thread counts, batch
geometries, or two repeats of one configuration.

    python scripts/diff_runs.py results/validation-cuda-run1 results/validation-cpu
"""
import argparse
import json
from pathlib import Path


def diff_one(a: dict, b: dict) -> dict:
    ia, ib = a["items"], b["items"]
    if len(ia) != len(ib):
        raise SystemExit("runs have different item counts; not comparable")
    if a.get("fingerprint") != b.get("fingerprint"):
        raise SystemExit("runs are on different item sets; not comparable")

    max_d = 0.0
    for p, q in zip(ia, ib):
        for x, y in zip(p["logprobs"], q["logprobs"]):
            max_d = max(max_d, abs(x - y))
    return {
        "acc": (a["acc"], b["acc"]),
        "acc_norm": (a["acc_norm"], b["acc_norm"]),
        "max_abs_logprob_delta": max_d,
        "pick_diffs": sum(p["pick"] != q["pick"] for p, q in zip(ia, ib)),
        "pick_norm_diffs": sum(p["pick_norm"] != q["pick_norm"]
                               for p, q in zip(ia, ib)),
        "bitwise_identical": all(p["logprobs"] == q["logprobs"]
                                 for p, q in zip(ia, ib)),
        "n": len(ia),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="directory of run records")
    ap.add_argument("right", help="directory of run records to compare against")
    a = ap.parse_args()

    left, right = Path(a.left), Path(a.right)
    names = sorted(p.name for p in left.glob("*.json"))
    if not names:
        raise SystemExit(f"no run records in {left}")

    print(f"| model | acc {left.name} -> {right.name} | acc_norm | "
          f"max abs dlogp | pick diffs | pick_norm diffs | bitwise |")
    print("|---|---|---|---|---|---|---|")
    total_picks = 0
    for name in names:
        r = right / name
        if not r.exists():
            print(f"| {name[:-5]} | missing in {right.name} | | | | | |")
            continue
        d = diff_one(json.loads((left / name).read_text()), json.loads(r.read_text()))
        total_picks += d["pick_diffs"] + d["pick_norm_diffs"]
        print(f"| {name[:-5]} | {d['acc'][0]:.3f} -> {d['acc'][1]:.3f} "
              f"| {d['acc_norm'][0]:.3f} -> {d['acc_norm'][1]:.3f} "
              f"| {d['max_abs_logprob_delta']:.6g} | {d['pick_diffs']}/{d['n']} "
              f"| {d['pick_norm_diffs']}/{d['n']} | {d['bitwise_identical']} |")
    print(f"\ntotal decisions changed: {total_picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
