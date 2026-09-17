"""Which length does lm-eval divide by for acc_norm?

Takes lm-eval's OWN log-likelihoods out of the fixture, applies each candidate
normalization rule, and checks which one reproduces lm-eval's per-item acc_norm.
No model is loaded: if a rule disagrees here, the scorer is exonerated and the
rule is the bug.
"""
import argparse
import glob
import json

import numpy as np

RULES = {
    "mine: bytes of ' ' + choice": lambda c: len((" " + c).encode("utf-8")),
    "chars of ' ' + choice":       lambda c: len(" " + c),
    "bytes of choice":             lambda c: len(c.encode("utf-8")),
    "chars of choice":             lambda c: len(c),
}


def _ll(x):
    """lm-eval nests resps as [[ll, is_greedy]] with varying depth; unwrap
    until the number. Values may be strings."""
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture",
                    default=sorted(glob.glob(
                        "validation/fixtures/samples_arc_easy_*.jsonl"))[-1])
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.fixture)]
    n = len(rows)
    ref_acc_norm = sum(int(r["acc_norm"]) for r in rows) / n
    print(f"{n} items from {a.fixture}")
    print(f"lm-eval acc_norm = {ref_acc_norm:.3f}\n")
    print(f"{'rule':32s} {'acc_norm':>9s} {'item agreement':>16s}")

    for name, length in RULES.items():
        hits = agree = 0
        for r in rows:
            lls = [_ll(x) for x in r["filtered_resps"]]
            choices = r["doc"]["choices"]["text"]
            gold = int(r["target"])
            pick = int(np.argmax([ll / length(c) for ll, c in zip(lls, choices)]))
            hits += pick == gold
            agree += (pick == gold) == bool(r["acc_norm"])
        print(f"{name:32s} {hits / n:9.3f} {f'{agree}/{n}':>16s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())