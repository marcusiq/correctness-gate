"""Cross-validate the scorer against an lm-evaluation-harness run, item for item.

The fixture (validation/fixtures/samples_arc_easy_*.jsonl) carries the exact
prompt strings lm-eval scored, its per-continuation log-likelihoods, and the
gold target. We score the same strings with our HF backend and compare.

Expect pick agreement on ~all 50 items and accuracy landing on the fixture's
0.58 / 0.64. Do NOT expect bit-identical log-likelihoods: the fixture ran
batch size 8 on CUDA, we run batch 1 on CPU. That residual delta is real
(batch geometry changes kernel reduction order) and is exactly what the M7
invariance signal quantifies.
"""
import argparse
import glob
import json

import numpy as np

from correctness_gate.backends import make_backend
from correctness_gate.config import load_models
from correctness_gate.mcq import continuation_logprob


def _ll(x):
    """lm-eval sample files nest resps as [[ll, is_greedy]] with varying
    depth across versions; unwrap until the number."""
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture",
                    default=sorted(glob.glob(
                        "validation/fixtures/samples_arc_easy_*.jsonl"))[-1])
    ap.add_argument("--model", default="qwen0.5b-hf")
    ap.add_argument("--models", default="configs/models.yaml")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.fixture)]
    specs, _ = load_models(a.models)
    be = make_backend(specs[a.model])

    agree = acc_mine = acc_ref = 0
    agree_norm = accn_mine = accn_ref = 0
    max_delta = 0.0
    for k, r in enumerate(rows, 1):
        args = r["arguments"]
        ctx = args["gen_args_0"]["arg_0"]
        conts = [args[f"gen_args_{i}"]["arg_1"] for i in range(len(args))]
        gold = int(r["target"])

        mine = [continuation_logprob(be.score(ctx, c)) for c in conts]
        theirs = [_ll(x) for x in r["resps"]]

        pm, pt = int(np.argmax(mine)), int(np.argmax(theirs))
        agree += pm == pt
        acc_mine += pm == gold
        acc_ref += pt == gold
        max_delta = max(max_delta, max(abs(m - t) for m, t in zip(mine, theirs)))

        # acc_norm: the gate's decision metric, so it needs validating too. The
        # length rule here must stay identical to mcq.score_item: UTF-8 bytes of
        # the continuation, leading space included. `conts` are lm-eval's own
        # continuation strings, which already carry that space.
        lens = [len(c.encode("utf-8")) for c in conts]
        pmn = int(np.argmax([m / L for m, L in zip(mine, lens)]))
        accn_mine += pmn == gold
        accn_ref += int(r["acc_norm"])          # lm-eval's per-item verdict
        agree_norm += (pmn == gold) == bool(r["acc_norm"])
        print(f"\r{k}/{len(rows)}", end="", flush=True)

    n = len(rows)
    print(f"\npick agreement: {agree}/{n}")
    print(f"acc  mine {acc_mine / n:.2f}   lm-eval {acc_ref / n:.2f}")
    print(f"max |ll_mine - ll_lmeval| = {max_delta:.4f}")
    print(f"\nacc_norm item agreement: {agree_norm}/{n}")
    print(f"acc_norm  mine {accn_mine / n:.2f}   lm-eval {accn_ref / n:.2f}")
    if agree_norm != n:
        print("MISMATCH: the normalized metric disagrees with lm-eval on "
              f"{n - agree_norm} item(s). The gate decides on acc_norm, so fix "
              "this before trusting any verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
