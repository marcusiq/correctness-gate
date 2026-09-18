"""Batch-geometry invariance on the items that can actually show it.

`cgate invariance` checks the first k items at n_batch 512 vs 32. Any
sequence shorter than the small batch is prefilled in one chunk under both
settings, so it cannot differ by construction. This script picks the k
longest items by token count instead and sweeps several small batch sizes.

    python scripts/invariance_longest.py --device cpu
    python scripts/invariance_longest.py --device cuda
"""
import argparse
import dataclasses
import json

from llama_cpp import Llama

from correctness_gate.config import load_models
from correctness_gate.items import load_items
from correctness_gate.signals import batch_invariance_llamacpp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b-q4km")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--items", default="data/arc_easy_200.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--small", type=int, nargs="+", default=[32, 8, 1])
    a = ap.parse_args()

    specs, _ = load_models(a.models)
    spec = dataclasses.replace(specs[a.model], device=a.device)
    items = load_items(a.items)

    tok = Llama(model_path=spec.path, vocab_only=True, verbose=False)
    lengths = [len(tok.tokenize((it.context() + it.continuation(it.gold))
                                .encode("utf-8"))) for it in items]
    order = sorted(range(len(items)), key=lambda i: -lengths[i])[: a.k]
    print(f"token lengths over all {len(items)} items: min {min(lengths)}, "
          f"median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}")
    print(f"checking the {a.k} longest: {[lengths[i] for i in order]}")

    for small in a.small:
        r = batch_invariance_llamacpp(spec, [items[i] for i in order],
                                      k=a.k, small=small)
        print(json.dumps({"device": a.device, **r}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
