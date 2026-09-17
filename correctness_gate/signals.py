"""Cheap correctness signals, ordered below task accuracy in cost.

compare_configs      candidate vs pinned reference on a small fixed prompt
                     set: per-token logprob deltas, greedy-flip rate, KL
                     (Kullback-Leibler) divergence in nats
batch_invariance_*   same model, same inputs, different batch geometry;
                     any logit delta is a reduction-order effect, the
                     home-scale version of the batch-invariance result

Teacher-forcing note: all signals force the gold continuation, so both
configurations are scored on identical token paths and every delta is
attributable to numerics, never to sampling.
"""
from __future__ import annotations

import dataclasses
import gc

import numpy as np

from .backends import make_backend
from .config import ModelSpec
from .items import Item
from .mcq import log_softmax_rows


def _pairs(items: list[Item], k: int):
    return [(it.context(), it.continuation(it.gold)) for it in items[:k]]


def compare_configs(ref_spec: ModelSpec, cand_spec: ModelSpec,
                    items: list[Item], k: int = 20) -> dict:
    ref, cand = make_backend(ref_spec), make_backend(cand_spec)
    lp_deltas, kls = [], []
    flips = positions = skipped = 0
    for ctx, cont in _pairs(items, k):
        r, c = ref.score(ctx, cont), cand.score(ctx, cont)
        if r.token_ids != c.token_ids:
            skipped += 1  # tokenizers disagree; per-token comparison would be meaningless
            continue
        lr, lc = log_softmax_rows(r.rows), log_softmax_rows(c.rows)
        idx = np.arange(len(r.token_ids))
        lp_deltas.extend(np.abs(lr[idx, r.token_ids]
                                - lc[idx, c.token_ids]).tolist())
        flips += int((r.rows.argmax(axis=1) != c.rows.argmax(axis=1)).sum())
        positions += len(idx)
        pr = np.exp(lr)
        kls.extend((pr * (lr - lc)).sum(axis=1).tolist())
    return {
        "reference": ref_spec.name,
        "candidate": cand_spec.name,
        "positions": positions,
        "tokenizer_mismatch_items": skipped,
        "logprob_delta_mean": float(np.mean(lp_deltas)) if lp_deltas else None,
        "logprob_delta_max": float(np.max(lp_deltas)) if lp_deltas else None,
        "greedy_flip_rate": flips / positions if positions else None,
        "kl_mean_nats": float(np.mean(kls)) if kls else None,
    }


def batch_invariance_llamacpp(spec: ModelSpec, items: list[Item],
                              k: int = 8, small: int = 32) -> dict:
    """Same model and inputs, prefill chunked at n_batch=spec.n_batch vs
    n_batch=small. The two backends are constructed sequentially (never
    co-resident) to keep memory flat."""
    rows_by = {}
    for label, nb in (("large", spec.n_batch), ("small", small)):
        be = make_backend(dataclasses.replace(spec, n_batch=nb))
        rows_by[label] = [be.score(ctx, cont).rows
                          for ctx, cont in _pairs(items, k)]
        del be
        gc.collect()
    deltas = [float(np.abs(a - b).max())
              for a, b in zip(rows_by["large"], rows_by["small"])]
    flips = sum(int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
                for a, b in zip(rows_by["large"], rows_by["small"]))
    return {
        "model": spec.name,
        "n_batch": [spec.n_batch, small],
        "items_checked": len(deltas),
        "max_abs_logit_delta": max(deltas) if deltas else None,
        "greedy_flips": flips,
        "bitwise_identical": (max(deltas) == 0.0) if deltas else None,
    }


def batch_invariance_hf(spec: ModelSpec, items: list[Item],
                        k: int = 4, copies: int = 4) -> dict:
    """Same prompt scored alone vs stacked `copies` times in one forward.
    Identical inputs at identical positions; any delta comes purely from
    batched kernel reduction order."""
    be = make_backend(spec)  # must be an hf spec
    torch = be.torch
    deltas = []
    for ctx, cont in _pairs(items, k):
        whole = be.tok.encode(ctx + cont)
        ids1 = torch.tensor([whole], device=spec.device)
        idsk = torch.tensor([whole] * copies, device=spec.device)
        with torch.no_grad():
            l1 = be.model(ids1).logits[0]
            lk = be.model(idsk).logits[0]
        deltas.append(float((l1 - lk).abs().max()))
    return {
        "model": spec.name,
        "copies": copies,
        "items_checked": len(deltas),
        "max_abs_logit_delta": max(deltas),
        "bitwise_identical": max(deltas) == 0.0,
    }
