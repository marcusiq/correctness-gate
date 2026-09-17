"""Multiple-choice scoring by summed continuation log-likelihood.

The scoring rule lm-evaluation-harness uses for arc_easy / hellaswag /
winogrande / MMLU, reimplemented:
  acc       argmax over choices of  sum(token logprobs)
  acc_norm  argmax over choices of  sum(token logprobs) / len(choice in UTF-8 bytes)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .backends import Scored
from .items import Item


def log_softmax_rows(rows: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the vocab axis: subtract the
    row max before exponentiating so exp never overflows."""
    m = rows.max(axis=-1, keepdims=True)
    return rows - (m + np.log(np.exp(rows - m).sum(axis=-1, keepdims=True)))


def continuation_logprob(scored: Scored) -> float:
    lp = log_softmax_rows(scored.rows)
    idx = np.arange(len(scored.token_ids))
    return float(lp[idx, scored.token_ids].sum())


@dataclass
class ItemScore:
    qid: str
    gold: int
    logprobs: list[float]   # one summed logprob per choice
    pick: int               # argmax of logprobs (acc)
    pick_norm: int          # argmax of byte-length-normalized logprobs (acc_norm)
    correct: bool
    correct_norm: bool


def score_item(backend, item: Item) -> ItemScore:
    lps, lens = [], []
    for i in range(len(item.choices)):
        s = backend.score(item.context(), item.continuation(i))
        lps.append(continuation_logprob(s))
        lens.append(len(item.continuation(i).encode("utf-8")))
    arr = np.array(lps)
    pick = int(arr.argmax())
    pick_norm = int((arr / np.array(lens)).argmax())
    return ItemScore(item.qid, item.gold, lps, pick, pick_norm,
                     pick == item.gold, pick_norm == item.gold)


def evaluate(backend, items: list[Item], progress: bool = True) -> dict:
    scores = []
    for k, it in enumerate(items, 1):
        scores.append(score_item(backend, it))
        if progress:
            print(f"\r{k}/{len(items)}", end="", flush=True)
    if progress:
        print()
    n = len(scores)
    return {
        "n": n,
        "acc": sum(s.correct for s in scores) / n,
        "acc_norm": sum(s.correct_norm for s in scores) / n,
        "items": [asdict(s) for s in scores],
    }
