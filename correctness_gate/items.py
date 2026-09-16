"""Frozen evaluation items and their identity.

An item set is a positional list: baselines store per-item correctness by
position, so the fingerprint is deliberately order-sensitive. Reordering
the file is a new item set, and that is correct behavior.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    qid: str
    question: str
    choices: list[str]
    gold: int  # index into choices

    def context(self) -> str:
        # Must match lm-evaluation-harness's arc template exactly, or
        # cross-validation against lm-eval compares different prompts.
        return f"Question: {self.question}\nAnswer:"

    def continuation(self, i: int) -> str:
        return " " + self.choices[i]


def normalize_gold(labels: list[str], answer_key: str) -> int:
    """ARC's answerKey is 'A'..'E' on most items and '1'..'5' on some;
    labels carries the same convention per item, so index lookup handles
    both without a special case."""
    if answer_key in labels:
        return labels.index(answer_key)
    raise ValueError(f"answerKey {answer_key!r} not in labels {labels}")


def load_items(path: str | Path) -> list[Item]:
    raw = json.loads(Path(path).read_text())
    return [Item(**r) for r in raw]


def fingerprint(items: list[Item]) -> str:
    canon = json.dumps([asdict(i) for i in items], sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
