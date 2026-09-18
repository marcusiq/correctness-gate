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
    gold: int

    def context(self) -> str:
        #watch arc template exactly
        return f"Question: {self.question}\nAnswer:"
    def continuation(self, i: int) -> str:
        return " " + self.choices[i]

def normalize_gold(labels: list[str], answer_key: str) -> int:
    # returns the index of the answer in the labels if it exists, if it does not raise an error
    if answer_key in labels:
        return labels.index(answer_key)
    raise ValueError(f"answerKey {answer_key!r} not in labels {labels}")

def load_items(path: str | Path) -> list[Item]:
    raw = json.loads(Path(path).read_text())
    return [Item(**r) for r in raw]
def hash_json(obj) -> str:
    """Stable 64-bit id for any JSON-able object.

    Canonical form first: sorted keys and no whitespace, so two equal objects
    always produce the same bytes and therefore the same id. Used for the item
    set and for the execution config, which are the two things a comparison is
    only valid within.
    """
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]

def fingerprint(items: list[Item]) -> str:
    return hash_json([asdict(i) for i in items])