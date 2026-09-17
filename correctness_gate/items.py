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
def fingerprint(items: list[Item]) -> str:
    canon = json.dumps([asdict(i) for i in items], sort_keys=True, separators=(",",":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]