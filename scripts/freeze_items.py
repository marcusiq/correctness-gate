import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset
from correctness_gate.items import Item, fingerprint, normalize_gold

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data/arc_easy_200.json")
    a = ap.parse_args()

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=a.split)
    rows = list(ds)
    random.Random(a.seed).shuffle(rows)

    frozen = []
    for r in rows[: a.n]:
        frozen.append({
            "qid": r["id"],
            "question": r["question"],
            "choices": r["choices"]["text"],
            "gold": normalize_gold(r["choices"]["label"], r["answerKey"]),
        })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=1))
    items = [Item(**f) for f in frozen]
    print(f"froze {len(items)} items -> {out}  fingerprint {fingerprint(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())