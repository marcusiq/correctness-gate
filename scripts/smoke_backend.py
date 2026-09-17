"""First-light check for a backend: teacher-force an obvious answer and
print each continuation position's greedy pick. This is scratch/probe_align.py
from the M0 audit, rebuilt on the backend contract."""
import argparse

from correctness_gate.backends import make_backend
from correctness_gate.config import load_models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b-q4km")
    ap.add_argument("--models", default="configs/models.yaml")
    a = ap.parse_args()

    specs, _ = load_models(a.models)
    be = make_backend(specs[a.model])
    s = be.score("Question: What color is a clear daytime sky?\nAnswer:",
                 " It is blue.")
    print("continuation token ids:", s.token_ids)
    print("rows shape:", s.rows.shape)
    print("greedy pick at each position:", s.rows.argmax(axis=1).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
