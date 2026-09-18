"""Download a model listed in configs/models.yaml if its file is missing.
Used by CI; harmless locally."""
import argparse
import sys
import urllib.request
from pathlib import Path

from correctness_gate.config import load_models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    specs, _ = load_models(a.models)
    spec = specs[a.model]
    dest = Path(spec.path).expanduser()
    if dest.exists():
        print(f"{dest} already present")
        return 0
    if not spec.url:
        print(f"{spec.name}: file missing and no url configured",
              file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {spec.url} -> {dest}")
    urllib.request.urlretrieve(spec.url, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
