"""Configuration loading.

Two files drive everything:
  configs/models.yaml   which model artifacts exist and how to load them
  configs/gate.yaml     the statistical contract of the gate

Design choice: plain dataclasses plus a strict YAML loader instead of a
validation framework. The config surface is a dozen keys; the failure mode
that matters is a misspelled key silently ignored, so unknown keys raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


def _strict(cls, raw: dict, where: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**raw)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str            # "llamacpp" or "hf"
    path: str               # .gguf path (llamacpp) or HF model id / local dir (hf)
    url: str = ""           # optional download source, used by scripts/fetch_model.py
    device: str = "cpu"     # sets n_gpu_layers=0 only; on a CUDA build set
                            # CUDA_VISIBLE_DEVICES= too, or large matmuls still run on the GPU
    dtype: str = "float32"  # hf only; float32 is the determinism-friendly default
    n_ctx: int = 640        # llamacpp only; the logits buffer is n_ctx x vocab floats
    n_threads: int = 4      # llamacpp only; stay at or below physical cores
    n_batch: int = 512      # llamacpp prefill chunk size; the invariance check varies this
    n_gpu_layers: int | None = None   # llamacpp only. None = derive from device
                                      # (-1 all layers on cuda, 0 on cpu). Set an
                                      # explicit count for partial offload when
                                      # two models must share the 12 GB card.


@dataclass(frozen=True)
class GateConfig:
    alpha: float = 0.05
    power: float = 0.80
    mde: float = 0.05
    hard_floor: float = 0.15
    n_boot: int = 10_000
    seed: int = 1234
    metrics: list[str] = field(default_factory=lambda: ["acc", "acc_norm"])


def load_models(path: str | Path) -> tuple[dict[str, ModelSpec], str | None]:
    raw = yaml.safe_load(Path(path).read_text())
    specs = {}
    for name, body in raw["models"].items():
        spec = _strict(ModelSpec, {"name":name, **body}, f"{path}:{name}")
        specs[name] = spec
    under_test = raw.get("under_test")
    if under_test is not None and under_test not in specs:
        raise ValueError(f"under_test {under_test!r} is not a defined model")
    return specs, under_test

def load_gate(path: str | Path) -> GateConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _strict(GateConfig, raw.get("gate", raw), str(path))
