from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ModelSpec

@dataclass
class Scored:
    token_ids: list[int] # continuation token ids, in order
    rows: np.ndarray # len(toen_ids), vocab float 32 lgits

class LlamaCppBackend:
    def __init__(self, spec: ModelSpec):
        from llama_cpp import Llama, llama_cpp

        self.spec = spec
        want = spec.n_gpu_layers
        if want is None:
            want = 0 if spec.device == "cpu" else -1

        if want != 0 and not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError(
                f"{spec.name}: device={spec.device} requests GPU offload but this "
                "llama-cpp-python build has no GPU support. Reinstall the CUDA wheel; "
                "a CPU build would silently produce CPU numbers labeled cuda.")

        self.llm = Llama(
            model_path=str(Path(spec.path).expanduser()),
            n_ctx=spec.n_ctx,
            n_batch=spec.n_batch,
            n_threads=spec.n_threads,
            n_gpu_layers=want,
            logits_all=True,
            seed=1234,
            verbose=False,
        )
        # What was requested, for the run record. Ground truth still comes from
        # the VRAM check below, not from this number.
        self.n_gpu_layers = want

    def score(self, context: str, continuation: str) -> Scored:
        whole = self.llm.tokenize((context+continuation).encode("utf-8"))
        ctx = self.llm.tokenize(context.encode("utf-8"))
        cont = whole[len(ctx):]
        if not cont:
            raise ValueError("continuation tokenized to zero tokens")
        if len(whole) > self.spec.n_ctx:
            raise ValueError(
                f"{len(whole)} tokens exceeds n_ctx={self.spec.n_ctx}; "
                "raise n_ctx in configs/models.yaml deliberately (the logits "
                "buffer is n_ctx x vocab floats)")
        self.llm.reset()
        self.llm.eval(whole)
        scores = np.asarray(self.llm.scores, dtype=np.float32)[: self.llm.n_tokens]
        # scores[k-1] is the distribution that predicts token k
        rows = scores[len(ctx) - 1 : len(whole) - 1]
        return Scored(token_ids=cont, rows=rows.copy())

class HFBackend:
    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.spec = spec
        path = str(Path(spec.path).expanduser())
        self.tok = AutoTokenizer.from_pretrained(path)
        # transformers v5 takes dtype=; on 4.x change this to torch_dtype=
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=getattr(torch, spec.dtype))
        self.model.to(spec.device).eval()

    def score(self, context: str, continuation: str) -> Scored:
        torch = self.torch
        whole = self.tok.encode(context + continuation)
        ctx = self.tok.encode(context)
        cont = whole[len(ctx):]
        if not cont:
            raise ValueError("continuation tokenized to zero tokens")
        ids = torch.tensor([whole], device=self.spec.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0]  # (len(whole), vocab)
        rows = logits[len(ctx) - 1 : len(whole) - 1].float().cpu().numpy()
        return Scored(token_ids=list(cont), rows=rows)


def make_backend(spec: ModelSpec):
    if spec.backend == "llamacpp":
        return LlamaCppBackend(spec)
    if spec.backend == "hf":
        return HFBackend(spec)
    raise ValueError(f"unknown backend {spec.backend!r}")