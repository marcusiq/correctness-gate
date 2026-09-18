# Determinism and devices

What the gate assumes about repeatability, and the measurements behind it. Every number
here comes from a run on this machine with the reproducing command beside it.

Hardware: RTX 3060 12 GB (compute 8.6), i5-11600KF 6C/12T, WSL2 Ubuntu 24.04,
llama-cpp-python 0.3.35 with CUDA support.

## Why this matters to the gate

A quality gate compares two runs and attributes any difference to the change under test.
That attribution is only valid if everything else was held constant. So the first question
is not "did accuracy drop" but "how much does this stack move when nothing changes at all".
That floor sets what a real regression has to clear.

It also decides what the determinism tier should test. If run-to-run repeats at a fixed
configuration are bit-identical, repeating them measures nothing, and the tier should vary
batch geometry, device and dtype instead.

## Measured: run-to-run on CUDA is bit-identical

Two independent invocations of the five-case validation, separate processes, fresh model
loads each time, GPU offload enabled for every model. 5 models x 200 items x 4 choices,
so 4,000 scored continuations per run.

```
python scripts/run_validation.py --cache results/validation-cuda-run1
python scripts/run_validation.py --cache results/validation-cuda-run2
python scripts/diff_runs.py results/validation-cuda-run1 results/validation-cuda-run2
```

| model | acc run1 -> run2 | acc_norm | max abs dlogp | pick diffs | pick_norm diffs | bitwise |
|---|---|---|---|---|---|---|
| qwen3b-f16 | 0.715 -> 0.715 | 0.685 -> 0.685 | 0 | 0/200 | 0/200 | True |
| qwen3b-q8 | 0.710 -> 0.710 | 0.680 -> 0.680 | 0 | 0/200 | 0/200 | True |
| qwen3b-q4km | 0.710 -> 0.710 | 0.635 -> 0.635 | 0 | 0/200 | 0/200 | True |
| qwen3b-q2k | 0.220 -> 0.220 | 0.260 -> 0.260 | 0 | 0/200 | 0/200 | True |
| qwen3b-q2k-imatrix | 0.710 -> 0.710 | 0.685 -> 0.685 | 0 | 0/200 | 0/200 | True |

Total decisions changed: 0.

Not merely equal accuracies: identical log-probabilities in every position. This matches
what the M0 audit found for the CPU path, and it is what licenses the determinism tier to
vary configuration rather than repeat runs.

## Measured: batch geometry

```
python scripts/invariance_longest.py --device cpu
python scripts/invariance_longest.py --device cuda
```

qwen0.5b-q4km, the 8 longest items (68 to 115 tokens; the set's median is 27, so the
first-k selection `cgate invariance` uses mostly tests sequences shorter than one chunk
and cannot show a difference by construction). Prefill chunked at n_batch 512 vs the
smaller size. Max abs logit delta, greedy flips in parentheses:

| build and device | 512 vs 32 | 512 vs 8 | 512 vs 1 |
|---|---|---|---|
| CPU-only wheel (GGML_NATIVE=OFF, AVX2) | 0.0 | 0.0 | 0.976 (4) |
| CUDA wheel, `CUDA_VISIBLE_DEVICES=`, 0 layers | 0.0 | 0.0 | 1.096 (1) |
| CUDA wheel, GPU visible, 0 layers | 0.966 (2) | 1.565 (4) | 1.611 (3) |
| CUDA wheel, all layers offloaded | 0.881 (0) | 3.419 (2) | 2.761 (2) |

- True CPU prefill is bitwise batch-invariant for chunks of 8 or more. Single-token
  chunks take a different (matrix-vector) path and differ. The gate scores each
  sequence in one chunk, so its numbers sit on the invariant path.
- The two CPU rows differ at n_batch 1 (0.976 vs 1.096): the native build on this
  AVX-512 machine and the pinned AVX2 build are different kernels.
- With a CUDA build and a visible GPU, `n_gpu_layers=0` behaves like a GPU run.
  "0 layers offloaded" is not "CPU".

## Measured: layer offload moves both scores and decisions

> **Correction (2026-09-18).** This section was first written as "CPU versus CUDA".
> The `validation-cpu` records were produced by the CUDA build of llama-cpp-python
> with `n_gpu_layers=0` and the GPU still visible to the process. That is not a CPU
> run: llama.cpp still sends large matrix multiplications to a visible GPU. Evidence:
> re-scoring 10 items of 3B Q4_K_M under that setup reproduces the record exactly
> (max |dlogp| 0.0); with `CUDA_VISIBLE_DEVICES=` set, the same code differs from it
> by up to 4.1 nats. The comparison below is therefore "0 layers offloaded on a CUDA
> build" versus "all layers offloaded". The measured divergence and decision counts
> stand; the attribution to CPU versus GPU kernels does not. See the batch-geometry
> section for the invariance measurement that exposed this.

Same weights, same items, same code. Only the offload configuration differs.

```
sed 's/device: cuda/device: cpu/' configs/models.yaml > configs/models.cpu.yaml
python scripts/run_validation.py --models configs/models.cpu.yaml --cache results/validation-cpu
python scripts/diff_runs.py results/validation-cpu results/validation-cuda-run1 --worst 3
```

All five models. Deltas are per summed continuation log-probability, 800 of them per
model (200 items x 4 choices), so 4,000 in total.

| model | acc cpu -> cuda | acc_norm | median abs dlogp | p95 | max | pick diffs | pick_norm diffs |
|---|---|---|---|---|---|---|---|
| qwen3b-f16 | 0.710 -> 0.715 | 0.680 -> 0.685 | 0.0139 | 0.280 | 0.747 | 1/200 | 2/200 |
| qwen3b-q8 | 0.715 -> 0.710 | 0.685 -> 0.680 | 0.0806 | 0.692 | 1.42 | 4/200 | 1/200 |
| qwen3b-q2k-imatrix | 0.730 -> 0.710 | 0.675 -> 0.685 | 0.212 | 1.97 | 7.29 | 8/200 | 7/200 |
| qwen3b-q4km | 0.705 -> 0.710 | 0.650 -> 0.635 | 0.385 | 3.34 | 7.15 | 6/200 | 7/200 |
| qwen3b-q2k | 0.240 -> 0.220 | 0.250 -> 0.260 | 0.386 | 5.98 | 21.1 | 27/200 | 26/200 |

89 of 2,000 gate-relevant decisions changed. Four things follow.

**Quantization amplifies it.** The f16 model is the control: identical weights, identical
items, different offload configuration, and its median divergence is 0.0139 with 3 of 400
decisions moved. Every quantized variant is an order of magnitude worse. Which kernels run
where under the hybrid configuration is not isolated here (see the correction above), so
this shows that quantized weights are far more sensitive to the execution path, not which
dequantization path is responsible.

**It scales with coarseness.** Median divergence runs f16 0.014 -> Q8_0 0.081 -> Q2_K
0.386, a 28x spread, and the tail grows faster than the median: Q2_K's p95 is 5.98 and
its worst single continuation moves 21.1 nats. Decisions moved follow the same order,
from 3 of 400 on f16 to 53 of 400 on Q2_K.

**An imatrix reduces cross-device divergence, not just perplexity.** Q2_K with an
importance matrix has a median of 0.212 against plain Q2_K's 0.386, and 15 decisions moved
against 53, from the same file size. quant-lab measured the imatrix rescuing perplexity
from 23190 to 10.48; this is a second, independent benefit of the same artifact.

**It is not confined to hopeless continuations.** The largest excursions do sit on very
unlikely answers, where the log scale is steep: `ACTAAP_2010_7_17` choice 2 on Q2_K goes
from -167.65 to -146.55. But the medians above are computed over all 800 continuations
per model, so the typical one moves too.

**The effect on the gate is material.** acc_norm on Q4_K_M moved 0.015 between the two configurations,
roughly a third of the configured minimum detectable effect of 0.05. Worse, the borderline
validation case (`subtle-open-question`) came out at p = 0.0987 fully offloaded and p = 0.4244 with
0 layers offloaded. Both PASS, but the strength of evidence differs by a factor of four from the
offload setting alone. A candidate and baseline scored under different offload settings spend a large part of
the gate's detection budget before any real change is considered.

That is the measured justification for the config fingerprint below.

**Caveat on these records.** The CPU cache was produced before config fingerprinting
existed, so those records cannot be gated, only diffed. Regenerating them costs about five
hours of CPU time and would prove nothing further: the cross-device claim is a
measurement, not a gate decision. A cross-device gating demo belongs on the 0.5B models,
where a CPU run takes minutes.

**Two things observed during the CPU run**, both worth knowing:

- llama.cpp printed `CUDA_Host compute buffer size of 18.4518 MiB, does not match
  expectation of 10.5098 MiB`. With a CUDA-enabled build, a run with `n_gpu_layers=0`
  still registers the CUDA backend and allocates pinned host memory. Weights and compute
  stay on the CPU, but "CPU run" does not mean "CUDA untouched". For hard isolation, set
  `CUDA_VISIBLE_DEVICES=` in the environment before the process starts.
- The run loaded `configs/gate.yaml` at start, so it gated under the single-metric config
  that was in force at 16:16 and was unaffected by the change to `metrics: [acc, acc_norm]`
  made while it ran. Long runs freeze their configuration at startup; check what a result
  was actually measured under before comparing it to anything.

## The config fingerprint

Every run record now carries the execution configuration that produced it and a hash of
it:

```json
"config": {"backend": "llamacpp", "device": "cuda", "n_gpu_layers": -1,
           "n_ctx": 640, "n_batch": 512, "n_threads": 4,
           "lib": "llama_cpp 0.3.35"},
"config_fingerprint": "..."
```

`gate.compare` refuses, failure-closed, when the two records' fingerprints differ, and
also when either record lacks one. The HF backend additionally records `dtype` and
`allow_tf32`, because TF32 silently changes float32 matmuls on Ampere with nothing
visible in the config file.

## A labeling failure worth recording

The first two validation runs were originally saved as `validation-cpu` and
`validation-cuda`. Both were in fact GPU runs: the 3B specs already carried
`device: cuda`, and nothing in a run record said which device produced it, so the mistake
was invisible in the output. It was caught only by noticing that two supposedly different
devices produced bit-identical log-probabilities.

Before that, a `--cache` flag that parsed but was never read caused a "new" run to silently
reuse the previous cache. No error, no warning, a complete and confident report.

Both failures have the same shape and the same fix: **a run record must carry the
configuration that produced it.** Recording `device`, `n_gpu_layers`, `n_threads`,
`n_batch`, `dtype` and the backend library version, hashing them into a
`config_fingerprint`, and refusing to compare runs whose fingerprints differ turns both of
these from silent mislabeling into a failed comparison.
