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

## Measured: batch geometry on CPU

```
cgate invariance --model qwen0.5b-q4km --k 8
```

n_batch 512 vs 32, 8 items: max abs logit delta 0.0, bitwise identical.

**Caveat, not yet resolved.** Item sequences here run roughly 10 to 120 tokens. Any
sequence shorter than 32 tokens is processed as a single chunk under both settings, so
those items cannot show a difference by construction. Until the check reports the token
count of each sequence it tested, this result is not evidence of invariance. Re-run with
a smaller `small` and with the longest items selected.

## Pending: CPU versus CUDA

The cross-device comparison, which is the one that can move numbers rather than confirm
they are stable:

```
sed 's/device: cuda/device: cpu/' configs/models.yaml > configs/models.cpu.yaml
python scripts/run_validation.py --models configs/models.cpu.yaml --cache results/validation-cpu
python scripts/diff_runs.py results/validation-cuda-run1 results/validation-cpu
```

Results go here when the run completes. The interesting quantity is not the log-prob
delta by itself but whether any of the 1,000 per-item decisions flip because of it.

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
