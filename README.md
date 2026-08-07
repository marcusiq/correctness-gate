# correctness-gate

A CI release gate for LLM quality that derives its thresholds from measured
noise, validates itself against known quantization regressions, and reports
every verdict with a confidence interval and its own detection limit.

Under construction. The statistical core lands first; see commit history.
