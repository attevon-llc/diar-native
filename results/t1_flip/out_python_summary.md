# Diarization Boundary Smoothing — Benchmark Summary

Generated 2026-08-19 15:14:25. Headline metric: WSER (lower is better).
Bootstrap = paired CI on pooled (OFF - ON) improvement; significant iff ci_low > 0.

## Per-dataset

| dataset | files | OFF WSER | ON WSER | Δ (OFF-ON) | islands OFF→ON | bootstrap 95% CI | sig |
|---|---:|---:|---:|---:|---:|---|:---:|
| tier 1 | 1 | 0.0123 | 0.0086 | +0.0037 | 81→21 | [+0.0037, +0.0037] | yes |
| **overall** | 1 | 0.0123 | 0.0086 | +0.0037 | 81→21 | [+0.0037, +0.0037] | yes |

## Regression flags

| file | model | flags |
|---|---|---|
| karpathy | large-v3-turbo | der_c0-moved (off=0.052361 on=0.047628) |

