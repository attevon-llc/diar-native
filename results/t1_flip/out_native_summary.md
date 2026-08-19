# Diarization Boundary Smoothing — Benchmark Summary

Generated 2026-08-19 15:11:52. Headline metric: WSER (lower is better).
Bootstrap = paired CI on pooled (OFF - ON) improvement; significant iff ci_low > 0.

## Per-dataset

| dataset | files | OFF WSER | ON WSER | Δ (OFF-ON) | islands OFF→ON | bootstrap 95% CI | sig |
|---|---:|---:|---:|---:|---:|---|:---:|
| tier 1 | 1 | 0.0194 | 0.0131 | +0.0063 | 64→7 | [+0.0063, +0.0063] | yes |
| **overall** | 1 | 0.0194 | 0.0131 | +0.0063 | 64→7 | [+0.0063, +0.0063] | yes |

## Regression flags

| file | model | flags |
|---|---|---|
| karpathy | large-v3-turbo | der_c0-moved (off=0.059992 on=0.050906) |

