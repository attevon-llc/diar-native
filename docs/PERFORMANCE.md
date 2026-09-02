# Performance and accuracy

A bridge between the README's headline figures and
[`validation/RESULTS.md`](../validation/RESULTS.md), which is the **append-only record of every
measurement ever taken** and the only source of truth. This page gives the numbers that matter
with the conditions attached, and points at whoever owns the detail. Where this page and RESULTS
disagree, RESULTS wins.

## Accuracy

`pyannote.metrics` DER, collar 0.25, overlap included, AMI with the official UEMs. These hold the
recorded gates exactly and **beat the production pyannote fork** they replace — accuracy parity
was gate G1 and the hard precondition for the whole project, since a faster diarizer that is
worse is not a win.

| corpus | DER |
|---|---|
| AMI test-16, full | **13.101%** |
| AMI test-16, exclusive | **17.813%** |
| Karpathy (66-min hand-labelled acceptance clip) | **8.219%** |
| VoxConverse dev-216 | **4.847%** |

## Speed

Warm engine, quiet machine, RTX A6000:

| workload | wall time | |
|---|---|---|
| Karpathy, 66.5 min of audio | **21.6 s** | **184× realtime** |
| AMI ES2004a, 36 min of audio | **6.6 s** | |
| Upload → transcript, reference file | **54.4 s** | down from 108.4 s on the Python path |

Concurrent requests share one engine's VRAM (T9a shared sessions — spans no longer double under
load), fbank runs pipelined against the GPU, and the sidecar ingests original media directly
rather than requiring a pre-transcoded WAV.

What this is measured against: the Python stack's clustering is ~41% of wall time on a 4.7 h file
and serialises with everything else under the GIL, and the embedding stage alone is 53% of GPU
time with no way to scale it independently.

> **Read this before optimising diarization.** Since T2, diarization **overlaps** transcription.
> On a 66.5-min file that is 37.5 s of diarization hidden inside 50.3 s of transcription — so
> **making diarization faster buys nothing for single-job latency until it exceeds the
> transcription window.** It still buys throughput under concurrency, and it still buys VRAM
> headroom. But a single-job stopwatch will not move, and people have been surprised by that.

### Levers deliberately not on the list

| lever | why not |
|---|---|
| A smaller model set | It is a **VRAM floor**, never a speed lever (RESULTS §7.5). |
| Batching the gender classifier | Batch-1 **by design** (RESULTS §7.20). |
| fp16 for diarization | **Rejected on measured DER collapse** (RESULTS §4.18). Not to be confused with fp16 *gender*, which is fine and shipped. |

## Memory

Owned by [`VRAM_AND_TIERS.md`](VRAM_AND_TIERS.md). The one figure worth carrying in your head:
the warm stack holds a **7 575 MiB floor** on a 12 GB card and each concurrent job adds only
**~490 MiB**, because weights are shared and only activations scale. That floor is warm-start
caching, not a leak. `SPEAKRS_ARENA_SHRINK=1` trades ~20% per-job speed for a much lower floor on
4 GB-tier cards.

## Determinism: CPU vs CUDA

The two engines produce **bit-identical centroids**; segment **boundaries can differ by one
frame**. Verified, not assumed — it is what makes one image serving both devices safe to offer
per request. Concurrent execution was likewise verified **output-identical to serial**, which is
the precondition that made shared sessions shippable.

## How a number gets into RESULTS

[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) is law and owns the procedure. The rules that
catch people: **one timed leg at a time on a quiet machine**, **sample VRAM during a run rather
than after**, **every speed claim ships with its accuracy check** (prove output identity by
diffing raw records — never assert it), and **never re-run a logged test**.

Corpora, reference paths and every number to beat are owned by
[`TEST_CORPORA_AND_BASELINES.md`](TEST_CORPORA_AND_BASELINES.md). The comparison is always the
production pyannote fork @ `a3f38afb` (run inside the backend image with the fork bind-mounted,
i.e. the exact production path) against speakrs @ `b0756b1` in `cuda` mode — 1.0 s step,
pyannote-equivalent, **never** `cuda-fast`. Gates G1-G5 are defined in
[`validation/TESTPLAN.md`](../validation/TESTPLAN.md).

> **A smoke test is not an accuracy check.** `verify-models` passes every stage on a build with
> ~52% DER. Anything touching the numeric substrate needs a real-corpus DER run on each published
> architecture, run natively. See
> [PROVISIONING.md](PROVISIONING.md#what-verification-does-not-cover) (issue #21).

## Decisions on record

Measured, then decided against — so nobody re-litigates them from intuition:

| decision | outcome | evidence |
|---|---|---|
| **TensorRT EP in-process** | **Rolled back.** 1.33-1.48× faster at +0.03 pp AMI DER, but the compatibility surface was not worth speed that hides behind transcription anyway. | RESULTS §7.26 (kept as a recipe if this changes) |
| **Native fbank** | **Superseded** by the pipelined fbank∥GPU design. | RESULTS §7.28 |
| **Sinc resampler** | **Rejected** — keep `FftFixedIn`. | RESULTS §7.29 |
| **Fused fbank in the embedding graph** | **Rejected** — ~6.7× cost on ORT CUDA. Ship fbank-outside plus a batched native fbank. | RESULTS §4.2 |
| **ubuntu 26.04 base** | **Reverted, pinned back to 24.04.** ~3× DER regression on arm64 (exclusive 18.7% → 52.4%), root-caused to OpenBLAS 0.3.32. | RESULTS §7.52, issue #18 |
| **fp16 gender on linux/arm64** | **Root-caused and fixed** — an ORT load-time GELU fusion, not an accuracy problem. | RESULTS §7.40 |
| **Triton (tier T2)** | **Opt-in, not default.** 2.14× throughput at 8 concurrent clients on one weight copy, but a heavier RAM and system footprint. | [ARCHITECTURE.md](ARCHITECTURE.md#deployment-tiers) |

---

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [DEPLOYMENT.md](DEPLOYMENT.md) ·
[README](../README.md)
