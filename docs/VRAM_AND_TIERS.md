# GPU VRAM: what is held, why, and what fits where

Measured 2026-08-19 on the live stack (RTX 3080 Ti, 12 288 MiB) after the native diarization
flip. Raw numbers and method: [validation/RESULTS.md](../validation/RESULTS.md) §7.12 and §7.14.
This file is the digest — read it before changing model sets, concurrency, or GPU placement.

## The short version

| | |
|---|---|
| idle floor, everything warm | **7 575 MiB** |
| peak during 3 concurrent jobs | 9 047 MiB |
| marginal cost per concurrent job | **~490 MiB** |
| settles back to | **7 575 MiB, exactly** |

**It is a floor, not a leak.** VRAM returns to the same figure after every job and does not creep
upward. What looks like "the GPU is still holding memory" is warm-start caching that is
deliberately never released.

## Who holds what

| holder | VRAM | composition | scales with concurrency? |
|---|---|---|---|
| **diar-native sidecar** | 4 136 MiB | ~547 MiB ONNX weights + ~3.6 GB ORT arena and cuDNN conv workspace | **no** — one copy serves every job |
| **celery-worker** (whisper large-v3-turbo, int8_float16) | 2 038 MiB | weights + CTranslate2 pool | **no** for weights — `--pool=threads` shares one ModelManager across all 8 threads |
| **celery-redaction** (toxic-bert / xlm-roberta) | 1 346 MiB | weights, resident even when idle | no |
| per concurrent job | ~490 MiB | activations only | **yes** |

Gender detection (wav2vec2) runs **on CPU**: 0 VRAM, ~380 MB RAM, 87-90 s per file.

## Why the floor is 7.5 GB, and why the sidecar is 16× its weights

Three deliberate warm-start decisions stack up. The interesting one is the third: it is *earned
on first inference*, not allocated at load.

| moment | GPU total | what changed |
|---|---|---|
| app only, no sidecar | 2 838 MiB | `PRELOAD_GPU_MODELS=true` pins whisper at worker startup |
| sidecar started | 3 385 MiB | +547 MiB — ONNX **weights only** |
| after a 30 s clip | 4 069 MiB | arena starts growing |
| after a 10 min clip | 6 979 MiB | arena sized to batch-32 activations |
| steady since | 7 575 MiB | high-water mark, never returned |

ONNX Runtime allocates through a **BFC arena per session** that grows to peak demand and never
shrinks — the analogue of PyTorch's caching allocator, but without an automatic
`torch.cuda.empty_cache()`. So 251 MB of ONNX on disk becomes 4.1 GB resident once a real batch
has run. `arena_extend_strategy` is **already** set to the lean `SameAsRequested`; the untried
levers are `with_conv_max_workspace(false)` with a cheaper `ConvAlgorithmSearch`, ORT's per-run
`memory.enable_memory_arena_shrinkage`, and a shared cross-session allocator (§4.25). Each trades
against speed and must be benchmarked, not assumed.

## Deployment tiers

| tier | GPU | configuration | status |
|---|---|---|---|
| server | ≥ 12 GB | fast set (`models_folded/`), everything co-resident | **shipping** — measured above |
| mid | 6-8 GB | fast set, move redaction to CPU or another GPU | expected to fit; unverified |
| **laptop** | **4 GB** | needs load/unload — see below | **NOT yet supported** |
| Apple Silicon | unified memory | CoreML EP path exists in speakrs; different constraints entirely | **future work — after the Linux build is stable** |

### Cheapest capacity wins (config only, no code)

1. `REDACTION_GPU_DEVICE_ID` → another GPU or CPU: frees **1 346 MiB**.
2. `DIAR_NATIVE_GPU` → another GPU: frees **4 136 MiB**.

On this host both are free wins because two A6000s sit idle at 15 MiB each. Doing both leaves
GPU 1 almost entirely to whisper — roughly 5 concurrent jobs instead of 2.

### What a 4 GB GPU actually needs

The floor (7.5 GB) is nearly twice a 4 GB card, so this is not a configuration change — it needs
work. Budget on a 4 GB card, assuming diarization and transcription do **not** stay co-resident:

| component | cost | note |
|---|---|---|
| whisper large-v3-turbo int8_float16 | ~2.0 GB | smaller models cut this further |
| diar-native, small set (`models_small/`) | 1.6 GB | 59× RT instead of 277× (RESULTS §4.27) |
| redaction | 1.3 GB | must move off-GPU entirely |
| **co-resident total** | **~4.9 GB** | **over budget before any activations** |

So a 4 GB tier requires, roughly in order of value:

1. **Redaction off the GPU** — mandatory, config only.
2. **Small model set** for the sidecar — config only, accuracy-neutral (§4.16 verified the
   batching toggle is RTTM-identical), costs 3.6× diarization speed.
3. **Load/unload between stages** — the real work. `ModelManager.release_transcriber()` already
   exists for exactly this and fires when total VRAM < 16 GB, but the sidecar is a *separate
   long-lived process* that never unloads. A 4 GB tier needs the sidecar to release sessions
   when idle (or to be started per job), which trades the 277× warm-serving win that made the
   flip worth doing. **Note this pulls directly against T2**: overlap requires both models
   resident at once, so a 4 GB tier is sequential-only.
4. **Shrink the arena** rather than the model — the sidecar's 3.6 GB of arena is a far better
   target than its 547 MB of weights.

Honest position: 4 GB is reachable but is a distinct operating mode — small set, redaction off
GPU, no transcribe∥diarize overlap, and sidecar session lifecycle management. It should be
measured as its own tier, not assumed from these numbers.

## Rules of thumb

- Don't reach for the small model set to buy concurrency: it saves 2.6 GB but costs 3.6×
  diarization speed. One fast engine at 277× RT beats four small ones at 59× (RESULTS §7.5).
- Concurrency is capped by *activations*, not weights — weights are shared.
- What actually gates parallel throughput is not VRAM but `diar-server`'s `Mutex<DiarEngine>`:
  requests serialize regardless of `DIAR_MAX_INFLIGHT`. That is T9a (Arc-shared sessions).
- Measure peaks **during** load, not after. Sampling a finished run reports the idle floor and
  tells you nothing.
