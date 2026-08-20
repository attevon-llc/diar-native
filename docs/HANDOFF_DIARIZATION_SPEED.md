# Handoff — remaining diarization speed levers

Companion to `HANDOFF_T9A_SHARED_SESSIONS.md`. That one is about **concurrency** (many jobs at
once); this one is about **single-job speed** and the work that feeds it. Both target
`diar-native`; neither touches `transcribe-app`.

**Corpora, file paths and every baseline number:** `docs/TEST_CORPORA_AND_BASELINES.md`.
**Read first:** `validation/RESULTS.md` §7.5, §7.13-7.15 (where the time actually goes), §4.16
(the fbank ladder), §4.4-4.6 (TensorRT evidence), `docs/BENCHMARK_PROTOCOL.md` (how to measure
anything here without producing numbers nobody can trust).

---

## Where diarization stands

Already done, and the reason the remaining levers are smaller than they look:

| | |
|---|---|
| warm serving | **277× RT** (7.9 s for a 36-min meeting), 3.4× the PyAnnote fork |
| accuracy | AMI 13.10 %, Karpathy 8.219 %, VoxConverse 4.847 % (beats the fork's 5.099 %) |
| exclusive-diarization bug | fixed — was resolving overlaps by cluster index (§7.7) |
| in the app | runs **inside transcription's window** (T2), so on single-job latency it is already free |

**The critical framing:** since T2, diarization overlaps transcription. On a 66.5-min file that is
37.5 s of diarization hidden inside 50.3 s of transcription. **Making diarization faster buys
nothing for single-job latency until it exceeds the transcription window.** It only pays for:

1. **throughput** — that is T9a, not this document;
2. **files where diarization exceeds transcription** — the 2.1 h seed file is close (113.8 s
   diarize vs ~71 s transcribe under load);
3. **the 4 GB laptop tier**, where the two cannot be co-resident and must run sequentially.

Measure against `docs/BENCHMARK_PROTOCOL.md`'s anchor before assuming any of these pays.

---

## Lever 1 — TensorRT EP inside ort (T11) — strongest remaining, highest risk

**STATUS 2026-08-19: implemented, measured, parked as opt-in (RESULTS §7.26).** 1.33-1.48×
warm speed; AMI full +0.030 pp / exclusive +0.006 pp / Karpathy −0.002 pp (bit-parity is
unachievable — TRT kernels shift boundaries ~1 frame); cache-hit restart 6 s; libs-absent
fallback byte-identical to CUDA. Default off (`SPEAKRS_TRT=1` + `WITH_TENSORRT=1` image to
enable) — operator decision: keep CUDA's exact recorded accuracy as the default.

Spec: `docs/DETAILED_SPECS.md` S-T11. Evidence: TRT measured **2.4× vs ORT-CUDA** on the embedding
model (§4.6), and the graphs are **fixed-shape (b32)**, so the phase-6 rebuild-storm precondition
is absent (§4.4).

De-risking facts already established:
- Provider libs must come from the **same ORT release tarball** as the runtime. `Dockerfile.server`
  copies the full 1.24.2 lib set — **verify `libonnxruntime_providers_tensorrt.so` is present
  first**; if the MS tarball lacks it, you need TensorRT libs in-image (NVIDIA apt or a TRT-bearing
  base) before anything else.
- Register providers `[TensorRT, CUDA, CPU]` so CUDA remains the fallback.
- `.with_engine_cache(true)` + `.with_engine_cache_path("/models/trt_cache")` on a **mounted
  volume**, or every restart pays the build.
- `.with_fp16(false)` — fp32 only. §4.18 records fp16 collapsing DER in the StatsPool
  variance/sqrt subgraph. (Note fp16 was fine for the *gender* model, §7.18 — different graph,
  different answer. Measure, do not generalise.)
- First run builds engines for seconds-to-minutes, so the server must warm each TRT session before
  `/healthz` reports ready.
- Apply to **multimask-b64(b32) + segmentation** first; leave fbank/tail on CPU.

**Gate:** RTTM bit-parity vs CUDA EP on clip30 + ES2004a + the 2.2 h file; warm E2E delta recorded;
restart shows a cache hit (log inspection); removing the cache volume falls back to CUDA EP cleanly.

## Lever 2 — native Rust fbank

The pooled ONNX-CPU fbank still costs ~10 s on a 4.7 h file. `kaldi-native-fbank` / `knf-rs` is
~3-5× faster per chunk and rayon-friendly. §4.16 is the context: fbank was **76 % of wall** before
the session-pool fan-out took ES2004a from 39.4 s to 12.9 s, so this is attacking what is left of
an already-optimised path.

**Gate:** RTTM identical (fbank is deterministic; there is no excuse for drift), and a measured
per-chunk improvement on the 4.7 h file, not just a microbenchmark.

## Lever 3 — symphonia decode inside diar-server

Today the worker decodes audio, writes a 16 kHz WAV to a shared volume, and the sidecar reads it
back. If the sidecar accepts the **original media path** and decodes with symphonia, that write and
read disappear, and mp4/mp3 ingest works directly. Preprocess ffmpeg decode is 1.30 s of a 54.8 s
job (§7.15), so this is polish — but it also removes a whole class of failure: the shared volume
being root-owned silently disabled the engine fast path for who knows how long (§7.10).

**Gate:** byte-identical PCM against the current ffmpeg path (compare decoded samples, not RTTMs),
then RTTM identity.

## Lever 4 — ORT arena / allocator (only for the 4 GB tier)

The sidecar holds 4 136 MiB for 251 MB of weights. **Already tested and ruled out:**
`ConvAlgorithmSearch::Heuristic` + `with_conv_max_workspace(false)` changes VRAM by **zero bytes**
(§7.23) — conv workspace is not the lever, and `arena_extend_strategy` is already the lean
`SameAsRequested`. Untested: per-run `memory.enable_memory_arena_shrinkage` on `RunOptions`, and a
shared cross-session allocator (§4.25 notes there is none today).

**Do not spend time here for the 12 GB target** — it has 4 328 MiB free and fits ~8 concurrent jobs
(§7.14). This matters only for the 4 GB laptop tier, where diarization and transcription cannot be
co-resident at all.

## Lever 5 — per-phase pipelining (adjacent to T9a)

Even with shared sessions, a job holds segmentation and embedding for its whole run. Releasing
per phase would let job B segment while job A embeds. This is a deeper change than T9a and should
only be considered **after** T9a's numbers show phase contention is what remains.

---

## What is deliberately *not* on this list

- **Smaller model set** (`models_small/`) — 3.6× slower diarization for 2.6 GB (§7.5). It is a
  VRAM floor for laptops, never a speed or concurrency lever.
- **Bigger batches for the small models** — the batch-32 multimask graph is already what delivers
  277× RT; gender is batch-1 by design because its windows are 5 s and batching would raise peak
  VRAM for ~1.5 s of work (§7.20).
- **fp16 for diarization** — §4.18. Rejected on measured DER collapse, not on principle.

## Standing rules for anyone picking this up

1. One timed leg at a time (`run_e2e_baseline.sh` holds an `flock`); sample VRAM **during** a run.
2. Every speed claim ships with its accuracy check — the gates are in `BENCHMARK_PROTOCOL.md`.
3. For anything that should not change outputs, **prove identity by diffing raw records** rather
   than asserting it. T2's shared audio buffer was exactly that risk and the diff is what settled it.
4. `ort` stays pinned `=2.0.0-rc.12`; regenerate `patches/0001-*.patch` after any vendored change.
