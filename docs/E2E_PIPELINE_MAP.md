# OpenTranscribe E2E Pipeline Map + Ranked Levers (researched 2026-08-19)

Source: two read-only code explorations (backend job lifecycle; presentation path). Every claim
carries a file:line in transcribe-app. This is the factual base for the north-star program
(upload → presented, per tier).

## The pipeline in one paragraph

Upload (`files/upload.py:455` → MinIO → dispatch) → Celery chain of 3
(`dispatch.py:181-199`): `preprocess` (cpu; ffmpeg extract, shared-volume WAV, waveform task
fired in parallel) → `gpu_transcribe` (gpu; audio mmap-load → whisper CT2 w/ Silero VAD
hardcoded on → diarize → assign/resegment/dedup → **segments bulk-INSERT + status=COMPLETED at
progress 0.78**, `finalize.py:331-344`) → `postprocess` (cpu; **synchronous speaker matching
gates the completion event**, `postprocess.py:6-10`) → `enrich_and_dispatch` fan-out (async:
indexing/facts/redaction/summary/topics/gender→LLM-speaker-ID/clustering). Frontend learns via
WebSocket (Redis pub/sub bridge) and only fetches the transcript on `completed` + 1 s
(`notificationHandler.ts:120-126`). Full progress ladder + task/queue inventory in the agent
reports (this file is the digest; raw reports in session history — regenerate via the two
Explore prompts if needed).

## Telemetry status (for the baseline run)

- **Rich machinery EXISTS**: Redis markers `benchmark:{task_id}` (40 call sites) + durable
  `file_pipeline_timing` table (~55 ms-columns incl. `user_perceived_duration_ms`) + admin
  read APIs + `TIMING:` logs + VRAMProfiler + engine metrics. Gated on
  **`ENABLE_BENCHMARK_TIMING`** — must be ON for the baseline.
- Gaps to fill for a complete E2E report (tiny diffs, post-PR): `summary_*`, `clustering_*`,
  `search_index_*` columns are never written; `stage_timings` dicts produced then discarded
  (`job.py:44` etc.); two markers missing from `_TIMESTAMP_MARKERS`.

## RANKED LEVERS (value ÷ effort, with file anchors)

| # | lever | anchor | expected effect | effort |
|---|---|---|---|---|
| L1 | **Native diarization flip** (sidecar, READY) | INSTALL_NATIVE.md | diarize stage ~13 s/h audio (277× RT); frees worker GIL+VRAM | hook + flip |
| L2 | **True transcribe ∥ diarize overlap** — with the sidecar, `_GpuRawStage` can POST /diarize at stage start and collect after whisper finishes (today's "overlap" is only *weights preload*, `stages.py:377-400`) | `stages.py:333-493` | GPU-stage wall → ≈ max(transcribe, diarize) instead of sum | small, after flip |
| L3 | **Progressive presentation** — transcript is durable + endpoints ungated at 0.78; emit `transcript_ready` (or fetch on the 0.75 tick); speaker labels late-attach via EXISTING handlers (`websocket.ts:448-461`); redaction stays gating | `finalize.py:344`, `notificationHandler.ts:120` | user sees transcript ~10-60 s earlier (skips sync speaker-matching + 1 s delay) | small; UX decision: brief "Speaker 1" labels |
| L4 | **Move `_FinalizeStage` off the GPU worker** — default fast path runs finalize (assign, resegment, dedup, bulk INSERT) INSIDE the GPU task, holding the concurrency-1 GPU slot (`pipelines.py:273-274`) | `finalize.py:266-354` | GPU slot freed minutes earlier per job → throughput under load | medium (the split machinery already exists) |
| L5 | **Gender detection demotion + relocation** — wav2vec2, minutes of CPU, runs on the `cpu` queue at PIPELINE_CRITICAL priority competing with pre/postprocess, and gates LLM speaker-ID | `speaker_attribute_task.py:153`, `celery.py:236` | de-prioritize now (config); absorb into diar-native later (shares PCM — RUST_SERVICES_PLAN §3) | tiny now / medium later |
| L6 | **`precompute_vad` is wired end-to-end but DEAD** (`vad_regions=None` hardcoded, `stages.py:327`) — implement in preprocess; feeds whisper AND (later) diarization window gating; natural Rust/sidecar item | `engine/config.py:25`, `stages.py:327` | skip silence work in BOTH GPU stages | medium |
| L7 | **Progress-event DB writes** — every emit() = session + UPDATE + WS (8-10 round-trips/job) | `pipelines.py:268-271` | batch/async; small per job, matters under load | tiny |
| L8 | **`vad_min_silence_ms=2000`** (4× faster-whisper default 500) | `config.py:59-63` | latency/accuracy knob — WSER-test at 500/1000 | config test |
| L9 | **Telemetry completeness** (dead columns, stage_timings persist) | §above | makes the baseline report complete | tiny |
| L10 | ASR quantization int8_float16 + batch re-tune with freed VRAM | `transcriber.py:196-253` | 1.3-1.7× the #1 stage | config + WSER validation |

Facts that close earlier questions: VAD is Silero-ONNX inside faster-whisper, always on
(torch-free ✓); **no mDeBERTa in the repo** (toxicity = toxic-bert/xlm-roberta on the redaction
worker); gender model = wav2vec2 (~380 MB) effectively always-CPU in default topology; redaction
device policy is per-scan auto GPU/CPU with a global inference lock.

## Baseline protocol binding (post-PR)

Run the SPEEDUP_ROADMAP E2E protocol with `ENABLE_BENCHMARK_TIMING=1`; primary metric =
`user_perceived_duration_ms` (already computed!) per tier per engine; secondary =
`fully_indexed_duration_ms` + per-stage columns. L9's tiny marker diffs land first so the
report is complete on run one.
