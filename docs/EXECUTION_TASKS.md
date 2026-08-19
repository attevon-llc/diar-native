# Execution Tasks — detailed specs + recommended model/effort per task

Companion to PLAN.md (sequence) and E2E_PIPELINE_MAP.md (anchors). Each task: steps, files,
completion gate, and a recommended Claude model + reasoning-effort tier. Rationale for model
choices at the bottom.

Legend: model ∈ {Sonnet 5, Opus 5, Fable 5} · effort ∈ {low, medium, high, xhigh}.

---

## T1 — The flip + E2E baseline (first action post-PR)
**Steps:** apply the ModelManager hook (INSTALL_NATIVE.md Step 1, try/except-fallback variant);
commit the two staged files + hook in transcribe-app; build/verify `diar-server:latest`; bring
up sidecar (compose overlay); `DIARIZER_ENGINE=native`; upload test_videos through the app;
kill-sidecar fallback drill; then baseline: `ENABLE_BENCHMARK_TIMING=1`, run the
SPEEDUP_ROADMAP corpus ×3 for python vs native (fast set) vs native (small set); pull
`user_perceived_duration_ms` + per-stage columns via admin_timing API.
**Gate:** app-level outputs verified (transcript, speakers, OpenSearch embeddings); **Karpathy
10-minute clip** (`karpathy_10m.wav`, hand-labelled `reference.rttm` midpoint-mapped, smoothing
ON, `benchmark_boundary.py`) WSER ≤ 0.27% — state the ASR model with the number, since the bar
was measured on that clip and not the 66.5-min one; on the full clip the gate is **parity with
the fork** (within noise), not an absolute threshold. Fallback drill passes; baseline tables land
in RESULTS §7.
**STATUS: DONE 2026-08-19** — both engines 0.27% on the 10-min clip, native 0.890% vs fork
0.859% on the full clip, E2E baseline 1.40× faster upload→presented on 2.1 h (RESULTS §7.1-7.10).
**Model/effort: Opus 5 / high** (multi-service orchestration + judgment on anomalies; the
procedure itself is fully written). Sonnet 5/high acceptable if runs go clean.

## T2 — L2 true transcribe∥diarize overlap
**Steps:** in `_GpuRawStage.run` (stages.py:333-493): when native engine active, POST /diarize
(async thread) right after audio load, join after transcription; remove diarizer-preload path
for native; keep sequential fallback. Respect timing markers.
**Gate:** outputs identical to sequential run (same RTTM/DiarizeResult); GPU-stage wall ≈
max(stages) on the 2.2h benchmark file; timing columns still populate.
**Model/effort: Fable 5 or Opus 5 / high** — touches the production engine's concurrency;
subtle failure modes (error propagation from the async diarize, OOM interplay).

## T3 — L3 progressive presentation
**Steps:** emit `transcript_ready` WS event at finalize.py:344; frontend: on that event (or
0.75+ progress tick) call fetchTranscriptData() (notificationHandler.ts); verify speaker-label
late-attach via existing `speaker_updated` handlers; keep redaction gating untouched
(crud.py:588 path); UX copy for interim "Speaker N" labels.
**Gate:** transcript visible at ~0.78 in a live run; labels update in place on completion;
redaction-enabled files unchanged; no double-fetch races.
**Model/effort: Sonnet 5 / high** — full-stack but small and precisely specified.

## T4 — L4 finalize off the GPU worker
**Steps:** default fast path currently calls run_gpu_stage()+run_cpu_finalize() both in the GPU
task (pipelines.py:273-274). Split: GPU task returns RawInferenceResult; finalize runs in the
existing cpu-side machinery (the split classes already exist). Verify Redis payload sizes
acceptable (job.py notes 0.6-2.5 MB for 4.7h).
**Gate:** identical outputs; GPU task wall drops by the finalize duration on benchmark files;
no regression in single-user latency (finalize starts immediately on cpu worker).
**Model/effort: Opus 5 / high** — plumbing change in production task topology.

## T5 — L5a gender queue-priority fix (immediate) + L5b gender-in-sidecar (post-flip)
**L5a:** change detect_speaker_attributes dispatch priority/queue (speaker_attribute_task.py:153,
celery.py:236) so it can't compete with PIPELINE_CRITICAL tasks. Gate: enrichment ordering
unchanged, no starvation of gender task itself. **Sonnet 5 / low.**
**L5b:** export Common-Voice-Gender-Detection (wav2vec2) to ONNX; parity fixture vs PyTorch;
add `/classify_gender` to diar-server (per-speaker windows from held PCM during diarize, or
standalone endpoint); rewire speaker_attribute_task to call sidecar; delete presigned-URL clip
fetching. Gate: same gender decisions on a fixture set; enrichment chain latency drop measured.
**Model/effort: Fable 5 or Opus 5 / high** (Rust + export parity + app rewiring).

## T6 — L9+L7 telemetry completeness + batched progress writes
**Steps:** add benchmark markers for summary/clustering/search_index; add the two missing
entries to `_TIMESTAMP_MARKERS`; persist `stage_timings` dict into file_pipeline_timing;
batch/async the per-emit DB writes (pipelines.py:268-271) — coalesce to ≤1 write/2s.
**Gate:** baseline report shows no dead columns; progress UX unchanged.
**Model/effort: Sonnet 5 / medium.**

## T7 — L6 precompute_vad implementation
**Steps:** implement Silero VAD in `_PreprocessStage` (or sidecar P2), populate
PreprocessResult.vad_regions (stages.py:327); consume in transcriber (pass clip list or use
regions to skip) and later in diarization window gating. Start Python-side (fastest), Rust
port with preprocessing consolidation later.
**Gate:** WSER unchanged on the acceptance clip; measurable GPU-stage reduction on
silence-heavy media; regions telemetry recorded.
**Model/effort: Opus 5 / high** (accuracy-sensitive interaction with whisper chunking).

## T8 — L8 VAD silence-knob test
**Steps:** WSER-harness sweep vad_min_silence_ms ∈ {500, 1000, 2000} on the acceptance corpus.
Config-only; pick by WSER + latency. **Model/effort: Sonnet 5 / medium.**

## T9 — Remaining M1 Rust items (speakrs internals)
**(a) Arc-shared sessions** (concurrency without N× VRAM), **(b) speaker-count constraints
port** (k-means path), **(c) supervisor/idle-recycling polish**, **(d) symphonia decode (P2)**.
Each gates on the Phase-B suite (DER ≤0.3% drift, determinism ×3).
**Model/effort: Fable 5 / xhigh for (a),(b)** — deep vendored-crate surgery where today's
session showed the value of maximal reasoning (VBx vectorization class of work);
**Opus 5 / high for (c),(d)**.

## T10 — Upstream PR submission (two PRs per gameplan)
**Steps:** UPSTREAM_PRS.md Step 0-4 (fork, rebase-check, intro issue, branch split, per-patch
isolation re-validation, submit PR-A then PR-B, file issues).
**Gate:** each branch passes isolated validation on upstream tip before submission.
**Model/effort: Opus 5 / high** (mechanics are scripted; judgment needed on rebase conflicts
and maintainer interaction drafts). Fable 5 if upstream tip has diverged heavily.

## T11 — TRT EP inside ort for the multimask session (post-flip perf experiment)
**Steps:** enable ort TensorRT EP + engine cache on the multimask/seg sessions (fixed shapes —
rebuild-storm precondition absent); measure warm E2E vs CUDA EP; keep CUDA EP fallback.
**Gate:** RTTM identity; warm E2E improvement recorded; engine-cache behavior across restarts.
**Model/effort: Fable 5 / xhigh** — highest-risk perf work (EP interplay, provider pairing;
we already hit the rc.12/rc.13 class of trap).

## T12 — Cross-file speaker clustering (RUST_SERVICES_PLAN §6)
**Steps:** FIRST locate user's prior research (GH issues/docs) and profile the current
matching service; then ANN index (usearch/hnsw_rs) prototype + decision-parity fixture vs
speaker_matching_service; batch re-resolution design after.
**Gate:** identical match decisions on fixture library; measured query/batch speedups.
**Model/effort: Fable 5 / high** for design+parity; Sonnet 5 for the profiling leg.

## T13 — Text-model ladder (profile → optimum-ORT → absorb)
**Steps:** profile NLP/redaction workers on the baseline corpus; optimum-ORT the provably hot
models in place; only then consider text-native per §3b.
**Model/effort: Sonnet 5 / medium** (profiling + mechanical conversions); escalate to Opus 5
if absorption (Rust) is justified.

---

## Model-choice rationale (honest guidance)

- **The heavy docs are the equalizer**: most tasks above are specified to the file:line level
  with written gates — that's exactly what makes **Sonnet 5 effective and economical** for the
  mechanical/config/telemetry/frontend tasks (T3, T5a, T6, T8, T13).
- **Opus 5 / high** for production-plumbing changes where the spec is complete but failure
  modes need judgment (T1, T2*, T4, T7, T10).
- **Fable 5 (this model) / xhigh** where today's session demonstrated its specific value:
  novel Rust internals surgery with numerics-parity stakes (T9a/b), EP/runtime trap-rich perf
  work (T11), and algorithm-design with correctness proofs (T12). These are the tasks where a
  wrong-but-plausible change silently costs accuracy — maximal reasoning pays for itself.
- Effort tiers: default **high** for anything touching production behavior; **xhigh** only for
  vendored-Rust/numerics/EP work; **medium/low** for config, docs, sweeps, and telemetry.
- Universal rule regardless of model: every task ends with its written gate (output-identity or
  accuracy harness) — the discipline, not the model, is what kept today error-free.
