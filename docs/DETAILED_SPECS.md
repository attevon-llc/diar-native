# Detailed Implementation Specs — encoding session knowledge so smaller models can execute

Purpose: these specs downgrade the required model tier for the hardest tasks (EXECUTION_TASKS
T2/T5b/T9/T11) by spelling out exact code shapes, traps already hit, and validation sequences.
Written by Fable 5 with the full session context loaded (2026-08-19). With these specs,
**Opus 5/high executes everything below; Sonnet 5/high suffices for T2 and T5b.**

---

## S-T2: transcribe∥diarize overlap in `_GpuRawStage.run` (stages.py:333-493)

Key insight that simplifies everything: **preprocess already wrote the 16 kHz WAV to the shared
volume** (`load_from_shared_volume`, audio_loader.py:68). The sidecar can read THAT path —
zero handoff cost, no new WAV write (diarizer_native's write-path is only for the
in-postprocess topology).

Shape (native engine active; else keep existing sequential code untouched):
```python
# after audio load (:360-368), before transcriber load:
diar_result: dict = {}
diar_error: list = []
def _diarize_async():
    try:
        diar_result["out"] = _post_json(f"{base}/diarize",
            {"wav_path": local_wav_path, "file_id": task_id}, timeout=...)
    except Exception as e:
        diar_error.append(e)
diar_thread = threading.Thread(target=_diarize_async, daemon=True); diar_thread.start()
# ... transcriber load + transcribe as today ...
diar_thread.join(timeout=remaining_soft_time_budget)
if diar_error or not diar_result:
    # FALLBACK: sequential in-process diarization exactly as the current code path
```
Rules: (1) do NOT hold the DB session across the join; (2) emit(0.55) at join start so progress
semantics keep meaning; (3) benchmark markers: `mark("diarize_request_sent")` at thread start,
`mark("diarize_joined")` after join — add both to `_TIMESTAMP_MARKERS`; (4) the VRAM-gate calls
(`_wait_for_vram`, :369/:442) are SKIPPED on the native path (diarization VRAM is the sidecar's);
(5) OOM-retry ladder (`diarizer.py:26`) is not applicable — sidecar owns memory. Map the JSON
response through the same conversion used in diarizer_native.diarize() — factor that mapping
into a shared helper in diarizer_native.py so both call sites share it.
**Gate:** RTTM/DiarizeResult identical to sequential native run on 2.2h file; GPU-stage wall ≈
max(transcribe, diarize) ± 5%; fallback drill (sidecar stopped mid-run) completes via fork path.

## S-T5b: gender-in-sidecar

1. Export: `optimum-cli export onnx --model prithivMLmods/Common-Voice-Gender-Detection
   --task audio-classification out/`. Wav2vec2 preprocessing = per-clip zero-mean/unit-variance
   normalization ONLY (verify `feature_extractor.do_normalize=true`, no fbank) — trivial in
   Rust: `x = (x - mean) / (std + 1e-7)`. Record `config.id2label` order in a constants file.
2. Parity fixture: 20 clips (mix of speakers from test corpus) → transformers pipeline labels +
   logits vs ONNX (Python ORT first, then Rust) — logits within 1e-4, labels identical.
3. diar-server: `POST /classify_gender {wav_path|samples_b64, windows: [[s,e],...]}` →
   `{per_window: [{label, score}], majority: label}`. Session: CPU EP, intra 4 threads.
   Mirror app semantics: up to 5 windows per speaker, majority vote
   (speaker_attribute_task.py:317-320).
4. Rewire `speaker_attribute_task._run_gender_inference_parallel` (:113) to call the sidecar
   with the segment windows + the file's audio path — deletes the presigned-URL fetch pool.
   Keep the old path behind the same DIARIZER_ENGINE flag family (`GENDER_ENGINE=native`).
**Gate:** identical gender decisions on the fixture set; enrichment-chain latency measured
before/after; LLM speaker-ID dispatch time improves.

## S-T9a: Arc-shared sessions in speakrs (concurrency without N× VRAM)

**IMPLEMENTED 2026-08-19 (RESULTS §7.25). CORRECTION:** option 1 as originally written
("mutex per session, buffers stay put") **cannot deliver concurrency on this code shape** —
`DiarizationPipeline` borrows `&'a mut SegmentationModel`/`&'a mut EmbeddingModel` for its
whole lifetime (pipeline.rs:197), so locks acquired through those borrows are held per-job and
two jobs serialize exactly as before. The weights/scratch split is not an escalation, it is
the precondition for any sharing at all.

What landed (same effect as the SharedModels/Scratch split, smaller blast radius): sessions
become `Arc<Mutex<Session>>` (`SharedSession`, `inference.rs`), locked for exactly one `run()`
per inference call; the model STRUCTS themselves become the per-request scratch. Each model
gains `clone_shared()` — Arc-clones the 13 sessions, re-allocates the ~14 staging buffers
(~130 MB host RAM) and a fresh `primary_batch_run_options` (its preallocated output tensor
makes RunOptions per-handle by construction — the original warning stands).
`DiarEngine::clone_shared()` composes seg+emb+gender handles; diar-server keeps a prototype
engine under a mutex it locks only long enough to clone a handle per request. No method
signature changes; pipeline code untouched.
Contention profile (measured): same-session jobs serialize per batch; cross-session work
overlaps. The unsafe `Sync` wrapper (concurrent `Run` on one session; ORT C API allows it)
remains the escalation if per-batch locking measures as the binding constraint on a QUIET
machine.
**Gate:** N=4 concurrent diarize jobs on one engine: outputs identical to serial — **PASSED
4/4, three separate runs**; VRAM ≈ single engine + N×scratch — **PASSED, 4 510 MiB peak under
4 jobs ≈ one engine, zero per-job VRAM growth**; throughput ≥ 2× serial on 4 short files —
**1.51× measured on a non-quiet machine (load 10-13), re-measure quiet before judging**;
harness: `validation/t9a_concurrency.sh`.

## S-T9b: speaker-count constraints (pyannote parity)

Reference semantics (pyannote clustering.py L1004-1024): after VBx produces clusters, let
`found = kept_speakers.len()`. If `num_speakers` forced and != found, or found < min_speakers,
or found > max_speakers → target = forced or clamp(found, min, max); re-cluster
`train_embeddings` (the SAME filtered+L2-normalized rows fed to AHC) with k-means k=target
(k-means++ init, seeded RNG — sklearn uses random_state=42; bit-parity with sklearn is NOT
required, gate on cluster count + DER); centroids = per-cluster MEANS (not gamma-weighted);
hard assignment for gamma (one-hot); then proceed to `assign_chunk_embeddings` UNCHANGED.
Plumbing: `PipelineConfig` gains `num_speakers/min_speakers/max_speakers: Option<usize>`;
diar-core `EngineConfig` + server request fields forward them; diarizer_native passes
config.num_speakers/min/max (removing the current warning path).
Note: with app defaults (1..20) this fires ~never — fixture tests use forced counts.
**Gate:** forced-count fixtures (force 2 on a 4-speaker file etc.) produce exactly k clusters;
unforced behavior bit-identical to current (feature is pure addition).

## S-T11: TensorRT EP inside ort (multimask + segmentation sessions)

**IMPLEMENTED + MEASURED 2026-08-19, then ROLLED BACK by operator decision (RESULTS §7.26 has the numbers and the recipe).**
Corrections learned: (a) the ORT 1.24.2 tarball DOES ship the TRT provider — what's missing is
`libnvinfer.so.10`/`libnvonnxparser.so.10` (now behind `--build-arg WITH_TENSORRT=1`, TRT
10.16.1 cuda12.9 — the repo's cuda13 default candidate will not load); (b) the bit-parity gate
below is unachievable in principle: TRT kernels shift logits at the last ulp and binarized
boundaries move ~1 frame. Measured cost/benefit: 1.33-1.48× warm diarization speed; AMI-16
full +0.030 pp, exclusive +0.006 pp, Karpathy −0.002 pp; runs byte-deterministic within TRT;
cache-hit restart 6 s vs ~7 min build; libs-absent fallback byte-identical to CUDA EP. Runtime
gate `SPEAKRS_TRT=1`, cache at `SPEAKRS_TRT_CACHE` (default `/tmp/diar-native/trt_cache` — the
persistent handoff volume, NOT the ro models mount). Original spec follows:

Config (ort rc.12): register providers in order [TensorRT, CUDA, CPU] on the target sessions:
```rust
TensorRTExecutionProvider::default()
    .with_engine_cache(true)
    .with_engine_cache_path("/models/trt_cache")   // MOUNTED VOLUME — survives restarts
    .with_fp16(false)                              // DER invariant: fp32 only
    .with_max_workspace_size(2 << 30)
```
Facts that de-risk this (hard-won): (1) our graphs are FIXED-SHAPE (b32) → no shape profiles
needed, the historical rebuild-storm precondition is absent; (2) provider libs must come from
the SAME ORT release tarball as the runtime pairing — Dockerfile.server already copies the full
1.24.2 lib set incl. `libonnxruntime_providers_tensorrt.so` (verify present; if the MS tarball
lacks it, TRT EP needs the TensorRT libs in-image: add `libnvinfer` from NVIDIA apt or switch
base to a TRT-bearing image — CHECK FIRST); (3) ort minor-version traps are real (rc.12 pin,
Cargo.lock) — do NOT bump ort while doing this; (4) first-run engine build takes ~seconds-min →
server must warm up at boot (run one dummy inference per TRT session before /healthz reports
ready); (5) apply to multimask-b64(b32) + segmentation sessions first; leave fbank/tail on CPU.
**Gate:** RTTM bit-parity vs CUDA EP on clip30 + ES2004a + 2.2h; warm E2E delta recorded;
restart → engine cache hit (no rebuild; log inspection); fallback to CUDA EP works when
trt_cache volume absent.

## S-T4 addendum (finalize off GPU worker)

> **CLOSED — negative result, measured 2026-08-19 (RESULTS §7.27).** The premise below is that
> the CPU-only tail holds a scarce GPU slot. It does not: the GPU worker runs
> `--pool=threads --concurrency=8`, and `ModelManager`'s lock guards model *loading*, not
> inference, so nothing serialises per task. The tail is 4.01 s of a 57.9 s job (finalize 2.18 s
> + save 1.83 s) and sits on a thread slot, not the GPU. Splitting would add a 0.6-2.5 MB Redis
> hop and put the user-visible save behind the CPU queue, which S-T4's own gate forbids. Revisit
> only under a prefork pool or a per-task GPU semaphore. Spec retained below for that case.


The chain already has a 3-link structure; the fast path merely fuses stages (pipelines.py:
273-274). Split = call `run_gpu_stage()` only in the GPU task, return `RawInferenceResult`
(job.py:117 — 0.6-2.5 MB JSON via Redis for 4.7h: acceptable), and let `postprocess` (cpu)
invoke `run_cpu_finalize()` before its speaker-matching step. Watch: `stage_timings` merge from
both halves; progress ladder handoff (0.65-0.78 moves to cpu task); `TIMING:` log continuity.
**Gate:** identical outputs; GPU task ends right after diarize; user_perceived_duration
unchanged or better; throughput test with 2 queued files shows GPU busy on file 2 while file 1
finalizes.

---

## Updated model/effort matrix (post-specs)

| task | was | NOW |
|---|---|---|
| T2 overlap | Fable/Opus high | **Sonnet 5 high** (S-T2) |
| T5b gender sidecar | Fable/Opus high | **Sonnet 5 high** (S-T5b; Rust endpoint is pattern-copy of embed_window) |
| T9a shared sessions | Fable xhigh | **Opus 5 high** (S-T9a option 1; escalate only for option 2) |
| T9b constraints | Fable xhigh | **Opus 5 high** (S-T9b) |
| T11 TRT EP | Fable xhigh | **Opus 5 high** (S-T11; escalate to Fable only if the provider-lib check fails and base-image surgery starts) |
| T4 finalize split | Opus high | Opus 5 high (S-T4 addendum) |
| T1 flip+baseline, T10 PRs | Opus high | unchanged |
| T3, T5a, T6, T8, T13 | Sonnet | unchanged (Sonnet 5, low-high per EXECUTION_TASKS) |
| T12 corpus clustering | Fable high | **still Fable-or-Opus** — design work pending the user's prior research; spec it in a future Fable session AFTER that research is located, then downgrade |

Escalation rule: any task whose gate fails twice on the assigned model → escalate one tier with
the failure transcript. Never skip a gate to save tokens — a silent accuracy regression costs
more than any model differential.

---

## S-T2 CORRECTION (measured 2026-08-19, during T1)

Two premises in S-T2 above are wrong for the deployment we actually ship, and T2 must be
re-scoped before it is written:

1. **The default path is `_GpuStage`, not `_GpuRawStage`.** `Engine.process()` (engine.py:87-90)
   runs the **fused** `_GpuStage`, and that is what the live compose profile uses.
   `_GpuRawStage` (stages.py:333-493) is only reached through `run_preprocess` +
   `run_gpu_stage`, i.e. the gpu-split profile and the `benchmark_boundary.py` harness. Editing
   only `_GpuRawStage` would leave production untouched.
2. **"Preprocess already wrote the 16 kHz WAV, so the handoff is free" does not hold there.**
   Only `_PreprocessStage` writes the shared-volume WAV (stages.py:315); the fused `_GpuStage`
   decodes `job.audio_path` itself and holds the audio in memory. In the default topology there
   is no shared-volume WAV for the sidecar to read.

   (Related: those shared volumes were root-owned in every deployment until RESULTS §7.10 — so
   even on the split path the write had been failing silently.)

**Correction to this correction (same day, after the volume was repaired).** Points 1 and 2
above described a *broken* deployment, not the intended one. `_run_engine_pipeline` only takes
the `_GpuRawStage` fast path when the shared-volume WAV exists (core.py:143-185) — and it never
did, because the volume was root-owned (§7.10). Once ownership was fixed the fast path engaged
and production moved to `_GpuRawStage` with the preprocess WAV present, exactly as S-T2
originally assumed. **S-T2 was right; the deployment was broken.**

T2 as implemented therefore covers both: `_GpuRawStage` (production) and `_GpuStage` (the fused
fallback), sharing one `_AsyncDiarization` helper. Verified in RESULTS §7.13.
