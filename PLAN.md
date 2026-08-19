# Execution Plan — Native Diarization for OpenTranscribe

The living roadmap. Companion docs: [README](README.md) (context/why/tiers),
[validation/TESTPLAN.md](validation/TESTPLAN.md) (test matrix + gates),
[validation/RESULTS.md](validation/RESULTS.md) (every measurement). Status markers updated as
work lands. Origin: approved plan of 2026-08-18/19 (Claude session), amended by product
decisions below.

## North star (stated 2026-08-19)

**Minimize upload→presented latency at maximum accuracy, on whatever machine the app is
installed on.** Every optimization is judged against the per-tier E2E job timeline.

**MEASURED at the flip (RESULTS §7.5, 3080 Ti, 5 files × 3 runs, controls held):** upload→presented
is **1.40× faster on a 2.1 h file** (206.7 s → 147.2 s, 37× → 51× RT) and 1.35× on 66.5 min;
the diarization stage itself is **2.01× faster** (116.6 s → 58.0 s) at unchanged transcription.
Bottleneck ordering has flipped exactly as predicted: on the 2.1 h file diarization was 62% of
the GPU stage and is now 45%, so (1) **transcription** is the clear #1 (levers: batch re-tune with
freed VRAM, VAD gating, Parakeet tier — int8_float16 is already on), (2) NLP/aux stages,
(3) preprocessing/handoff, (4) diarization (SOLVED). The stages still run **sequentially**, so
L2 transcribe∥diarize overlap is worth a further ~45% of the GPU stage on its own — the largest
remaining lever, now quantified. Also on the list: **progressive presentation** (transcript shown
at ASR-complete, speaker labels attach after) — the biggest perceived-latency lever costs no
model speed at all.

## Product decisions (locked)

1. **Adopt speakrs** (validated: G1/G2 accuracy parity, patched-build speed > fork) as the engine
   inside a new wrapper, vendored + pinned; upstream PRs for our fixes.
2. **Two deployment tiers** (README §6): T1 shared-weights sidecar = DEFAULT open-source path
   (laptops → single-GPU boxes); T2 Triton + TRT = opt-in for large home servers + AWS.
3. **Celery owns ALL job orchestration** (queue/retry/priority/routing). The sidecar is a
   stateless executor with only an admission semaphore (max in-flight) — mirrors the current
   worker model-manager. speakrs' internal queue is not exposed.
4. **Shared weights are a core requirement**, not an option: Arc-shared ORT sessions (thread-safe
   `run()`, weights loaded once) + per-request scratch buffers → N concurrent diarizations
   without N× VRAM (parity with the old Celery shared-weights PyTorch pattern).
5. **End-state removes the pyannote fork** for diarization: Rust binary + ORT + ~33 MB ONNX
   models replace the pinned fork; fork path stays config-flagged until shadow-mode proves T1,
   then retires (ends fork maintenance). Torch may remain in the backend image for OTHER
   features (alignment/NLP) — separate audit later.
6. Production repos (`transcribe-app`, `pyannote-audio-fork`) are read-only from this project.
7. Model artifacts derive from gated community-1 weights: self-export, private hosting, never
   committed/redistributed.

## Phase status

- **Phase A — stock-fork baselines: DONE** (Gate 0 passed; AMI 13.093%, Karpathy 8.194%,
  VoxConverse dev 5.099%, duration curve, CPU leg — RESULTS §2).
- **Phase B — speakrs validation: DONE (2026-08-19). FINAL VERDICT: GO** (validation/REPORT.md).
  G1 ✅ AMI 13.100 vs 13.093; G2 ✅ Karpathy 8.219 vs 8.194; G5 ✅ bit-determinism; G3 synthetic
  miss adjudicated by the **VoxConverse arbiter: speakrs 4.847% BEATS fork 5.099%** (211/216
  speaker-count agreement, 95-38 per-file DER wins); G4 quiet-machine: **1.26× faster (2.2h),
  1.2× faster (Karpathy)**, 4.7h extreme file 1.39× slower — attributed to AHC linkage at
  N≈50k (74% of wall). Patched build verified accuracy-preserving on every corpus.
- **Phase B5 — go/no-go report: DONE** — REPORT.md carries the final gate table + verdict.

## Phase C — adoption (next; ~3-4 weeks part-time)

Repo layout target (crates to be created here):

```
crates/diar-core/    # speakrs wrapper: Arc-shared sessions + per-request buffers (decision #4),
                     #   per-speaker centroids out, embed_window(), num/min/max-speaker
                     #   constraints (port VBxClustering L1004-1024 k-means path),
                     #   dual outputs (full + exclusive diarization)
crates/diar-server/  # T1 sidecar: HTTP/gRPC /diarize /embed_window /healthz, admission
                     #   semaphore, CUDA + CPU-only builds
crates/diar-cli/     # bench/ops runner (RTTM out, --dump-stages)
crates/diar-ffi/     # C-ABI cdylib → T2 Triton custom backend (Triton backend API is C)
```

Milestones:
- **M1 diar-core: CORE LANDED 2026-08-19** (crates/diar-{core,cli,server} built + gate passed:
  AMI 13.101% identical, 16/16 content-identical RTTMs, centroids/embed_window/exclusive verified,
  31 MB binaries — RESULTS §4.26). Remaining M1: speaker-count constraints, Arc-shared sessions,
  lazy session loading, server supervisor. + upstream PRs (list below). **T1 SHIP-GATE: CLEARED 2026-08-19** —
  clustering optimization (VBx vectorization + threaded pdist) makes speakrs 2.03× faster than
  the fork on the 4.7h file (RESULTS §4.22-4.24); G4 = clean sweep on all files. Remaining M1
  gate: re-run full Phase-B suite after diar-core wrapper lands — DER ≤ 0.3%, determinism,
  constraint-path fixture tests.
- **M2 diar-server + OpenTranscribe integration: DONE 2026-08-19 (T1)** — flipped on the live
  stack, app-level outputs verified (transcript, speakers, OpenSearch embeddings, profile
  auto-match), kill-sidecar drill passes in both directions, E2E baseline measured (RESULTS
  §7.1-7.7). Three defects found and fixed: root-owned handoff volume, one-way fallback, and
  speakrs' exclusive-diarization overlap resolution (§7.7 — the accuracy gate paid for itself).
  Accuracy at app level now within 0.03 pp WSER of the fork. Original M2 scope below:
- **M2 (original scope)**: compose service; `diarizer_native.py`
  implementing the `SpeakerDiarizer` contract (`DiarizeResult`, 256-d L2-normalized
  `native_embeddings` via existing `build_native_embeddings`, `overlap_info`), selected by
  `DIARIZER_ENGINE=native|python` (default python). Gate: TRUE E2E — full compose stack, upload
  `test_videos/` through API → Celery `gpu-diarize` → OpenSearch; kill-sidecar fallback drill;
  Karpathy **10-min** clip WSER ≤ 0.27% smoothed at app level (that clip is where the number
  comes from — RESULTS §7.9), and parity with the fork on the 66.5-min clip.
- **M3 hardening**: shadow mode (both engines, log diffs) → default flip after a clean week;
  CUDA + CPU-only images; laptop-class validation.
- **M4 (T2)**: Triton repo productionization — accuracy-correct community-1 TRT engines
  (rebuild from our exports), `diar-ffi` backend or sidecar-calls-Triton, per-arch engine build
  job (sm_86 local+g5 / sm_89 g6), AWS compose profile. M11 full-pipeline concurrency measured
  here.

## Post-flip optimization sequence (ACCEPTED 2026-08-19 — full detail: docs/E2E_PIPELINE_MAP.md)

User-confirmed direction: progressive updates, async task loads, and downstream overlap
wherever outputs allow. Execution order after the flip + baseline run:
1. L2 true transcribe∥diarize overlap (sidecar makes it a small change; GPU stage → max not sum)
2. L3 progressive presentation (transcript_ready at 0.78; labels late-attach via existing
   handlers; redaction stays gating)
3. L4 finalize off the GPU worker (stop holding the GPU slot for CPU work)
4. L5a gender-task QUEUE-PRIORITY fix now (one line — stop competing with the critical path);
   L5b **gender-in-sidecar promoted to early post-flip** (shares held PCM, kills presigned-URL
   clip fetches, unblocks LLM speaker-ID sooner — visible enrichment latency)
5. L9+L7 telemetry completeness + batched progress writes (baseline hygiene)
6. L6 precompute_vad implementation; L8 VAD silence knob WSER-test
7. Text-model Rust absorption stays PROFILE-GATED (ladder in RUST_SERVICES_PLAN §3)
- **L10 ASR int8 quant: TESTED BY USER — REAL DEGRADATION OBSERVED → last-if-at-all**
  (user's measurement overrides literature expectations, per evidence policy).
Each lever gates on output-identity/accuracy checks (WSER/DER harness) and is judged by
`user_perceived_duration_ms` per tier.

## Upstream PR queue (speakrs, with our benchmark receipts)

1. **Multimask batching fix** — exporter/loader b32-vs-b64 filename mismatch silently disables
   batching (RESULTS §4.15).
2. **fbank session-pool fan-out** — 3.1× E2E on CUDA (RESULTS §4.16; patches already in
   `vendor/speakrs`, gated by `SPEAKRS_FBANK_POOL`/`SPEAKRS_FBANK_THREADS`).
3. **Folded segmentation graph** in export script (bit-exact, −7% E2E, 2× on ORT-CUDA serving).
4. **Arc-shared sessions / concurrent pipeline** (decision #4) — after M1 design settles.
4b. **VBx vectorization + threaded blocked pdist** — DONE in the local patch set (8× clustering,
   4.7h 474→171.6 s, RTTMs bit-identical, parity fixtures green; RESULTS §4.22-4.24). This is
   now the flagship perf PR alongside the fbank pool.
5. Bug report: ORT-CUDA teardown crash (`corrupted double-linked list`); batched-fbank-graph
   numeric deviation vs single-chunk graph.
6. **Exclusive-diarization correctness** (RESULTS §7.7, UPSTREAM_PRS PR-7) — `make_exclusive`
   runs on binarized activations, so every overlap tie resolves to the highest cluster index
   (100.0% of 22 297 sampled AMI overlap frames). Fixed by resolving overlaps on the continuous
   `frame_activations`: AMI exclusive DER 18.654% → **17.813%** (pyannote control 17.828%),
   confusion 2.655% → 1.814%, full diarization bit-identical on 16/16 files. Found by T1's
   app-level accuracy gate — the first defect the flip surfaced.

## Reference pointers

- Fork bugs found (report to fork owner = us): CPU-only Linux crash in `_gpu_empty_cache`
  (RESULTS §5.6); production image ORT-GPU cu13/cu12 mismatch (RESULTS §4.3); stale ONNX
  artifacts exported from wrong checkpoints (RESULTS §1).
- Decision history + failure post-mortems of the 2026-Q1 ONNX attempt:
  `transcribe-app/docs/upstream-patches/` (treat as hypotheses; every load-bearing claim was
  re-verified here — see RESULTS "evidence policy").
