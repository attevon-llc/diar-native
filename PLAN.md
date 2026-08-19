# Execution Plan — Native Diarization for OpenTranscribe

The living roadmap. Companion docs: [README](README.md) (context/why/tiers),
[validation/TESTPLAN.md](validation/TESTPLAN.md) (test matrix + gates),
[validation/RESULTS.md](validation/RESULTS.md) (every measurement). Status markers updated as
work lands. Origin: approved plan of 2026-08-18/19 (Claude session), amended by product
decisions below.

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
- **Phase B — speakrs validation: NEARLY DONE.** G1 ✅ G2 ✅ G5 ✅ (bit-determinism);
  G3 partial (2.2h 0.18% A/B ✓, 1.0h ✓, 0.5h phantom-5th-speaker flagged; 3.2h/4.7h scoring in
  progress); G4 flipped to expected-pass by the patch set (3.1× E2E; 89× RT on 3080 Ti vs fork
  80× on A6000) — final verdict from the **quiet-machine timing pass** (mandatory; see RESULTS
  §4.11 contamination lesson). Outstanding: speakrs VoxConverse dev leg; 0.5h speaker diagnosis;
  batched-fbank-graph numeric deviation quantification (RESULTS §4.16 area).
- **Phase B5 — go/no-go report: pending** (assembled from RESULTS once the above land).

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
- **M1 diar-core** + upstream PRs (list below). Gate: re-run full Phase-B suite vs our own
  RTTMs — DER ≤ 0.3%, determinism, constraint-path fixture tests.
- **M2 diar-server + OpenTranscribe integration**: compose service; `diarizer_native.py`
  implementing the `SpeakerDiarizer` contract (`DiarizeResult`, 256-d L2-normalized
  `native_embeddings` via existing `build_native_embeddings`, `overlap_info`), selected by
  `DIARIZER_ENGINE=native|python` (default python). Gate: TRUE E2E — full compose stack, upload
  `test_videos/` through API → Celery `gpu-diarize` → OpenSearch; kill-sidecar fallback drill;
  Karpathy WSER ≤ 0.27% smoothed at app level.
- **M3 hardening**: shadow mode (both engines, log diffs) → default flip after a clean week;
  CUDA + CPU-only images; laptop-class validation.
- **M4 (T2)**: Triton repo productionization — accuracy-correct community-1 TRT engines
  (rebuild from our exports), `diar-ffi` backend or sidecar-calls-Triton, per-arch engine build
  job (sm_86 local+g5 / sm_89 g6), AWS compose profile. M11 full-pipeline concurrency measured
  here.

## Upstream PR queue (speakrs, with our benchmark receipts)

1. **Multimask batching fix** — exporter/loader b32-vs-b64 filename mismatch silently disables
   batching (RESULTS §4.15).
2. **fbank session-pool fan-out** — 3.1× E2E on CUDA (RESULTS §4.16; patches already in
   `vendor/speakrs`, gated by `SPEAKRS_FBANK_POOL`/`SPEAKRS_FBANK_THREADS`).
3. **Folded segmentation graph** in export script (bit-exact, −7% E2E, 2× on ORT-CUDA serving).
4. **Arc-shared sessions / concurrent pipeline** (decision #4) — after M1 design settles.
5. Bug report: ORT-CUDA teardown crash (`corrupted double-linked list`); batched-fbank-graph
   numeric deviation vs single-chunk graph.

## Reference pointers

- Fork bugs found (report to fork owner = us): CPU-only Linux crash in `_gpu_empty_cache`
  (RESULTS §5.6); production image ORT-GPU cu13/cu12 mismatch (RESULTS §4.3); stale ONNX
  artifacts exported from wrong checkpoints (RESULTS §1).
- Decision history + failure post-mortems of the 2026-Q1 ONNX attempt:
  `transcribe-app/docs/upstream-patches/` (treat as hypotheses; every load-bearing claim was
  re-verified here — see RESULTS "evidence policy").
