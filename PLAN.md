# Execution Plan — Native Diarization for OpenTranscribe

The living roadmap. Companion docs: [README](README.md) (context/why/tiers),
[validation/TESTPLAN.md](validation/TESTPLAN.md) (test matrix + gates),
[validation/RESULTS.md](validation/RESULTS.md) (every measurement). Status markers updated as
work lands. Origin: approved plan of 2026-08-18/19 (Claude session), amended by product
decisions below.

## North star (stated 2026-08-19)

**Minimize upload→presented latency at maximum accuracy, on whatever machine the app is
installed on.** Every optimization is judged against the per-tier E2E job timeline.

**MEASURED after flip + overlap (RESULTS §7.13, 3080 Ti, 5 files × 3 runs, controls held):**
upload→presented is **1.99× faster on the 66.5-min reference** (108.4 s → 54.4 s, 37× → **73× RT**)
and **1.72× on 2.1 h** (206.7 s → 120.3 s, 37× → 63×). Contributions: the engine flip made
diarization itself 2.01× faster, then T2 overlap made the GPU stage cost max(transcribe, diarize)
instead of the sum. T3 shows the transcript ~1.6 s before completion on top of that.

Bottleneck ordering now: (1) **transcription** is the clear #1 and is no longer hidden behind
diarization — remaining levers are batch re-tune with freed VRAM, VAD gating (T8), and the
Parakeet tier; int8_float16 is already on. (2) NLP/aux stages, (3) preprocessing/handoff,
(4) diarization — **SOLVED, and now free**: it runs inside transcription's window, so further
diarization speed buys nothing on single-job latency. It only matters again for throughput,
which is gated by T9a, not by the engine. The short-file floor is ~4 s of fixed per-job
overhead, which is what limits media under a minute.

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
5. **End-state removes the pyannote fork** for diarization: Rust binary + ORT + ONNX models
   replace the pinned fork. (The two core graphs are ~33 MB, but a deployable `fast` set is
   **~484 MB** — the batched variants and the 189.5 MB gender classifier dominate; `--skip-gender`
   brings it to ~294 MB. Still against a ~9 GB image with full torch.)
   Fork path stays config-flagged until shadow-mode proves T1,
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
                     #   dual outputs (full + exclusive diarization); logging.rs, ort_compat.rs,
                     #   provision/ (the model exporter, marker and five-stage smoke test —
                     #   here, not in diar-server, so it is integration-testable)
crates/diar-server/  # T1 sidecar: HTTP /diarize /embed_window /healthz /readyz, admission
                     #   semaphore, CUDA + CPU-only builds; engines.rs = device registry,
                     #   one process serves cuda + cpu, selected per request (issue #1);
                     #   cli.rs = provision-models / verify-models / check-token subcommands
                     #   and the startup model gate (no subcommand = serve)
crates/diar-cli/     # bench/ops runner (RTTM out, --dump-stages)
crates/diar-ffi/     # C-ABI cdylib → T2 Triton custom backend (Triton backend API is C)
```

Milestones:
- **M1 diar-core: CORE LANDED 2026-08-19** (crates/diar-{core,cli,server} built + gate passed:
  AMI 13.101% identical, 16/16 content-identical RTTMs, centroids/embed_window/exclusive verified,
  31 MB binaries — RESULTS §4.26). **Arc-shared sessions (T9a): LANDED 2026-08-19** — all 13
  ORT sessions shared, per-request scratch handles, N=4 concurrent outputs identical to serial
  at one engine's VRAM (RESULTS §7.25; app-level flip + quiet-machine throughput leg pending).
  Remaining M1: speaker-count constraints (T9b), lazy session loading, server supervisor. + upstream PRs (list below). **T1 SHIP-GATE: CLEARED 2026-08-19** —
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
  - **Multi-device serving (issue #1): DONE 2026-08-31** — the CUDA image is a SUPERSET of the
    CPU image on amd64 and always was: `ort-sys` statically links the ORT CPU EP (no
    `libonnxruntime` NEEDED entry in the binary at all) and `ort/cuda` is purely additive, so
    serving CPU from the GPU image costs zero extra bytes and zero extra libraries. Verified
    empirically against the *already-shipped* `davidamacey/diar-native:0.2.0`: `DIAR_MODE=cpu`,
    no `--gpus`, no `/dev/nvidia*` → `/healthz`, `/diarize` and `/embed_window` all succeed.
    Added a startup-loaded engine registry (`crates/diar-server/src/engines.rs`), `DIAR_DEVICES`
    (comma list, first = default), a per-request `device` field, an `x-diar-device` response
    header, and JSON `/healthz` reporting loaded vs compiled-in devices. Defaults unchanged:
    unset ⇒ one engine from `DIAR_MODE`, one global semaphore, as before. Engine loads stay
    serial in `run()` before bind — `DiarEngine::load` calls `set_var`, which cannot race live
    tokio workers, so lazy per-request loading is unsound, not merely unoptimized.
    `docker/Dockerfile.server-cpu` STAYS: it is the arm64 / 189 MB artifact, not a correctness
    carve-out (the CUDA base image and its ORT tarball are x86-64 only).
  - **Deferred timed legs (issue #5): DONE 2026-09-01 (RESULTS §7.44–§7.48)** — the three timed
    legs issue #1 could not run on a loud box, all PASS, plus two bonus legs. B4 mixed-device
    concurrency: 4 CUDA `/diarize` beside 4 CPU `/embed_window` costs the CUDA leg **+0.5%**
    (median 38.13 → 38.31 s) against its own 4.3% spread, outputs identical, and reaching that
    contention at all required `DIAR_MAX_INFLIGHT=8` — **at the shipped default of 2 the outer
    semaphore serialises the mix, so the risk is not reachable without opting in.** B1: the
    `--features cuda` build costs CPU-mode inference nothing (52.54 → 51.98 s, inside a
    0.8 s spread), one RTTM MD5 across both builds. B2: a resident CPU engine costs the CUDA
    path +0.8% latency and **−4 MiB** VRAM. B5 **corrected a shipped claim** — fp16 gender saves
    **252 MiB, not the ~500 MiB** the CHANGELOG asserted by borrowing §7.18's whole-container AMI
    figure; `docs/ORT_FUSION_FP16_AARCH64.md` had been overstating the aarch64 fp32 fallback's
    cost by 2× as a result. B6: the aarch64 Level1 gender cap is free within resolution, because
    gender is **~0.16 s** of the call and not the ~1.5 s §7.18 records — so §7.41's cap stands on
    a measured bound, and `GeluFusionL2` does not need revisiting. Harnesses committed as
    `validation/b{1,2,4,5,6}_*.sh`.
  - **Structured server logging: DONE 2026-08-31 (RESULTS §7.37)** — `diar-server` shipped with
    NO `tracing-subscriber` and never installed one, so speakrs' 40 events and diar-core's 2
    warnings went nowhere and the operator saw two `eprintln!` lines and crashes. `RUST_LOG`
    now defaults to `info,ort::logging=warn` (unset must not mean silent; ORT's native bridge
    emits 5797 INFO lines per CUDA startup and would bury everything), `DIAR_LOG_FORMAT` selects
    the rendering, logs go to stdout, and every `/diarize` / `/embed_window` request gets a
    span (`request_id`, device, duration, outcome, `error_class`) that speakrs' own events
    nest under. Policy lives in `diar_core::logging` so the server and the CLI cannot drift.
  - **Model provisioning (issue #2): DONE 2026-08-31 (RESULTS §7.35, §7.36, §7.38)** — the last
    blocker to self-hosted OpenTranscribe running the native diarizer: there was no supported way
    for a third party to obtain the weights. `diar-server` gained `provision-models`,
    `verify-models` and `check-token`, all writing machine-readable JSON. Working out what
    `models_folded/` actually IS came first — the recipe is **5 steps, not 1**, and step 2b
    (onnxsim constant-folding the segmentation graphs under the plain filenames) was mandatory
    and undocumented, costing ~2x and the ORT-CUDA `Sin`/`Cos` CPU-fallback tax if skipped.
    A cold run reproduces all 15 diarization graphs with **bit-identical initializers** and an
    **identical RTTM sha256** (119.5 s, ~484 MB at recipe 2).
    Provenance lands in `diar-provision.json`; `EXPORT_RECIPE_VERSION` is 2 (fp16 gender
    restored — §7.39), and recipe-1 directories are `stale`, which is non-fatal.
  - **Startup model gate + `/readyz`: DONE 2026-08-31** — a missing or zero-length required file
    is now fatal (**exit 8**) with remediation text naming the provisioning command and the HF
    gate URL, instead of one "CUDA session load failed" per device inside a crash loop that also
    fails `up --wait`. A missing MARKER is deliberately only a warning: every directory deployed
    before this shipped has none. `/healthz` gained the `models_*` fields and **must stay 200 in
    every state** (the compose healthcheck and `diarizer_native.py` gate on status alone);
    `/readyz` is the endpoint allowed to 503. `DIAR_ALLOW_UNVERIFIED_MODELS=1` is the escape
    hatch. Ten audit defects fixed in §7.38 — the load-bearing two: provisioning **defaulted to a
    device**, so a GPU-less host wrote a marker declaring good models known-bad and bricked
    startup permanently; and the smoke test verified graphs production does not run, so a zeroed
    production multimask graph passed all five stages green.
  - **fp16 gender on linux/arm64 (issue #14): DONE 2026-09-01 (`c06fa15`; RESULTS §7.40,
    docs/ORT_FUSION_FP16_AARCH64.md)** — `gender-wav2vec2.onnx` (fp16) fails to
    LOAD on linux/arm64, silently disabling speaker gender there while requests still answer
    200. Not a model defect and not "aarch64 lacks a kernel amd64 has": ORT fuses `Erf`-GELU
    into `com.microsoft.Gelu` *at session load*, and **every** aarch64 build checked lacks the
    fp16 kernel for it — the builds differ only in whether the fusion FIRES. macOS arm64
    declines to fuse fp16, so it never creates the node and works fine (verified natively:
    default and `coreml` builds, `cpu`/`coreml`/`coreml_fast`, all pass end-to-end).
    Fix shipped: cap the GENDER session at `GraphOptimizationLevel::Level1` on aarch64 only
    (`crates/diar-core/src/ort_compat.rs`) — bitwise identical to the unoptimized graph and
    leaves zero contrib ops on fp16 tensors, while the diarization graphs keep full
    optimization. The surgical alternative is
    `disable_specified_optimizers=GeluFusionL2` (9.58e-04 max Δlogit, 6/6 labels); note the
    issue's proposed `GeluFusion` does NOT work, an unknown optimizer name is silently ignored,
    and the separator is `;` not `,`. Nothing under `vendor/` is involved and `ort` needs no
    bump. Latent risk on record: 11 of 15 diarization graphs are rewritten to
    `com.microsoft::FusedConv` (fp32-only kernel) and are safe purely by being fp32, so any
    future fp16 export needs a LOAD gate on aarch64, not just an accuracy gate.
    Reproduction harness committed: `validation/ort_fusion_probe/run_probe.sh`.
- **M4 (T2)**: Triton repo productionization — accuracy-correct community-1 TRT engines
  (rebuild from our exports), `diar-ffi` backend or sidecar-calls-Triton, per-arch engine build
  job (sm_86 local+g5 / sm_89 g6), AWS compose profile. M11 full-pipeline concurrency measured
  here.

## Benchmark anchor (do not re-measure — docs/BENCHMARK_PROTOCOL.md)

Every lever from here is judged against the committed pre-flip PyAnnote numbers in
`results/e2e_baseline/python/`. Primary reference file: **Karpathy 66.5 min** (`01a01aba-d9d2…`)
— **108.4 s python anchor, 54.4 s now** — chosen because it carries both a speed baseline and
hand-labelled ground truth, so speed and accuracy regress on the same file. Long-media reference:
the 2.1 h seed file, 206.7 s → 120.3 s. Protocol, gates and the accuracy checks that must
accompany any speed claim: `docs/BENCHMARK_PROTOCOL.md`.

## Post-flip optimization sequence (ACCEPTED 2026-08-19 — full detail: docs/E2E_PIPELINE_MAP.md)

User-confirmed direction: progressive updates, async task loads, and downstream overlap
wherever outputs allow. Execution order after the flip + baseline run:
1. L2 true transcribe∥diarize overlap — **DONE 2026-08-19 (T2, RESULTS §7.13)**, outputs
   byte-identical, GPU stage = max not sum
2. L3 progressive presentation — **DONE 2026-08-19 (T3)**, `transcript_ready` emitted at both
   finalize commit points; redaction still gates server-side
3. L4 finalize off the GPU worker (stop holding the GPU slot for CPU work)
4. L5a gender-task QUEUE-PRIORITY fix — **DONE 2026-08-19**, folded into the CPU-pool wedge fix
   (RESULTS §7.11): priority dropped to USER_TRIGGERED plus per-task time limits, because at
   PIPELINE_CRITICAL with no limit it could deadlock the whole ingest pipeline;
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
