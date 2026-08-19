# Rust Services Plan — aux models, shared memory, and load-safety capacity math

Extends SPEEDUP_ROADMAP with the detailed design for (a) absorbing aux models (gender, text)
into native serving, (b) memory-sharing choices, (c) behavior under extreme load (1000s of
queued requests; multi-user; small computers).

## 1. Memory-sharing facts and rules

- **tmpfs (and /dev/shm) = RAM.** Files consume page cache up to the mount's `size=` cap;
  pages may spill to swap. A plain Docker volume = disk (no RAM pressure; pays I/O + wear).
- 16 kHz mono int16 WAV sizes: 10 min ≈ 19 MB · 30 min ≈ 58 MB · 2 h ≈ 230 MB · 4.7 h ≈ 540 MB.
- **Backpressure is the load-safety mechanism, not the filesystem**: requests wait in the
  Celery queue as tiny messages; only `DIAR_MAX_INFLIGHT` jobs materialize audio at once.
  Peak handoff footprint = `inflight × largest_file`, INDEPENDENT of queue depth. 1000s of
  queued requests cost queue metadata only.
- Rules: (1) handoff volume on **disk by default**; tmpfs opt-in
  (`o=size=2g` hard cap) on RAM-rich servers; (2) client deletes after POST returns
  (diarizer_native already does, `finally: unlink`); (3) sidecar enforces a max-duration guard
  (reject > N hours with 413 rather than OOM); (4) end-state removes the handoff entirely —
  symphonia decode in the sidecar reads the ORIGINAL media path (no PCM copy exists anywhere).

## 2. Sizing table (per deployment tier)

| host | engine set | mode | inflight | handoff | approx peak RAM (engine RSS + handoff) | VRAM |
|---|---|---|---|---|---|---|
| laptop 8-16 GB, small/no GPU | small | cpu | 1 | disk | ~3.5 GB | 0 / 1.6 GB |
| workstation, 12 GB GPU | small or fast | cuda | 2 | disk | ~4 GB | 1.6-4.2 GB |
| beefy server (A6000-class) | fast | cuda | 2-4 | tmpfs 2 GB cap | ~5-6 GB | 4.2 GB |
| AWS g5/g6 multi-user | fast, N replicas or T2 Triton | cuda | 2/replica | tmpfs | scale by replica | 4.2 GB/replica |

Measured anchors: engine RSS 3.1 GB (4.7 h job); VRAM per MODELS_SETS.md.

## 3. Aux-model absorption plan (gender, text/NLP) — per model

Principle (NATIVE_INFERENCE_NOTES ladder): profile share → optimum-ORT in Python → absorb only
hot small models into the sidecar. Shared memory is IRRELEVANT for text models (inputs are KB);
it only ever mattered for PCM.

| model | type | native path | notes |
|---|---|---|---|
| Common-Voice-Gender-Detection | small audio classifier | **best sidecar candidate**: it consumes the SAME PCM the sidecar already holds — add `/classify_gender` that runs per-speaker on segment windows during diarization (zero extra decode/transfer; per-speaker majority vote lands in the diarize response) | ONNX export trivial; CPU-fast |
| gliner-pii (redaction) | token classifier | stage 1 optimum-ORT in the redaction worker (GPU already assigned); sidecar absorption only if profiling shows it hot | tokenizer via HF `tokenizers` (native Rust) if absorbed |
| MiniLM cross-encoder (re-rank) | text pair scorer | optimum-ORT first; strong absorb candidate later (tiny, high call volume at search time) — micro-batching applies | |
| mdeberta classifier / Tiny-Toxic | text classifiers | optimum-ORT; absorb only if hot | |

**Micro-batching for tiny models under flood load** (the 1000s-of-requests case): the sidecar
collects requests for ≤5-10 ms or until batch=N, runs one batched ORT call, fans results back.
This is the same dynamic-batching principle Triton proved at 2.14×, in-process, and it converts
request floods into GPU/CPU-efficient batches with bounded memory (queue caps + 429 backpressure
when full). Implement as a generic `micro_batch<T>` worker in diar-core when the first text
model is absorbed.

## 3a2. CPU-vs-GPU placement bench protocol (run once text-native components exist)

For EACH aux model (VAD, gender, PII, re-ranker, toxicity, mdeberta) in Rust/ort:
1. Export to ONNX (optimum or native export); verify output parity vs the PyTorch original on
   a fixture set (same discipline as diarization: parity BEFORE speed).
2. Bench matrix: {CPU EP (intra-threads 1/4/8), CUDA EP} × batch {1, 8, 32, 128} ×
   realistic input lengths. Record latency p50/p95 + throughput + VRAM.
3. Decision rule per model: GPU only if throughput at its REAL production batch size beats CPU
   by >2× (PCIe + transfer overhead rule; tiny models at batch 1 almost always stay CPU).
4. VAD-specific: bench per-chunk streaming (the actual usage) not just batch — the Rust gain is
   per-call overhead removal (reused session, no GIL) + new capability (gate diarization
   windows); absolute seconds/job are small but scale with volume and enable preprocessing
   consolidation.
5. Results land in validation/RESULTS.md; placement table goes here.

## 3b. Service topology decision (2026-08-19): TWO sidecars

- **diar-native** (audio, GPU): diarization + gender classifier (shares held PCM).
- **text-native** (text, CPU-first): PII / re-ranker / toxicity / classifiers — SEPARATE service:
  different resource shape (small models, high request rate, CPU replicas scale independently),
  failure isolation (text crash must not cost the warm GPU engine), independent upgrade cadence.
  Same cargo workspace → shared micro_batch/ORT/tokenizers code; two ~30-50 MB single-purpose
  containers. Build order: flip → profile → optimum-ORT interim → text-native for proven-hot
  models only.
- **Torch-free backend endgame**: whisper=CT2 (torch-free), diarization=Rust, text=ORT.
  Audit clarifications (user, 2026-08-19):
  - **VAD is not a torch pin**: faster-whisper's VAD is Silero-ONNX via onnxruntime already.
    Rust move = same Silero ONNX in `ort` (~2 MB, CPU, trivially fast) — lands in sidecar
    preprocessing (P2) so silence-gating happens once at the PCM and feeds BOTH diarization
    (skip empty windows) and transcription.
  - **Alignment is OUT of scope by decision**: removed from the pipeline in favor of ASR-native
    word timestamps; reinstating would require a per-language wav2vec2 model + an extra model
    pass per job — contrary to the model-zoo-shrinking goal. Revisit only if a future feature
    needs sub-word precision the ASR timestamps can't provide.
  - Remaining audit targets: transitive torch imports in NLP/redaction paths.
- **Task overlap**: with diarization off-worker, transcribe+diarize run concurrently without
  stages.py VRAM gating → job latency approaches max(stage) not sum(stages); measured by the
  post-PR E2E protocol. Output-identity gates every migration step, as throughout.

## 4. Rust preprocessing service (from SPEEDUP_ROADMAP, detailed)

Phase P1: tmpfs opt-in flag for the existing handoff (config only, servers).
Phase P2: symphonia decode in diar-server → accept `{media_path}` for mp3/mp4/flac/wav;
  resample via rubato; delete the WAV handoff path from diarizer_native (keep as fallback).
Phase P3 (only if profiling justifies): standalone `prep-server` producing decoded PCM once
  per media for ALL consumers via mmap on a capped tmpfs; consumers: diarize (sidecar),
  transcribe (CT2 accepts numpy from mmap zero-copy in Python), waveform-peaks for UI,
  gender classifier. Eviction: delete on last-consumer-done + LRU cap; same backpressure rule —
  materialize only inflight jobs, never the queue.

## 5. Failure-mode checklist (extreme load)

- Queue flood → Celery queue grows (Redis memory: ~KB/task — monitor, cap with rate limits);
  sidecar untouched (semaphore).
- tmpfs full → single job fails loudly (write error) → retried/fallback; box never OOMs
  (hard `size=` cap).
- Sidecar crash under load → `restart: unless-stopped` + worker fallback to fork path
  (INSTALL_NATIVE hook variant); in-flight jobs retried by Celery.
- Multi-user fairness → Celery priorities/rate limits (existing app machinery — decision #3:
  orchestration stays in Celery, never in the sidecar).

## 6. Cross-file speaker clustering (large-corpus speaker identification) — user-flagged lever

Distinct from within-file VBx (solved, 8×): resolving speakers ACROSS the library
(speaker_matching_service + OpenSearch kNN, thresholds 0.5/0.75; batch re-resolution jobs).
User has prior research on large-corpus embedding clustering methods (CPU/GPU) — locate and
cite it (GH issues/docs) before design; do not re-derive.

Rust shape:
- **Warm ANN index in-process** (usearch or hnsw_rs): sub-ms nearest-centroid queries;
  OpenSearch stays the durable store, index = rebuildable cache. Fits micro-batching.
- **Batch global re-resolution**: incremental centroid clustering reusing the vectorized
  distance machinery from the VBx/pdist work; CPU to ~1e5 vectors, GPU (cuVS-class) only past
  that per the >2× placement rule (§3a2).
- Gate: decision-parity vs the current matching service on a fixture library BEFORE speed
  claims (same discipline as diarization).
