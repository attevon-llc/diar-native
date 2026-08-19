# Pipeline Speedup Roadmap — preprocessing / diarization / transcription

Ranked by measured-or-estimated value ÷ effort. Statuses: MEASURED (number exists),
READY (implemented, awaiting flip), CANDIDATE (concrete plan, unmeasured), RESEARCH.

## Preprocessing

1. **tmpfs handoff volume** — CANDIDATE, config-only (~minutes). The gpu-split WAV handoff and
   the diar-native handoff currently touch disk; tmpfs removes seconds per long file.
2. **symphonia decode inside diar-server** — CANDIDATE (small Rust change). Worker POSTs the
   ORIGINAL media path; no WAV write at all. Also enables direct mp4/mp3 ingestion for T1.
3. **Single-decode + shared-mmap PCM service** — RESEARCH; only if profiling shows multiple
   consumers re-decoding the same audio. UI waveform-peaks generation folds in here for free.
4. Reality check: decode is seconds/job — these are polish tiers, not multipliers.

## Diarization (engine = diar-native sidecar)

1. **Warm-engine serving — MEASURED, READY: 277× RT** (7.9 s / 36-min meeting, 3.4× the fork).
   The single biggest remaining action is simply flipping the app to the sidecar.
2. **ORT TensorRT EP on the multimask session** — CANDIDATE, strongest remaining lever.
   speakrs graphs are FIXED-SHAPE (the phase-6 rebuild-storm precondition is gone) and TRT
   measured 2.4× vs ORT-CUDA on the embedding model (RESULTS §4.6). `ort` exposes the TRT EP +
   engine cache. Estimated additional ~1.3-1.6× E2E warm.
3. **Native Rust fbank (kaldi-native-fbank / knf-rs)** — CANDIDATE. The pooled ONNX-CPU fbank
   still costs ~10 s on a 4.7 h file; native fbank is ~3-5× faster per chunk and rayon-friendly.
4. **Arc-shared sessions** — READY-adjacent (PLAN M1): N concurrent jobs on one weight copy —
   throughput per GPU, not single-job latency.
5. VRAM/speed set selection — MEASURED, READY (MODELS_SETS.md): 4.2 GB@277× vs 1.6 GB@59×.

## Transcription (faster-whisper / CTranslate2)

CT2 is already native C++ — Rust offers nothing there. Real levers, cheapest first:
1. **int8_float16 quantization** — CANDIDATE, one config value; typically 1.3-1.7× on Ampere
   with ~0 WER cost. Validate with the existing WSER harness (word timestamps must hold).
2. **VRAM headroom → bigger whisper batches** — free side-effect of moving diarization off the
   worker; re-tune BatchedInferencePipeline batch size after the flip.
3. **VAD gating** (skip silence) if not already enabled for long quiet media.
4. **TRT-LLM Whisper on Triton** — RESEARCH (docs/ASR_TRITON_NOTES.md): throughput winner but
   word-timestamp gap; only for T2/AWS scale-out.
5. **Calibrated Parakeet TDT** — MEASURED spike: ~4× faster inference, 50 ms median word-start
   after bias correction (vs FW's 10 ms corrected). English-heavy throughput tier candidate;
   judged by WSER-through-pipeline in the formal bake-off (user's GH issue).

## Aux models next in line (gender detection, text/NLP models)

Ladder per docs/NATIVE_INFERENCE_NOTES.md: profile share first → `optimum[onnxruntime]`
in-place (2-4× typical, zero architecture) → absorb small hot models into the sidecar
(HF `tokenizers` is native Rust) → GPU placement only where batch size justifies PCIe.

## Post-PR E2E comparison protocol (the "full workflow timing" run)

When the pending transcribe-app PR lands and the native flip is authorized:
1. Instrument once: the app already logs per-stage timings (upload → decode → transcribe →
   diarize → align/assign → NLP → index). Capture Celery task spans for BOTH engines.
2. Corpus: test_videos/ ×3 + karpathy_10m + one 2h seed file; 3 runs each, quiet machine.
3. Report per-task wall + queue time, GPU util, VRAM peaks, and end-to-end job latency:
   `DIARIZER_ENGINE=python` vs `native` (fast set) vs `native` (small set).
4. Accuracy guards alongside: WSER on karpathy, speaker counts, OpenSearch embedding parity.
5. Output → validation/RESULTS.md §5.x + a summary table in REPORT.md.
