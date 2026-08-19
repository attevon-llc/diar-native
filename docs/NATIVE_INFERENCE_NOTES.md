# Native (Rust/ONNX) Inference for the Rest of the Model Zoo — future-work note

Question (2026-08-19): beyond diarization, can OpenTranscribe's other models run via Rust/ONNX
instead of PyTorch, and on GPU where appropriate?

## Landscape

- **Already native:** faster-whisper = CTranslate2 (C++); no torch at ASR inference.
- **Torch consumers (aux models):** GLiNER PII, MiniLM cross-encoder re-ranker,
  sentence-transformers embeddings, mdeberta classifier, toxicity/gender detectors.

## Cost ladder (do in order)

1. **Profile first** — measure per-model share of worker time/VRAM before porting anything
   (same evidence-first protocol as the diarization work; see RESULTS §4.11 hygiene rules).
2. **ONNX-in-Python (`optimum[onnxruntime]`)** — 2-4× typical for transformer classifiers on
   CPU, zero architecture change. Captures most of the win.
3. **Absorb into the Rust sidecar** — once `diar-server` exists, adding small ONNX sessions
   (PII, re-ranker) to the same process is near-free marginal cost; HF `tokenizers` is natively
   Rust. Justified only for high-call-volume small models where deployment weight matters.
4. **GPU-vs-CPU placement is measured, never assumed** — tiny models on short inputs are often
   faster on CPU (PCIe round-trip dominates; cf. the fork's GPU aggregate/reconstruct negative
   results); batch jobs (whole-transcript embeddings) favor GPU.

Goal end-state alignment: every model the T1 sidecar absorbs shrinks the torch footprint of the
backend image (diarization's removal is the first and largest step; see PLAN.md decision #5).
