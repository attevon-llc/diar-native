# ASR (Whisper) Serving Research Notes — future T2 companion workstream

Scope note: diar-native owns diarization; this file captures the parallel question "can we serve
faster-whisper/Whisper the same way (batching, self-hosted, private) for AWS?" Answer: yes —
three routes, verified 2026-08.

## Routes

1. **Triton + TensorRT-LLM Whisper** (official): `tensorrtllm_backend/docs/whisper.md` —
   encoder+decoder TRT-LLM engines, `whisper_bls` orchestrator, `inflight_fused_batching`
   (continuous batching), cross-KV cache config. Encoder-decoder IFB is first-class (NVIDIA blog).
   Newer `tritonserver:*-trtllm-python-py3` images can serve HF models without manual engine
   builds. Max throughput; most engineering. k2-fsa/sherpa also ships TRT-LLM-based Triton
   whisper recipes.
2. **vLLM Whisper**: `vllm serve openai/whisper-large-v3` → OpenAI-compatible
   `/v1/audio/transcriptions` with continuous batching (docs.vllm.ai speech_to_text). KEY FIT:
   OpenTranscribe's cloud-ASR provider factory can point at a self-hosted vLLM endpoint —
   near-zero app changes, full privacy. Red Hat 2026 guide documents productionizing this.
3. **Triton python backend wrapping faster-whisper/CT2**: least work, exact current behavior
   (word timestamps, VAD), instance-group concurrency but NO cross-request batching. Stepping
   stone only.

## Watch-items

- **Word-level timestamps** (speaker assignment + WSER depend on them): TRT-LLM/vLLM Whisper are
  segment-oriented. Solution owned already: WhisperX's wav2vec2 alignment model is a simple
  encoder → trivial ONNX/TRT export, batches perfectly on Triton. T2 end-state = Triton hosting
  diarization + alignment + ASR on one GPU with dynamic batching.
- vLLM beam search for whisper currently inefficient (encoder/decoder cache work ongoing);
  greedy is the production setting anyway.
- Licensing: all three routes are OSS (Apache/BSD-class). NVIDIA Riva/NIM exists as a managed
  alternative but adds licensing.

## Decision protocol (when picked up)

One-day bake-off on the existing harness corpus (duration curve + seeds), same GPU:
CT2-in-worker (baseline) vs vLLM vs TRT-LLM Triton — measure jobs/hour/GPU at concurrency
1/4/8, WER parity, word-timestamp quality (WSER through the app pipeline), VRAM. Quiet-machine
rules apply (RESULTS §4.11). Cost model: jobs/hour/GPU × AWS g5/g6 hourly vs per-minute API
pricing.

Sources: tensorrtllm_backend whisper.md; NVIDIA enc-dec IFB blog; docs.vllm.ai speech_to_text;
Red Hat AI private transcription guide (2026-03).
