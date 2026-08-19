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

## DECISION UPDATE (2026-08-19, after word-timestamp verification)

**Hard constraint from the app:** WhisperX alignment was REMOVED from the pipeline (too slow);
word-level timestamps come natively from faster-whisper's cross-attention DTW and are REQUIRED
for speaker assignment. Verified: vLLM's word-level timestamps are an open gap (API param exists,
DTW alignment unimplemented — vllm#25750, vllm#13400; only segment-level landed via PR #24209);
TRT-LLM Whisper is segment-oriented too.

**→ Re-ranked: faster-whisper/CTranslate2 REMAINS the ASR engine** (only route with production
word timestamps + already fast; app also uses faster_CrisperWhisper — the accurate-word-timestamp
model family). Serving evolution = the diarization-T1 shared-weights pattern applied to CT2
(one model instance, N concurrent Celery jobs; optionally hosted as a Triton python-backend for
ops consolidation — no cross-request batching, accepted). Routes 1/2 move to a watch-list keyed
to vLLM #25750 (word-level DTW) — revisit only if that lands with quality parity.

## Why Whisper serving stacks have worse word timestamps (user-confirmed by testing)

Whisper has no native word timings — every implementation reconstructs them post-hoc via DTW
over cross-attention. faster-whisper/CrisperWhisper carry the most battle-tuned implementation;
vLLM/TRT-LLM/Triton recipes that skip that reimplementation ship segment-only or degraded word
timings. This is architectural, not a maturity gap that waits itself out.

**The one alternative class that sidesteps it: transducer/CTC models (NVIDIA Parakeet TDT,
Canary).** Tokens are frame-aligned by construction → word timestamps are a native byproduct;
several-× faster than Whisper-class; Triton/NIM-servable with real batching. Bake-off entrant
alongside CT2 when the ASR workstream opens — judge on (a) WER on our real corpora (noisy/
accented audio vs large-v3-turbo), (b) word-timestamp quality THROUGH the app's WSER pipeline,
(c) multilingual needs, (d) jobs/hour/GPU. Until something beats CT2 on (a)+(b), CT2 stays.

## Measured spike (2026-08-19): word-timestamp accuracy vs hand labels

Karpathy 10-min clip, 2282 hand-labeled words (`karpathy_10m.ref.words.json`), A6000, harness
`validation/asr_timestamp_spike.py` (SequenceMatcher word alignment, |Δstart| on matches):

| engine | match rate | Δstart median/mean/p95 (ms) | inference |
|---|---|---|---|
| faster-whisper large-v3-turbo (fp16, prod model) | **97.37%** | **40 / 69.8 / 240** | 24.9 s (incl. load) ≈ 24× RT |
| parakeet-tdt-0.6b-v2 (NeMo, timestamps=True) | 95.88% | 110 / 142.1 / 390 | 5.5 s pure ≈ 109× RT (71.6 s incl. NeMo load+download) |

→ Parakeet ≈ 4× faster inference, but word timings ≈ 2.75× worse (median) and match rate lower —
confirms user's prior testing. **CT2/faster-whisper remains the engine.** Re-run protocol for
bigger variants (tdt-1.1b, v3 multilingual, canary): swap the model name in `run_parakeet()`.
NeMo installs cleanly on the backend image's torch 2.11 (`pip install -U nemo_toolkit[asr]`,
per model card; NGC nemo container is the fallback). Raw outputs:
`results/asr_spike_{faster_whisper,parakeet}.json`.

## Timing recovery for NVIDIA models (measured 2026-08-19)

Signed-error analysis of the spike data shows the errors are mostly SYSTEMATIC lateness
(emission delay), not noise:

| engine | signed bias | raw abs median | bias-corrected abs median / p95 |
|---|---|---|---|
| parakeet-tdt-0.6b-v2 | +110 ms | 110 ms | **50 ms** / 280 ms |
| faster-whisper large-v3-turbo | +40 ms | 40 ms | **10 ms** / 210 ms |

Levers, in value order: (1) **constant-offset calibration** per model (one labeled clip) —
recovers Parakeet to ≈ uncorrected-FW level, and improves FW itself to 10 ms median (**apply in
the app regardless of engine**); (2) larger variants (tdt-1.1b/v3) improve match-rate, not the
~80 ms TDT frame-grid timing floor; (3) NeMo Forced Aligner (fast CTC alignment — NOT the slow
wav2vec2 path that was removed) as an optional sharpening stage. Final judge = WSER through the
app pipeline. Conclusion unchanged (FW stays), but calibrated Parakeet is a legitimate
throughput-tier contender for the bake-off.

## Watch-items

- **Word-level timestamps** (speaker assignment + WSER depend on them): TRT-LLM/vLLM Whisper are
  segment-oriented. Solution owned already: WhisperX's wav2vec2 alignment model is a simple
  encoder → trivial ONNX/TRT export, batches perfectly on Triton. T2 end-state = Triton hosting
  diarization + alignment + ASR on one GPU with dynamic batching.
- vLLM beam search for whisper currently inefficient (encoder/decoder cache work ongoing);
  greedy is the production setting anyway.
- Licensing: all three routes are OSS (Apache/BSD-class). NVIDIA Riva/NIM exists as a managed
  alternative but adds licensing.

## Proper ASR investigation spec (for the OpenTranscribe GH issue — expands the spike)

The 2026-08-19 spike was ONE clip + match-rate; the formal investigation must add:
1. **True WER** (jiwer or meeteval, normalized text) — not SequenceMatcher match-rate — on a
   multi-domain corpus: Karpathy (clean podcast), AMI meetings (far-field/overlap), Earnings-21
   (telephone/finance jargon), 2-3 seed_t* files (noisy real-world). All already staged locally.
2. **Timestamp scoring with calibration applied** (per-model constant from a held-out clip;
   measured biases: FW +40 ms, parakeet-tdt-0.6b +110 ms) — report corrected median/p95.
3. **Model sweep** via the existing harness (one-line swap): faster-whisper large-v3-turbo
   (baseline) + distil variants; parakeet-tdt-0.6b-v2/v3 + tdt-1.1b; canary-1b; CrisperWhisper.
4. **WSER through the actual app pipeline** (the metric that matters): plug each engine's words
   into the speaker-assignment path on the Karpathy acceptance clip and score WSER vs the
   0.27% production bar.
5. **Throughput/cost leg**: jobs/hour/GPU at concurrency 1/4/8 (serving route per engine:
   CT2 in-process vs NeMo vs Triton/TRT), × AWS g5/g6 hourly → $/audio-hour table.
6. Quiet-machine rules (RESULTS §4.11) and per-engine determinism check ×3.

## Decision protocol (when picked up)

One-day bake-off on the existing harness corpus (duration curve + seeds), same GPU:
CT2-in-worker (baseline) vs vLLM vs TRT-LLM Triton — measure jobs/hour/GPU at concurrency
1/4/8, WER parity, word-timestamp quality (WSER through the app pipeline), VRAM. Quiet-machine
rules apply (RESULTS §4.11). Cost model: jobs/hour/GPU × AWS g5/g6 hourly vs per-minute API
pricing.

Sources: tensorrtllm_backend whisper.md; NVIDIA enc-dec IFB blog; docs.vllm.ai speech_to_text;
Red Hat AI private transcription guide (2026-03).
