# GitHub Issue Drafts (ready to file)

Drafts produced 2026-08-19 during Phase B/C-M1 validation. File with `gh issue create` in the
respective repos (David's call on timing; nothing filed automatically).

## 1. davidamacey/pyannote-audio — bug: CPU-only Linux crash in `_gpu_empty_cache`

**Title:** `_gpu_empty_cache()` crashes all CPU-only Linux runs (MPS branch taken via hasattr)

**Body:**
`src/pyannote/audio/pipelines/speaker_diarization.py:559-564`: when CUDA is unavailable the
fallback branch calls `torch.mps.empty_cache()` guarded only by `hasattr(torch, "mps")`, which
is True on Linux torch builds → `RuntimeError: Cannot execute emptyCache() without MPS backend`.
Every pure-CPU Linux deployment (docker-compose.lite class) crashes at the first stage boundary.
Repro: run the community-1 pipeline with `--device cpu` in a container with no GPU visible.
Fix: guard with `torch.backends.mps.is_available()`. Found during diar-native validation
(RESULTS §5.6); workaround used there: expose a GPU but pass device=cpu.

## 2. attevon-llc/OpenTranscribe — bug: production image's onnxruntime-gpu cannot load CUDA EP

**Title:** onnxruntime-gpu 1.28 (CUDA 13 build) vs cu12 libs — CUDA EP unusable in backend image

**Body:**
`opentranscribe-backend` ships `onnxruntime-gpu==1.28` built against CUDA 13
(`libcublasLt.so.13`) while the image's pip CUDA stack is cu12 (torch 2.11+cu128, cuDNN 9.19).
ORT's CUDA provider fails to load: `Failed to load libonnxruntime_providers_cuda.so ...
libcublasLt.so.13: cannot open shared object file`. Dormant today (no app code uses ORT-GPU) but
any future ORT-GPU work silently falls back to CPU. Fix: pin an ORT cu12 build or add cu13 libs.
Evidence: diar-native RESULTS §4.3.

## 3. attevon-llc/OpenTranscribe — cleanup: stale ONNX artifacts exported from wrong checkpoints

**Title:** models/onnx/* were exported from segmentation-3.0 + standalone wespeaker, not community-1

**Body:**
sha256 comparison proves `models/onnx/segmentation.onnx` + `embedding.onnx` derive from
`pyannote/segmentation-3.0` and standalone `wespeaker-voxceleb-resnet34-LM` — NOT the
community-1 subfolder checkpoints the app actually runs (weights differ). Any historical E2E
measurements using these artifacts measured a different model. Recommend: delete or clearly
mark; regenerate from community-1 subfolders if ONNX artifacts are needed again.
Evidence: diar-native RESULTS §1.

## 4. attevon-llc/OpenTranscribe — comment for the existing Parakeet/NVIDIA-ASR future-work issue

**Comment:**
Measured spike (2026-08-19, diar-native `docs/ASR_TRITON_NOTES.md`): on the hand-labeled
Karpathy 10-min clip, faster-whisper large-v3-turbo = 97.4% word match, 40 ms median word-start
error (24× RT); parakeet-tdt-0.6b-v2 = 95.9% match, 110 ms median (≈109× RT pure inference).
Signed-bias analysis: both engines are systematically LATE (FW +40 ms, parakeet +110 ms);
constant-offset calibration recovers parakeet to 50 ms median and FW to 10 ms — calibration is
worth applying in the app regardless of engine. Full investigation spec (true WER, model sweep,
WSER-through-pipeline, $/audio-hour) is in diar-native `docs/ASR_TRITON_NOTES.md`.

## 5. avencera/speakrs — see docs/UPSTREAM_PRS.md for the PR/issue series
