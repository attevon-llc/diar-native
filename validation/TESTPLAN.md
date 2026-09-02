# Test Plan — speakrs vs pyannote fork (community-1)

Companion to [RESULTS.md](RESULTS.md) (measurements) and the repo [README](../README.md)
(context/why). This file defines **what** is tested, **how**, and **what passes**.

## 1. Engines under test

| id | engine | version pin | runtime | invocation |
|---|---|---|---|---|
| **A** | pyannote fork (production baseline) | `davidamacey/pyannote-audio` `gpu-optimizations` @ `a3f38afb` | `opentranscribe-backend:latest` image, fork bind-mounted ro over site-packages (exact production code path) | `validation/run_fork_baseline.py` + transcribe-app's own `benchmark-pyannote-direct.py` for the canonical duration curve |
| **B** | speakrs | `avencera/speakrs` @ `b0756b1` | `diar-bench:latest` (docker/Dockerfile.bench: ORT 1.24.2 GPU, CUDA 12.8, cuDNN 9) | `validation/run_speakrs.sh` → `xtask diarize` |
| **S** | Triton serving spike — **RETIRED**, see RESULTS §7.26 | tritonserver 26.06-py3 (ORT backend; TRT backend planned) | model repo `triton/models/` | harness deleted; §7.26 *is* the recipe |

**Models (both engines):** self-exported from the production HF cache of
`pyannote/speaker-diarization-community-1` (segmentation 4-layer-LSTM checkpoint, WeSpeaker
ResNet34 embedding, PLDA/VBx params). Export = `vendor/speakrs/scripts/export_models.py` run
inside the backend image, offline (`HF_HUB_OFFLINE=1`). Checkpoint identity verified by sha256
(RESULTS §1) — the community-1 weights differ from segmentation-3.0 / standalone wespeaker-LM.

**Engine B config parity:** mode `cuda` (segmentation step = 1.0 s, same as pyannote;
`cuda-fast` = 2.0 s is excluded from all accuracy comparisons), AHC threshold 0.6,
VBx Fa 0.07 / Fb 0.8 — speakrs defaults match the community-1 `config.yaml`.

## 2. Test matrix

| # | corpus | files | ground truth | engine A | engine B | metric(s) | purpose |
|---|---|---|---|---|---|---|---|
| M1 | Frozen-baseline repro | 0.5h + 2.2h wavs ×5 runs | frozen RTTMs `baseline_a6000_20260421_*` | GPU 0 | — | DER vs frozen, spk, segs | Gate 0: environment still reproduces April baseline |
| M2 | AMI test-16 | 16 Mix-Headset wavs (~9.3 h) | official `only_words` RTTMs + UEMs | GPU 2 ×1 | GPU 2 ×1 | DER@0.25 (overlap incl., UEM), spk ±, RTF | G1: real-corpus accuracy parity |
| M3 | Karpathy acceptance clip | 66.5-min podcast | hand-labeled RTTM (maintainer) | GPU 0 ×3 | GPU 2 ×3 | DER@0.25, spk exact, determinism | G2: production-style content; never tuned on |
| M4 | Duration curve | 0.5/1.0/2.2/3.2/4.7 h wavs ×3 | frozen RTTMs (A/B cross-scoring) | GPU 0 | GPU 2 | A/B DER, spk exact, RTF, peak RSS/VRAM | G3+G4: long-file behavior, memory, speed |
| M5 | Determinism | 2.2h + Karpathy ×3 | self | ✓ | ✓ | identical RTTMs across runs | G5 |
| M6 | VoxConverse dev | 216 wavs (~20 h) | official RTTMs | ×1 | ×1 | DER@0.25 + DER@0 (published-comparable), RTF | apples-to-apples vs speakrs' published 7.0–7.1% claim |
| M7 | CPU-only leg | 0.5h wav (+clip30 smoke) | frozen RTTM | `--device cpu` | mode `cpu` | RTF, DER vs GPU run | lite/CPU deployment matrix |
| M8 | Serving spike | synthetic batch-32 tensors | parity vs eager | — | — | ms/batch server-side, EP placement, folding effect | Triton/ORT serving viability (RESULTS §4) |
| M9 | DONE (RESULTS §4.4/§4.6) TRT plans on Triton | batch-32 | parity gates | — | — | ms/batch vs M8 | tensorrt backend upside |
| M10 | DONE (RESULTS §4.6) A6000 re-run of M8 headline numbers | — | — | — | — | — | remove 3080 Ti confound |
| M11 | **A6000 Triton concurrency/throughput** — model-level DONE (RESULTS §4.7: 2.14× at 8 clients); full-pipeline deferred to Phase C M4 | N∈{1,2,4,8} concurrent clients | RTTM parity vs serial | — | — | throughput, p50/p95, GPU util | does parallel+batched Triton beat serial throughput on one A6000? |

Scoring: `validation/score_der.py` — `pyannote.metrics.DiarizationErrorRate`, **collar 0.25,
skip_overlap=False**, AMI cropped by official UEMs, hypothesis `<label>_run<N>.rttm` vs reference
`<label>.rttm`. Aggregate DER = corpus-level accumulation (not mean of per-file).
RTF = audio_seconds / wall_seconds (engine B wall includes ~2–4 s container+model-load startup per
file — noted where material). Memory: engine A `torch.cuda.max_memory_allocated` + reported peaks;
engine B via `nvidia-smi` poll / `/usr/bin/time -v` where run uncontainerized.

**M11 design (A6000 Triton throughput vs serial).** Serial baselines process one file at a time and
leave the GPU idle during CPU stages (clustering ≈ 41% of wall). The serving hypothesis is that
**concurrency + batching converts that idle time into throughput**: N files in flight keep the GPU
fed (dynamic batching coalesces embedding requests across files) while CPU stages of other files
overlap. Protocol: deploy the model repo on an A6000; drive with a client orchestrator (per-file
window extraction → batched seg/embedding requests → local clustering), sweep N∈{1,2,4,8}
concurrent files over AMI-16; record corpus wall-clock vs the serial engine-A sweep (M2), p50/p95
request latency, `nvidia-smi` utilization, and Triton per-model stats; verify RTTM parity vs the
serial run so speed never silently costs accuracy. Compare ORT backend now, TRT plans when M9 lands.

## 3. Gates (accept/reject for adopting speakrs — decided on A6000 numbers)

| gate | criterion | rationale |
|---|---|---|
| **G0** | Engine A reproduces frozen baselines: DER drift ≈ 0, spk/segs identical | invalid environment otherwise |
| **G1** | M2: engine B aggregate AMI DER ≤ engine A + **0.1 pp**; per-file speakers within ±1 | historical acceptance bar of the transcribe-app benchmark program |
| **G2** | M3: engine B Karpathy DER ≤ engine A + 0.1 pp; speaker count exactly 2 | acceptance fixture, production-style audio |
| **G3** | M4: speaker count exact on all 5 duration files; A/B median DER ≤ 0.5%, max ≤ 2.0% | cross-implementation tolerance (bit-parity not expected between numpy and Rust) |
| **G4** | M4 4.7 h: engine B RTF ≥ 1.0× engine A; peak RSS < 8 GB; VRAM < 4 GB | must not be slower or heavier than production |
| **G5** | M5: engine B RTTMs identical across 3 runs | reproducibility requirement (engine A is proven deterministic) |

**Decision rule:** all gates pass → adopt speakrs (Phase C-pass: `diar-core` wrapper, sidecar,
OpenTranscribe integration). Accuracy gates fail → per-stage `.npy` dump bisection (speakrs exposes
all intermediates) to localize divergence; if unfixable at the wrapper/config level → Phase C-fail:
Triton BLS architecture with the fork's own orchestration code (fully specified in the approved
plan). **Speed-only miss on G4 is not an automatic reject**: the serving-spike findings (fused-fbank
ORT-CUDA tax, RESULTS §4.2) identify a concrete optimization path (fbank-outside split) that would
be Milestone 1 work; decision then weighs effort vs the Triton-BLS branch.

## 4. Reproduction quick-reference

```bash
# Engine A — AMI test-16 (GPU 2)
docker run --rm --gpus '"device=2"' --entrypoint python \
  -v /path/to/diar-native:/work \
  -v /path/to/pyannote-audio-fork/src/pyannote/audio:/home/appuser/.local/lib/python3.13/site-packages/pyannote/audio:ro \
  -v /path/to/transcribe-app/models/huggingface:/home/appuser/.cache/huggingface:ro \
  -v /path/to/datasets/diarization-boundary:/data:ro -e HF_HUB_OFFLINE=1 \
  opentranscribe-backend:latest /work/validation/run_fork_baseline.py \
  --out /work/results/rttm/fork_ami_test16 --label-mode first-dot --device cuda \
  /data/ami_audio/<MEETING>.Mix-Headset.wav ...

# Engine B — same corpus (GPU 2)
validation/run_speakrs.sh cuda 2 results/rttm/speakrs_ami_test16 1 \
  <LABEL>:/path/to/datasets/diarization-boundary/ami_audio/<MEETING>.Mix-Headset.wav ...

# Score any pair
docker run --rm --entrypoint python -v /path/to/diar-native:/work \
  -v /path/to/pyannote-audio-fork/src/pyannote/audio:/home/appuser/.local/lib/python3.13/site-packages/pyannote/audio:ro \
  opentranscribe-backend:latest /work/validation/score_der.py \
  --ref-dir /work/refs/ami --hyp-dir /work/results/rttm/<TAG> --uem-dir /work/refs/ami --json-out /work/results/<TAG>_der.json

# Model export (regenerate gitignored models/)
docker run --rm --entrypoint bash -v /path/to/diar-native:/work \
  -v /path/to/transcribe-app/models/huggingface:/home/appuser/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 opentranscribe-backend:latest \
  -c "pip install -q --user onnxscript; python /work/vendor/speakrs/scripts/export_models.py /work/models"

# speakrs image
docker build -f docker/Dockerfile.bench -t diar-bench:latest vendor/speakrs

# Triton spike (GPU 1) — RETIRED. TensorRT/Triton was measured and rolled back; RESULTS §7.26
# is the reproduction recipe and the cost-benefit to re-judge. The gRPC bench harness that
# produced the spike numbers was deleted (recover: `git show <sha>:validation/triton_bench.py`).
docker run -d --name diar-triton-spike --gpus '"device=1"' -p 8610:8000 -p 8611:8001 -p 8612:8002 \
  -v /path/to/diar-native/triton/models:/models:ro nvcr.io/nvidia/tritonserver:26.06-py3 \
  tritonserver --model-store=/models
```

Corpora locations: AMI + VoxConverse + Karpathy ground truth under
`/path/to/datasets/diarization-boundary/`; duration-curve wavs under
`transcribe-app/benchmark/test_audio/`; frozen baselines under
`transcribe-app/benchmark/results/rttm/baseline_a6000_*`.
