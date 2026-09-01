# `ort-fusion-probe` — what ORT's optimizer does to a graph on this build

Built for **issue #14**. Explanation: [`docs/ORT_FUSION_FP16_AARCH64.md`](../../docs/ORT_FUSION_FP16_AARCH64.md).
Measurements: `validation/RESULTS.md` **§7.40** (append-only — add a section, never edit one).

**Read the doc before re-running anything here.** The questions this tool was built to answer
are already answered; it is kept so a *fix* can be verified, or so the same question can be
asked of a new graph or a new platform — not so the 2026-09-01 investigation gets repeated.

## Why it exists

`gender-wav2vec2.onnx` contains no `com.microsoft.Gelu`. ORT **creates one at session load**
by fusing the `Erf`-GELU pattern, then on linux/arm64 fails to find an fp16 kernel for the
node it just made. Nothing about that is visible by inspecting the model file, and a session
either opens or it doesn't — there is no accuracy number to look at. This tool makes the
load-time rewrite observable: it serializes the *optimized* graph and reports load success or
failure per configuration.

Deliberately **not a workspace member**, so a root `cargo build` never builds it. It depends
on nothing in this repo — only on the same pinned `ort =2.0.0-rc.12`, which is the point:
it must observe what `diar-server` observes.

## Run it

```bash
# one command, both platforms, from an Apple Silicon Mac:
validation/ort_fusion_probe/run_probe.sh <models-dir>
```

Or by hand:

```bash
cargo build --release --manifest-path validation/ort_fusion_probe/Cargo.toml
python validation/ort_fusion_probe/make_clips.py /tmp/clips.bin

# does every graph load, and what did the optimizer rewrite it into?
./target/release/ort-fusion-probe load /tmp/dumps <models-dir>/*.onnx
python validation/ort_fusion_probe/inspect_dumps.py /tmp/dumps

# does a candidate fix load, and does it change the numbers?   L0 = the reference.
./target/release/ort-fusion-probe run <models-dir>/gender-wav2vec2.onnx /tmp/clips.bin /tmp/d \
    L3 L3:GeluFusionL2 L1 L0
```

`make_clips.py` and `inspect_dumps.py` need `numpy` + `onnx`. Any venv with those works; the
`scripts/provision/requirements.txt` environment already has them.

## Traps this tool exists to keep you out of

All three measured, all three cost real time:

1. **The optimizer is `GeluFusionL2`, not `GeluFusion`.** ORT registers the pass twice (L1 and
   L2 instances) under suffixed names. `GeluFusion` and `GeluFusionL1` both leave the failure
   in place.
2. **An unrecognized optimizer name is silently ignored.** No error, no warning.
   `disable_specified_optimizers=NotARealOptimizerName` loads fine and changes nothing — so a
   misspelled name ships a config entry that *looks* applied and does nothing. Never assume a
   disable took; check the load outcome or the dump.
3. **The separator is `;`, not `,`** — `GeluFusionL2;BiasGeluFusion` disables both,
   `BiasGeluFusion,GeluFusionL2` disables neither, despite `ort`'s doc comment saying
   "comma-separated".

## Reproducing the FAILING platform without a Linux box

Docker Desktop on Apple Silicon runs `linux/arm64` natively, so the failure reproduces on a
Mac. Two hard requirements found the hard way:

- Base image **`rust:1-trixie`**, not `bookworm`. This ORT needs glibc ≥ 2.38
  (`__isoc23_strtol`); bookworm's 2.36 fails at link with a wall of undefined references.
- **`RUSTFLAGS="-C link-arg=-lstdc++"`**, or the static ORT lib fails on `__cxa_call_terminate`.

Native macOS builds need `LIBRARY_PATH=/opt/homebrew/opt/openblas/lib` (both the default and
the `coreml` build). `coreml` is macOS-only and never goes through Docker — Docker on macOS
has no Metal access regardless of image arch.

## Getting a models directory on a machine that has none

The diarization weights are **gated** (pyannote community-1) — never commit or redistribute
them; copy them from a machine that already has a provisioned set.

The **gender classifier is ungated** (`prithivMLmods/Common-Voice-Gender-Detection`), so it
needs no HuggingFace token and can always be exported locally:

```bash
python -c "import sys; sys.path.insert(0,'scripts/provision'); \
           import export_gender; print(export_gender.export('<models-dir>'))"
```

A freshly exported gender model is not byte-identical to the shipped `models_folded/` one if
the local `transformers` differs (node count and size shift), but the properties this bug
turns on — opset 17, plain `ai.onnx`, 20 `Erf`, fp16 initializers, fp32 I/O — are stable, and
it reproduces the linux/arm64 failure with the exact error text from the issue.
