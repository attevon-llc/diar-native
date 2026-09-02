# Model sets

Both sets are produced by one export. `--set small` runs the full recipe and then deletes
the fast-only files, which is why the 19 shared files are **md5-identical** across the two
sets (verified).

| set | directory | files | on disk | VRAM | throughput |
|---|---|---|---|---|---|
| fast (default, >=6 GB GPUs) | `models_folded/` | 24 | 483 MB | 4.2 GB | 277x RT warm |
| small (laptops) | `models_small/` | 19 | 393 MB | 1.6 GB | 59x RT |

Sizes are the shipped directories measured (`models_folded/` = 483,411,782 bytes). A freshly
provisioned `fast` set is within ~57 KB of that; see the fp16 note below.

## The actual difference: 5 files, not 1

An earlier version of this file claimed small was fast "minus
`wespeaker-multimask-tail-b64.onnx`". That was wrong. The real difference:

- `segmentation-3.0-b64.onnx`
- `wespeaker-voxceleb-resnet34-b64.onnx`
- `wespeaker-multimask-tail-b64.onnx`
- `wespeaker-voxceleb-resnet34-tail-b64.onnx`
- `gender-wav2vec2.meta.json`

All four `.onnx` entries are the batch-64 graphs; the small tier never takes the batched
path, so it does not carry them. `gender-wav2vec2.meta.json` is documentation only —
`crates/diar-core/src/gender.rs` hardcodes its labels and never reads the file — but
provisioning cross-checks it against that compiled-in constant, so it is not merely
decorative.

`gender-wav2vec2.onnx` itself (189.5 MB, ~40% of the directory) is present in **both** sets.
`--skip-gender` omits it; the sidecar then returns no speaker genders, silently — which is why
`/healthz` reports `models_gender`, so that a `--skip-gender` deployment answering
`diarize(gender=true)` with 200 and no genders is a decision rather than a mystery.

## The gender model is fp16, and that is load-bearing

`gender-wav2vec2.onnx` must be **fp16** (213/213 FLOAT16 initializers, fp32 in and out, opset
17). The fp32 fallback is 378.5 MB and costs roughly **+500 MiB VRAM**.

fp16 conversion broke on torch 2.13 — the graph it emits carries two **no-op `Cast` nodes** that
made `onnxconverter_common.float16` produce something ORT rejected, so provisioning silently fell
back to fp32. The exporter now elides them; `EXPORT_RECIPE_VERSION` was bumped to **2** to mark
the change. Directories provisioned by recipe 1 are reported `stale` — they still serve, but they
carry the heavy classifier. `provision-models --force` brings them current.

On **aarch64** the fp16 model needs an ORT optimization-level cap to load at all — see `docs/DEPLOYMENT.md`.
That is handled automatically in `crates/diar-core/src/ort_compat.rs`; nothing to configure.

The authoritative list lives in `crates/diar-core/src/provision/files.rs`
(`SHARED_REQUIRED` / `FAST_ONLY_REQUIRED`), which is unit-tested against these counts.

## Getting them

The models are not distributed. Export them locally — see
[`docs/INSTALL_NATIVE.md`](docs/INSTALL_NATIVE.md) step 0:

```bash
HF_TOKEN=<your token> diar-server provision-models --models-dir ./models_folded --set fast
```

## Two artifacts that look like mistakes and are not

- `wespeaker-multimask-tail-b64.onnx` is a **byte copy** of the b32 graph. speakrs asks for
  the b64 filename but sizes its multimask buffers for 32, so a genuine batch-64 graph there
  crashes the worker (RESULTS §4.15).
- `segmentation-3.0*.onnx` are **constant-folded** graphs written under the plain filenames.
  Unfolded graphs are ~2x slower on CUDA and silently fall back to CPU for `Sin`/`Cos`
  (RESULTS §4.1).

Both are asserted by `diar-server verify-models`, so neither can be "tidied up" by accident.
