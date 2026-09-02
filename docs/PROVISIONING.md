# Provisioning the models

Why diar-native cannot ship you its models, how to get them, and exactly what the verification
does and does not prove. For running the service, start at the [README](../README.md).

- [Why you have to do this](#why-you-have-to-do-this)
- [The token](#the-token)
- [The three subcommands](#the-three-subcommands)
- [What the export produces](#what-the-export-produces)
- [The provenance marker](#the-provenance-marker)
- [What verification checks](#what-verification-checks)
- [What verification does NOT cover](#what-verification-does-not-cover)

---

## Why you have to do this

The models are **not distributed and cannot be**: they are derivatives of the gated
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
weights. There is no `.onnx` on Hugging Face for any of them — upstream ships
`pytorch_model.bin` plus `plda/*.npz` — so **conversion is mandatory, not an optimisation**. Each
operator exports locally with their own token; nothing is redistributed.

`install.sh` does this for you on first run. Everything below is what it is doing, and what to do
when you are driving the binary yourself.

## The token

Two free, instant, **auto-approved** steps — no waiting list, no human review, and the pipeline
itself is CC-BY-4.0:

1. Create a **read** token at <https://huggingface.co/settings/tokens>.
2. **Signed in as that same account**, accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.

> **Step 2 catches people out.** A perfectly valid token whose account never accepted the gate
> fails with **HTTP 403**, and accepting the terms while signed in as a *different* account fails
> identically. The gate is per-account. A rejected or revoked token is **HTTP 401**. Both exit
> **5**.

The token belongs to *you* — this project has no token of its own to fall back on. It is used
only for that one download; **serving never touches the network**. `install.sh` and `start.sh`
prompt for it with hidden input, store it in `.env` at mode 600, and never put it on a command
line. `provision-models` scrubs it out of the exporter's stdout *and* stderr and marks its
`--hf-token` argument `hide_env_values`.

Check a token any time without downloading anything (~200 ms):

```bash
diar-server check-token
```

Token sources, in order: `--hf-token`, then `HF_TOKEN`, `HUGGINGFACE_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`. Empty values are skipped rather than treated as a token. See
[CONFIGURATION.md](CONFIGURATION.md#provisioning-provision-models-verify-models-check-token) for
`HF_ENDPOINT` (mirrors and air-gapped proxies) and `HF_HOME`.

## The three subcommands

```bash
diar-server check-token                                    # ~200 ms, no download
export HF_TOKEN=<your huggingface read token>
diar-server provision-models --models-dir /models --set fast
diar-server verify-models   --models-dir /models           # deep: full sha256 + smoke test
```

All three write **machine-readable JSON to stdout** and install no log subscriber, so their
output is never interleaved with log records. They branch on the
[exit codes](DEPLOYMENT.md#exit-codes), which are a stable contract.

### You need a Python interpreter, but only for this

Provisioning shells out to a python with **torch and pyannote.audio** installed
(`DIAR_EXPORT_PYTHON`, default `python3`), which `diar-server` deliberately does not bundle.
**Serving needs no Python at all** — that is why the CPU serving image is 195 MB rather than
~2 GB — so running `provision-models` against a serving image exits **6** with `No python
interpreter at 'python3'`. That is what the separate `-provision` image tag is for. See
[DEPLOYMENT.md](DEPLOYMENT.md#no-python-in-the-serving-images).

The export does `pipeline.to(torch.device("cpu"))` and never touches an accelerator, which is why
the provisioning image is CPU-based even for GPU deployments. Provisioning also defaults to
`cpu` for its smoke stage, *not* the serving default of `cuda` — provisioning defaulting to a GPU
is what used to brick GPU-less hosts. An unusable device here is exit **9**, and it writes **no
marker**: a device problem must never mark the models known-bad.

In OpenTranscribe's backend image most export dependencies are already present — only
`onnxscript` and a constant folder are missing.

> **CPython 3.13 note.** `onnxsim` publishes no wheel for 3.13 at any version and is a C++
> extension, so it cannot install on a 3.13 image without cmake and a toolchain. The exporter
> falls back to `onnxslim` (a pure-python wheel), which is numerically **bit-exact** and
> eliminates the same ops, but emits a **differently-shaped graph** — so a directory folded with
> onnxslim is functionally equivalent to `models_folded/` but **not byte-comparable** to it.
> Which folder ran is recorded in the marker's `toolchain.folder`.

## What the export produces

Expect roughly **484 MB** written for the `fast` set with gender, in a couple of minutes (the
acceptance run measured **119.5 s**). Only about **32 MB is downloaded**; the rest is produced
locally by the export.

The gender classifier is **189.5 MB of the output (~40%)**. `--skip-gender` omits it, at the cost
of speaker gender detection. Note that gender is enabled by **file presence**, so a
`--skip-gender` deployment answers `diarize(gender=true)` with a 200 and no genders —
`/healthz`'s `models_gender` field is the difference between that being a decision and a mystery.

> **If you are reading an older number:** the RESULTS §7.36 acceptance run reports **673 MB**,
> not 484 MB. That run predates the fp16 gender fix (RESULTS §7.39) and hit the 378.5 MB fp32
> fallback. Export recipe 2 restores fp16, so a directory provisioned by the current build is
> ~484 MB. `models_folded/` is the reference: 483,411,782 bytes.

`--set fast` (default) and `--set small` select the model tier; what differs between them is in
[`MODELS_SETS.md`](../MODELS_SETS.md), which matches the authoritative file lists in
`crates/diar-core/src/provision/files.rs`.

The models directory is checked for writability **up front, before** a multi-hundred-MB export
(exit **7**), not after.

## The provenance marker

`provision-models` writes `diar-provision.json` recording the export-recipe version, the upstream
pipeline revision, toolchain versions, and every file's size and sha256.

**Startup checks it `stat`-only**: the marker parses, the recipe version is current, the smoke
test passed, and every recorded file is present at its recorded length. There is deliberately
**no hashing at startup** — re-reading ~484 MB on every boot is unacceptable.

So startup answers *"is this the directory that passed?"*, **not** *"is this directory still
byte-perfect?"* The latter is what `verify-models` is for. Claiming more would itself be a
fail-open.

The state this produces is surfaced on `/healthz` as `models_state`
(`verified | stale | unverified | failed`), with `models_reason` carrying a human sentence plus
the remediation command for every non-verified state. See [API.md](API.md#get-healthz).

### The startup gate, and why it is asymmetric

Before loading any engine the server does that `stat`-only pass. What it does with the result is
deliberately uneven:

- **A missing or zero-length model file is fatal** (exit **8**), with a message naming
  provisioning and the gate URL. Without this, a half-provisioned directory surfaces as "CUDA
  session load failed" once per configured device, inside a `restart: unless-stopped` crash loop
  that also fails `up --wait` — and the operator's actual problem never appears in the logs. A
  marker that records a **failed** smoke test, or that vouches for a file which is now the wrong
  length, is equally fatal.
- **A missing marker is only a warning.** Every models directory deployed before this feature
  shipped has no marker; refusing to start on those would turn a provenance improvement into an
  outage. An unparseable marker, and one written to a newer schema, are likewise warnings.
- **A stale marker is only a warning.** `stale` means the recipe version differs from the one
  this build ships (`EXPORT_RECIPE_VERSION`, currently **2**), or a `small` set is being asked to
  serve `fast`. Stale directories serve normally — but they return 503 from `/readyz`, which
  gates on `verified` exactly.

Which tier the gate requires is read from the directory's **own marker**, falling back to `fast`
when there is none. `DIAR_MODEL_SET` overrides it, for an operator who wants to assert that a
directory ought to be a given tier and get a loud complaint when it is not.
`DIAR_ALLOW_UNVERIFIED_MODELS=1` downgrades the fatal cases to warnings.

`verify-models` **re-attests by default**, which is the recovery path out of a marker that
records a stale failure.

## What verification checks

`verify-models` runs a **five-stage smoke test**. Stages 1-3 and 5 always run on the **CPU**
execution provider — zero VRAM, no device required, runnable in CI and on a laptop. Only stage 4
uses the configured mode, because it is the only stage whose purpose is to exercise the real
serving path.

1. **Parse** every `.onnx`. Non-obvious: live compose sets `SPEAKRS_LAZY_SESSIONS=1`, and speakrs
   then skips the batch-64 sessions at startup — so a corrupt `*-b64.onnx` is invisible to a
   normal server start. This stage also carries the **aarch64 load gate** described in
   [DEPLOYMENT.md](DEPLOYMENT.md#the-fp16-gender-model-on-linuxarm64); it must stay a load check,
   never a numeric one.
2. **I/O contract** against a compiled-in table of names and shapes. Catches the
   right-filename/wrong-model case.
3. **Cross-path numeric agreement**: fbank b1 vs b32; the fused embedding graph vs the split
   fbank→tail path; multimask vs single tail; the b64 multimask is a byte copy of b32; the b64
   tail is batch-invariant; and the b32 multimask graph agrees with the b1 multimask graph under
   an identical mask.
4. **End-to-end** diarization of a 26 s fixture, with sanity bounds on speakers, segments,
   centroids and gender verdicts.
5. **PLDA** `.npy` headers: exact dtype and shape.

### Why there are no golden files

Nothing here compares against committed reference outputs. That is a **licensing constraint, not
laziness** — reference activations from gated weights would themselves be a derivative we cannot
redistribute. Instead every numeric check is a **cross-path agreement**, which is strictly
stronger than a golden file for the failure we actually care about, because it cannot be
satisfied by a consistently-wrong export.

## What verification does NOT cover

> **`verify-models` proves the models are USABLE. It does not prove they are ACCURATE.**
> This is a real, demonstrated gap, tracked as **issue #21**.

The proof is on the record. On an ubuntu 26.04 / linux-arm64 build, `verify-models` **passes
every one of the five stages**, reporting a plausible-looking 2 speakers and 7 segments on the
smoke fixture — while AMI-16 exclusive DER on that same build is **~52%** instead of 18.7%
(RESULTS §7.52; the full story is in
[DEPLOYMENT.md](DEPLOYMENT.md#the-base-image-is-pinned-to-ubuntu-2404)).

The reason is structural rather than a missing assertion: the cross-path checks in stage 3 verify
that the *graphs agree with each other*, and they did. The regression was in a BLAS kernel
underneath the clustering stage, which every path shares equally — so consistency was preserved
while correctness was not. Embeddings were fine (centroids matched at ~1.0000 cosine); the
clustering grouped them into the wrong number of speakers.

**Practical consequence.** Anything that changes the numeric substrate — a base-image bump, an
OpenBLAS version, a new ORT release, a different export recipe — needs a **DER check on a real
corpus, on each published architecture, run natively**. A smoke pass is not evidence. QEMU is not
evidence either, since OpenBLAS picks kernels by runtime CPU detection. The protocol for such a
run is [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) and the numbers to beat are in
[`TEST_CORPORA_AND_BASELINES.md`](TEST_CORPORA_AND_BASELINES.md).

---

Deeper procedure, prerequisites and the CPython 3.13 caveat:
[`INSTALL_NATIVE.md`](INSTALL_NATIVE.md).

See also: [DEPLOYMENT.md](DEPLOYMENT.md) · [CONFIGURATION.md](CONFIGURATION.md) ·
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [README](../README.md)
