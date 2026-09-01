# ORT graph fusion vs. fp16 on aarch64 — why the gender model fails to load on linux/arm64

**Status:** investigated, root-caused, fix chosen. Not yet implemented.
**Issue:** #14. **Full measurements:** `validation/RESULTS.md` §7.40 (append-only; that
section is the record, this file is the explanation).

---

## The one-paragraph version

`models_folded/gender-wav2vec2.onnx` is fp16. On **linux/arm64** it fails to *load* — the
session never opens, so speaker gender is silently unavailable on that platform. The model
file is not at fault: it is plain opset-17 `ai.onnx` with no contrib ops in it. ONNX Runtime
*rewrites* it at load time — an optimizer folds the `Erf`-based GELU pattern into
`com.microsoft.Gelu` — and then discovers it has no fp16 kernel for the node it just created.
The fix is to stop ORT performing that one rewrite on that one session. Diarization is
unaffected; only the gender classifier is.

---

## What actually goes wrong

```
gender-wav2vec2.onnx            ORT load-time optimizer          ORT kernel lookup
  opset 17, plain ai.onnx  ──▶  GeluFusionL2 rewrites       ──▶  com.microsoft.Gelu
  20 × Erf                      20 × Erf into                     fp16?  ✗ no kernel
  213 fp16 initializers         com.microsoft.Gelu                fp32?  ✓ exists
                                                                      │
                                                                      ▼
                                            Failed to find kernel for com.microsoft.Gelu(1)
                                            … implemented only for (tensor(float),), but the
                                            node in the model has (tensor(float16))
```

The node named in the error **does not exist in the file on disk**. It is created by ORT
during session load. That is why grepping the model for `com.microsoft.Gelu` finds nothing
and why the error is confusing on first read.

## The part that was surprising

The natural theory — *"amd64 has an fp16 kernel for the fused op and aarch64 doesn't"* — is
only half right, and the wrong half is the half that matters.

**Every aarch64 ORT build checked is missing the fp16 kernel.** `nm` on the ORT 1.24.2 static
library that the `ort` crate links on macOS arm64 shows exactly one instantiation,
`onnxruntime::Gelu<float>`, and zero for `MLFloat16` — the same gap as linux/arm64. Yet the
model loads fine on macOS.

The difference is **whether the fusion fires at all**:

| platform | fp16 kernel for fused Gelu | does the fusion rewrite fp16? | result |
| --- | --- | --- | --- |
| linux/amd64 | yes | yes | loads |
| **linux/arm64** | **no** | **yes** | **fails** |
| macOS arm64 (native, CPU or CoreML) | no | **no** — declines fp16 | loads |

macOS is not rescued by having the kernel. It is rescued because its ORT build refuses to
create the node in the first place. So this is a **build-configuration divergence between two
targets of the same ORT 1.24.2 release**, not an Apple-silicon property and not an
architecture property. Upstream-reportable against onnxruntime: an optimizer should not emit
a node the same build cannot execute.

Evidence (optimized-graph dumps via `with_optimized_model_path`, macOS arm64):

| graph | opt level | nodes | `Erf` left | contrib ops emitted |
| --- | --- | --- | --- | --- |
| gender **fp16** | Level3 (default) | 994 | 20 | none |
| gender **fp16** | Level1 | 994 | 20 | none |
| gender **fp32** | Level3 (default) | 496 | 0 | `Gelu` ×8, `BiasGelu` ×12, `FusedMatMul` ×12 |

The fp32 row proves the dump does capture Level-2 fusions, so the fp16 row's surviving 20
`Erf` is a real negative rather than a blind spot in the measurement.

## Choosing the fix — and two traps in the obvious one

Both candidates are one line in `GenderModel::load_optional`
(`crates/diar-core/src/gender.rs`). Nothing under `vendor/` is involved, and `ort`
`=2.0.0-rc.12` already exposes what is needed (`SessionBuilder::with_disabled_optimizers`,
and the generic `with_config_entry`) — no version bump.

| candidate | what it does | loads on linux/arm64? | numerics vs. unoptimized reference |
| --- | --- | --- | --- |
| **(a)** `optimization.disable_specified_optimizers=GeluFusionL2` | kills just that one rewrite | ✅ | max \|Δ logit\| 9.58e-04, labels 6/6 |
| **(b)** cap the gender session at `GraphOptimizationLevel::Level1` | skips all Level-2 fusions | ✅ | **0.000e+00 — bitwise identical**, labels 6/6 |
| (c) fp32 gender on this platform | avoids fp16 entirely | ✅ | +190 MB disk, ~500 MiB VRAM (§7.18) |

### Trap 1 — the optimizer is called `GeluFusionL2`, not `GeluFusion`

ORT registers the Erf-GELU pass **twice**, an L1 and an L2 instance, under suffixed names.
Only disabling `GeluFusionL2` works:

```
disable_specified_optimizers=GeluFusion      → still FAILS
disable_specified_optimizers=GeluFusionL1    → still FAILS
disable_specified_optimizers=GeluFusionL2    → loads ✅
```

### Trap 2 — a wrong name is silently ignored

`disable_specified_optimizers=NotARealOptimizerName` loads fine and changes nothing. There is
no error, no warning. A misspelled name ships a config entry that *looks* applied and does
nothing — which is exactly the failure mode candidate (a) would have had with the name from
the issue.

### Trap 3 — the separator is `;`, not `,`

```
GeluFusionL2;BiasGeluFusion    → both disabled ✅
BiasGeluFusion,GeluFusionL2    → neither disabled ✗
GeluFusionL1,GeluFusionL2      → neither disabled ✗
```

The `ort` crate's own doc comment on `with_disabled_optimizers` says *"Accepts a
comma-separated list of optimizers to disable"*, which is wrong for this build. Worth an
upstream `ort` documentation issue. Practically: pass a single name, or separate with `;`.

### Recommendation: **(b), Level1 on the gender session only**

(a) is more surgical and is what issue #14 preferred, and it is validated and safe. But (b)
is bitwise identical to the unoptimized graph, and it does not depend on an ORT-internal
optimizer name that is undocumented, silently ignored when wrong, and *already renamed once*.
Ship (b); (a) is the recorded alternative if the other Level-2 optimizations are ever wanted
back.

Whichever ships is **inert on macOS**, where Level3 and Level0 already produce bitwise
identical output on this graph.

## The latent risk this uncovered

"Diarization is fine on aarch64" is true but is **luck, not immunity**. Dumping the optimized
form of all 15 diarization graphs shows **11 of them are rewritten into a contrib op by the
same machinery**:

| graphs | contrib ops after Level3 | initializer dtypes |
| --- | --- | --- |
| `wespeaker-voxceleb-resnet34{,-b32,-b64}`, `…-tail{,-b3,-b32,-b64}`, `wespeaker-multimask-tail{,-b32,-b64}` (11) | `com.microsoft::FusedConv` ×33 each | FLOAT only |
| `segmentation-3.0{,-b32,-b64}`, `wespeaker-fbank{,-b32}` (4) | none | FLOAT only |

The only `FusedConv` kernel in the build is `FusedConv_kMSDomain_ver1_float`. **Those graphs
are safe today solely because they are fp32.** §4.18 rejected fp16 for the embedding graphs on
accuracy grounds, not this one — so if fp16 is ever revisited there, 11 of 15 graphs land on
the identical failure.

> **Rule to carry forward:** any future fp16 export must be gated on *loading* on aarch64, not
> only on numerical accuracy. An accuracy gate cannot catch this, because the session never
> opens far enough to produce a number.

## Also worth knowing: fp16 logits are not portable across ORT builds

The same fp16 model, the same six inputs, two aarch64 ORT builds: logits differ by up to
**0.29**, purely from arithmetic ordering. All six labels still agree. Within a single build,
every optimization level agrees bitwise.

So an fp16 gender **logit** is not a cross-build-reproducible number — only the **label** is.
Gates should assert on labels (as `scripts/provision/export_gender.py` already does), never
on logit equality across platforms.

## Reproducing this

The failing platform is reachable from an Apple Silicon Mac: Docker Desktop runs `linux/arm64`
containers natively, so no Linux box is needed to reproduce or to test a fix.

- Base image must be **`rust:1-trixie`**, not `bookworm` — this ORT needs glibc ≥ 2.38
  (`__isoc23_strtol`); bookworm's 2.36 fails at link.
- Needs `RUSTFLAGS="-C link-arg=-lstdc++"`.
- The native macOS side needs `LIBRARY_PATH=/opt/homebrew/opt/openblas/lib` for both the
  default and the `coreml` build. `--mode coreml` additionally needs the `.mlmodelc` assets
  in the models directory.
- The gender classifier repo (`prithivMLmods/Common-Voice-Gender-Detection`) is **ungated** —
  exporting it needs no HuggingFace token, so this whole investigation reproduces without one.

Platform matrix as measured (`diar-server verify-models --set fast`):

| build | `--mode` | result |
| --- | --- | --- |
| default (CPU), native macOS arm64 | `cpu` | 16 graphs loaded; 2 speakers, 7 segments, 8 exclusive, gender=2 |
| `--features coreml`, native macOS arm64 | `cpu` | same |
| `--features coreml`, native macOS arm64 | `coreml` | same |
| `--features coreml`, native macOS arm64 | `coreml_fast` | same |
| linux/arm64 (Docker) | `cpu` | gender session **fails to load**; diarization graphs fine |
