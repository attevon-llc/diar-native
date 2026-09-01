# Upstream report drafts — ORT fp16 GELU fusion (issue #14)

**STATUS: DRAFTS. NOTHING HAS BEEN FILED.** Anything outward-facing against `onnxruntime` or
the `ort` crate needs the operator's explicit approval, per the ground rules in `CLAUDE.md`.
Two separate reports against two separate projects; they are independent and either can be
filed without the other.

Evidence behind both: `validation/RESULTS.md` §7.40, explained in
`docs/ORT_FUSION_FP16_AARCH64.md`, reproducible with `validation/ort_fusion_probe/run_probe.sh`.

Before filing, re-check that neither is already reported, and confirm the claims still hold on
the ORT version being filed against (both were measured on **ORT 1.24.2** as vendored by
`ort =2.0.0-rc.12`).

---

## Draft 1 — microsoft/onnxruntime

**Title:** GeluFusion emits `com.microsoft.Gelu(float16)` on aarch64 where no fp16 kernel is
registered, so a valid fp32-in/fp32-out fp16 model cannot load

**Type:** bug

### What happens

An opset-17 model containing no contrib ops at all fails to load with an error naming a contrib
op:

```
Failed to find kernel for com.microsoft.Gelu(1) (node:'Gelu' ep:'CPUExecutionProvider').
Op with name (Gelu) domain (com.microsoft) and type (Gelu) kernel is not supported in
CPUExecutionProvider. Encountered following errors: (Kernel found kernel in the supported
version range (node_version: 1). However the types are incompatible. This op has been
implemented only for the following types (tensor(float),), but the node in the model has the
following type (tensor(float16))
```

The node does not exist in the file. ORT's own Level-2 `GeluFusionL2` creates it during session
load by folding the `Erf`-based GELU pattern, then kernel resolution fails on the node the
optimizer just synthesized. **The optimizer and the kernel registry in the same build disagree
about which dtypes are supported.**

### Why this is a bug rather than a missing kernel

The same build is internally inconsistent: a graph that loads at
`GraphOptimizationLevel::Level1` fails at `Level2`/`Level3`. Optimizations are supposed to be
transparent, so raising the level should never turn a loadable model into an unloadable one.

It is also **not** simply "aarch64 lacks the kernel". We measured a second aarch64 target —
macOS arm64, ORT 1.24.2, same `ort` crate — which **also has no fp16 `Gelu` kernel**
(`nm` shows `onnxruntime::Gelu<float>` as the only instantiation) and which **loads the model
fine**, because its build does not apply the fusion to an fp16 graph. So two aarch64 builds of
the same release differ in whether the fusion fires, and only one of them is self-consistent.

### Reproduction

- Model: a wav2vec2 audio classifier exported to opset 17 with `torch.onnx.export`, then
  converted with `onnxconverter_common.float16` using `keep_io_types=True` (fp32 in and out,
  213 fp16 initializers, 20 `Erf` nodes, no contrib domain on any node).
- Load it on linux/aarch64 with default session options → the error above.
- `GraphOptimizationLevel::Level1` → loads.
- `optimization.disable_specified_optimizers=GeluFusionL2` → loads.
- Dumping the optimized graph via `optimized_model_filepath` shows 20 `Erf` replaced by
  `com.microsoft::Gelu` ×8 + `com.microsoft::BiasGelu` ×12 on the fp32 build, and untouched on
  the macOS arm64 build.

### Suggested fix

Have `GeluFusion` (and the other contrib-op fusions) check that a kernel is registered for the
node's dtype on the target EP before rewriting — or gate the rewrite on the same dtype list the
kernel registers. Failing that, the error should say the node was synthesized by an optimizer
and name the optimizer, because as written it sends the reader to look for a corrupt model.

---

## Draft 2 — pykeio/ort

**Title:** `SessionBuilder::with_disabled_optimizers` docs say "comma-separated"; ORT actually
splits on `;`, and a comma-joined list silently disables nothing

**Type:** documentation bug (with a possible ergonomics fix)

### What the docs say

```rust
/// Accepts a comma-separated list of optimizers to disable.
pub fn with_disabled_optimizers(mut self, optimizers: impl AsRef<str>) -> BuilderResult
```

### What actually happens

Measured against ORT 1.24.2 via `ort =2.0.0-rc.12`, on linux/aarch64, using load success of a
model that only loads when a specific optimizer is disabled as the observable:

```
with_disabled_optimizers("GeluFusionL2")               -> optimizer disabled   (model loads)
with_disabled_optimizers("GeluFusionL1;GeluFusionL2")  -> both disabled        (model loads)
with_disabled_optimizers("GeluFusionL1,GeluFusionL2")  -> NEITHER disabled     (model fails)
with_disabled_optimizers("BiasGeluFusion,GeluFusionL2") -> NEITHER disabled    (model fails)
```

The comma-joined string is taken as a **single optimizer name**, matches nothing, and disables
nothing. Confirmed independently on real linux/arm64 hardware against a second model file.

### Why it matters more than a typo

ORT **silently ignores an unrecognized optimizer name** — no error, no warning, the session
opens and behaves exactly as if the option were unset. Combined with the wrong separator in the
docs, a user following the documentation gets a config entry that looks applied, does nothing,
and reports nothing. We hit exactly this: the first fix we tried was a comma-joined list and it
appeared to be a no-op for reasons that took a while to attribute correctly.

### Suggested fix

1. Correct the doc comment to name `;` as the separator.
2. Optionally, take `impl IntoIterator<Item = impl AsRef<str>>` and join with `;` internally, so
   the separator cannot be got wrong; or reject a value containing `,` with a message naming
   `;`, since a comma is never valid inside a single optimizer name.

A note that unrecognized names are silently ignored would also save the next person the same
detour.
