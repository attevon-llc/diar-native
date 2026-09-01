//! The smoke test: five stages that separate "the files exist" from "the files are usable".
//!
//! This runs in **Rust, against the same ORT build the server uses**, deliberately. A
//! python-side check would validate a different onnxruntime than the one that serves
//! traffic, and would leave the whole "a subtly-wrong export does not fail loudly" hole
//! open — which is the reason the issue asks for this at all.
//!
//! Stages 1-3 and 5 always run on the **CPU** execution provider: zero VRAM, no device
//! required, runnable in CI and on a laptop. Only stage 4 (the end-to-end run) uses the
//! configured mode, because it is the only stage whose purpose is to exercise the real
//! serving path.
//!
//! Nothing here compares against committed reference outputs. That is a licensing
//! constraint, not laziness — reference activations from gated weights would themselves be
//! a derivative we cannot redistribute. Instead every numeric check is a **cross-path
//! agreement**: two graphs that were exported from the same checkpoint by different routes
//! must agree with each other. That is strictly stronger than a golden file for the failure
//! we actually care about, because it cannot be satisfied by a consistently-wrong export.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::{Tensor, ValueType};

use super::files::{
    io_spec, onnx_files, required_files, IoSpec, ModelSet, GENDER_META, GENDER_MODEL, PLDA_SPECS,
};
use crate::{DiarEngine, EngineConfig, Mode};

/// Agreement tolerance for cross-path numeric checks.
///
/// Generous relative to what a correct export actually achieves — RESULTS §7.33 measured
/// 7.8e-08 for the b64 tail batch-invariance check, and §7.16 measured 5.96e-06 for the
/// gender ONNX-vs-torch parity gate. A real disagreement is orders of magnitude larger than
/// this, so the bar separates "different graph" from "different float ordering" cleanly.
/// If a check ever lands near this bar, that is a finding to root-cause — NOT a number to
/// relax.
const TOL: f32 = 1e-4;

const SAMPLES_10S: usize = 160_000;
const FBANK_FRAMES: usize = 998;
const FBANK_FEATURES: usize = 80;
const MASK_FRAMES: usize = 589;
const EMBED_DIM: usize = 256;
/// Segmentation output is `[batch, 589, 7]` for a 10 s chunk.
const SEG_CLASSES: usize = 7;

/// The row compared in every fixed-batch check. Deliberately NOT 0: a graph exported with a
/// batch-coupling bug (or one that only wires row 0 to the real compute) would pass a row-0
/// comparison and fail on every other row, which is the shape of the bug these checks exist
/// to find.
const PROBE_ROW: usize = 5;

/// The device asked for is not usable on this machine — as distinct from the models being
/// wrong.
///
/// This distinction is load-bearing, not cosmetic. `provision-models` writes a `fail` marker
/// when the smoke test rejects a directory, and a `fail` marker makes every later
/// `diar-server` start exit non-zero. Conflating "this box has no GPU" with "these 470 MB of
/// models are bad" therefore takes a perfectly good models directory permanently out of
/// service (and, on the CPU-only image, would do so on EVERY provisioning run). Whenever this
/// error is raised, the same directory has just been proven to load on the CPU EP.
#[derive(Debug)]
pub struct DeviceUnavailable {
    pub mode: String,
    pub detail: String,
}

impl std::fmt::Display for DeviceUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the `{}` execution device is not usable on this machine, so the end-to-end \
             stage could not run here. The models themselves are NOT implicated — the same \
             directory loaded successfully on the CPU execution provider moments ago. \
             Re-run with `--mode cpu` (the default for provisioning), or give the container \
             a device (`docker run --gpus all …`).\n\nUnderlying error: {}",
            self.mode, self.detail
        )
    }
}

impl std::error::Error for DeviceUnavailable {}

/// Was this failure caused by a missing/unusable device rather than by the models?
///
/// Walks the whole `anyhow` chain, so it keeps working when callers add context.
pub fn is_device_unavailable(err: &anyhow::Error) -> bool {
    err.chain().any(|e| e.is::<DeviceUnavailable>())
}

#[derive(Debug, Clone)]
pub struct SmokeOptions {
    pub models_dir: PathBuf,
    pub set: ModelSet,
    pub with_gender: bool,
    /// Mode for stage 4 only.
    pub mode: Mode,
    /// 16 kHz mono WAV. Defaults to the baked-in fixture.
    pub clip: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StageResult {
    pub stage: String,
    pub detail: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SmokeReport {
    pub stages: Vec<StageResult>,
    pub num_speakers: usize,
    pub segments: usize,
    pub duration_ms: u64,
    pub clip_sha256: String,
    pub mode: String,
}

/// Load a session on the CPU EP. Every stage-1..3 session goes through here.
fn cpu_session(path: &Path) -> Result<Session> {
    // Must go through ort_compat, or the smoke test verifies a session the server will never
    // build. On aarch64 this is what lets the fp16 gender graph load at all (issue #14).
    crate::ort_compat::session_for(path).map_err(|e| {
        // "this file is broken" and "this runtime cannot execute this graph" need opposite
        // remediations, and conflating them sends the operator to re-export a file that is
        // perfectly good. The aarch64 fp16 GELU case (issue #14) is exactly this: ORT's own
        // optimizer synthesizes a contrib op the same build has no kernel for, and the
        // resulting message named an operator our export does not even contain.
        let detail = e.to_string();
        let unsupported = detail.contains("Failed to find kernel")
            || detail.contains("kernel is not supported")
            || detail.contains("NOT_IMPLEMENTED");
        if unsupported {
            e.context(format!(
                "STAGE 1 FAILED: {} parsed, but THIS ONNX Runtime build cannot execute it. \
                 The file is not corrupt and re-exporting will produce the same one — the \
                 runtime is missing a kernel for an operator or dtype on this platform. \
                 Try `DIAR_ORT_OPT_LEVEL=basic` (ORT's optimizer can fuse graphs into ops it \
                 then cannot run), and see issue #14.",
                path.display()
            ))
        } else {
            e.context(format!(
                "STAGE 1 FAILED: {} could not be loaded as an ONNX graph. The file is \
                 present but not a usable model — it is truncated, corrupt, or not ONNX. \
                 Re-run `provision-models --force`.",
                path.display()
            ))
        }
    })
}

fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening smoke clip {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        bail!(
            "smoke clip {} must be 16 kHz mono, got {} Hz {} ch",
            path.display(),
            spec.sample_rate,
            spec.channels
        );
    }
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    })
}

/// Deterministic mask values. A seeded generator rather than all-ones: a constant mask
/// makes the weighted-pooling arithmetic degenerate, and several plausible pooling bugs
/// cancel out exactly when every weight is equal.
fn seeded_mask(n: usize, seed: u64) -> Vec<f32> {
    // Mix the seed with an odd multiplier and an xor rather than `seed | 1`. That earlier
    // form mapped every EVEN seed onto its odd successor, so `seeded_mask(n, 42)` and
    // `seeded_mask(n, 43)` were byte-identical — which would have made half the 64 mask
    // rows in stage 3e duplicates and quietly halved the strength of the batch-invariance
    // check. Both operations here are bijections, so distinct seeds stay distinct.
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_BABE;
    if s == 0 {
        s = 0x1234_5678_9ABC_DEF0;
    }
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Keep strictly inside (0,1] so no row is fully masked out.
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 0.9 + 0.1
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_f32(
    session: &mut Session,
    inputs: Vec<(
        std::borrow::Cow<'_, str>,
        ort::session::SessionInputValue<'_>,
    )>,
    out_name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let outputs = session
        .run(inputs)
        .map_err(|e| anyhow!("inference failed: {e}"))?;
    let (shape, data) = outputs[out_name]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("extracting output `{out_name}`: {e}"))?;
    Ok((shape.to_vec(), data.to_vec()))
}

/// STAGE 1 — every graph in the set parses.
///
/// Non-obvious and load-bearing: live compose sets `SPEAKRS_LAZY_SESSIONS=1`, and speakrs'
/// loader then SKIPS the batch-64 sessions entirely at startup. So a corrupted
/// `wespeaker-voxceleb-resnet34-b64.onnx` is completely invisible to a normal server start
/// and only surfaces on the first batched job in production. This stage loads every file
/// unconditionally.
fn stage1_parse_all(opts: &SmokeOptions) -> Result<StageResult> {
    let files = onnx_files(opts.set, opts.with_gender);
    for name in &files {
        let p = opts.models_dir.join(name);
        let _session = cpu_session(&p)?;
    }
    Ok(StageResult {
        stage: "1-parse".into(),
        detail: format!("{} ONNX graphs loaded on the CPU EP", files.len()),
    })
}

fn shape_of(vt: &ValueType) -> Option<Vec<i64>> {
    match vt {
        ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
        _ => None,
    }
}

fn check_io(session: &Session, spec: &IoSpec) -> Result<()> {
    for (kind, outlets, expected) in [
        ("input", session.inputs(), spec.inputs),
        ("output", session.outputs(), spec.outputs),
    ] {
        if outlets.len() != expected.len() {
            bail!(
                "STAGE 2 FAILED: {} declares {} {kind}(s), expected {}. This is the \
                 signature of a right-filename/wrong-model mix-up (RESULTS §1).",
                spec.file,
                outlets.len(),
                expected.len()
            );
        }
        for (outlet, (want_name, want_shape)) in outlets.iter().zip(expected) {
            if outlet.name() != *want_name {
                bail!(
                    "STAGE 2 FAILED: {} {kind} is named `{}`, expected `{}`. Names are \
                     load-bearing — the engine binds tensors by name, so this model cannot \
                     be driven even though it loaded.",
                    spec.file,
                    outlet.name(),
                    want_name
                );
            }
            let Some(got) = shape_of(outlet.dtype()) else {
                bail!(
                    "STAGE 2 FAILED: {} {kind} `{want_name}` is not a tensor",
                    spec.file
                );
            };
            if got.len() != want_shape.len() {
                bail!(
                    "STAGE 2 FAILED: {} {kind} `{want_name}` has rank {}, expected {}",
                    spec.file,
                    got.len(),
                    want_shape.len()
                );
            }
            for (axis, (g, w)) in got.iter().zip(want_shape.iter()).enumerate() {
                // `None` == dynamic/batch, and ORT reports dynamic dims as -1. Only assert
                // dimensions the recipe actually fixes.
                if let Some(w) = w {
                    if *g != *w {
                        bail!(
                            "STAGE 2 FAILED: {} {kind} `{want_name}` axis {axis} is {g}, \
                             expected {w}. Wrong model under the right filename.",
                            spec.file
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// STAGE 2 — the I/O contract, against a compiled-in table.
///
/// Catches the exact incident recorded in RESULTS §1: ONNX artifacts sitting at the
/// expected paths that had been exported from the wrong (fallback 3.1) checkpoints, so
/// every downstream number silently measured a different model.
fn stage2_io_contract(opts: &SmokeOptions) -> Result<StageResult> {
    let mut checked = 0;
    for name in onnx_files(opts.set, opts.with_gender) {
        let Some(spec) = io_spec(name) else { continue };
        let session = cpu_session(&opts.models_dir.join(name))?;
        check_io(&session, spec)?;
        checked += 1;
    }

    // The gender sidecar is documentation-only at runtime (`gender.rs` hardcodes its
    // labels), which makes it exactly the sort of file that rots unnoticed. Checking it
    // against the compiled-in constant turns it into a real guard: if upstream ever
    // reorders id2label, the engine would silently report every speaker's gender inverted.
    if opts.with_gender && opts.set == ModelSet::Fast {
        let p = opts.models_dir.join(GENDER_META);
        let raw =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", p.display()))?;
        let map = v
            .get("id2label")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("{} has no id2label object", p.display()))?;
        for (idx, expected) in crate::gender::ID2LABEL.iter().enumerate() {
            let got = map
                .get(&idx.to_string())
                .and_then(|s| s.as_str())
                .ok_or_else(|| anyhow!("{} id2label has no index {idx}", p.display()))?;
            if got != *expected {
                bail!(
                    "STAGE 2 FAILED: {} maps label {idx} to `{got}`, but diar-core is \
                     compiled with `{expected}`. Upstream has relabelled the classifier — \
                     every gender verdict this build produces would be wrong.",
                    p.display()
                );
            }
        }
        checked += 1;
    }

    Ok(StageResult {
        stage: "2-io-contract".into(),
        detail: format!("{checked} signatures matched the compiled-in contract"),
    })
}

/// STAGE 3 — cross-path numeric agreement.
fn stage3_numeric(opts: &SmokeOptions, audio: &[f32]) -> Result<StageResult> {
    let dir = &opts.models_dir;
    if audio.len() < SAMPLES_10S {
        bail!(
            "smoke clip is {} samples, need at least {SAMPLES_10S} (10 s) for the \
             embedding graphs",
            audio.len()
        );
    }
    let clip: Vec<f32> = audio[..SAMPLES_10S].to_vec();
    let mask = seeded_mask(MASK_FRAMES, 0x5EED);
    let mut details: Vec<String> = Vec::new();

    // --- 3a: fbank batch-1 vs batch-32, row 0 ---
    let mut fbank = cpu_session(&dir.join("wespeaker-fbank.onnx"))?;
    let wave1 = Array3::from_shape_vec((1, 1, SAMPLES_10S), clip.clone())?;
    let (_, fb1) = run_f32(
        &mut fbank,
        ort::inputs!["waveform" => Tensor::from_array(wave1)?],
        "fbank",
    )?;

    let mut fbank32 = cpu_session(&dir.join("wespeaker-fbank-b32.onnx"))?;
    let mut wide = Vec::with_capacity(32 * SAMPLES_10S);
    for _ in 0..32 {
        wide.extend_from_slice(&clip);
    }
    let wave32 = Array3::from_shape_vec((32, 1, SAMPLES_10S), wide)?;
    let (_, fb32) = run_f32(
        &mut fbank32,
        ort::inputs!["waveform" => Tensor::from_array(wave32)?],
        "fbank",
    )?;
    let row = FBANK_FRAMES * FBANK_FEATURES;
    let d3a = max_abs_diff(&fb1[..row], &fb32[..row]);
    if d3a > TOL {
        bail!(
            "STAGE 3a FAILED: wespeaker-fbank.onnx and wespeaker-fbank-b32.onnx disagree \
             on identical audio by {d3a:.3e} (bar {TOL:.0e}). One of the two fbank graphs \
             is wrong."
        );
    }
    details.push(format!("3a fbank b1-vs-b32 {d3a:.2e}"));

    // --- 3b: fused embedding vs split fbank -> tail (the strongest single check) ---
    // Couples three graphs at once. If any one of the fused graph, the fbank graph, or the
    // tail graph is from a different checkpoint or a botched export, this disagrees.
    let mut fused = cpu_session(&dir.join("wespeaker-voxceleb-resnet34.onnx"))?;
    let w1 = Array2::from_shape_vec((1, MASK_FRAMES), mask.clone())?;
    let (_, fused_out) = run_f32(
        &mut fused,
        ort::inputs![
            "waveform" => Tensor::from_array(Array3::from_shape_vec((1, 1, SAMPLES_10S), clip.clone())?)?,
            "weights" => Tensor::from_array(w1.clone())?
        ],
        "output",
    )?;

    let mut tail = cpu_session(&dir.join("wespeaker-voxceleb-resnet34-tail.onnx"))?;
    let fb_arr = Array3::from_shape_vec((1, FBANK_FRAMES, FBANK_FEATURES), fb1[..row].to_vec())?;
    let (_, split_out) = run_f32(
        &mut tail,
        ort::inputs![
            "fbank" => Tensor::from_array(fb_arr.clone())?,
            "weights" => Tensor::from_array(w1.clone())?
        ],
        "output",
    )?;
    let d3b = max_abs_diff(&fused_out, &split_out);
    if d3b > TOL {
        bail!(
            "STAGE 3b FAILED: the fused wespeaker-voxceleb-resnet34.onnx and the split \
             wespeaker-fbank.onnx -> wespeaker-voxceleb-resnet34-tail.onnx path disagree by \
             {d3b:.3e} (bar {TOL:.0e}) on identical input. At least one of those three \
             graphs was exported from a different checkpoint or is corrupt."
        );
    }
    details.push(format!("3b fused-vs-split {d3b:.2e}"));

    // --- 3c: multimask tail vs single tail, identical mask ---
    // The ONNX counterpart of the torch-side parity assert in export_models.py.
    let mut multimask = cpu_session(&dir.join("wespeaker-multimask-tail.onnx"))?;
    let mut masks3 = Vec::with_capacity(3 * MASK_FRAMES);
    for _ in 0..3 {
        masks3.extend_from_slice(&mask);
    }
    let (_, mm_out) = run_f32(
        &mut multimask,
        ort::inputs![
            "fbank" => Tensor::from_array(fb_arr.clone())?,
            "masks" => Tensor::from_array(Array2::from_shape_vec((3, MASK_FRAMES), masks3)?)?
        ],
        "output",
    )?;
    let d3c = max_abs_diff(&mm_out[..EMBED_DIM], &split_out);
    if d3c > TOL {
        bail!(
            "STAGE 3c FAILED: wespeaker-multimask-tail.onnx row 0 disagrees with \
             wespeaker-voxceleb-resnet34-tail.onnx by {d3c:.3e} (bar {TOL:.0e}) under an \
             identical mask."
        );
    }
    details.push(format!("3c multimask-vs-tail {d3c:.2e}"));

    // --- 3d: the multimask b64 file must be a byte COPY of b32 ---
    // NOT an export. RESULTS §4.15: speakrs' loader asks for the b64 filename while its
    // runtime buffers are sized 32, so a genuine batch-64 graph there kills the worker with
    // "receiver disconnected". The b32 graph under the b64 name is the fix, and this check
    // is what stops a well-meaning future change from "correcting" it.
    if opts.set == ModelSet::Fast {
        let a = super::sha256_file(&dir.join("wespeaker-multimask-tail-b32.onnx"))
            .map_err(|e| anyhow!("{e}"))?;
        let b = super::sha256_file(&dir.join("wespeaker-multimask-tail-b64.onnx"))
            .map_err(|e| anyhow!("{e}"))?;
        if a != b {
            bail!(
                "STAGE 3d FAILED: wespeaker-multimask-tail-b64.onnx is not a byte-for-byte \
                 copy of wespeaker-multimask-tail-b32.onnx (sha256 {a:.16} vs {b:.16}). A \
                 real batch-64 graph under that filename crashes the worker — speakrs sizes \
                 its multimask buffers for 32 (RESULTS §4.15)."
            );
        }
        details.push("3d multimask-b64 is a byte copy of b32".to_string());
    }

    // --- 3e: the batch-32 MULTIMASK tail, against the batch-1 multimask ---
    //
    // THE PRODUCTION HOT PATH, and until now verified by nothing but its I/O shape. Live
    // compose sets `SPEAKRS_LAZY_SESSIONS=1`; the multimask sessions are the ones speakrs
    // loads REGARDLESS of that flag (`inference/embedding/load/sessions.rs`, which gates the
    // primary/split-tail batched sessions on `!lazy_sessions` but not this one). So this
    // graph runs on every batched job in production, while stage 3d only proved the b64 file
    // is a copy of it and stage 3c only checked the *batch-1* multimask.
    //
    // Feeding 32 DISTINCT masks (rather than 32 copies of one) is what makes this a real
    // probe: it is comparing row PROBE_ROW of a genuinely heterogeneous batch.
    let mut mm32 = cpu_session(&dir.join("wespeaker-multimask-tail-b32.onnx"))?;
    let mut fb32_in = Vec::with_capacity(32 * row);
    let mut mm_masks = Vec::with_capacity(96 * MASK_FRAMES);
    for i in 0..32 {
        fb32_in.extend_from_slice(&fb1[..row]);
        for s in 0..3 {
            // Row PROBE_ROW, speaker 0 gets the same mask stage 3c used, so the two graphs
            // are being asked the identical question.
            if i == PROBE_ROW && s == 0 {
                mm_masks.extend_from_slice(&mask);
            } else {
                mm_masks.extend(seeded_mask(MASK_FRAMES, 0xB100 + (i * 3 + s) as u64));
            }
        }
    }
    let (_, mm32_out) = run_f32(
        &mut mm32,
        ort::inputs![
            "fbank" => Tensor::from_array(Array3::from_shape_vec((32, FBANK_FRAMES, FBANK_FEATURES), fb32_in)?)?,
            "masks" => Tensor::from_array(Array2::from_shape_vec((96, MASK_FRAMES), mm_masks)?)?
        ],
        "output",
    )?;
    let probe = PROBE_ROW * 3 * EMBED_DIM;
    let d3e = max_abs_diff(&mm32_out[probe..probe + EMBED_DIM], &mm_out[..EMBED_DIM]);
    if d3e > TOL {
        bail!(
            "STAGE 3e FAILED: wespeaker-multimask-tail-b32.onnx row {PROBE_ROW} disagrees \
             with wespeaker-multimask-tail.onnx by {d3e:.3e} (bar {TOL:.0e}) under an \
             identical mask. This is the graph production actually executes on every \
             batched job — a wrong one degrades DER on every file longer than one window, \
             silently."
        );
    }
    details.push(format!("3e multimask-b32-vs-b1 {d3e:.2e}"));

    // --- 3f: the fixed-batch SEGMENTATION graph, against the batch-1 graph ---
    //
    // Folding is applied to all three segmentation graphs by `fold_segmentation.py`, and the
    // b32/b64 graphs have STATIC shapes where the b1 graph is dynamic — so they are not the
    // same protobuf and a folder bug can hit one and not the other. On CPython 3.13 the
    // folder is `onnxslim`, which rewrites `Gemm` into `MatMul`+`Add`; that rewrite landing
    // wrong on only the static-shaped graphs would leave every multi-window file segmented
    // by a broken graph with every other check still green.
    let mut seg1 = cpu_session(&dir.join("segmentation-3.0.onnx"))?;
    let (_, seg1_out) = run_f32(
        &mut seg1,
        ort::inputs!["input" => Tensor::from_array(Array3::from_shape_vec((1, 1, SAMPLES_10S), clip.clone())?)?],
        "output",
    )?;
    let mut seg32 = cpu_session(&dir.join("segmentation-3.0-b32.onnx"))?;
    let mut seg_wide = Vec::with_capacity(32 * SAMPLES_10S);
    for _ in 0..32 {
        seg_wide.extend_from_slice(&clip);
    }
    let (_, seg32_out) = run_f32(
        &mut seg32,
        ort::inputs!["input" => Tensor::from_array(Array3::from_shape_vec((32, 1, SAMPLES_10S), seg_wide)?)?],
        "output",
    )?;
    let seg_row = MASK_FRAMES * SEG_CLASSES;
    if seg1_out.len() < seg_row || seg32_out.len() < seg_row {
        bail!(
            "STAGE 3f FAILED: segmentation output is {} / {} values, expected at least \
             {seg_row} ({MASK_FRAMES}x{SEG_CLASSES})",
            seg1_out.len(),
            seg32_out.len()
        );
    }
    let d3f = max_abs_diff(&seg1_out[..seg_row], &seg32_out[..seg_row]);
    if d3f > TOL {
        bail!(
            "STAGE 3f FAILED: segmentation-3.0.onnx and segmentation-3.0-b32.onnx disagree \
             on identical audio by {d3f:.3e} (bar {TOL:.0e}). One of the two segmentation \
             graphs was folded or exported wrongly; every file longer than a single window \
             is segmented by the b32 graph."
        );
    }
    details.push(format!("3f segmentation b1-vs-b32 {d3f:.2e}"));

    // --- 3g: the fixed-batch tails (b3, b32) against the batch-1 tail ---
    // b3 is what the 3-speaker boundary path uses and b32 is the batched embedding tail.
    // Cheap to fold in here: the b1 reference (`split_out` above) is already computed, so
    // each costs exactly one forward pass.
    for (file, batch) in [
        ("wespeaker-voxceleb-resnet34-tail-b3.onnx", 3usize),
        ("wespeaker-voxceleb-resnet34-tail-b32.onnx", 32usize),
    ] {
        let probe_row = PROBE_ROW.min(batch - 1);
        let mut sess = cpu_session(&dir.join(file))?;
        let mut fb = Vec::with_capacity(batch * row);
        let mut wts = Vec::with_capacity(batch * MASK_FRAMES);
        for i in 0..batch {
            fb.extend_from_slice(&fb1[..row]);
            if i == probe_row {
                wts.extend_from_slice(&mask);
            } else {
                wts.extend(seeded_mask(MASK_FRAMES, 0xC200 + i as u64));
            }
        }
        let (_, out) = run_f32(
            &mut sess,
            ort::inputs![
                "fbank" => Tensor::from_array(Array3::from_shape_vec((batch, FBANK_FRAMES, FBANK_FEATURES), fb)?)?,
                "weights" => Tensor::from_array(Array2::from_shape_vec((batch, MASK_FRAMES), wts)?)?
            ],
            "output",
        )?;
        let at = probe_row * EMBED_DIM;
        let d = max_abs_diff(&out[at..at + EMBED_DIM], &split_out);
        if d > TOL {
            bail!(
                "STAGE 3g FAILED: {file} row {probe_row} disagrees with \
                 wespeaker-voxceleb-resnet34-tail.onnx by {d:.3e} (bar {TOL:.0e}) on \
                 identical input. That graph was exported from a different checkpoint, is \
                 batch-coupled, or was corrupted."
            );
        }
        details.push(format!("3g {} b{batch}-vs-b1 {d:.2e}", short(file)));
    }

    // --- 3h: the fused batch-32 embedding graph against the fused batch-1 graph ---
    // `wespeaker-voxceleb-resnet34-b32.onnx` is its own export (fbank ∘ tail baked into one
    // graph), not a composition of graphs checked elsewhere, so nothing else covers it.
    let mut fused32 = cpu_session(&dir.join("wespeaker-voxceleb-resnet34-b32.onnx"))?;
    let mut wave_wide = Vec::with_capacity(32 * SAMPLES_10S);
    let mut fused_w = Vec::with_capacity(32 * MASK_FRAMES);
    for i in 0..32 {
        wave_wide.extend_from_slice(&clip);
        if i == PROBE_ROW {
            fused_w.extend_from_slice(&mask);
        } else {
            fused_w.extend(seeded_mask(MASK_FRAMES, 0xD300 + i as u64));
        }
    }
    let (_, fused32_out) = run_f32(
        &mut fused32,
        ort::inputs![
            "waveform" => Tensor::from_array(Array3::from_shape_vec((32, 1, SAMPLES_10S), wave_wide)?)?,
            "weights" => Tensor::from_array(Array2::from_shape_vec((32, MASK_FRAMES), fused_w)?)?
        ],
        "output",
    )?;
    let at = PROBE_ROW * EMBED_DIM;
    let d3h = max_abs_diff(&fused32_out[at..at + EMBED_DIM], &fused_out);
    if d3h > TOL {
        bail!(
            "STAGE 3h FAILED: wespeaker-voxceleb-resnet34-b32.onnx row {PROBE_ROW} \
             disagrees with wespeaker-voxceleb-resnet34.onnx by {d3h:.3e} (bar {TOL:.0e}) \
             on identical input."
        );
    }
    details.push(format!("3h fused b32-vs-b1 {d3h:.2e}"));

    // --- 3i: the batch-64 family, LAST and deliberately minimal ---
    //
    // Cost rebalance, on purpose. This is the most expensive check in the whole smoke test
    // (a 64x998x80 CPU forward pass) and it covers a graph that live compose does NOT load:
    // `SPEAKRS_LAZY_SESSIONS=1` gates `split_primary_tail_batched_session` on
    // `!lazy_sessions`. It still has to be CORRECT — someone turning lazy sessions off must
    // not get silently wrong embeddings — so it keeps a real numeric check (RESULTS §7.33
    // measured 7.8e-08). But it runs after every graph production actually executes has
    // already been checked, so a failure in the hot path is reported first and the operator
    // does not wait through this to hear about it.
    if opts.set == ModelSet::Fast {
        let mut tail64 = cpu_session(&dir.join("wespeaker-voxceleb-resnet34-tail-b64.onnx"))?;
        let mut fb64 = Vec::with_capacity(64 * row);
        let mut mk64 = Vec::with_capacity(64 * MASK_FRAMES);
        for i in 0..64 {
            fb64.extend_from_slice(&fb1[..row]);
            mk64.extend(seeded_mask(MASK_FRAMES, 0xA5A5 + i as u64));
        }
        let (_, out64) = run_f32(
            &mut tail64,
            ort::inputs![
                "fbank" => Tensor::from_array(Array3::from_shape_vec((64, FBANK_FRAMES, FBANK_FEATURES), fb64)?)?,
                "weights" => Tensor::from_array(Array2::from_shape_vec((64, MASK_FRAMES), mk64.clone())?)?
            ],
            "output",
        )?;
        let row7_mask: Vec<f32> = mk64[7 * MASK_FRAMES..8 * MASK_FRAMES].to_vec();
        let (_, out1) = run_f32(
            &mut tail,
            ort::inputs![
                "fbank" => Tensor::from_array(fb_arr.clone())?,
                "weights" => Tensor::from_array(Array2::from_shape_vec((1, MASK_FRAMES), row7_mask)?)?
            ],
            "output",
        )?;
        let d3i = max_abs_diff(&out64[7 * EMBED_DIM..8 * EMBED_DIM], &out1);
        if d3i > TOL {
            bail!(
                "STAGE 3i FAILED: wespeaker-voxceleb-resnet34-tail-b64.onnx row 7 disagrees \
                 with the batch-1 tail on the same row by {d3i:.3e} (bar {TOL:.0e}). The \
                 b64 graph is batch-coupled, which would make split-primary batching \
                 numerically wrong rather than merely absent."
            );
        }
        details.push(format!("3i tail-b64 batch-invariance {d3i:.2e}"));
    }

    Ok(StageResult {
        stage: "3-numeric".into(),
        detail: details.join("; "),
    })
}

/// Trim the shared `wespeaker-` prefix and the `.onnx` suffix for compact stage detail lines.
fn short(file: &str) -> &str {
    file.trim_start_matches("wespeaker-")
        .trim_end_matches(".onnx")
}

/// STAGE 5 — PLDA parameter files.
fn stage5_plda(opts: &SmokeOptions) -> Result<StageResult> {
    for (name, dtype, shape) in PLDA_SPECS {
        let p = opts.models_dir.join(name);
        let head = read_npy_header(&p)?;
        if head.descr != dtype.descr() {
            bail!(
                "STAGE 5 FAILED: {} has dtype {}, expected {}. Byte size alone cannot \
                 catch this — plda_tr.npy and plda_lda.npy are both 131200 bytes with \
                 different dtypes AND different shapes.",
                p.display(),
                head.descr,
                dtype.descr()
            );
        }
        if head.shape != *shape {
            bail!(
                "STAGE 5 FAILED: {} has shape {:?}, expected {:?}",
                p.display(),
                head.shape,
                shape
            );
        }
        let want = 128 + shape.iter().product::<usize>() * dtype.size();
        let got = std::fs::metadata(&p)?.len() as usize;
        if got != want {
            bail!(
                "STAGE 5 FAILED: {} is {got} bytes, expected {want} for a {:?} {} array",
                p.display(),
                shape,
                head.descr
            );
        }
    }

    let p = opts
        .models_dir
        .join("wespeaker-voxceleb-resnet34.min_num_samples.txt");
    let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    let n: u64 = raw
        .trim()
        .parse()
        .with_context(|| format!("STAGE 5 FAILED: {} is not an integer", p.display()))?;
    if n == 0 {
        bail!("STAGE 5 FAILED: {} is 0; it must be positive", p.display());
    }

    Ok(StageResult {
        stage: "5-plda".into(),
        detail: format!("{} PLDA arrays + min_num_samples={n}", PLDA_SPECS.len()),
    })
}

#[derive(Debug)]
struct NpyHeader {
    descr: String,
    shape: Vec<usize>,
}

/// Parse a `.npy` v1/v2 header. Only the two fields that matter, without a numpy dependency.
fn read_npy_header(path: &Path) -> Result<NpyHeader> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        bail!(
            "STAGE 5 FAILED: {} is not a .npy file (bad magic)",
            path.display()
        );
    }
    let major = bytes[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        )
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + hlen)
            .ok_or_else(|| anyhow!("STAGE 5 FAILED: {} header is truncated", path.display()))?,
    )?;

    let descr = extract_between(header, "'descr':", ",")
        .ok_or_else(|| anyhow!("STAGE 5 FAILED: no descr in {}", path.display()))?
        .trim()
        .trim_matches('\'')
        .to_string();
    let shape_raw = extract_between(header, "'shape':", ")")
        .ok_or_else(|| anyhow!("STAGE 5 FAILED: no shape in {}", path.display()))?;
    let shape = shape_raw
        .trim()
        .trim_start_matches('(')
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    Ok(NpyHeader { descr, shape })
}

fn extract_between<'a>(hay: &'a str, after: &str, until: &str) -> Option<&'a str> {
    let i = hay.find(after)? + after.len();
    let rest = &hay[i..];
    let j = rest.find(until)?;
    Some(&rest[..j])
}

/// STAGE 4 — the real thing: load the engine and diarize the clip.
fn stage4_end_to_end(opts: &SmokeOptions, audio: &[f32]) -> Result<(StageResult, usize, usize)> {
    let gender_present = opts.models_dir.join(GENDER_MODEL).exists();
    let mut engine = match DiarEngine::load(&EngineConfig::new(&opts.models_dir, opts.mode)) {
        Ok(e) => e,
        Err(e) if opts.mode != Mode::Cpu => {
            // Discriminate DEVICE from MODELS by experiment rather than by string-matching an
            // ORT error. The CPU execution provider is statically linked into every build, so
            // if the SAME directory loads on CPU, the files are fine and the accelerator is
            // not. Only the second case may be recorded as a smoke failure — see
            // `DeviceUnavailable` for why conflating them takes good models out of service.
            match DiarEngine::load(&EngineConfig::new(&opts.models_dir, Mode::Cpu)) {
                Ok(_) => {
                    return Err(anyhow::Error::new(DeviceUnavailable {
                        mode: mode_name(opts.mode).to_string(),
                        detail: format!("{e:#}"),
                    }))
                }
                Err(_) => {
                    return Err(e).context("STAGE 4 FAILED: the engine could not load these models")
                }
            }
        }
        Err(e) => return Err(e).context("STAGE 4 FAILED: the engine could not load these models"),
    };
    let out = engine
        .diarize_with(audio, "smoke", gender_present)
        .context("STAGE 4 FAILED: diarization errored on the smoke clip")?;

    let duration_s = audio.len() as f64 / 16_000.0;

    if out.num_speakers == 0 || out.num_speakers > 10 {
        bail!(
            "STAGE 4 FAILED: found {} speakers in a {duration_s:.1} s clip; expected 1-10. \
             Clustering is not producing sane output.",
            out.num_speakers
        );
    }
    if out.segments.is_empty() {
        bail!("STAGE 4 FAILED: no speech segments found in the smoke clip");
    }
    if out.exclusive_segments.is_empty() {
        bail!("STAGE 4 FAILED: no exclusive segments produced");
    }
    if out.centroids.len() != out.num_speakers {
        bail!(
            "STAGE 4 FAILED: {} centroids for {} speakers",
            out.centroids.len(),
            out.num_speakers
        );
    }
    for (i, c) in out.centroids.iter().enumerate() {
        if c.len() != EMBED_DIM {
            bail!(
                "STAGE 4 FAILED: centroid {i} has {} dimensions, expected {EMBED_DIM}",
                c.len()
            );
        }
        if !c.iter().all(|v| v.is_finite()) {
            bail!("STAGE 4 FAILED: centroid {i} contains NaN or infinity");
        }
        let l2 = c.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Written as an explicit finite-and-positive test rather than `!(l2 > 0.0)`: the
        // negated form reads as a NaN guard, but line 854 already proved every component
        // finite, so the only reachable bad values are 0.0 and (on overflow) infinity.
        if !l2.is_finite() || l2 <= 0.0 {
            bail!("STAGE 4 FAILED: centroid {i} has zero or non-finite magnitude ({l2})");
        }
    }

    // Exclusive segments are one-speaker-per-frame by construction; overlap here means the
    // resolver is broken in a way the full segment list would hide.
    let mut sorted = out.exclusive_segments.clone();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start));
    for w in sorted.windows(2) {
        if w[1].start < w[0].end - 1e-6 {
            bail!(
                "STAGE 4 FAILED: exclusive segments overlap ({:.3}-{:.3} and {:.3}-{:.3})",
                w[0].start,
                w[0].end,
                w[1].start,
                w[1].end
            );
        }
    }
    let speech: f64 = sorted.iter().map(|s| s.end - s.start).sum();
    if speech > duration_s + 1e-3 {
        bail!("STAGE 4 FAILED: {speech:.2} s of exclusive speech in a {duration_s:.2} s clip");
    }

    if let Some(genders) = &out.speaker_gender {
        for (spk, v) in genders {
            if !crate::gender::ID2LABEL.contains(&v.label.as_str()) {
                bail!(
                    "STAGE 4 FAILED: gender label `{}` for {spk} is not one of {:?}",
                    v.label,
                    crate::gender::ID2LABEL
                );
            }
            if !(v.confidence > 0.0 && v.confidence <= 1.0) {
                bail!(
                    "STAGE 4 FAILED: gender confidence {} for {spk} is outside (0, 1]",
                    v.confidence
                );
            }
        }
    }

    // embed_window is a separate public entry point (the boundary-resolver recheck path);
    // it has its own padding logic and is not exercised by diarize at all.
    let two_s = &audio[..(32_000).min(audio.len())];
    let emb = engine
        .embed_window(two_s)
        .context("STAGE 4 FAILED: embed_window errored")?;
    if emb.len() != EMBED_DIM || !emb.iter().all(|v| v.is_finite()) {
        bail!(
            "STAGE 4 FAILED: embed_window returned {} values (finite: {})",
            emb.len(),
            emb.iter().all(|v| v.is_finite())
        );
    }

    let detail = format!(
        "{} speakers, {} segments, {} exclusive, gender={}",
        out.num_speakers,
        out.segments.len(),
        out.exclusive_segments.len(),
        out.speaker_gender.as_ref().map_or(0, |g| g.len())
    );
    Ok((
        StageResult {
            stage: "4-end-to-end".into(),
            detail,
        },
        out.num_speakers,
        out.segments.len(),
    ))
}

/// Run all five stages. Returns the report the marker records.
pub fn run(opts: &SmokeOptions) -> Result<SmokeReport> {
    let started = std::time::Instant::now();

    // Fail early and specifically on absence, rather than letting ORT say "No such file".
    let missing = super::marker::missing_required(&opts.models_dir, opts.set, opts.with_gender);
    if !missing.is_empty() {
        bail!(
            "models directory {} is missing {} required file(s): {}",
            opts.models_dir.display(),
            missing.len(),
            missing.join(", ")
        );
    }

    let audio = read_wav_16k_mono(&opts.clip)?;
    let clip_sha256 = super::sha256_file(&opts.clip).map_err(|e| anyhow!("{e}"))?;

    // Execution order is by COST AND PRECISION, not by stage number: every structural
    // check runs before the end-to-end run, so the most specific message wins.
    //
    // Measured, not assumed: with stage 5 last, truncating `plda_tr.npy` by 64 bytes was
    // reported as "STAGE 4 FAILED: pipeline init: reached EOF before reading all data" —
    // no filename, no hint that PLDA was involved. Running the header check first turns
    // that into "plda_tr.npy is 131136 bytes, expected 131200". Producing unactionable
    // errors is the exact failure mode this command exists to eliminate, so it would be
    // perverse for the verifier itself to emit one.
    let mut stages = vec![
        stage1_parse_all(opts)?,
        stage2_io_contract(opts)?,
        stage5_plda(opts)?,
        stage3_numeric(opts, &audio)?,
    ];
    let (s4, num_speakers, segments) = stage4_end_to_end(opts, &audio)?;
    stages.push(s4);

    Ok(SmokeReport {
        stages,
        num_speakers,
        segments,
        duration_ms: started.elapsed().as_millis() as u64,
        clip_sha256,
        mode: mode_name(opts.mode).to_string(),
    })
}

pub fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Cpu => "cpu",
        Mode::Cuda => "cuda",
        Mode::CoreMl => "coreml",
        Mode::CoreMlFast => "coreml_fast",
    }
}

/// What a deep verification found.
///
/// `hashed` is reported separately from `drift` because "zero files differ" and "zero files
/// were compared" are the same value of `drift.len()` and radically different facts.
#[derive(Debug, Clone)]
pub struct DeepReport {
    pub smoke: SmokeReport,
    /// Files whose sha256 no longer matches the marker.
    pub drift: Vec<String>,
    /// How many files were actually compared against a recorded hash. `0` with no marker.
    pub hashed: usize,
    /// Was there a marker to compare against at all?
    pub marker_present: bool,
    /// Marker recorded a FAILED smoke test before this run.
    pub marker_recorded_failure: bool,
}

impl DeepReport {
    /// Did every tier pass — including the tier that needs a marker to exist?
    pub fn fully_verified(&self) -> bool {
        self.marker_present && self.drift.is_empty()
    }
}

/// Deep verification: everything the smoke test does, plus full sha256 of every file
/// against the marker. This is the tier that actually detects a silent rewrite.
///
/// A MISSING MARKER IS NOT SUCCESS. `Marker::read` returns `Ok(None)` for an absent file, and
/// the obvious `if let Some(marker)` here used to skip the entire drift loop, leaving `drift`
/// empty — which the CLI printed as "OK: … verified." after comparing exactly zero bytes
/// against exactly zero recorded hashes. An operator who copied a `/models` directory from an
/// unknown source got a clean bill of health from the one command whose stated job is
/// detecting a silent rewrite. The marker's presence is now part of the result.
pub fn verify_deep(opts: &SmokeOptions) -> Result<DeepReport> {
    let mut drift = Vec::new();
    let mut hashed = 0usize;
    let marker = super::marker::Marker::read(&opts.models_dir).map_err(|e| anyhow!("{e}"))?;
    let marker_recorded_failure = marker.as_ref().is_some_and(|m| !m.smoke.passed());
    if let Some(marker) = &marker {
        for rec in &marker.files {
            let p = opts.models_dir.join(&rec.name);
            hashed += 1;
            match super::sha256_file(&p) {
                Ok(h) if h == rec.sha256 => {}
                Ok(h) => drift.push(format!(
                    "{}: sha256 {} != recorded {}",
                    rec.name,
                    &h[..16],
                    &rec.sha256[..16.min(rec.sha256.len())]
                )),
                Err(e) => drift.push(format!("{}: {e}", rec.name)),
            }
        }
        // A marker that vouches for nothing is the same hole wearing a hat.
        if marker.files.is_empty() {
            drift.push(format!(
                "{}: the marker records NO file hashes, so nothing could be compared. It \
                 was written by a run that failed before hashing. Re-run \
                 `diar-server provision-models --models-dir {} --force`.",
                super::MARKER_FILE,
                opts.models_dir.display()
            ));
        }
    }
    let smoke = run(opts)?;
    Ok(DeepReport {
        smoke,
        drift,
        hashed,
        marker_present: marker.is_some(),
        marker_recorded_failure,
    })
}

/// The files a caller should hash when writing a marker.
pub fn marker_file_list(set: ModelSet, with_gender: bool) -> Vec<&'static str> {
    required_files(set, with_gender)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_parsing_matches_the_real_shapes() {
        // Uses the shipped models dir when present; skipped in a clean checkout since
        // model artifacts are gated and never committed.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models_folded")
            .canonicalize();
        let Ok(dir) = dir else { return };
        for (name, dtype, shape) in PLDA_SPECS {
            let p = dir.join(name);
            if !p.exists() {
                continue;
            }
            let h = read_npy_header(&p).unwrap();
            assert_eq!(h.descr, dtype.descr(), "{name} dtype");
            assert_eq!(h.shape, *shape, "{name} shape");
        }
    }

    #[test]
    fn seeded_mask_is_deterministic_and_in_range() {
        let a = seeded_mask(589, 42);
        let b = seeded_mask(589, 42);
        assert_eq!(a, b);
        // Adjacent seeds must differ. The stage-3e probe walks 0xA5A5..0xA5A5+64, so a
        // seeding scheme that collapsed neighbours would duplicate half those rows.
        assert_ne!(a, seeded_mask(589, 43));
        for i in 0..8u64 {
            for j in (i + 1)..8 {
                assert_ne!(
                    seeded_mask(16, 0xA5A5 + i),
                    seeded_mask(16, 0xA5A5 + j),
                    "seeds {i} and {j} collide"
                );
            }
        }
        assert!(a.iter().all(|v| *v > 0.0 && *v <= 1.0), "mask out of range");
        // A constant mask would make several pooling bugs cancel; ensure real variation.
        let min = a.iter().copied().fold(f32::INFINITY, f32::min);
        let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 0.5, "mask is too flat to be a useful probe");
    }

    #[test]
    fn max_abs_diff_is_actually_max_abs() {
        assert_eq!(max_abs_diff(&[1.0, -2.0], &[1.0, 3.0]), 5.0);
        assert_eq!(max_abs_diff(&[], &[]), 0.0);
    }

    #[test]
    fn npy_header_rejects_a_non_npy_file() {
        let p = std::env::temp_dir().join("diar-not-a-npy.bin");
        std::fs::write(&p, b"definitely not numpy").unwrap();
        let err = read_npy_header(&p).unwrap_err().to_string();
        assert!(err.contains("bad magic"), "{err}");
        let _ = std::fs::remove_file(&p);
    }
}
