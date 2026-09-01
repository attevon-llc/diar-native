//! Per-speaker gender classification (wav2vec2), run against audio the engine already holds.
//!
//! The app used to do this in a separate Celery task that re-fetched clips over presigned URLs
//! and ran wav2vec2 on one CPU core — 87-90 s per file, the second-largest task in the whole
//! pipeline (RESULTS §7.15). Here the PCM is already decoded and resident for diarization, so
//! classification is windows of an existing buffer: no refetch, no re-decode, no second copy.
//!
//! Optional by construction: with no `gender-wav2vec2.onnx` in the models dir the engine loads
//! exactly as before and the app keeps its own path.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Model filename in the models dir; absence disables the feature.
///
/// THE CANONICAL DEFINITION. Every other site names this constant rather than repeating the
/// literal — [`crate::provision::files::GENDER_MODEL`] re-exports it, and
/// [`crate::ort_compat`] imports it. There used to be three independent copies of the string,
/// which was a live hazard rather than a tidiness complaint: **two behaviours are scoped by
/// this exact filename** — the aarch64 fp16 optimization cap
/// ([`crate::ort_compat::apply_workarounds`]) and the fp16 load gate
/// (`provision::verify` stage 1). Renaming the export while updating only one copy would
/// silently stop both from applying, on arm64 only, where the symptom is a server that starts
/// fine and returns no genders. `crates/diar-core/tests/model_filenames.rs` pins the value
/// against the Python exporter that writes it.
pub const GENDER_MODEL_FILE: &str = "gender-wav2vec2.onnx";
/// `config.id2label` of prithivMLmods/Common-Voice-Gender-Detection, index-ordered.
///
/// Public so provisioning can cross-check it against the `gender-wav2vec2.meta.json` written
/// beside the model. Nothing reads that file at runtime, so without that check an upstream
/// relabelling would invert every verdict silently.
pub const ID2LABEL: [&str; 2] = ["female", "male"];
/// Windows per speaker to vote over — mirrors the app (speaker_attribute_task.py:317-320).
const MAX_WINDOWS_PER_SPEAKER: usize = 5;
/// Clips shorter than this are too unreliable to vote with (the app uses the same floor).
const MIN_SAMPLES: usize = 16_000;
/// Longest clip fed to the model, taken from the middle of the window.
///
/// Gender is decided from a few seconds of voice, but speaker turns run to a minute or more —
/// and wav2vec2 activations scale with input length, so passing a whole turn cost **6.3 GB of
/// VRAM** for no accuracy. The app never hit this because it runs on CPU, where an oversized
/// clip is merely slow.
/// Override with `DIAR_GENDER_MAX_SECONDS` — laptop tiers will want this lower.
const DEFAULT_MAX_SECONDS: usize = 5;

fn max_samples(sample_rate: usize) -> usize {
    std::env::var("DIAR_GENDER_MAX_SECONDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MAX_SECONDS)
        * sample_rate
}

pub struct GenderModel {
    /// Shared across engine handles (T9a): weights + arena load once, `run` locks per window.
    session: Arc<Mutex<Session>>,
    /// Reused across windows so classification allocates once per engine, not once per clip.
    scratch: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenderVerdict {
    pub label: String,
    pub confidence: f32,
    pub windows: usize,
}

impl GenderModel {
    /// Load from `models_dir`, or `Ok(None)` when the model is not deployed.
    pub fn load_optional(models_dir: &Path, cuda: bool) -> Result<Option<Self>> {
        let path = models_dir.join(GENDER_MODEL_FILE);
        if !path.exists() {
            return Ok(None);
        }
        // ort's builder errors are not Send+Sync, so they are stringified rather than
        // propagated with `?` straight into anyhow.
        let builder =
            Session::builder().map_err(|e| anyhow::anyhow!("gender session builder: {e}"))?;
        let builder = if cuda {
            // Same provider the rest of the engine uses; CPU stays the fallback so a box
            // without a working CUDA EP still classifies rather than failing the request.
            builder
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .map_err(|e| anyhow::anyhow!("gender CUDA provider: {e}"))?
        } else {
            builder
        };
        // Without this the fp16 gender model does not load AT ALL on aarch64 — see
        // `ort_compat` for the mechanism (ORT's GeluFusion synthesizes a contrib op the
        // aarch64 build has no fp16 kernel for) and issue #14.
        let mut builder = crate::ort_compat::apply_workarounds(builder, &path)?;
        let session = builder
            .commit_from_file(&path)
            .map_err(|e| anyhow::anyhow!("gender session commit: {e}"))
            .with_context(|| format!("loading gender model from {}", path.display()))?;
        Ok(Some(Self {
            session: Arc::new(Mutex::new(session)),
            scratch: Vec::new(),
        }))
    }

    /// Cheap handle over the same session with a fresh scratch buffer (T9a).
    pub fn clone_shared(&self) -> Self {
        Self {
            session: Arc::clone(&self.session),
            scratch: Vec::new(),
        }
    }

    /// Classify one clip, borrowed from the caller's decoded audio.
    ///
    /// Preprocessing is wav2vec2's `do_normalize`: zero mean, unit variance over the clip — no
    /// fbank, which is why this needs no feature-extractor port. The normalized copy reuses a
    /// per-engine scratch buffer, so a file's worth of windows allocates once rather than once
    /// each. The remaining host→device transfer is ORT's; sharing a resident GPU buffer between
    /// this model and diarization would need IoBinding plus a shared CUDA allocator, which is
    /// tracked separately and is not what this buys.
    pub fn classify(&mut self, clip: &[f32]) -> Result<(usize, f32)> {
        let n = clip.len();
        let mean = clip.iter().sum::<f32>() / n as f32;
        let var = clip.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n as f32;
        let denom = (var + 1e-7).sqrt();

        self.scratch.clear();
        self.scratch.reserve(n);
        self.scratch.extend(clip.iter().map(|s| (s - mean) / denom));

        let input = Array2::from_shape_vec((1, n), self.scratch.clone())?;
        let tensor =
            Tensor::from_array(input).map_err(|e| anyhow::anyhow!("gender input tensor: {e}"))?;
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = session
            .run(ort::inputs!["input_values" => tensor])
            .map_err(|e| anyhow::anyhow!("gender inference: {e}"))?;
        let (_, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("gender logits: {e}"))?;

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let (idx, p) = exp
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, p)| (i, *p / sum))
            .unwrap_or((0, 0.0));
        Ok((idx, p))
    }

    /// Majority vote per speaker over their longest windows, weighted by confidence — the same
    /// shape as the app's aggregation, so decisions are comparable.
    pub fn classify_speakers(
        &mut self,
        audio: &[f32],
        segments: &[crate::SegmentOut],
        sample_rate: usize,
    ) -> HashMap<String, GenderVerdict> {
        let mut by_speaker: HashMap<&str, Vec<&crate::SegmentOut>> = HashMap::new();
        for seg in segments {
            by_speaker
                .entry(seg.speaker.as_str())
                .or_default()
                .push(seg);
        }

        let cap = max_samples(sample_rate);
        let mut out = HashMap::new();
        for (speaker, mut segs) in by_speaker {
            // Longest windows first: short turns classify poorly, and the app picks the same way.
            segs.sort_by(|a, b| (b.end - b.start).total_cmp(&(a.end - a.start)));

            let mut scores = [0.0f32; ID2LABEL.len()];
            let mut used = 0usize;
            for seg in segs.into_iter().take(MAX_WINDOWS_PER_SPEAKER) {
                let start = (seg.start * sample_rate as f64).max(0.0) as usize;
                let end = ((seg.end * sample_rate as f64) as usize).min(audio.len());
                if end <= start || end - start < MIN_SAMPLES {
                    continue;
                }
                // Centre-crop over-long turns: the middle of a turn is the cleanest voice,
                // away from the boundaries where the previous speaker may still be trailing.
                let (start, end) = if end - start > cap {
                    let mid = start + (end - start) / 2;
                    (mid - cap / 2, mid + cap / 2)
                } else {
                    (start, end)
                };
                match self.classify(&audio[start..end]) {
                    Ok((idx, conf)) => {
                        scores[idx] += conf;
                        used += 1;
                    }
                    // One bad window must not cost the speaker a verdict.
                    Err(e) => tracing::warn!("gender window failed for {speaker}: {e}"),
                }
            }
            if used == 0 {
                continue;
            }
            let (idx, score) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, s)| (i, *s))
                .unwrap();
            out.insert(
                speaker.to_string(),
                GenderVerdict {
                    label: ID2LABEL[idx].to_string(),
                    confidence: score / used as f32,
                    windows: used,
                },
            );
        }
        out
    }
}
