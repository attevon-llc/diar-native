//! diar-core — OpenTranscribe's diarization engine, wrapping speakrs.
//!
//! Adds the integration surface the app contract requires (see PLAN.md decision #4 and
//! `validation/TESTPLAN.md`): per-speaker centroids, `embed_window`, dual full/exclusive
//! segment outputs, and deployment-tier config (fbank pool defaults off in CPU mode —
//! measured core-contention regression, RESULTS §4.12).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use speakrs::inference::{EmbeddingModel, MaskedEmbeddingInput, SegmentationModel};
use speakrs::pipeline::{DiarizationPipeline, RuntimeConfig, segmentation_step_seconds};
use speakrs::ExecutionMode;

pub mod audio;
pub mod gender;
use gender::{GenderModel, GenderVerdict};

/// Frame count of the segmentation mask grid (10 s window @ SincNet stride).
const MASK_FRAMES: usize = 589;
/// Embedding dimension of the WeSpeaker ResNet34 head.
pub const EMBEDDING_DIM: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Cpu,
    Cuda,
}

impl Mode {
    fn execution_mode(self) -> ExecutionMode {
        match self {
            Mode::Cpu => ExecutionMode::Cpu,
            Mode::Cuda => ExecutionMode::Cuda,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Directory holding the self-exported community-1 ONNX models + PLDA params.
    pub models_dir: PathBuf,
    pub mode: Mode,
    /// fbank session-pool size. `None` = auto: cores/4 (clamped 1-8) on CUDA, 1 on CPU
    /// (pool contends with inference for cores in CPU mode — RESULTS §4.12).
    pub fbank_pool: Option<usize>,
}

impl EngineConfig {
    pub fn new(models_dir: impl Into<PathBuf>, mode: Mode) -> Self {
        Self {
            models_dir: models_dir.into(),
            mode,
            fbank_pool: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentOut {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizeOutput {
    /// Overlap-aware speaker segments (pyannote `speaker_diarization` equivalent).
    pub segments: Vec<SegmentOut>,
    /// One-speaker-per-frame segments (pyannote `exclusive_speaker_diarization` equivalent).
    pub exclusive_segments: Vec<SegmentOut>,
    /// Gamma-weighted, UN-normalized per-speaker centroids; row i == SPEAKER_{i:02}.
    /// Consumers (OpenSearch kNN path) apply their own L2 normalization.
    pub centroids: Vec<Vec<f32>>,
    pub num_speakers: usize,
    /// Full-diarization RTTM (harness-compatible).
    pub rttm: String,
    /// Per-speaker gender, when the gender model is deployed and the caller asked for it.
    /// Classified from the audio already decoded for diarization — no refetch, no re-decode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_gender: Option<std::collections::HashMap<String, GenderVerdict>>,
}

pub struct DiarEngine {
    seg: SegmentationModel,
    emb: EmbeddingModel,
    gender: Option<GenderModel>,
    models_dir: PathBuf,
}

impl DiarEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let mode = config.mode.execution_mode();
        // Pool sizing is read from env by the (patched) speakrs loader; pin it before load.
        let pool = config.fbank_pool.unwrap_or(match config.mode {
            Mode::Cpu => 1,
            Mode::Cuda => std::thread::available_parallelism()
                .map(|c| (c.get() / 4).clamp(1, 8))
                .unwrap_or(1),
        });
        // SAFETY: called before any speakrs session exists; single-threaded load path.
        std::env::set_var("SPEAKRS_FBANK_POOL", pool.to_string());

        let step = segmentation_step_seconds(mode) as f32;
        let seg = SegmentationModel::with_mode(
            config.models_dir.join("segmentation-3.0.onnx"),
            step,
            mode,
        )
        .context("loading segmentation model")?;
        let emb = EmbeddingModel::with_mode_and_config(
            config.models_dir.join("wespeaker-voxceleb-resnet34.onnx"),
            mode,
            &RuntimeConfig::default(),
        )
        .context("loading embedding model")?;
        // Optional: absent model means the app keeps its own gender path untouched.
        let gender = GenderModel::load_optional(&config.models_dir, config.mode == Mode::Cuda)
            .context("loading gender model")?;
        Ok(Self {
            seg,
            emb,
            gender,
            models_dir: config.models_dir.clone(),
        })
    }

    /// Cheap handle over the same ORT sessions with fresh per-request scratch (T9a,
    /// PLAN decision #4). Weights + arenas (the VRAM) load once; each handle carries
    /// only staging buffers (~130 MB host RAM), so N handles run N jobs concurrently
    /// without N × engine. Every inference call locks its session for exactly one run.
    pub fn clone_shared(&self) -> Result<Self> {
        Ok(Self {
            seg: self.seg.clone_shared(),
            emb: self
                .emb
                .clone_shared()
                .map_err(|e| anyhow::anyhow!("cloning embedding handle: {e}"))?,
            gender: self.gender.as_ref().map(GenderModel::clone_shared),
            models_dir: self.models_dir.clone(),
        })
    }

    /// Diarize 16 kHz mono f32 samples. `&mut` because the handle carries per-request
    /// scratch buffers; for concurrency give each job its own [`Self::clone_shared`] handle.
    pub fn diarize(&mut self, audio: &[f32], file_id: &str) -> Result<DiarizeOutput> {
        self.diarize_with(audio, file_id, false)
    }

    /// `with_gender` classifies each speaker from this same buffer before it is dropped.
    pub fn diarize_with(
        &mut self,
        audio: &[f32],
        file_id: &str,
        with_gender: bool,
    ) -> Result<DiarizeOutput> {
        let mut pipeline = DiarizationPipeline::new(&mut self.seg, &mut self.emb, &self.models_dir)
            .map_err(|e| anyhow::anyhow!("pipeline init: {e}"))?;
        let result = pipeline
            .run_with_file_id(audio, file_id)
            .map_err(|e| anyhow::anyhow!("diarization: {e}"))?;

        let segments = result.segments.iter().map(to_segment_out).collect();
        let rttm = result.rttm(file_id);
        let centroids: Vec<Vec<f32>> = result
            .centroids
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let num_speakers = centroids.len();

        // Exclusive variant: the engine resolves overlaps by activation score and applies the
        // same duration filter and gap merging the full segments get.
        let exclusive_segments: Vec<SegmentOut> =
            result.exclusive_segments.iter().map(to_segment_out).collect();

        // Classify while the decoded audio is still here: the caller's buffer is the same one
        // diarization just used, so this costs windows of an existing allocation rather than a
        // second fetch, decode and copy.
        let speaker_gender = match (with_gender, self.gender.as_mut()) {
            (true, Some(model)) => Some(model.classify_speakers(audio, &exclusive_segments, 16_000)),
            (true, None) => {
                tracing::warn!("gender requested but {} is not deployed", gender::GENDER_MODEL_FILE);
                None
            }
            _ => None,
        };

        Ok(DiarizeOutput {
            segments,
            exclusive_segments,
            centroids,
            num_speakers,
            rttm,
            speaker_gender,
        })
    }

    /// Embed an arbitrary audio window (16 kHz mono), unmasked — the `boundary_resolver`
    /// acoustic-recheck contract. Center-pads short clips to the model minimum.
    /// Returns the RAW (un-normalized) 256-d embedding; callers normalize as needed.
    pub fn embed_window(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let min = self.emb.min_num_samples().max(1);
        let padded: Vec<f32>;
        let clip: &[f32] = if audio.len() >= min {
            audio
        } else {
            let pad = min - audio.len();
            let left = pad / 2;
            let mut v = vec![0.0f32; min];
            v[left..left + audio.len()].copy_from_slice(audio);
            padded = v;
            &padded
        };
        let mask = vec![1.0f32; MASK_FRAMES];
        let input = MaskedEmbeddingInput {
            audio: clip,
            mask: &mask,
            clean_mask: None,
        };
        let out = self
            .emb
            .embed_batch(&[input])
            .map_err(|e| anyhow::anyhow!("embed_window: {e}"))?;
        Ok(out.row(0).to_vec())
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }
}

fn to_segment_out(segment: &speakrs::Segment) -> SegmentOut {
    SegmentOut {
        start: segment.start,
        end: segment.end,
        speaker: segment.speaker.clone(),
    }
}
