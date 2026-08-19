//! diar-server — T1 sidecar (PLAN.md deployment tier 1).
//!
//! Stateless executor: Celery (or any client) POSTs a job; orchestration/queueing stays
//! upstream (PLAN decision #3). One engine instance (shared weights); an admission
//! semaphore bounds in-flight jobs; compute runs on a blocking thread.
//! Audio arrives BY PATH on a shared volume (matches the compose topology).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use diar_core::{DiarEngine, DiarizeOutput, EngineConfig, Mode};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

struct AppState {
    engine: Mutex<DiarEngine>,
    gate: Semaphore,
}

#[derive(Deserialize)]
struct DiarizeRequest {
    /// Path to a 16 kHz mono WAV on a shared volume.
    wav_path: PathBuf,
    #[serde(default)]
    file_id: Option<String>,
    /// Also classify each speaker's gender from this same decoded audio, before it is
    /// dropped — saves the caller a second fetch, decode and model host.
    #[serde(default)]
    gender: bool,
}

#[derive(Deserialize)]
struct EmbedRequest {
    /// Path to a 16 kHz mono WAV on a shared volume…
    #[serde(default)]
    wav_path: Option<PathBuf>,
    /// …or raw 16 kHz mono f32 little-endian samples, base64 (small clips).
    #[serde(default)]
    samples_b64: Option<String>,
    /// Optional window within the file, in seconds (wav_path input only).
    #[serde(default)]
    start_s: Option<f64>,
    #[serde(default)]
    end_s: Option<f64>,
}

fn decode_samples_b64(b64: &str) -> anyhow::Result<Vec<f32>> {
    // minimal base64 decode without an extra dependency
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rev = [255u8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        rev[c as usize] = i as u8;
    }
    let clean: Vec<u8> = b64.bytes().filter(|b| !b" \n\r\t=".contains(b)).collect();
    let mut bytes = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let vals: Vec<u8> = chunk
            .iter()
            .map(|&c| rev[c as usize])
            .collect();
        anyhow::ensure!(vals.iter().all(|&v| v != 255), "invalid base64");
        let mut acc: u32 = 0;
        for &v in &vals {
            acc = (acc << 6) | v as u32;
        }
        acc <<= 6 * (4 - vals.len());
        let out = [(acc >> 16) as u8, (acc >> 8) as u8, acc as u8];
        bytes.extend_from_slice(&out[..vals.len().saturating_sub(1)]);
    }
    anyhow::ensure!(bytes.len() % 4 == 0, "sample byte length not a multiple of 4");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

fn load_wav(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "expected 16 kHz mono, got {} Hz {} ch",
        spec.sample_rate,
        spec.channels
    );
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    })
}

async fn diarize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DiarizeRequest>,
) -> Result<Json<DiarizeOutput>, (StatusCode, String)> {
    let _permit = state
        .gate
        .acquire()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
    let state2 = Arc::clone(&state);
    let out = tokio::task::spawn_blocking(move || -> anyhow::Result<DiarizeOutput> {
        let audio = load_wav(&req.wav_path)?;
        let file_id = req
            .file_id
            .clone()
            .or_else(|| {
                req.wav_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "file".into());
        let mut engine = state2.engine.lock().expect("engine mutex poisoned");
        engine.diarize_with(&audio, &file_id, req.gender)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;
    Ok(Json(out))
}

async fn embed_window(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    let _permit = state
        .gate
        .acquire()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
    let state2 = Arc::clone(&state);
    let embedding = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
        let clip: Vec<f32> = if let Some(b64) = &req.samples_b64 {
            decode_samples_b64(b64)?
        } else if let Some(path) = &req.wav_path {
            let audio = load_wav(path)?;
            let sr = 16_000.0;
            match (req.start_s, req.end_s) {
                (Some(s), Some(e)) if e > s => {
                    let a = ((s * sr) as usize).min(audio.len());
                    let b = ((e * sr) as usize).min(audio.len());
                    audio[a..b].to_vec()
                }
                _ => audio,
            }
        } else {
            anyhow::bail!("embed_window requires wav_path or samples_b64");
        };
        let mut engine = state2.engine.lock().expect("engine mutex poisoned");
        engine.embed_window(&clip)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;
    Ok(Json(EmbedResponse { embedding }))
}

async fn healthz() -> &'static str {
    "ok"
}

fn main() -> anyhow::Result<()> {
    // speakrs pipeline threads + ORT need more than the 2 MiB default thread stack
    // (measured: stack overflow in worker threads; same finding as the test suite).
    if std::env::var("RUST_MIN_STACK").is_err() {
        std::env::set_var("RUST_MIN_STACK", "16777216");
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024)
        .build()?
        .block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let models_dir =
        std::env::var("DIAR_MODELS_DIR").unwrap_or_else(|_| "/models".to_string());
    let mode = match std::env::var("DIAR_MODE").as_deref() {
        Ok("cpu") => Mode::Cpu,
        _ => Mode::Cuda,
    };
    let max_inflight: usize = std::env::var("DIAR_MAX_INFLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let bind = std::env::var("DIAR_BIND").unwrap_or_else(|_| "0.0.0.0:8701".to_string());

    let engine = DiarEngine::load(&EngineConfig::new(models_dir, mode))?;
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        gate: Semaphore::new(max_inflight),
    });

    let app = Router::new()
        .route("/diarize", post(diarize))
        .route("/embed_window", post(embed_window))
        .route("/healthz", get(healthz))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("diar-server listening on {bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
