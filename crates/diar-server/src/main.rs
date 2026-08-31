//! diar-server — T1 sidecar (PLAN.md deployment tier 1).
//!
//! Stateless executor: Celery (or any client) POSTs a job; orchestration/queueing stays
//! upstream (PLAN decision #3). One set of shared ORT sessions per device (weights + arenas
//! load once — PLAN decision #4 / T9a); each request runs on its own cheap engine handle, so
//! jobs execute concurrently up to the admission semaphore instead of serializing on an
//! engine mutex. Compute runs on blocking threads.
//! Audio arrives BY PATH on a shared volume (matches the compose topology).
//!
//! One process can serve several execution devices at once (see [`engines`]): the CUDA build
//! is a superset of the CPU build, so `{"device": "cpu"}` on a GPU deployment runs the same
//! code over the same weights the CPU-only image runs. `DIAR_DEVICES` picks which engines get
//! loaded; with it unset the server loads exactly one engine from `DIAR_MODE`, which is what
//! it has always done.

mod cli;
mod engines;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use diar_core::provision::marker::ModelsStatus;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use diar_core::{DiarEngine, DiarizeOutput};
use engines::{Device, EngineRegistry};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

/// Response header naming the device a request actually ran on. A header, not a body field:
/// `DiarizeOutput` is the consumer's parsed schema and adding to it is not free, whereas this
/// costs nothing and is readable by anything that can see response headers.
const DEVICE_HEADER: &str = "x-diar-device";

/// `Ok` side carries the device header alongside the JSON body.
type ApiResult<T> = Result<([(&'static str, &'static str); 1], Json<T>), (StatusCode, String)>;

struct AppState {
    /// Every loaded engine, keyed by device. Each prototype holds that device's shared
    /// sessions and never runs jobs itself.
    engines: EngineRegistry,
    /// Global admission gate (`DIAR_MAX_INFLIGHT`). Unchanged semantics and default: it bounds
    /// TOTAL inflight work across all devices, so enabling a second engine cannot silently
    /// double concurrency and oversubscribe the box.
    gate: Semaphore,
    /// Optional inner sub-gate for CPU work (`DIAR_MAX_INFLIGHT_CPU`). `None` — the default —
    /// means no inner gate at all and therefore no behaviour change. When set, CPU requests
    /// take the global permit FIRST and this one SECOND, always in that order, so there is no
    /// lock-ordering hazard against requests that only take the global one.
    cpu_gate: Option<Semaphore>,
    /// Provisioning state of the models directory, decided ONCE at startup by a `stat`-only
    /// pass. Not re-read per request: `/healthz` is polled by a compose healthcheck on a
    /// short interval, and re-hashing (or even re-stat-ing) 470 MB on every poll would be a
    /// self-inflicted load source. It answers "is this the directory that passed", which is
    /// a startup-time fact.
    models: ModelsStatus,
}

impl AppState {
    /// Run `f` against `device`'s engine. Off coreml, this clones a cheap per-request handle and
    /// releases the mutex immediately, so jobs run concurrently up to the admission
    /// semaphore (T9a). speakrs cfg's `clone_shared` out under `coreml` — CoreML models
    /// aren't wrapped in diar-native's `Arc<Mutex<Session>>` scheme, and speakrs' own
    /// `SegmentationModel`/`EmbeddingModel` document themselves as single-thread-at-a-time
    /// under coreml (`unsafe impl Send`, not `Sync`) — so this path holds the mutex for the
    /// whole request instead. Correct, but serializes all coreml jobs; DIAR_MAX_INFLIGHT has
    /// no effect in this mode. The mutex is per-engine, so the serialization is per-device.
    #[cfg(not(feature = "coreml"))]
    fn with_engine_on<R>(
        &self,
        device: Device,
        f: impl FnOnce(&mut DiarEngine) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut handle = self
            .engine_for(device)?
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone_shared()?;
        f(&mut handle)
    }

    #[cfg(feature = "coreml")]
    fn with_engine_on<R>(
        &self,
        device: Device,
        f: impl FnOnce(&mut DiarEngine) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut guard = self
            .engine_for(device)?
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut guard)
    }

    /// Only reachable with a device that came from `EngineRegistry::resolve`, so a miss here is
    /// an internal bug rather than a client error.
    fn engine_for(&self, device: Device) -> anyhow::Result<&std::sync::Mutex<DiarEngine>> {
        self.engines
            .engine(device)
            .ok_or_else(|| anyhow::anyhow!("no {device} engine loaded"))
    }
}

#[derive(Deserialize)]
struct DiarizeRequest {
    /// Path to media on a shared volume. A 16 kHz mono WAV takes the exact handoff fast
    /// path the app relies on; anything else (mp3/m4a/flac/ogg/aac/mp4, any-rate wav) is
    /// decoded and resampled in-process.
    ///
    /// `media_path` and `audio_path` are accepted as aliases. The `wav_path` name is kept
    /// because the live OpenTranscribe caller sends it, but it undersells the field and has
    /// led third parties to transcode to WAV first for no reason — the decoder handles
    /// anything symphonia can read.
    #[serde(alias = "media_path", alias = "audio_path")]
    wav_path: PathBuf,
    #[serde(default)]
    file_id: Option<String>,
    /// Also classify each speaker's gender from this same decoded audio, before it is
    /// dropped — saves the caller a second fetch, decode and model host.
    #[serde(default)]
    gender: bool,
    /// Execution device for this request ("cuda", "cpu", …). Omitted/null = the server's
    /// default device. Deliberately `Option<String>` and not a derived enum: axum 0.7 turns a
    /// serde variant mismatch into a bare 422 with no useful body, whereas parsing it
    /// ourselves yields a 400 that names the devices this build actually serves.
    #[serde(default)]
    device: Option<String>,
}

#[derive(Deserialize)]
struct EmbedRequest {
    /// Path to media on a shared volume — any symphonia-decodable format, not only WAV.
    /// `media_path` and `audio_path` are accepted as aliases; see `DiarizeRequest::wav_path`.
    #[serde(default, alias = "media_path", alias = "audio_path")]
    wav_path: Option<PathBuf>,
    /// …or raw 16 kHz mono f32 little-endian samples, base64 (small clips).
    #[serde(default)]
    samples_b64: Option<String>,
    /// Optional window within the file, in seconds (wav_path input only).
    #[serde(default)]
    start_s: Option<f64>,
    #[serde(default)]
    end_s: Option<f64>,
    /// Execution device for this request; see `DiarizeRequest::device`.
    #[serde(default)]
    device: Option<String>,
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

/// 16 kHz mono WAVs keep the hound fast path (byte-identical to the app handoff);
/// everything else decodes via symphonia and resamples to 16 kHz mono.
fn load_audio(path: &PathBuf) -> anyhow::Result<Vec<f32>> {
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        if let Ok(samples) = load_wav(path) {
            return Ok(samples);
        }
    }
    diar_core::audio::decode_to_16k_mono(path)
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

/// Acquire the admission permits for `device`: the global gate always, then the CPU sub-gate
/// if one is configured and this is CPU work. Order is fixed (global → device) so a mixed
/// deployment has no lock-ordering hazard.
async fn admit<'a>(
    state: &'a AppState,
    device: Device,
) -> Result<
    (
        tokio::sync::SemaphorePermit<'a>,
        Option<tokio::sync::SemaphorePermit<'a>>,
    ),
    (StatusCode, String),
> {
    let global = state
        .gate
        .acquire()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
    let device_permit = match (device, &state.cpu_gate) {
        (Device::Cpu, Some(cpu_gate)) => Some(
            cpu_gate
                .acquire()
                .await
                .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?,
        ),
        _ => None,
    };
    Ok((global, device_permit))
}

async fn diarize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DiarizeRequest>,
) -> ApiResult<DiarizeOutput> {
    // Resolve BEFORE admission: a bad device name must cost a 400, not an admission permit.
    let device = state
        .engines
        .resolve(req.device.as_deref())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let _permits = admit(&state, device).await?;
    let state2 = Arc::clone(&state);
    let out = tokio::task::spawn_blocking(move || -> anyhow::Result<DiarizeOutput> {
        let audio = load_audio(&req.wav_path)?;
        let file_id = req
            .file_id
            .clone()
            .or_else(|| {
                req.wav_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "file".into());
        state2.with_engine_on(device, |engine| {
            engine.diarize_with(&audio, &file_id, req.gender)
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;
    Ok(([(DEVICE_HEADER, device.as_str())], Json(out)))
}

async fn embed_window(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> ApiResult<EmbedResponse> {
    let device = state
        .engines
        .resolve(req.device.as_deref())
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let _permits = admit(&state, device).await?;
    let state2 = Arc::clone(&state);
    let embedding = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
        let clip: Vec<f32> = if let Some(b64) = &req.samples_b64 {
            decode_samples_b64(b64)?
        } else if let Some(path) = &req.wav_path {
            let audio = load_audio(path)?;
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
        state2.with_engine_on(device, |engine| engine.embed_window(&clip))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;
    Ok((
        [(DEVICE_HEADER, device.as_str())],
        Json(EmbedResponse { embedding }),
    ))
}

/// `/healthz` body. Was the bare string "ok"; the compose healthcheck is `curl -sf .../healthz`,
/// which only inspects the HTTP status, so returning JSON is non-breaking.
///
/// Consumers should gate on this before sending a `device` field: neither request struct uses
/// `deny_unknown_fields`, so an OLD diar-server silently IGNORES `"device": "cpu"` and runs the
/// job on its default device. `supported_devices` is the negotiation point.
///
/// Additive by design — new fields are safe to append.
#[derive(Serialize)]
struct HealthResponse {
    /// Always "ok" today; a field rather than an implicit 200 so richer states can be added.
    status: &'static str,
    /// Device used when a request omits `device`.
    default_device: &'static str,
    /// Devices loaded in THIS process and serving requests right now. First entry is the
    /// default. A device here is guaranteed to work.
    devices: Vec<&'static str>,
    /// Devices this BUILD can serve — a compile-time capability, a superset of `devices`.
    /// One listed here but absent from `devices` needs a `DIAR_DEVICES` change, not a rebuild.
    supported_devices: Vec<&'static str>,

    // ---- provisioning state (issue #2) ----
    // FLAT fields rather than a nested object, so appending them stays additive for anything
    // already parsing this body.
    /// True only when `models_state == "verified"`. This is what `/readyz` gates on.
    models_verified: bool,
    /// `verified` | `stale` | `unverified` | `failed`.
    ///
    /// `unverified` is NOT an error: every models directory provisioned before this feature
    /// existed has no marker, and the server serves them exactly as before.
    models_state: &'static str,
    models_dir: String,
    /// Tier recorded in the marker (`fast`/`small`), when there is one.
    models_set: Option<String>,
    models_exporter_version: Option<u32>,
    /// Upstream pipeline commit the weights came from.
    models_pipeline_revision: Option<String>,
    models_smoke_at: Option<String>,
    /// Whether the gender classifier was provisioned. Gender is enabled by FILE PRESENCE, so
    /// a `--skip-gender` deployment answers `diarize(gender=true)` with 200 and no genders;
    /// reporting it here is the difference between that being a decision and a mystery.
    models_gender: bool,
    /// Human sentence plus the remediation command, for every non-verified state.
    models_reason: Option<String>,
}

fn health_body(state: &AppState) -> HealthResponse {
    let m = &state.models;
    HealthResponse {
        status: "ok",
        default_device: state.engines.default_device().as_str(),
        devices: state.engines.devices().iter().map(|d| d.as_str()).collect(),
        supported_devices: engines::supported().iter().map(|d| d.as_str()).collect(),
        models_verified: m.state.is_verified(),
        models_state: m.state.as_str(),
        models_dir: m.dir.display().to_string(),
        models_set: m.set.clone(),
        models_exporter_version: m.exporter_version,
        models_pipeline_revision: m.pipeline_revision.clone(),
        models_smoke_at: m.smoke_at.clone(),
        models_gender: m.dir.join(diar_core::provision::files::GENDER_MODEL).exists(),
        models_reason: m.reason.clone(),
    }
}

/// ALWAYS 200 while the process is serving, in every model state.
///
/// This is a hard compatibility constraint, not a stylistic choice. Verified callers inspect
/// the STATUS ONLY — `docker-compose.diar-native.yml` runs `curl -sf .../healthz || exit 1`,
/// and `diarizer_native.py` checks `resp.status == 200`. Every models directory deployed
/// today has no marker, so a 503 for "unverified" would, on the day this ships, fail every
/// existing healthcheck, fail `up --wait` for the whole stack, and make OpenTranscribe fall
/// back to in-process PyAnnote — the exact silent quality regression this work exists to
/// prevent, caused by the fix for it. Changing the BODY is safe; changing the CODE is not.
async fn healthz(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(health_body(&state))
}

/// 200 only when the models are verified; 503 otherwise, with the same body.
///
/// This is where "still provisioning" is distinguished from "broken", with zero blast radius
/// on existing callers. Compose healthchecks move here AFTER provisioning once.
async fn readyz(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthResponse>) {
    let body = health_body(&state);
    let code = if body.models_verified {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body))
}

fn main() -> anyhow::Result<()> {
    // speakrs pipeline threads + ORT need more than the 2 MiB default thread stack
    // (measured: stack overflow in worker threads; same finding as the test suite).
    if std::env::var("RUST_MIN_STACK").is_err() {
        std::env::set_var("RUST_MIN_STACK", "16777216");
    }

    // `None` => serve. The live deployment runs `diar-server` with no arguments and both
    // Dockerfiles have ENTRYPOINT with no CMD, so this path must stay exactly as it was.
    match cli::Cli::parse().command {
        None | Some(cli::Command::Serve) => {}
        // The provisioning subcommands never construct an engine: no ORT, no VRAM, no
        // device. They must work on a box with no GPU and an empty models directory —
        // which is precisely the situation they exist to fix.
        Some(cli::Command::ProvisionModels(args)) => {
            std::process::exit(cli::run_provision(args))
        }
        Some(cli::Command::VerifyModels(args)) => std::process::exit(cli::run_verify(args)),
        Some(cli::Command::CheckToken(args)) => std::process::exit(cli::run_check_token(args)),
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
    let max_inflight: usize = std::env::var("DIAR_MAX_INFLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    // Opt-in inner gate; a 0 would deadlock every CPU request, so treat it as unset.
    let max_inflight_cpu: Option<usize> = std::env::var("DIAR_MAX_INFLIGHT_CPU")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let bind = std::env::var("DIAR_BIND").unwrap_or_else(|_| "0.0.0.0:8701".to_string());

    // Provisioning gate FIRST — before any engine is constructed. A `stat`-only pass, so it
    // costs nothing and fires exactly once. Its job is to name the real problem (no models)
    // instead of letting it surface as one "session load failed" per configured device.
    let models = cli::startup_gate_or_exit(std::path::Path::new(&models_dir));

    // Every engine loads here, serially, before `axum::serve` — DiarEngine::load calls
    // std::env::set_var, which cannot safely run alongside live tokio workers. See engines.rs.
    let engines = EngineRegistry::load_from_env(&models_dir)?;
    eprintln!(
        "diar-server devices: [{}] (default {})",
        engines
            .devices()
            .iter()
            .map(|d| d.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        engines.default_device()
    );
    let state = Arc::new(AppState {
        engines,
        gate: Semaphore::new(max_inflight),
        cpu_gate: max_inflight_cpu.map(Semaphore::new),
        models,
    });

    let app = Router::new()
        .route("/diarize", post(diarize))
        .route("/embed_window", post(embed_window))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    eprintln!("diar-server listening on {bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
