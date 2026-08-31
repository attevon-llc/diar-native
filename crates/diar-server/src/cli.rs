//! Command-line surface for `diar-server`.
//!
//! ## Backward compatibility is a hard requirement
//!
//! The live invocation is `command: ["diar-server"]` with NO arguments, and both Dockerfiles
//! set `ENTRYPOINT ["/usr/local/bin/diar-server"]` with no CMD. So the subcommand is
//! `Option<Command>` and `None` means *serve*, exactly as before.
//!
//! One behaviour change worth calling out: today `main()` never reads `std::env::args()`, so
//! stray arguments are silently ignored. With clap they become usage errors (exit 2). That is
//! the right trade — silently ignoring an argument someone thought was doing something is its
//! own class of bug — but it is a change.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use diar_core::provision::{
    self, exit as exit_code, files::ModelSet, marker, preflight, verify,
};
use diar_core::Mode;

/// Where the smoke clip is baked in the images. Overridable with `--smoke-clip`.
const BAKED_CLIP: &str = "/usr/local/share/diar-native/smoke.wav";
/// Fallback for running out of a source checkout.
const REPO_CLIP: &str = "vendor/speakrs/fixtures/test.wav";

#[derive(Parser, Debug)]
#[command(
    name = "diar-server",
    version,
    about = "OpenTranscribe native diarization sidecar",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the HTTP sidecar (the default when no subcommand is given).
    Serve,
    /// Download, export, verify and vouch for the model set.
    ProvisionModels(ProvisionArgs),
    /// Deep-verify an existing models directory: full sha256 plus the whole smoke test.
    VerifyModels(VerifyArgs),
    /// Check the HuggingFace token and repo gate. Two HTTPS calls, no download.
    CheckToken(TokenArgs),
}

#[derive(clap::Args, Debug)]
pub struct ProvisionArgs {
    /// HuggingFace read token.
    ///
    /// PREFER the environment: a token on the command line is visible to every process on
    /// the box via `ps`. HF_TOKEN, HUGGINGFACE_TOKEN and HUGGING_FACE_HUB_TOKEN are all
    /// consulted, first one set wins.
    #[arg(long, hide_env_values = true)]
    pub hf_token: Option<String>,

    /// Directory to write the models into. Must be WRITABLE — the serving compose file
    /// mounts it read-only, so provision against the host path.
    #[arg(long, env = "DIAR_MODELS_DIR", default_value = "/models")]
    pub models_dir: PathBuf,

    /// Model tier: `fast` (default, adds the batch-64 graphs) or `small` (laptops).
    #[arg(long, default_value = "fast")]
    pub set: String,

    /// Execution mode the end-to-end smoke stage runs under. The other stages always run on
    /// CPU, so this needs a working device only if you set it to one.
    #[arg(long, env = "DIAR_MODE")]
    pub mode: Option<String>,

    /// Re-export even when the existing marker is valid.
    #[arg(long)]
    pub force: bool,

    /// Skip the gender classifier (saves ~189 MB, ~40% of the set).
    ///
    /// DEPLOYMENT-WIDE: gender is enabled by the model FILE being present, so skipping it
    /// disables speaker gender for every device and every request — `diarize(gender=true)`
    /// then logs one warning and returns no genders while still answering 200.
    #[arg(long)]
    pub skip_gender: bool,

    /// HuggingFace cache directory (sets HF_HOME for the exporter).
    #[arg(long, env = "HF_HOME")]
    pub hf_cache: Option<PathBuf>,

    /// Python interpreter that has torch + pyannote.audio.
    #[arg(long, env = "DIAR_EXPORT_PYTHON")]
    pub python: Option<String>,

    /// 16 kHz mono WAV used by the smoke test.
    #[arg(long)]
    pub smoke_clip: Option<PathBuf>,

    /// Emit the result as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
pub struct VerifyArgs {
    #[arg(long, env = "DIAR_MODELS_DIR", default_value = "/models")]
    pub models_dir: PathBuf,
    #[arg(long, default_value = "fast")]
    pub set: String,
    #[arg(long, env = "DIAR_MODE")]
    pub mode: Option<String>,
    #[arg(long)]
    pub smoke_clip: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
pub struct TokenArgs {
    #[arg(long, hide_env_values = true)]
    pub hf_token: Option<String>,
    #[arg(long)]
    pub json: bool,
}

/// `--mode` / `DIAR_MODE` -> engine mode. Mirrors the server's own default (cuda) so the
/// smoke test exercises what the server will actually run.
pub fn parse_mode(raw: Option<&str>) -> Mode {
    match raw {
        Some("cpu") => Mode::Cpu,
        Some("coreml") => Mode::CoreMl,
        Some("coreml_fast") => Mode::CoreMlFast,
        _ => Mode::Cuda,
    }
}

fn parse_set(raw: &str) -> Result<ModelSet, i32> {
    ModelSet::parse(raw).ok_or_else(|| {
        eprintln!("error: unknown --set '{raw}'; expected 'fast' or 'small'");
        exit_code::USAGE
    })
}

fn resolve_clip(explicit: Option<PathBuf>) -> Result<PathBuf, i32> {
    if let Some(p) = explicit {
        if !p.exists() {
            eprintln!("error: --smoke-clip {} does not exist", p.display());
            return Err(exit_code::USAGE);
        }
        return Ok(p);
    }
    for candidate in [PathBuf::from(BAKED_CLIP), PathBuf::from(REPO_CLIP)] {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    eprintln!(
        "error: no smoke clip found (looked at {BAKED_CLIP} and {REPO_CLIP}). Pass \
         --smoke-clip with a 16 kHz mono WAV of at least 10 seconds."
    );
    Err(exit_code::USAGE)
}

pub fn run_provision(args: ProvisionArgs) -> i32 {
    let set = match parse_set(&args.set) {
        Ok(s) => s,
        Err(c) => return c,
    };
    let clip = match resolve_clip(args.smoke_clip) {
        Ok(c) => c,
        Err(c) => return c,
    };

    let opts = provision::ProvisionOptions {
        models_dir: args.models_dir,
        set,
        mode: parse_mode(args.mode.as_deref()),
        token: args.hf_token,
        hf_cache: args.hf_cache,
        python: args.python,
        force: args.force,
        skip_gender: args.skip_gender,
        clip,
    };

    match provision::provision(&opts) {
        Ok(outcome) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&outcome).unwrap_or_default());
            } else if let Some(msg) = &outcome.message {
                println!("{msg}");
            } else {
                println!(
                    "Provisioned {} ({} set, {:.0} MB) in {:.1}s. Smoke test passed.",
                    outcome.models_dir,
                    outcome.model_set,
                    outcome.bytes as f64 / 1e6,
                    outcome.elapsed_ms as f64 / 1000.0
                );
            }
            exit_code::OK
        }
        Err(fail) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "error",
                        "exit_code": fail.code,
                        "message": fail.message,
                    })
                );
            }
            eprintln!("\nerror: {}", fail.message);
            fail.code
        }
    }
}

pub fn run_verify(args: VerifyArgs) -> i32 {
    let set = match parse_set(&args.set) {
        Ok(s) => s,
        Err(c) => return c,
    };
    let clip = match resolve_clip(args.smoke_clip) {
        Ok(c) => c,
        Err(c) => return c,
    };
    let with_gender = args
        .models_dir
        .join(diar_core::provision::files::GENDER_MODEL)
        .exists();

    let opts = verify::SmokeOptions {
        models_dir: args.models_dir.clone(),
        set,
        with_gender,
        mode: parse_mode(args.mode.as_deref()),
        clip,
    };

    match verify::verify_deep(&opts) {
        Ok((report, drift)) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": if drift.is_empty() { "ok" } else { "drift" },
                        "smoke": report,
                        "drift": drift,
                    })
                );
            } else {
                for stage in &report.stages {
                    println!("  {:<16} {}", stage.stage, stage.detail);
                }
                if drift.is_empty() {
                    println!("\nOK: {} verified.", args.models_dir.display());
                } else {
                    println!("\n{} file(s) differ from the marker:", drift.len());
                    for d in &drift {
                        println!("  {d}");
                    }
                }
            }
            // Content drift means the directory is no longer the one that was vouched for,
            // even though it still works. That is a smoke-test-level failure, not an "ok".
            if drift.is_empty() {
                exit_code::OK
            } else {
                exit_code::SMOKE_FAILED
            }
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            exit_code::SMOKE_FAILED
        }
    }
}

pub fn run_check_token(args: TokenArgs) -> i32 {
    let token = args
        .hf_token
        .or_else(|| preflight::token_from_env().map(|(_, t)| t));
    match preflight::check(&preflight::UreqTransport, token.as_deref()) {
        Ok(pre) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": "ok",
                        "user": pre.user,
                        "pipeline_repo": pre.pipeline_repo,
                        "pipeline_revision": pre.pipeline_revision,
                    })
                );
            } else {
                println!(
                    "OK: signed in as `{}`, and this account has accepted the terms for {}.",
                    pre.user, pre.pipeline_repo
                );
                if let Some(rev) = &pre.pipeline_revision {
                    println!("    pipeline revision {rev}");
                }
            }
            exit_code::OK
        }
        Err(e) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({"status": "denied", "message": e.message()})
                );
            }
            eprintln!("error: {}", e.message());
            exit_code::TOKEN_DENIED
        }
    }
}

/// The pre-engine startup gate. Returns the model status to publish on `/healthz`, or exits.
///
/// Runs BEFORE any engine is constructed, and uses `stat` only — no ORT, no VRAM, no device.
/// Without it, a half-provisioned directory surfaces as "CUDA session load failed" once per
/// configured device, inside a `restart: unless-stopped` crash loop that also fails
/// `up --wait`, and the operator's real problem (no models) never appears.
pub fn startup_gate_or_exit(models_dir: &std::path::Path) -> marker::ModelsStatus {
    // Which set is required follows the DEVICE-independent question "which graphs must
    // exist". Serving asks for the fast set unless told otherwise, matching the default.
    let set = std::env::var("DIAR_MODEL_SET")
        .ok()
        .and_then(|v| ModelSet::parse(&v))
        .unwrap_or(ModelSet::Fast);

    match provision::startup_gate(models_dir, set, set) {
        provision::StartupGate::Fatal(msg) => {
            eprintln!("\n{msg}\n");
            eprintln!(
                "(Set {}=1 to start anyway.)",
                provision::ALLOW_UNVERIFIED_ENV
            );
            std::process::exit(exit_code::NO_EXPORTER_ENV);
        }
        provision::StartupGate::Proceed { status, warning } => {
            // Non-fatal, so it belongs in the log stream with everything else. The Fatal arm
            // above deliberately stays on stderr: it is a multi-line remediation block printed
            // on the way to `exit()`, and it must survive any log configuration.
            if let Some(w) = warning {
                tracing::warn!(models_state = status.state.as_str(), "{w}");
            }
            status
        }
    }
}
