//! Model provisioning: obtain, export, verify and vouch for a models directory.
//!
//! Lives in `diar-core` rather than `diar-server` on purpose. `diar-server` is a binary
//! crate with no `tests/` directory, so nothing in it can be imported by an integration
//! test. Putting the logic here keeps it testable and keeps the diff on
//! `crates/diar-server/src/main.rs` down to clap types, a `match`, and two route handlers.
//!
//! Division of labour, and the reason for each split:
//!
//! | step | owner | why |
//! |---|---|---|
//! | token/gate preflight | Rust | must work in the plain runtime image, which has no python; produces the actionable message before anything heavy runs |
//! | download + ONNX/PLDA export | python subprocess | torch/onnxscript have no Rust equivalent |
//! | smoke test | **Rust** | must exercise the EXACT ORT build the server uses; a python-side check would validate a different runtime |
//! | marker write | Rust | the same code reads it at startup |
//! | idempotency | Rust | runs before the export, so a no-op needs no python at all |

pub mod exporter;
pub mod files;
pub mod marker;
pub mod preflight;
pub mod verify;

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub use files::{ModelSet, EXPORT_RECIPE_VERSION, MARKER_FILE, MARKER_SCHEMA};
pub use marker::{Marker, ModelsState, ModelsStatus};

/// Process exit codes. Stable: scripts and compose healthchecks branch on these.
pub mod exit {
    /// Success, including a no-op on an already-valid directory.
    pub const OK: i32 = 0;
    /// Bad arguments.
    pub const USAGE: i32 = 2;
    /// Files were produced but the smoke test rejected them.
    pub const SMOKE_FAILED: i32 = 3;
    /// The export subprocess failed.
    pub const EXPORT_FAILED: i32 = 4;
    /// Token missing/invalid, or the repo gate has not been accepted.
    pub const TOKEN_DENIED: i32 = 5;
    /// No usable python export environment — and, at serve time, a models directory too
    /// broken to start against.
    pub const NO_EXPORTER_ENV: i32 = 6;
    /// The models directory is not writable.
    pub const NOT_WRITABLE: i32 = 7;
}

/// Escape hatch for the startup gate. Set to `1`/`true` to downgrade its fatal cases to
/// warnings — for operators who know their directory is fine and would rather serve.
pub const ALLOW_UNVERIFIED_ENV: &str = "DIAR_ALLOW_UNVERIFIED_MODELS";

pub fn sha256_file(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Outcome of the startup gate.
pub enum StartupGate {
    /// Serve. `status` carries the state to report on `/healthz` and `/readyz`.
    Proceed { status: ModelsStatus, warning: Option<String> },
    /// Do not serve. Exit with `exit::NO_EXPORTER_ENV` after printing this.
    Fatal(String),
}

/// Decide whether to start, using `stat` only — no ORT, no VRAM, no device.
///
/// This is deliberately positioned BEFORE any engine construction. Without it, a
/// half-provisioned directory surfaces as "CUDA session load failed", once per configured
/// device, inside a `restart: unless-stopped` crash loop that also fails `up --wait` for the
/// whole stack. The operator's actual problem — missing models — never appears in the logs.
/// One cheap loop here names the real fix instead.
///
/// Note the asymmetry, which is the crux of not regressing existing deployments: a MISSING
/// FILE is fatal (the server genuinely cannot serve), but a MISSING MARKER is only a
/// warning. Every models directory deployed before this feature shipped has no marker, and
/// refusing to start on those would turn a provenance improvement into an outage.
pub fn startup_gate(models_dir: &Path, set: ModelSet, wanted: ModelSet) -> StartupGate {
    startup_gate_with(models_dir, set, wanted, allow_unverified_from_env())
}

pub fn allow_unverified_from_env() -> bool {
    std::env::var(ALLOW_UNVERIFIED_ENV)
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false)
}

/// `allow_unverified` is a PARAMETER, not read from the environment here.
///
/// Process environment is global mutable state, and two tests exercising the escape hatch
/// and the fatal path in parallel raced on it — one flipped the variable while the other was
/// mid-decision. Threading it through means the decision function is pure and the tests are
/// deterministic; only [`startup_gate`] touches the environment.
pub fn startup_gate_with(
    models_dir: &Path,
    set: ModelSet,
    wanted: ModelSet,
    allow_unverified: bool,
) -> StartupGate {
    // Gender is optional by construction (`GenderModel::load_optional`), so it is never
    // part of the can-we-start question.
    let missing = marker::missing_required(models_dir, set, false);
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(8).map(String::as_str).collect();
        let msg = format!(
            "Cannot serve: {} required model file(s) are missing from {}:\n  {}{}\n\n\
             The models are not shipped with diar-server — they are exported locally from \
             the gated pyannote pipeline. Provision them with:\n\n  \
             HF_TOKEN=<your token> diar-server provision-models --models-dir {} --set {set}\n\n\
             You will need a HuggingFace read token \
             (https://huggingface.co/settings/tokens) and to accept the terms at \
             https://huggingface.co/{} (free, CC-BY-4.0, auto-approved).",
            missing.len(),
            models_dir.display(),
            shown.join("\n  "),
            if missing.len() > shown.len() {
                format!("\n  … and {} more", missing.len() - shown.len())
            } else {
                String::new()
            },
            models_dir.display(),
            preflight::PIPELINE_REPO,
        );
        return if allow_unverified {
            StartupGate::Proceed {
                status: marker::evaluate(models_dir, wanted),
                warning: Some(format!(
                    "{msg}\n\n({ALLOW_UNVERIFIED_ENV} is set — starting anyway. The engine \
                     will very likely fail to load.)"
                )),
            }
        } else {
            StartupGate::Fatal(msg)
        };
    }

    let status = marker::evaluate(models_dir, wanted);
    match status.state {
        ModelsState::Failed => {
            let msg = format!(
                "Cannot serve: the models in {} are recorded as known-bad. {}",
                models_dir.display(),
                status.reason.clone().unwrap_or_default()
            );
            if allow_unverified {
                StartupGate::Proceed {
                    warning: Some(format!("{msg}\n\n({ALLOW_UNVERIFIED_ENV} is set — starting anyway.)")),
                    status,
                }
            } else {
                StartupGate::Fatal(msg)
            }
        }
        ModelsState::Unverified | ModelsState::Stale => {
            let warning = status.reason.clone();
            StartupGate::Proceed { status, warning }
        }
        ModelsState::Verified => StartupGate::Proceed {
            status,
            warning: None,
        },
    }
}

#[derive(Debug, Clone)]
pub struct ProvisionOptions {
    pub models_dir: PathBuf,
    pub set: ModelSet,
    pub mode: crate::Mode,
    pub token: Option<String>,
    pub hf_cache: Option<PathBuf>,
    pub python: Option<String>,
    pub force: bool,
    pub skip_gender: bool,
    pub clip: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProvisionOutcome {
    pub status: String,
    pub models_dir: String,
    pub model_set: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smoke: Option<verify::SmokeReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub elapsed_ms: u64,
    pub bytes: u64,
}

/// A provisioning failure, carrying the exit code the CLI should use.
pub struct ProvisionFailure {
    pub code: i32,
    pub message: String,
}

impl ProvisionFailure {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The full provisioning flow. Ordered so that the cheapest and most likely failure comes
/// first: a no-op costs a marker read, a bad token costs two HTTPS calls, and nothing
/// downloads until both have passed.
pub fn provision(opts: &ProvisionOptions) -> Result<ProvisionOutcome, ProvisionFailure> {
    let started = std::time::Instant::now();
    let with_gender = !opts.skip_gender;

    // 1. Writable? Up front — never discover a read-only mount after a 470 MB download.
    exporter::check_writable(&opts.models_dir)
        .map_err(|e| ProvisionFailure::new(exit::NOT_WRITABLE, e))?;

    // 2. Idempotency, before python and before the network.
    if !opts.force {
        let status = marker::evaluate(&opts.models_dir, opts.set);
        if status.state.is_verified() {
            return Ok(ProvisionOutcome {
                status: "up-to-date".into(),
                models_dir: opts.models_dir.display().to_string(),
                model_set: opts.set.to_string(),
                smoke: None,
                message: Some(format!(
                    "Models in {} are already provisioned and verified (recipe v{}). \
                     Nothing to do; pass --force to re-export.",
                    opts.models_dir.display(),
                    EXPORT_RECIPE_VERSION
                )),
                elapsed_ms: started.elapsed().as_millis() as u64,
                bytes: dir_bytes(&opts.models_dir),
            });
        }
    }

    // 3. Token + gate. The python subprocess is NEVER launched until this passes, which is
    //    what keeps the most likely failure off the traceback path.
    let token = opts
        .token
        .clone()
        .or_else(|| preflight::token_from_env().map(|(_, t)| t));
    let pre = preflight::check(&preflight::UreqTransport, token.as_deref())
        .map_err(|e| ProvisionFailure::new(exit::TOKEN_DENIED, e.message()))?;

    // 4. Is there anything to run the export with?
    let python = exporter::resolve_python(opts.python.as_deref());
    let python_version = exporter::check_python_env(&python, with_gender)
        .map_err(|e| ProvisionFailure::new(exit::NO_EXPORTER_ENV, e.message()))?;

    // 5. Export.
    let report = exporter::run_export(&exporter::ExportRequest {
        python: &python,
        models_dir: &opts.models_dir,
        set: opts.set,
        with_gender,
        token: token.as_deref(),
        hf_cache: opts.hf_cache.as_deref(),
    })
    .map_err(|e| match e {
        exporter::ExportError::NoExporterEnv(m) => {
            ProvisionFailure::new(exit::NO_EXPORTER_ENV, m)
        }
        exporter::ExportError::Failed(m) => ProvisionFailure::new(exit::EXPORT_FAILED, m),
    })?;

    // 6. Smoke test, in Rust, against the real ORT.
    let smoke_opts = verify::SmokeOptions {
        models_dir: opts.models_dir.clone(),
        set: opts.set,
        with_gender,
        mode: opts.mode,
        clip: opts.clip.clone(),
    };
    let smoke = match verify::run(&smoke_opts) {
        Ok(r) => r,
        Err(e) => {
            // Record the failure so a later startup reports `failed` with the reason,
            // rather than silently serving models we already know are bad.
            let failed = marker::SmokeRecord {
                status: "fail".into(),
                mode: verify::mode_name(opts.mode).into(),
                clip_sha256: sha256_file(&opts.clip).unwrap_or_default(),
                num_speakers: 0,
                segments: 0,
                duration_ms: 0,
                checked_at: now_rfc3339(),
                error: Some(format!("{e:#}")),
            };
            let m = Marker {
                schema: MARKER_SCHEMA,
                generated_at: now_rfc3339(),
                generated_by: generated_by(),
                model_set: opts.set.to_string(),
                exporter_version: EXPORT_RECIPE_VERSION,
                with_gender,
                upstream: marker::Upstream {
                    pipeline_repo: pre.pipeline_repo.clone(),
                    pipeline_revision: pre
                        .pipeline_revision
                        .clone()
                        .or_else(|| report.pipeline_revision.clone()),
                    gender_repo: report.gender_repo.clone(),
                    gender_revision: report.gender_revision.clone(),
                },
                toolchain: toolchain_of(&report, &python_version),
                speakrs: marker::SpeakrsPin::default(),
                files: Vec::new(),
                smoke: failed,
            };
            let _ = m.write(&opts.models_dir);
            return Err(ProvisionFailure::new(
                exit::SMOKE_FAILED,
                format!(
                    "The models were exported but FAILED verification, so they have been \
                     marked known-bad rather than left to degrade diarization silently.\n\n{e:#}"
                ),
            ));
        }
    };

    // 7. Vouch for the result.
    let names = verify::marker_file_list(opts.set, with_gender);
    let files = marker::file_records(&opts.models_dir, &names)
        .map_err(|e| ProvisionFailure::new(exit::EXPORT_FAILED, e))?;
    let bytes = files.iter().map(|f| f.bytes).sum();

    let m = Marker {
        schema: MARKER_SCHEMA,
        generated_at: now_rfc3339(),
        generated_by: generated_by(),
        model_set: opts.set.to_string(),
        exporter_version: EXPORT_RECIPE_VERSION,
        with_gender,
        upstream: marker::Upstream {
            pipeline_repo: pre.pipeline_repo.clone(),
            pipeline_revision: pre
                .pipeline_revision
                .clone()
                .or_else(|| report.pipeline_revision.clone()),
            gender_repo: report.gender_repo.clone(),
            gender_revision: report.gender_revision.clone(),
        },
        toolchain: toolchain_of(&report, &python_version),
        speakrs: marker::SpeakrsPin::default(),
        files,
        smoke: marker::SmokeRecord {
            status: "pass".into(),
            mode: smoke.mode.clone(),
            clip_sha256: smoke.clip_sha256.clone(),
            num_speakers: smoke.num_speakers,
            segments: smoke.segments,
            duration_ms: smoke.duration_ms,
            checked_at: now_rfc3339(),
            error: None,
        },
    };
    m.write(&opts.models_dir)
        .map_err(|e| ProvisionFailure::new(exit::NOT_WRITABLE, e))?;

    Ok(ProvisionOutcome {
        status: "provisioned".into(),
        models_dir: opts.models_dir.display().to_string(),
        model_set: opts.set.to_string(),
        smoke: Some(smoke),
        message: None,
        elapsed_ms: started.elapsed().as_millis() as u64,
        bytes,
    })
}

fn generated_by() -> String {
    format!("diar-server {}", env!("CARGO_PKG_VERSION"))
}

fn toolchain_of(r: &exporter::ExportReport, python_version: &str) -> marker::Toolchain {
    marker::Toolchain {
        python: r
            .python
            .clone()
            .or_else(|| Some(python_version.to_string())),
        torch: r.torch.clone(),
        torchaudio: r.torchaudio.clone(),
        onnx: r.onnx.clone(),
        onnxscript: r.onnxscript.clone(),
        onnxsim: r.onnxsim.clone(),
        pyannote_audio: r.pyannote_audio.clone(),
        transformers: r.transformers.clone(),
        folder: r.folder.clone(),
    }
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "diar-gate-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_all_required(dir: &Path, set: ModelSet) {
        for f in files::required_files(set, false) {
            std::fs::write(dir.join(f), b"xx").unwrap();
        }
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        let p = std::env::temp_dir().join(format!("diar-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn missing_models_are_fatal_and_the_message_teaches_the_fix() {
        let d = scratch();
        match startup_gate_with(&d, ModelSet::Fast, ModelSet::Fast, false) {
            StartupGate::Fatal(m) => {
                assert!(m.contains("provision-models"), "{m}");
                assert!(m.contains("huggingface.co/settings/tokens"), "{m}");
                assert!(m.contains(preflight::PIPELINE_REPO), "{m}");
            }
            _ => panic!("a directory with no models must not start"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The regression this whole design exists to avoid: every models directory deployed
    /// today has no marker, and refusing to start on those would take the live stack down.
    #[test]
    fn a_complete_directory_with_no_marker_still_starts() {
        let d = scratch();
        write_all_required(&d, ModelSet::Fast);
        match startup_gate_with(&d, ModelSet::Fast, ModelSet::Fast, false) {
            StartupGate::Proceed { status, warning } => {
                assert_eq!(status.state, ModelsState::Unverified);
                assert!(warning.is_some(), "operator should be told, but not blocked");
            }
            StartupGate::Fatal(m) => panic!("must not be fatal: {m}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_recorded_smoke_failure_is_fatal() {
        std::env::remove_var(ALLOW_UNVERIFIED_ENV);
        let d = scratch();
        write_all_required(&d, ModelSet::Fast);
        Marker {
            schema: MARKER_SCHEMA,
            generated_at: now_rfc3339(),
            generated_by: "test".into(),
            model_set: "fast".into(),
            exporter_version: EXPORT_RECIPE_VERSION,
            with_gender: false,
            upstream: Default::default(),
            toolchain: Default::default(),
            speakrs: Default::default(),
            files: vec![],
            smoke: marker::SmokeRecord {
                status: "fail".into(),
                mode: "cpu".into(),
                clip_sha256: String::new(),
                num_speakers: 0,
                segments: 0,
                duration_ms: 0,
                checked_at: now_rfc3339(),
                error: Some("stage 1 could not parse wespeaker-fbank.onnx".into()),
            },
        }
        .write(&d)
        .unwrap();
        match startup_gate_with(&d, ModelSet::Fast, ModelSet::Fast, false) {
            StartupGate::Fatal(m) => assert!(m.contains("known-bad"), "{m}"),
            _ => panic!("known-bad models must not serve"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_escape_hatch_downgrades_fatals_to_warnings() {
        let d = scratch();
        let got = startup_gate_with(&d, ModelSet::Fast, ModelSet::Fast, true);
        match got {
            StartupGate::Proceed { warning, .. } => {
                assert!(warning.unwrap().contains(ALLOW_UNVERIFIED_ENV));
            }
            StartupGate::Fatal(m) => panic!("escape hatch did not apply: {m}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_empty_file_counts_as_missing() {
        let d = scratch();
        write_all_required(&d, ModelSet::Small);
        std::fs::write(d.join("wespeaker-fbank.onnx"), b"").unwrap();
        match startup_gate_with(&d, ModelSet::Small, ModelSet::Small, false) {
            StartupGate::Fatal(m) => assert!(m.contains("wespeaker-fbank.onnx"), "{m}"),
            _ => panic!("a zero-length model must be treated as missing"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        let t = now_rfc3339();
        assert!(t.ends_with('Z'), "{t}");
        assert!(t.len() >= 20, "{t}");
    }
}
