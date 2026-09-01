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
    /// No usable python export environment: the interpreter is missing, or it cannot import
    /// torch / pyannote.audio / onnx. The fix is `pip install`.
    pub const NO_EXPORTER_ENV: i32 = 6;
    /// The models directory is not writable.
    pub const NOT_WRITABLE: i32 = 7;
    /// Serve time only: the models directory is too broken to start against. Split out of
    /// `NO_EXPORTER_ENV`, which it used to share — a supervisor branching on exit codes could
    /// not tell "install torch into the exporter" from "provision the models", and those have
    /// nothing to do with each other. Serving does not need python at all.
    pub const MODELS_UNUSABLE: i32 = 8;
    /// The requested execution device is not usable here (no GPU visible, backend not
    /// compiled in). Distinct from `SMOKE_FAILED` on purpose: it says nothing about the
    /// models, and unlike a smoke failure it never marks them known-bad.
    pub const DEVICE_UNAVAILABLE: i32 = 9;
    /// `verify-models` only: the directory works, but there is no marker to verify it
    /// AGAINST, so no content was compared to any recorded hash. Not `OK` — the deep tier
    /// did not run — and not `SMOKE_FAILED` — nothing is known to be wrong.
    pub const UNVERIFIABLE: i32 = 10;
}

/// Where the smoke clip lives, in preference order.
///
/// Resolved LATE (just before the export, after every cheap check) rather than at argument
/// parse time. `INSTALL_NATIVE.md` steers operators to provision from OpenTranscribe's
/// backend image, and that image copies only the binary and three ORT `.so`s out of the
/// diar-native image — not the clip. Resolving first meant the documented command died with
/// "no smoke clip found" (exit 2) before it had so much as looked at the token.
pub const CLIP_CANDIDATES: &[&str] = &[
    "/usr/local/share/diar-native/smoke.wav",
    "vendor/speakrs/fixtures/test.wav",
];

/// Find the smoke clip, or explain exactly how to supply one.
pub fn resolve_clip(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return if p.exists() {
            Ok(p.to_path_buf())
        } else {
            Err(format!("--smoke-clip {} does not exist", p.display()))
        };
    }
    for candidate in CLIP_CANDIDATES {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(format!(
        "no smoke clip found (looked at {}). The end-to-end verification stage needs one 16 \
         kHz mono WAV of at least 10 seconds containing speech; it is baked into the \
         diar-server images but NOT into images that merely copy the binary out of them \
         (OpenTranscribe's backend image is one). Pass `--smoke-clip /path/to/clip.wav` — any \
         short recording will do, it is never redistributed and only ever read.",
        CLIP_CANDIDATES.join(" and ")
    ))
}

/// Escape hatch for the startup gate. Set to `1`/`true` to downgrade its fatal cases to
/// warnings — for operators who know their directory is fine and would rather serve.
pub const ALLOW_UNVERIFIED_ENV: &str = "DIAR_ALLOW_UNVERIFIED_MODELS";

pub fn sha256_file(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
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
    Proceed {
        status: ModelsStatus,
        warning: Option<String>,
    },
    /// Do not serve. Print this, then exit with [`exit::MODELS_UNUSABLE`] — NOT
    /// `NO_EXPORTER_ENV`, which the two used to share. Serving needs no python at all, so
    /// "install torch" and "provision the models" must not arrive at a supervisor as the same
    /// number. `cli::startup_gate_or_exit` is the only caller and owns the exit.
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
pub fn startup_gate(models_dir: &Path, explicit: Option<ModelSet>) -> StartupGate {
    let set = serving_set(models_dir, explicit);
    startup_gate_with(models_dir, set, set, allow_unverified_from_env())
}

/// Which model set this directory should be judged against.
///
/// Read from the MARKER, not defaulted to `Fast`. Defaulting to `Fast` made
/// `provision-models --set small` self-defeating: provisioning exits 0 having deliberately
/// deleted the four batch-64 graphs (`provision.py` FAST_ONLY), and the server then refused
/// to start because four "required" files were missing — with remediation text that
/// interpolated `--set fast`, i.e. telling a laptop operator to build the tier they had just
/// declined. The directory itself records which tier it is; believing it is strictly better
/// than guessing, and the guess was wrong in the one case anybody would notice.
///
/// `explicit` (from `DIAR_MODEL_SET`) still wins, for the operator who wants to assert that a
/// directory *ought* to be the fast set and get a loud complaint when it is not.
pub fn serving_set(models_dir: &Path, explicit: Option<ModelSet>) -> ModelSet {
    if let Some(set) = explicit {
        return set;
    }
    match marker::Marker::read(models_dir) {
        Ok(Some(m)) => m.model_set().unwrap_or(ModelSet::Fast),
        _ => ModelSet::Fast,
    }
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
             https://huggingface.co/{} (free, CC-BY-4.0, auto-approved).\n\n\
             (This server is requiring the '{set}' set. That comes from the directory's own \
             marker when it has one, else '{}'; override it with DIAR_MODEL_SET=fast|small.)",
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
            ModelSet::Fast,
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
                    warning: Some(format!(
                        "{msg}\n\n({ALLOW_UNVERIFIED_ENV} is set — starting anyway.)"
                    )),
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
    /// `None` => resolve from [`CLIP_CANDIDATES`] when the smoke stage needs it, which is
    /// after the writability / idempotency / token / python checks and before the download.
    pub clip: Option<PathBuf>,
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
    /// `"fp16"` / `"fp32"` / `None` when the gender classifier was skipped. Surfaced here
    /// because it is a ~500 MiB VRAM difference (RESULTS §7.18) that was measured by the
    /// exporter and then reported to nobody.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender_precision: Option<String>,
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
                gender_precision: marker::Marker::read(&opts.models_dir)
                    .ok()
                    .flatten()
                    .and_then(|m| m.toolchain.gender_precision),
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

    // 4b. Is there a clip to verify WITH? Late enough that the operator has already been told
    //     about a bad token or a missing torch, early enough that nobody downloads 470 MB
    //     only to be told at the end that the last step cannot run.
    let clip =
        resolve_clip(opts.clip.as_deref()).map_err(|e| ProvisionFailure::new(exit::USAGE, e))?;

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
        exporter::ExportError::NoExporterEnv(m) => ProvisionFailure::new(exit::NO_EXPORTER_ENV, m),
        exporter::ExportError::Failed(m) => ProvisionFailure::new(exit::EXPORT_FAILED, m),
    })?;

    // 6. Smoke test, in Rust, against the real ORT.
    let smoke_opts = verify::SmokeOptions {
        models_dir: opts.models_dir.clone(),
        set: opts.set,
        with_gender,
        mode: opts.mode,
        clip: clip.clone(),
    };
    let smoke = match verify::run(&smoke_opts) {
        Ok(r) => r,
        // An unusable DEVICE is an environment problem, not a verdict on the files. Returning
        // early here — before any marker is written — is the whole point: the previous code
        // wrote `smoke.status: "fail"` into the models directory, and `startup_gate` then
        // refused to serve those (perfectly good) models forever after. On a GPU-less host,
        // or in the CPU-only image where CUDA can never validate, that made a successful
        // export permanently self-destruct.
        Err(e) if !should_record_failure(&e) => {
            return Err(ProvisionFailure::new(
                exit::DEVICE_UNAVAILABLE,
                format!(
                    "The models in {} were exported successfully and have NOT been marked \
                     bad — but the end-to-end verification stage could not run here.\n\n{e:#}\n\n\
                     Nothing has been recorded about their correctness. Re-run \
                     `diar-server verify-models --models-dir {} --mode cpu` to finish \
                     verifying and stamp the marker.",
                    opts.models_dir.display(),
                    opts.models_dir.display(),
                ),
            ));
        }
        Err(e) => {
            // Record the failure so a later startup reports `failed` with the reason,
            // rather than silently serving models we already know are bad.
            let failed = marker::SmokeRecord {
                status: "fail".into(),
                mode: verify::mode_name(opts.mode).into(),
                clip_sha256: sha256_file(&clip).unwrap_or_default(),
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
        gender_precision: report.gender_precision.clone(),
    })
}

fn generated_by() -> String {
    format!("diar-server {}", env!("CARGO_PKG_VERSION"))
}

/// May this smoke failure be written into the marker as `fail`?
///
/// `fail` is a permanent, load-bearing accusation: [`startup_gate`] treats it as known-bad and
/// refuses to serve, so a directory stamped with it is out of service until someone re-exports
/// it. That is the right response to "the models are wrong" and exactly the wrong response to
/// "this machine has no GPU" — and the two arrived at the same code path.
pub fn should_record_failure(err: &anyhow::Error) -> bool {
    !verify::is_device_unavailable(err)
}

/// Outcome of trying to stamp a passing deep verification onto the existing marker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "result", content = "detail")]
pub enum Attestation {
    /// The marker's smoke record was refreshed from this run.
    Updated(String),
    /// Nothing to attest to — there is no marker. Deliberately NOT auto-created: a marker
    /// records provenance (which pipeline revision, which toolchain, which folder), and none
    /// of that is knowable from a directory that is merely sitting there. Inventing one would
    /// manufacture exactly the false confidence this whole feature exists to remove.
    NoMarker,
    /// The directory is read-only, or the write failed. Verification still succeeded.
    NotWritten(String),
}

/// Record a passing deep verification in the existing marker.
///
/// This is the RECOVERY PATH, and its absence was a dead end. `verify-models` used to be
/// read-only with respect to the marker, so a directory carrying `smoke.status: "fail"` — from
/// a transient stage-4 failure, or from the GPU-less provisioning run described above — could
/// not be rehabilitated by the obvious command. `verify-models` would print "OK: … verified.",
/// exit 0, and the server would go on exiting non-zero with "recorded as known-bad" about
/// models that had just passed every check. The only escape was a full `--force` re-export:
/// re-download, re-export, minutes, for files that were never wrong.
///
/// Only the smoke record moves. Provenance (`upstream`, `toolchain`, `speakrs`) is left
/// exactly as the exporting run wrote it, because this run did not export anything and has
/// nothing truthful to say about where the bytes came from.
pub fn attest(models_dir: &Path, deep: &verify::DeepReport) -> Attestation {
    let mut marker = match marker::Marker::read(models_dir) {
        Ok(Some(m)) => m,
        Ok(None) => return Attestation::NoMarker,
        Err(e) => return Attestation::NotWritten(e),
    };
    let was = marker.smoke.status.clone();
    marker.smoke = marker::SmokeRecord {
        status: "pass".into(),
        mode: deep.smoke.mode.clone(),
        clip_sha256: deep.smoke.clip_sha256.clone(),
        num_speakers: deep.smoke.num_speakers,
        segments: deep.smoke.segments,
        duration_ms: deep.smoke.duration_ms,
        checked_at: now_rfc3339(),
        error: None,
    };
    // Idempotent: verify-models can be run any number of times without the field growing a
    // tail of identical suffixes.
    const REATTESTED: &str = " (re-attested by verify-models)";
    if !marker.generated_by.ends_with(REATTESTED) {
        marker.generated_by.push_str(REATTESTED);
    }
    match marker.write(models_dir) {
        Ok(()) => Attestation::Updated(format!(
            "marker smoke record updated: {was} -> pass (mode {})",
            deep.smoke.mode
        )),
        Err(e) => Attestation::NotWritten(e),
    }
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
        gender_precision: r.gender_precision.clone(),
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
                assert!(
                    warning.is_some(),
                    "operator should be told, but not blocked"
                );
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

    fn marker_recording(dir: &Path, set: ModelSet, smoke_status: &str) {
        let names = files::required_files(set, false);
        Marker {
            schema: MARKER_SCHEMA,
            generated_at: now_rfc3339(),
            generated_by: "test".into(),
            model_set: set.to_string(),
            exporter_version: EXPORT_RECIPE_VERSION,
            with_gender: false,
            upstream: Default::default(),
            toolchain: Default::default(),
            speakrs: Default::default(),
            files: marker::file_records(dir, &names).unwrap(),
            smoke: marker::SmokeRecord {
                status: smoke_status.into(),
                mode: "cpu".into(),
                clip_sha256: "abc".into(),
                num_speakers: 2,
                segments: 9,
                duration_ms: 10,
                checked_at: now_rfc3339(),
                error: (smoke_status != "pass").then(|| "device was busy".to_string()),
            },
        }
        .write(dir)
        .unwrap();
    }

    /// C2: `provision-models --set small` deletes the four batch-64 graphs on purpose. The
    /// startup gate used to demand the FAST set regardless, so the server refused to start on
    /// a directory provisioning had just declared complete — and told the operator to run
    /// `--set fast`, the tier they had deliberately not asked for.
    #[test]
    fn a_small_set_directory_starts_and_is_verified() {
        let d = scratch();
        write_all_required(&d, ModelSet::Small);
        marker_recording(&d, ModelSet::Small, "pass");

        match startup_gate_with(&d, ModelSet::Small, ModelSet::Small, false) {
            StartupGate::Proceed { status, warning } => {
                assert_eq!(status.state, ModelsState::Verified, "{:?}", status.reason);
                assert!(warning.is_none(), "{warning:?}");
            }
            StartupGate::Fatal(m) => panic!("a small-set directory must serve: {m}"),
        }
        // And the set really is read off the marker rather than assumed.
        assert_eq!(serving_set(&d, None), ModelSet::Small);
        assert_eq!(
            serving_set(&d, Some(ModelSet::Fast)),
            ModelSet::Fast,
            "DIAR_MODEL_SET must still be able to override"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn with_no_marker_the_serving_set_still_defaults_to_fast() {
        let d = scratch();
        assert_eq!(serving_set(&d, None), ModelSet::Fast);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// C1: a device that is not usable HERE says nothing about the models, and must never be
    /// recorded as a smoke failure — a `fail` marker takes the directory out of service until
    /// someone re-exports 470 MB.
    #[test]
    fn an_unusable_device_is_never_recorded_as_a_model_failure() {
        let device = anyhow::Error::new(verify::DeviceUnavailable {
            mode: "cuda".into(),
            detail: "CUDA session load failed".into(),
        })
        .context("STAGE 4 FAILED: wrapped in context, as the real call site does");
        assert!(!should_record_failure(&device));
        let msg = format!("{device:#}");
        assert!(msg.contains("not usable on this machine"), "{msg}");
        assert!(msg.contains("NOT implicated"), "{msg}");

        // A genuine model problem is still recorded.
        let models = anyhow::anyhow!("STAGE 3b FAILED: fused and split paths disagree");
        assert!(should_record_failure(&models));
    }

    /// C3: the recovery path. A directory carrying a stale `fail` record must be
    /// rehabilitatable without a full re-export.
    #[test]
    fn a_passing_verification_clears_a_stale_failure_marker() {
        let d = scratch();
        write_all_required(&d, ModelSet::Small);
        marker_recording(&d, ModelSet::Small, "fail");
        assert_eq!(
            marker::evaluate(&d, ModelSet::Small).state,
            ModelsState::Failed
        );

        let deep = verify::DeepReport {
            smoke: verify::SmokeReport {
                stages: vec![],
                num_speakers: 2,
                segments: 7,
                duration_ms: 1234,
                clip_sha256: "deadbeef".into(),
                mode: "cpu".into(),
            },
            drift: vec![],
            hashed: 18,
            marker_present: true,
            marker_recorded_failure: true,
        };
        assert!(deep.fully_verified());
        match attest(&d, &deep) {
            Attestation::Updated(detail) => assert!(detail.contains("fail -> pass"), "{detail}"),
            other => panic!("expected an update, got {other:?}"),
        }
        assert_eq!(
            marker::evaluate(&d, ModelSet::Small).state,
            ModelsState::Verified
        );

        // Provenance is NOT invented or overwritten, and repeat runs do not grow the field.
        let m = Marker::read(&d).unwrap().unwrap();
        assert_eq!(m.generated_by, "test (re-attested by verify-models)");
        attest(&d, &deep);
        let m = Marker::read(&d).unwrap().unwrap();
        assert_eq!(m.generated_by, "test (re-attested by verify-models)");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn attesting_a_directory_with_no_marker_does_not_invent_one() {
        let d = scratch();
        let deep = verify::DeepReport {
            smoke: verify::SmokeReport {
                stages: vec![],
                num_speakers: 1,
                segments: 1,
                duration_ms: 1,
                clip_sha256: String::new(),
                mode: "cpu".into(),
            },
            drift: vec![],
            hashed: 0,
            marker_present: false,
            marker_recorded_failure: false,
        };
        assert!(!deep.fully_verified(), "no marker is not full verification");
        assert_eq!(attest(&d, &deep), Attestation::NoMarker);
        assert!(
            !Marker::path_in(&d).exists(),
            "must not fabricate provenance"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    /// C4: the message has to name the flag, because the primary documented environment
    /// (OpenTranscribe's backend image) copies the binary out of our image without the clip.
    #[test]
    fn a_missing_smoke_clip_names_the_flag_that_fixes_it() {
        let err = resolve_clip(Some(Path::new("/nonexistent/clip.wav"))).unwrap_err();
        assert!(err.contains("--smoke-clip"), "{err}");

        // The no-candidates path only reproduces where neither baked path exists.
        if !CLIP_CANDIDATES.iter().any(|c| Path::new(c).exists()) {
            let err = resolve_clip(None).unwrap_err();
            assert!(err.contains("--smoke-clip"), "{err}");
            assert!(err.contains("16 kHz mono"), "{err}");
        }
    }

    /// C6: a supervisor must be able to tell "install torch" from "provision the models".
    #[test]
    fn exit_codes_are_distinct() {
        let codes = [
            exit::OK,
            exit::USAGE,
            exit::SMOKE_FAILED,
            exit::EXPORT_FAILED,
            exit::TOKEN_DENIED,
            exit::NO_EXPORTER_ENV,
            exit::NOT_WRITABLE,
            exit::MODELS_UNUSABLE,
            exit::DEVICE_UNAVAILABLE,
            exit::UNVERIFIABLE,
        ];
        let mut seen = std::collections::BTreeSet::new();
        for c in codes {
            assert!(
                seen.insert(c),
                "exit code {c} is used for two different things"
            );
        }
    }

    /// F3: the precision the exporter measured has to survive the trip into the marker.
    #[test]
    fn gender_precision_reaches_the_marker() {
        let report: exporter::ExportReport = serde_json::from_str(
            r#"{"python":"3.12.3","torch":"2.13.0","gender_precision":"fp32"}"#,
        )
        .unwrap();
        assert_eq!(report.gender_precision.as_deref(), Some("fp32"));
        let tc = toolchain_of(&report, "Python 3.12.3");
        assert_eq!(
            tc.gender_precision.as_deref(),
            Some("fp32"),
            "measured by export_gender.py, dropped by serde, reported to nobody"
        );
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        let t = now_rfc3339();
        assert!(t.ends_with('Z'), "{t}");
        assert!(t.len() >= 20, "{t}");
    }
}
