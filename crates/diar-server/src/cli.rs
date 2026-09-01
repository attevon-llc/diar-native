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
use diar_core::provision::{self, exit as exit_code, files::ModelSet, marker, preflight, verify};
use diar_core::Mode;

// The smoke-clip search path lives in `diar_core::provision::CLIP_CANDIDATES`, next to the
// code that resolves it. Provisioning resolves it LATE (after the writability, token and
// python checks) rather than at argument-parse time — see `provision::resolve_clip`.

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

    /// Execution device the end-to-end verification stage runs on. DEFAULTS TO `cpu`.
    ///
    /// Stages 1, 2, 3 and 5 always run on the CPU execution provider, which is statically
    /// linked into every build, so provisioning needs no GPU at all — and this default is
    /// what makes that true. It used to fall through to `cuda`, which meant the documented
    /// `docker run … diar-provision` line (no `--gpus`) exported 470 MB of correct models
    /// and then recorded them as known-bad because the verification stage could not open a
    /// device. Set this to `cuda` only if you specifically want the models exercised on the
    /// accelerator; `DIAR_DEVICES`' first entry is honoured too, so serving and provisioning
    /// share one vocabulary.
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

    /// 16 kHz mono WAV of at least 10 s used by the end-to-end verification stage.
    ///
    /// The diar-server images bake one in. Images that only COPY the binary out of them
    /// (OpenTranscribe's backend image does exactly that) have no clip, so provisioning from
    /// there needs this flag — any short speech recording works, it is only ever read.
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
    /// Tier to verify against. Defaults to whatever the directory's own marker records,
    /// falling back to `fast` when there is no marker.
    #[arg(long)]
    pub set: Option<String>,
    /// Execution device for the end-to-end stage. Defaults to `cpu` — see
    /// `provision-models --help`.
    #[arg(long, env = "DIAR_MODE")]
    pub mode: Option<String>,
    /// 16 kHz mono WAV for the end-to-end stage. See `provision-models --help`.
    #[arg(long)]
    pub smoke_clip: Option<PathBuf>,
    /// Do NOT update the marker's smoke record even if verification passes.
    ///
    /// By default a fully-passing deep verification re-stamps the marker, which is the
    /// documented way to rehabilitate a directory carrying a stale `fail` record without a
    /// full re-export. Provenance (upstream revision, toolchain) is never touched.
    #[arg(long)]
    pub no_attest: bool,
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

// There is deliberately no `parse_mode` here any more. It mirrored the SERVING default
// (unset/unrecognized -> cuda) and both provisioning subcommands used it, which is how
// `provision-models` came to require a GPU it does not need. Serving still resolves its own
// devices through `engines::plan_devices`; provisioning uses `provisioning_mode` below, which
// defaults to CPU and rejects an unrecognized name instead of guessing.

fn parse_set(raw: &str) -> Result<ModelSet, i32> {
    ModelSet::parse(raw).ok_or_else(|| {
        eprintln!("error: unknown --set '{raw}'; expected 'fast' or 'small'");
        exit_code::USAGE
    })
}

/// Execution device for the PROVISIONING subcommands. Defaults to CPU, honours `DIAR_DEVICES`.
///
/// Deliberately NOT the serving vocabulary, which must keep falling through to `cuda` when
/// `DIAR_MODE` is unset or unrecognized. Provisioning has the opposite requirement:
/// it must succeed on a machine with no accelerator, because that is the machine most people
/// self-hosting OpenTranscribe are provisioning on, and because `Dockerfile.server-cpu` has
/// no CUDA backend compiled in at all. Three sources, in order:
///
/// 1. `--mode` / `DIAR_MODE` — an explicit request. An unrecognised value is a usage error
///    here rather than a silent fall-through to cuda; silently ignoring `--mode cpu-typo` and
///    then failing to open a GPU is not a diagnosis anyone can act on.
/// 2. `DIAR_DEVICES` first entry — the knob the README teaches as primary and which "wins
///    over DIAR_MODE" for serving. An operator who sets `DIAR_DEVICES=cpu` gets a CPU server;
///    they should not also get a provisioning run that quietly targets CUDA.
/// 3. CPU.
fn provisioning_mode(explicit: Option<&str>) -> Result<Mode, i32> {
    let named = explicit.map(str::to_string).or_else(|| {
        std::env::var("DIAR_DEVICES").ok().and_then(|v| {
            v.split(',')
                .map(str::trim)
                .find(|s| !s.is_empty())
                .map(str::to_string)
        })
    });
    let Some(name) = named else {
        return Ok(Mode::Cpu);
    };
    match name.parse::<crate::engines::Device>() {
        Ok(d) => Ok(d.to_mode()),
        Err(e) => {
            eprintln!(
                "error: {e}\n\nProvisioning does not need a device at all — every check \
                 except the end-to-end stage runs on the CPU execution provider. Omit \
                 --mode/DIAR_MODE (or set it to `cpu`) unless you specifically want the \
                 models exercised on an accelerator."
            );
            Err(exit_code::USAGE)
        }
    }
}

pub fn run_provision(args: ProvisionArgs) -> i32 {
    let set = match parse_set(&args.set) {
        Ok(s) => s,
        Err(c) => return c,
    };
    let mode = match provisioning_mode(args.mode.as_deref()) {
        Ok(m) => m,
        Err(c) => return c,
    };

    let opts = provision::ProvisionOptions {
        models_dir: args.models_dir,
        set,
        mode,
        token: args.hf_token,
        hf_cache: args.hf_cache,
        python: args.python,
        force: args.force,
        skip_gender: args.skip_gender,
        // Resolved inside `provision()`, after the cheap checks.
        clip: args.smoke_clip,
    };

    match provision::provision(&opts) {
        Ok(outcome) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&outcome).unwrap_or_default()
                );
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
                // Say which gender classifier was produced. fp32 is the normal outcome under
                // the pinned torch and costs ~500 MiB more VRAM than fp16 (RESULTS §7.18) —
                // a number the operator needs BEFORE the first OOM, not after.
                if let Some(p) = &outcome.gender_precision {
                    println!(
                        "  gender classifier: {p}{}",
                        match p.as_str() {
                            "fp32" =>
                                " (onnxconverter_common could not convert this graph; \
                                   ~500 MiB more VRAM than fp16 — RESULTS §7.18)",
                            _ => "",
                        }
                    );
                }
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
    let explicit_set = match args.set.as_deref().map(parse_set).transpose() {
        Ok(s) => s,
        Err(c) => return c,
    };
    // Same rule as the startup gate: believe the directory's own marker rather than assuming
    // `fast`, or `verify-models` on a `--set small` directory reports four missing files.
    let set = provision::serving_set(&args.models_dir, explicit_set);
    let clip = match provision::resolve_clip(args.smoke_clip.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return exit_code::USAGE;
        }
    };
    let mode = match provisioning_mode(args.mode.as_deref()) {
        Ok(m) => m,
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
        mode,
        clip,
    };

    match verify::verify_deep(&opts) {
        Ok(deep) => {
            // Three outcomes, three exit codes, because they need three different actions:
            //
            //   ok          every file matched a recorded hash and every stage passed
            //   drift       a file is no longer the one the marker vouched for
            //   unverified  the directory WORKS but there is no marker, so nothing was
            //               compared against anything. Reporting that as "ok" was the bug:
            //               `Marker::read` returns Ok(None) for an absent marker, the drift
            //               loop was skipped entirely, and the command that exists to detect
            //               a silent rewrite printed a clean bill of health having hashed
            //               zero bytes. An operator handed an unknown /models directory got
            //               exit 0 and a false guarantee.
            let status = if !deep.marker_present {
                "unverified"
            } else if deep.drift.is_empty() {
                "ok"
            } else {
                "drift"
            };
            let attested = if args.no_attest || !deep.fully_verified() {
                None
            } else {
                Some(provision::attest(&args.models_dir, &deep))
            };

            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "status": status,
                        "marker_present": deep.marker_present,
                        "files_hashed": deep.hashed,
                        "smoke": deep.smoke,
                        "drift": deep.drift,
                        "attested": attested,
                        "gender_precision": marker::Marker::read(&args.models_dir)
                            .ok()
                            .flatten()
                            .and_then(|m| m.toolchain.gender_precision),
                    })
                );
            } else {
                for stage in &deep.smoke.stages {
                    println!("  {:<16} {}", stage.stage, stage.detail);
                }
                match status {
                    "ok" => println!(
                        "\nOK: {} verified ({} file(s) matched their recorded sha256).",
                        args.models_dir.display(),
                        deep.hashed
                    ),
                    "unverified" => println!(
                        "\nUNVERIFIED: {} passed every smoke stage, but there is no {} in \
                         it, so NOTHING was compared against a recorded hash. The smoke \
                         tier says these files work; it cannot say they are the files \
                         anyone vouched for, nor where they came from. To get a verified \
                         directory, re-export it:\n\n  \
                         diar-server provision-models --models-dir {} --force",
                        args.models_dir.display(),
                        diar_core::provision::MARKER_FILE,
                        args.models_dir.display()
                    ),
                    _ => {
                        println!("\n{} file(s) differ from the marker:", deep.drift.len());
                        for d in &deep.drift {
                            println!("  {d}");
                        }
                    }
                }
                match &attested {
                    Some(provision::Attestation::Updated(d)) => println!("  {d}"),
                    Some(provision::Attestation::NotWritten(e)) => println!(
                        "  (marker not updated: {e} — verification still passed; this is \
                         expected on a read-only mount)"
                    ),
                    _ => {}
                }
            }

            match status {
                "ok" => exit_code::OK,
                "unverified" => exit_code::UNVERIFIABLE,
                _ => exit_code::SMOKE_FAILED,
            }
        }
        Err(e) if verify::is_device_unavailable(&e) => {
            eprintln!("error: {e:#}");
            exit_code::DEVICE_UNAVAILABLE
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
    // exist". `DIAR_MODEL_SET` is an OVERRIDE, not the source of truth: when it is unset the
    // set comes from the directory's own marker, because assuming `fast` made
    // `provision-models --set small` produce a directory the server then refused to start
    // against — complaining about the four batch-64 graphs that provisioning had just been
    // told to leave out.
    let explicit = std::env::var("DIAR_MODEL_SET")
        .ok()
        .and_then(|v| ModelSet::parse(&v));

    match provision::startup_gate(models_dir, explicit) {
        provision::StartupGate::Fatal(msg) => {
            eprintln!("\n{msg}\n");
            eprintln!(
                "(Set {}=1 to start anyway.)",
                provision::ALLOW_UNVERIFIED_ENV
            );
            // NOT `NO_EXPORTER_ENV`. That code means "this python cannot run the export" and
            // its fix is `pip install`; this one means "provision the models" and serving
            // needs no python at all. A supervisor branching on exit codes could not tell
            // them apart while they shared 6.
            std::process::exit(exit_code::MODELS_UNUSABLE);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `DIAR_DEVICES` is process-global. Serialize the tests that write it so they cannot
    /// observe each other's value — the same race `startup_gate_with`'s doc comment describes.
    static ENV: Mutex<()> = Mutex::new(());

    /// C1: the default. `provision-models` with no flags must NOT ask for an accelerator.
    ///
    /// This is the whole bug in one assertion. The old `parse_mode` fell through to
    /// `Mode::Cuda`, so the `docker run --rm -e HF_TOKEN=… -v …:/models diar-provision` line
    /// that `Dockerfile.provision` itself documents — no `--gpus` — exported 470 MB of
    /// correct models and then stamped them known-bad because stage 4 could not open a
    /// device. In the CPU-only image, where CUDA is not compiled in at all, it could never
    /// have succeeded.
    #[test]
    fn provisioning_defaults_to_cpu_and_never_to_a_device() {
        let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DIAR_DEVICES");
        assert_eq!(provisioning_mode(None).unwrap(), Mode::Cpu);
    }

    /// C5: `DIAR_DEVICES` is the knob the README teaches as primary and it "wins over
    /// DIAR_MODE" for serving. An operator who sets `DIAR_DEVICES=cpu` must not get a
    /// provisioning run that silently targets CUDA.
    #[test]
    fn provisioning_honours_the_first_entry_of_diar_devices() {
        let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("DIAR_DEVICES", "cpu,cuda");
        assert_eq!(provisioning_mode(None).unwrap(), Mode::Cpu);
        // Blank/whitespace entries are skipped, matching plan_devices.
        std::env::set_var("DIAR_DEVICES", " , cpu ");
        assert_eq!(provisioning_mode(None).unwrap(), Mode::Cpu);
        // An explicit --mode/DIAR_MODE still wins over the device list.
        assert_eq!(provisioning_mode(Some("cpu")).unwrap(), Mode::Cpu);
        std::env::remove_var("DIAR_DEVICES");
    }

    /// An unrecognized device name is a usage error, not a silent fall-through to cuda.
    #[test]
    fn an_unknown_provisioning_device_is_rejected_rather_than_guessed() {
        let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DIAR_DEVICES");
        assert_eq!(
            provisioning_mode(Some("cpu-with-a-typo")),
            Err(exit_code::USAGE)
        );
    }

    #[test]
    fn cuda_is_still_available_when_asked_for_explicitly() {
        let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DIAR_DEVICES");
        // Only meaningful in a build that has the backend; in a CPU-only build asking for
        // cuda is correctly a usage error.
        let got = provisioning_mode(Some("cuda"));
        if cfg!(feature = "cuda") {
            assert_eq!(got.unwrap(), Mode::Cuda);
        } else {
            assert_eq!(got, Err(exit_code::USAGE));
        }
    }

    /// The clap surface itself is part of the contract: the live invocation is
    /// `command: ["diar-server"]` with no arguments and must keep meaning "serve".
    #[test]
    fn the_cli_still_parses_a_bare_invocation_as_serve() {
        use clap::Parser;
        let cli = Cli::parse_from(["diar-server"]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn verify_accepts_the_new_flags() {
        use clap::Parser;
        let cli = Cli::parse_from([
            "diar-server",
            "verify-models",
            "--models-dir",
            "/models",
            "--no-attest",
            "--json",
        ]);
        match cli.command {
            Some(Command::VerifyModels(a)) => {
                assert!(a.no_attest);
                assert!(a.json);
                // `--set` is now optional: unset means "believe the marker".
                assert!(a.set.is_none());
            }
            other => panic!("{other:?}"),
        }
    }
}
