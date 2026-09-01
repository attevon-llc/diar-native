//! Driving the python export subprocess.
//!
//! ## Why the exporter lives inside the binary
//!
//! The export scripts are `include_str!`d into `diar-server` (~60 KB on a 33 MB binary,
//! +0.2%) and materialized to a private temp directory at run time. Two consequences, both
//! deliberate:
//!
//! 1. **Version skew becomes structurally impossible.** The exporter IS the running
//!    binary's own bytes, so `marker.exporter_version` can never disagree with the server
//!    that reads it. A scripts-on-disk design has to defend against a stale checkout; this
//!    one cannot have one.
//! 2. **Zero bytes are added to either runtime image.** Production does not run the
//!    diar-native image at all — `transcribe-app/backend/Dockerfile.prod` pins it only to
//!    `COPY --from` the binary and three ORT `.so`s. A provisioning image as the primary
//!    path would have shipped a tool the real topology never sees. The CPU image staying at
//!    189 MB is a hard regression check.
//!
//! The python interpreter is therefore the HOST image's (`DIAR_EXPORT_PYTHON`, default
//! `python3` on PATH), not one we ship. `docker/Dockerfile.provision` exists as the fallback
//! for operators running the plain standalone image, which has no python at all.
//!
//! ## Token hygiene
//!
//! The token is passed through the environment, NEVER argv — anything on a command line is
//! world-readable via `ps`. It is never logged, never written to the marker, and every line
//! of subprocess output is scrubbed before it reaches a terminal or a log, because a python
//! traceback will happily print the `Authorization` header it was constructed with.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use super::files::ModelSet;

/// The export scripts, baked in. Adapted copies of the vendored originals — see
/// `scripts/provision/UPSTREAM.md` for the vendor pin and the diff command. They are NOT
/// edited in `vendor/speakrs/`, because any vendored edit forces a regeneration of
/// `patches/0001-cuda-performance-patch-set.patch`, which feeds seven upstream PR-prep
/// branches whose whole purpose is clean review of CUDA *performance* work.
const SCRIPTS: &[(&str, &str)] = &[
    (
        "provision.py",
        include_str!("../../../../scripts/provision/provision.py"),
    ),
    (
        "export_models.py",
        include_str!("../../../../scripts/provision/export_models.py"),
    ),
    (
        "fold_segmentation.py",
        include_str!("../../../../scripts/provision/fold_segmentation.py"),
    ),
    (
        "export_tail_b64.py",
        include_str!("../../../../scripts/provision/export_tail_b64.py"),
    ),
    (
        "export_gender.py",
        include_str!("../../../../scripts/provision/export_gender.py"),
    ),
];

/// What the python side reports back, via a JSON file rather than stdout scraping (stdout
/// carries progress and torch's own warnings, and parsing it would be fragile).
/// `Serialize` is here for the round-trip test that pins "every key provision.py writes is a
/// field of this struct" — the check that would have caught `gender_precision` being dropped.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ExportReport {
    #[serde(default)]
    pub python: Option<String>,
    #[serde(default)]
    pub torch: Option<String>,
    #[serde(default)]
    pub torchaudio: Option<String>,
    #[serde(default)]
    pub onnx: Option<String>,
    #[serde(default)]
    pub onnxscript: Option<String>,
    #[serde(default)]
    pub onnxsim: Option<String>,
    #[serde(default)]
    pub pyannote_audio: Option<String>,
    #[serde(default)]
    pub transformers: Option<String>,
    #[serde(default)]
    pub folder: Option<String>,
    #[serde(default)]
    pub pipeline_revision: Option<String>,
    #[serde(default)]
    pub gender_repo: Option<String>,
    #[serde(default)]
    pub gender_revision: Option<String>,
    /// `"fp16"` or `"fp32"` — which precision the gender classifier was actually written at.
    ///
    /// `export_gender.py` measures this and its own docstring says it is "REPORTED, so the
    /// marker and /healthz can say so rather than implying fp16". It was reported into a
    /// struct with no such field and (serde ignoring unknown keys by default) silently
    /// dropped, so it reached nothing. It still matters now that fp16 is reachable again
    /// (RESULTS §7.39): the fp32 fallback is 378 MB and costs ~500 MiB more VRAM than the
    /// fp16 build (RESULTS §7.18: 5396 -> 4890 MiB), which on a 6 GB card is the difference
    /// between working and OOM — and nothing records which one you got.
    #[serde(default)]
    pub gender_precision: Option<String>,
}

#[derive(Debug)]
pub enum ExportError {
    /// No usable python, or it lacks the export dependencies. Exit 6.
    NoExporterEnv(String),
    /// The export itself failed. Exit 4.
    Failed(String),
}

impl ExportError {
    pub fn message(&self) -> &str {
        match self {
            ExportError::NoExporterEnv(m) | ExportError::Failed(m) => m,
        }
    }
}

/// One import requirement: any ONE of `alternatives` satisfies it.
pub struct ModuleReq {
    /// Interchangeable modules. Satisfied if at least one imports.
    pub alternatives: &'static [&'static str],
    /// Why the export needs it — printed so a missing one produces a reason, not an
    /// ImportError three minutes into a download.
    pub why: &'static str,
    /// Only needed when the gender classifier is being exported.
    pub gender_only: bool,
}

/// Modules the export genuinely needs.
///
/// The point of this list is that `check_python_env` runs BEFORE anything downloads. Three
/// modules the scripts actually import were missing from it, and each failed in its own bad
/// way:
///
/// * `onnxconverter_common` — absence raises NOTHING. `export_gender.py` catches it and
///   silently produces the fp32 classifier, ~500 MiB more VRAM (RESULTS §7.18) with no
///   diagnostic anywhere.
/// * `onnxsim`/`onnxslim` — absence raises a good error, but only from step 2b, i.e. after
///   the full ~470 MB download and the entire base export have already run.
/// * `onnxruntime` — lazily imported by the fold parity check; same wasted work.
///
/// `torchaudio` and `huggingface_hub` are also imported (`export_models.py` uses
/// `torchaudio.compliance.kaldi.get_mel_banks` and `hf_hub_download`) and are separate
/// distributions from torch and pyannote.audio, so they are probed rather than assumed.
pub const REQUIRED_MODULES: &[ModuleReq] = &[
    ModuleReq {
        alternatives: &["torch"],
        why: "runs the checkpoints and drives torch.onnx.export",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["torchaudio"],
        why: "supplies get_mel_banks for the fbank graph (a separate wheel from torch)",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["pyannote.audio"],
        why: "loads the community-1 pipeline",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["huggingface_hub"],
        why: "resolves the gated PLDA npz files and the pipeline revision",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["numpy"],
        why: "writes the PLDA .npy parameter files",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["onnx"],
        why: "reads back and rewrites the exported graphs",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["onnxscript"],
        why: "required by torch.onnx.export(dynamo=True)",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["onnxruntime"],
        why: "runs the fold parity check that proves constant-folding was bit-exact",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["onnxsim", "onnxslim"],
        why: "the constant folder (step 2b). onnxsim is preferred and reproduces the shipped \
              bytes exactly; onnxslim is the pure-python fallback for CPython 3.13. EITHER \
              satisfies this — folding itself is NOT optional (an unfolded segmentation graph \
              is ~2x slower on CUDA and silently falls back to CPU for Sin/Cos)",
        gender_only: false,
    },
    ModuleReq {
        alternatives: &["transformers"],
        why: "loads the gender classifier checkpoint",
        gender_only: true,
    },
    ModuleReq {
        alternatives: &["onnxconverter_common"],
        why: "fp16-quantizes the gender classifier. WITHOUT IT THERE IS NO ERROR — the export \
              silently falls back to the 378 MB fp32 model, ~500 MiB more VRAM (RESULTS \
              §7.18). Pass --skip-gender if you do not want the classifier at all",
        gender_only: true,
    },
];

pub fn resolve_python(explicit: Option<&str>) -> String {
    explicit
        .map(str::to_string)
        .or_else(|| {
            std::env::var("DIAR_EXPORT_PYTHON")
                .ok()
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_else(|| "python3".to_string())
}

/// Confirm the interpreter exists and can import what the export needs, BEFORE downloading
/// anything. Produces the pip line for whatever is missing.
pub fn check_python_env(python: &str, need_gender: bool) -> Result<String, ExportError> {
    let probe = Command::new(python).arg("--version").output();
    let version = match probe {
        Ok(o) if o.status.success() => String::from_utf8_lossy(if o.stdout.is_empty() {
            &o.stderr
        } else {
            &o.stdout
        })
        .trim()
        .to_string(),
        Ok(o) => {
            return Err(ExportError::NoExporterEnv(format!(
                "`{python}` exited {} when asked for its version. Set DIAR_EXPORT_PYTHON to \
                 a working interpreter, or use the provisioning image \
                 (docker/Dockerfile.provision).",
                o.status
            )))
        }
        Err(e) => {
            return Err(ExportError::NoExporterEnv(format!(
                "No python interpreter at `{python}` ({e}). The model export needs torch and \
                 pyannote.audio, which diar-server does not bundle. Either point \
                 DIAR_EXPORT_PYTHON at an interpreter that has them, or run the provisioning \
                 image built from docker/Dockerfile.provision."
            )))
        }
    };

    let mut missing: Vec<String> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();
    for req in REQUIRED_MODULES {
        if req.gender_only && !need_gender {
            continue;
        }
        let satisfied = req.alternatives.iter().any(|module| {
            Command::new(python)
                .arg("-c")
                .arg(format!("import {module}"))
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });
        if !satisfied {
            let label = req.alternatives.join(" or ");
            reasons.push(format!("  {label} — {}", req.why));
            missing.push(label);
        }
    }
    if !missing.is_empty() {
        return Err(ExportError::NoExporterEnv(format!(
            "`{python}` ({version}) cannot import: {}\n{}\n\nInstall them into that \
             interpreter (see scripts/provision/requirements.txt), or run the provisioning \
             image built from docker/Dockerfile.provision, which ships a pinned CPU-only \
             environment. Note onnxsim has no wheel for CPython 3.13 — on 3.13 the exporter \
             falls back to onnxslim, which is numerically bit-exact but emits a \
             differently-shaped segmentation graph; see scripts/provision/UPSTREAM.md.",
            missing.join(", "),
            reasons.join("\n")
        )));
    }
    Ok(version)
}

/// Materialize the embedded scripts into a private directory.
fn materialize(dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o700))?;
    }
    for (name, body) in SCRIPTS {
        std::fs::write(dest.join(name), body)?;
    }
    Ok(())
}

/// Remove the token from anything about to be printed. Called on EVERY line of subprocess
/// output — huggingface_hub and requests both embed the bearer token in exception text.
pub fn scrub(line: &str, token: Option<&str>) -> String {
    let mut out = line.to_string();
    if let Some(t) = token {
        if t.len() >= 6 {
            out = out.replace(t, "***REDACTED***");
        }
    }
    // Belt and braces: redact anything that looks like an HF token even if it is not the
    // one we were handed (a stale token in the environment, say).
    redact_tokenish(&out)
}

fn redact_tokenish(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("hf_") {
        let (head, tail) = rest.split_at(i);
        out.push_str(head);
        let end = tail
            .char_indices()
            .position(|(_, c)| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        if end > 12 {
            out.push_str("***REDACTED***");
            rest = &tail[end..];
        } else {
            out.push_str(&tail[..3]);
            rest = &tail[3..];
        }
    }
    out.push_str(rest);
    out
}

pub struct ExportRequest<'a> {
    pub python: &'a str,
    pub models_dir: &'a Path,
    pub set: ModelSet,
    pub with_gender: bool,
    pub token: Option<&'a str>,
    pub hf_cache: Option<&'a Path>,
}

/// Run the export. Streams scrubbed output to stderr as it happens — an export takes
/// minutes, and a silent process for minutes is indistinguishable from a hung one.
pub fn run_export(req: &ExportRequest<'_>) -> Result<ExportReport, ExportError> {
    let work = std::env::temp_dir().join(format!(
        "diar-provision-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    materialize(&work).map_err(|e| {
        ExportError::Failed(format!(
            "could not stage the export scripts in {}: {e}",
            work.display()
        ))
    })?;
    let report_path = work.join("report.json");

    let mut cmd = Command::new(req.python);
    cmd.arg(work.join("provision.py"))
        .arg("--models-dir")
        .arg(req.models_dir)
        .arg("--set")
        .arg(req.set.as_str())
        .arg("--report")
        .arg(&report_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Unbuffered, or progress arrives in one lump at the end.
        .env("PYTHONUNBUFFERED", "1")
        .env("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        // The export scripts gate real invariants on `assert` (the multi-mask parity check in
        // export_models.py among them). `PYTHONOPTIMIZE` compiles every assert out, so a
        // value inherited from the environment or baked into a base image would delete those
        // gates silently — the checks would not fail, they would not exist. The asserts that
        // matter have been converted to explicit `raise`, and this makes the inherited-flag
        // hazard impossible for any that are added back.
        .env_remove("PYTHONOPTIMIZE");
    if !req.with_gender {
        cmd.arg("--skip-gender");
    }
    // Environment, never argv: argv is visible to every process on the box via `ps`.
    if let Some(t) = req.token {
        cmd.env("HF_TOKEN", t);
    }
    if let Some(cache) = req.hf_cache {
        cmd.env("HF_HOME", cache);
    }

    let mut child = cmd.spawn().map_err(|e| {
        ExportError::NoExporterEnv(format!("could not start `{}`: {e}", req.python))
    })?;

    let token = req.token.map(str::to_string);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let t2 = token.clone();
    let pump_err = std::thread::spawn(move || {
        let mut tail = Vec::new();
        if let Some(s) = stderr {
            for line in BufReader::new(s).lines().map_while(Result::ok) {
                let line = scrub(&line, t2.as_deref());
                eprintln!("[export] {line}");
                tail.push(line);
                if tail.len() > 40 {
                    tail.remove(0);
                }
            }
        }
        tail
    });
    if let Some(s) = stdout {
        for line in BufReader::new(s).lines().map_while(Result::ok) {
            eprintln!("[export] {}", scrub(&line, token.as_deref()));
        }
    }
    let tail = pump_err.join().unwrap_or_default();

    let status = child
        .wait()
        .map_err(|e| ExportError::Failed(format!("waiting for the exporter: {e}")))?;

    let report = read_report(&report_path);
    let _ = std::fs::remove_dir_all(&work);

    if !status.success() {
        return Err(ExportError::Failed(format!(
            "the model export failed ({status}). Last output:\n{}",
            tail.join("\n")
        )));
    }

    // A successful export that produced no readable report is a FAILED export, not a
    // successful one with blank provenance.
    //
    // The previous code was `read(..).ok().and_then(|b| from_slice(..).ok())` followed by
    // `unwrap_or_default()`: two swallowed errors and an all-`None` report. Provisioning then
    // printed "Provisioned … Smoke test passed.", exited 0, and wrote a marker claiming
    // `verified` whose `toolchain.folder` was `null` — the single field
    // `fold_segmentation.py` records "because an unexplained byte difference months later is
    // exactly the kind of thing that costs a day". The one artifact whose entire job is
    // saying where the bytes came from would have said nothing, and said it confidently.
    report.map_err(ExportError::Failed)
}

/// Read the exporter's provenance report, distinguishing "no file" from "bad file".
///
/// They have different causes (the write never happened vs. the write was truncated or the
/// schema drifted) and different fixes, and a message that conflated them would send the
/// operator looking in the wrong place.
fn read_report(path: &Path) -> Result<ExportReport, String> {
    let bytes = std::fs::read(path).map_err(|e| {
        format!(
            "the export reported success but wrote no readable provenance report at {} \
             ({e}). Its own record of which torch, which onnx and which constant folder \
             produced these bytes is therefore missing, so the models cannot be vouched for. \
             This usually means the temp filesystem filled up or the process was interrupted \
             between the last export step and the report write. Re-run \
             `provision-models --force`.",
            path.display()
        )
    })?;
    serde_json::from_slice::<ExportReport>(&bytes).map_err(|e| {
        format!(
            "the export wrote a provenance report at {} that could not be parsed ({e}). The \
             file exists but is truncated or is not the JSON this build expects — an \
             exporter/binary version mismatch would look exactly like this. Refusing to write \
             a marker with empty provenance. Re-run `provision-models --force`; if it \
             persists, the embedded scripts and this binary have diverged, which should be \
             impossible (they are the same bytes) unless --dump-scripts output is being run \
             by hand.",
            path.display()
        )
    })
}

/// Is `dir` writable? Checked UP FRONT, because live compose mounts `/models` read-only and
/// discovering that after a 470 MB download would be gratuitous.
pub fn check_writable(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            return Err(format!(
                "models directory {} does not exist and could not be created: {e}",
                dir.display()
            ));
        }
    }
    let probe = dir.join(".diar-provision-write-test");
    match std::fs::write(&probe, b"1") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(e) => Err(format!(
            "models directory {} is not writable ({e}). Provisioning needs read-write \
             access, but the serving compose file mounts it READ-ONLY (`/models:ro`). Run \
             provisioning against the host path directly, or mount it `:rw` for this one \
             command; serving should stay `:ro`.",
            dir.display()
        )),
    }
}

/// Path the scripts are written to on disk for inspection (`--dump-scripts`).
pub fn dump_scripts(dest: &Path) -> std::io::Result<Vec<PathBuf>> {
    materialize(dest)?;
    Ok(SCRIPTS.iter().map(|(n, _)| dest.join(n)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_supplied_token_is_scrubbed_from_output() {
        let line = "Authorization: Bearer hf_averyrealtokenvalue123 failed";
        let out = scrub(line, Some("hf_averyrealtokenvalue123"));
        assert!(!out.contains("hf_averyrealtokenvalue123"), "{out}");
        assert!(out.contains("***REDACTED***"), "{out}");
    }

    #[test]
    fn a_token_we_were_not_given_is_still_scrubbed() {
        // A stale token in the subprocess environment must not leak either.
        let out = scrub("token=hf_someOtherTokenAAAAAAAA end", None);
        assert!(!out.contains("hf_someOtherTokenAAAAAAAA"), "{out}");
        assert!(
            out.contains("end"),
            "must not eat the rest of the line: {out}"
        );
    }

    #[test]
    fn scrubbing_leaves_ordinary_text_alone() {
        let s = "exported wespeaker-fbank.onnx (0.1 MB)";
        assert_eq!(scrub(s, Some("hf_xxxxxxxxxxxx")), s);
        // Short hf_ prefixes (e.g. a variable named hf_home) are not tokens.
        assert_eq!(scrub("hf_home is set", None), "hf_home is set");
    }

    #[test]
    fn scrubbing_handles_several_tokens_in_one_line() {
        let out = scrub("a hf_AAAAAAAAAAAAAAAA b hf_BBBBBBBBBBBBBBBB c", None);
        assert!(!out.contains("hf_AAAA"), "{out}");
        assert!(!out.contains("hf_BBBB"), "{out}");
        assert!(out.contains(" b "), "{out}");
        assert!(out.ends_with(" c"), "{out}");
    }

    #[test]
    fn a_missing_interpreter_is_an_env_error_naming_the_fallback() {
        let err = check_python_env("/nonexistent/python-that-is-not-here", false).unwrap_err();
        assert!(matches!(err, ExportError::NoExporterEnv(_)));
        let m = err.message();
        assert!(m.contains("DIAR_EXPORT_PYTHON"), "{m}");
        assert!(m.contains("Dockerfile.provision"), "{m}");
    }

    #[test]
    fn read_only_directory_is_reported_with_the_ro_mount_hint() {
        // Needs a directory that EXISTS (or `check_writable` would create it) and that even
        // root cannot write to — CI runs these as root in a container, so a chmod'd temp dir
        // would not fail there. `/proc/self` is that directory on Linux; it does not exist on
        // macOS, where the test used to fail on the "does not exist" path instead and made
        // `cargo test -p diar-core` red on the Apple Silicon dev machine. `/System` is the
        // macOS equivalent: SIP-protected, unwritable even as root.
        #[cfg(target_os = "linux")]
        let dir = Path::new("/proc/self");
        #[cfg(target_os = "macos")]
        let dir = Path::new("/System");
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        return;

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let err = check_writable(dir).unwrap_err();
            assert!(err.contains("not writable"), "{err}");
            assert!(err.contains(":ro"), "must name the read-only mount: {err}");
        }
    }

    #[test]
    fn writable_directory_passes_and_leaves_no_probe_behind() {
        let d = std::env::temp_dir().join(format!("diar-writable-{}", std::process::id()));
        check_writable(&d).unwrap();
        assert!(!d.join(".diar-provision-write-test").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn every_embedded_script_is_non_empty_and_is_python() {
        for (name, body) in SCRIPTS {
            assert!(!body.trim().is_empty(), "{name} is empty");
            assert!(
                body.contains("import") || body.contains("def "),
                "{name} does not look like python"
            );
        }
        assert!(SCRIPTS.iter().any(|(n, _)| *n == "provision.py"));
    }

    #[test]
    fn materialize_writes_all_scripts() {
        let d = std::env::temp_dir().join(format!("diar-scripts-{}", std::process::id()));
        let paths = dump_scripts(&d).unwrap();
        assert_eq!(paths.len(), SCRIPTS.len());
        for p in &paths {
            assert!(p.exists(), "{} missing", p.display());
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Top-level packages the embedded scripts import at run time, extracted from their
    /// source. Nested (function-local) imports count — `onnxruntime` and `onnxconverter_common`
    /// are both function-local, and being function-local is precisely why their absence
    /// surfaced late (or, for `onnxconverter_common`, not at all).
    fn imported_top_level_packages() -> std::collections::BTreeSet<String> {
        const STDLIB: &[&str] = &[
            "__future__",
            "argparse",
            "collections",
            "importlib",
            "json",
            "os",
            "shutil",
            "sys",
            "typing",
        ];
        // The scripts import each other; those are files we materialize, not dependencies.
        let local: Vec<&str> = SCRIPTS
            .iter()
            .map(|(n, _)| n.trim_end_matches(".py"))
            .collect();

        let mut found = std::collections::BTreeSet::new();
        for (_, body) in SCRIPTS {
            for line in body.lines() {
                let t = line.trim();
                let spec = if let Some(rest) = t.strip_prefix("from ") {
                    rest.split(" import ").next().unwrap_or("")
                } else if let Some(rest) = t.strip_prefix("import ") {
                    rest.split(" as ").next().unwrap_or("")
                } else {
                    continue;
                };
                for part in spec.split(',') {
                    let root = part.trim().split('.').next().unwrap_or("").trim();
                    if root.is_empty()
                        || STDLIB.contains(&root)
                        || local.contains(&root)
                        || !root.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        continue;
                    }
                    found.insert(root.to_string());
                }
            }
        }
        found
    }

    /// F4: the preflight probe exists so a missing dependency produces a pip line BEFORE a
    /// ~470 MB download. Three modules the scripts import were absent from it — most damaging
    /// `onnxconverter_common`, whose absence raises nothing at all and silently downgrades the
    /// gender classifier to fp32. This pins the list against the scripts themselves, so adding
    /// an import without adding a probe fails here rather than in the field.
    #[test]
    fn the_probe_covers_every_module_the_scripts_import() {
        let probed: std::collections::BTreeSet<String> = REQUIRED_MODULES
            .iter()
            .flat_map(|r| r.alternatives)
            .map(|m| m.split('.').next().unwrap_or(m).to_string())
            .collect();
        let imported = imported_top_level_packages();
        assert!(
            !imported.is_empty(),
            "the import scanner found nothing — it has stopped working"
        );
        let unprobed: Vec<&String> = imported.difference(&probed).collect();
        assert!(
            unprobed.is_empty(),
            "the export scripts import {unprobed:?}, which check_python_env never probes for. \
             A missing one is discovered mid-export (or, for onnxconverter_common, never). \
             Add it to REQUIRED_MODULES. Probed: {probed:?}"
        );
    }

    #[test]
    fn the_constant_folder_requirement_is_satisfied_by_either_wheel() {
        let folder = REQUIRED_MODULES
            .iter()
            .find(|r| r.alternatives.contains(&"onnxsim"))
            .expect("the constant folder must be probed for");
        assert!(
            folder.alternatives.contains(&"onnxslim"),
            "onnxsim has no CPython 3.13 wheel; onnxslim must satisfy the same requirement"
        );
        // And the gender-only modules must not be demanded when gender is skipped.
        let gender_only: Vec<&str> = REQUIRED_MODULES
            .iter()
            .filter(|r| r.gender_only)
            .flat_map(|r| r.alternatives)
            .copied()
            .collect();
        assert!(gender_only.contains(&"transformers"), "{gender_only:?}");
        assert!(
            gender_only.contains(&"onnxconverter_common"),
            "{gender_only:?}"
        );
    }

    /// F11: `assert` is not a gate. `PYTHONOPTIMIZE=1` in the environment or a base image
    /// compiles every one of them out, and `run_export` inherits the environment — so a
    /// parity check written as an assert would not fail, it would cease to exist.
    #[test]
    fn no_embedded_script_gates_an_invariant_on_a_bare_assert() {
        for (name, body) in SCRIPTS {
            for (i, line) in body.lines().enumerate() {
                assert!(
                    !line.trim_start().starts_with("assert "),
                    "{name}:{} uses `assert` as a gate: {}. Use an explicit `raise` — under \
                     PYTHONOPTIMIZE this check silently disappears.",
                    i + 1,
                    line.trim()
                );
            }
        }
    }

    /// F6: two swallowed errors used to turn "the report is unreadable" into an all-`None`
    /// report and a `verified` marker with blank provenance.
    #[test]
    fn an_unreadable_export_report_is_a_hard_error_that_names_the_cause() {
        let missing = std::env::temp_dir().join("diar-report-that-does-not-exist.json");
        let _ = std::fs::remove_file(&missing);
        let err = read_report(&missing).unwrap_err();
        assert!(err.contains("wrote no readable provenance report"), "{err}");
        assert!(err.contains("--force"), "must name the remedy: {err}");

        let bad = std::env::temp_dir().join(format!("diar-report-bad-{}.json", std::process::id()));
        std::fs::write(&bad, b"{ this is not json").unwrap();
        let err2 = read_report(&bad).unwrap_err();
        assert!(err2.contains("could not be parsed"), "{err2}");
        assert_ne!(
            err.contains("could not be parsed"),
            err2.contains("could not be parsed"),
            "an IO error and a parse error must not read the same — they have different fixes"
        );
        let _ = std::fs::remove_file(&bad);
    }

    /// F3: every key the python driver writes must land in a field. serde drops unknown keys
    /// silently, which is how a measured `gender_precision` reached nothing at all.
    #[test]
    fn every_key_the_exporter_writes_is_consumed() {
        // The full key set written by provision.py's `report` dict plus the dict
        // export_gender.export() returns into it.
        let full = serde_json::json!({
            "python": "3.12.3",
            "torch": "2.13.0",
            "torchaudio": "2.11.0",
            "onnx": "1.22.0",
            "onnxscript": "0.7.1",
            "onnxsim": "0.7.3",
            "pyannote_audio": "4.0.7",
            "transformers": "4.57.0",
            "folder": "onnxsim",
            "pipeline_revision": "abc123",
            "gender_repo": "some/repo",
            "gender_revision": "def456",
            "gender_precision": "fp32",
        });
        let obj = full.as_object().unwrap().clone();
        let report: ExportReport = serde_json::from_value(full.clone()).unwrap();
        // Round-trip each key through the struct: anything serde dropped comes back as None.
        let back = serde_json::to_value(&report).unwrap();
        let back = back.as_object().unwrap();
        for (key, value) in &obj {
            assert_eq!(
                back.get(key),
                Some(value),
                "provision.py writes `{key}`, but ExportReport does not carry it — serde \
                 dropped it silently, exactly as it dropped gender_precision"
            );
        }
    }

    #[test]
    fn python_resolution_prefers_the_explicit_flag() {
        assert_eq!(
            resolve_python(Some("/usr/bin/python3.11")),
            "/usr/bin/python3.11"
        );
        std::env::remove_var("DIAR_EXPORT_PYTHON");
        assert_eq!(resolve_python(None), "python3");
        std::env::set_var("DIAR_EXPORT_PYTHON", "/opt/env/bin/python");
        assert_eq!(resolve_python(None), "/opt/env/bin/python");
        std::env::remove_var("DIAR_EXPORT_PYTHON");
    }
}
