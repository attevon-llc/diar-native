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

use serde::Deserialize;

use super::files::ModelSet;

/// The export scripts, baked in. Adapted copies of the vendored originals — see
/// `scripts/provision/UPSTREAM.md` for the vendor pin and the diff command. They are NOT
/// edited in `vendor/speakrs/`, because any vendored edit forces a regeneration of
/// `patches/0001-cuda-performance-patch-set.patch`, which feeds seven upstream PR-prep
/// branches whose whole purpose is clean review of CUDA *performance* work.
const SCRIPTS: &[(&str, &str)] = &[
    ("provision.py", include_str!("../../../../scripts/provision/provision.py")),
    ("export_models.py", include_str!("../../../../scripts/provision/export_models.py")),
    ("fold_segmentation.py", include_str!("../../../../scripts/provision/fold_segmentation.py")),
    ("export_tail_b64.py", include_str!("../../../../scripts/provision/export_tail_b64.py")),
    ("export_gender.py", include_str!("../../../../scripts/provision/export_gender.py")),
];

/// What the python side reports back, via a JSON file rather than stdout scraping (stdout
/// carries progress and torch's own warnings, and parsing it would be fragile).
#[derive(Debug, Clone, Deserialize, Default)]
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

/// Modules the export genuinely needs, with the reason each is required — so a missing one
/// produces an install line rather than an ImportError.
const REQUIRED_MODULES: &[(&str, &str)] = &[
    ("torch", "runs the checkpoints and drives torch.onnx.export"),
    ("pyannote.audio", "loads the community-1 pipeline"),
    ("numpy", "writes the PLDA .npy parameter files"),
    ("onnx", "reads back and rewrites the exported graphs"),
    ("onnxscript", "required by torch.onnx.export(dynamo=True)"),
    ("transformers", "loads the gender classifier checkpoint"),
];

pub fn resolve_python(explicit: Option<&str>) -> String {
    explicit
        .map(str::to_string)
        .or_else(|| std::env::var("DIAR_EXPORT_PYTHON").ok().filter(|v| !v.is_empty()))
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

    let mut missing: Vec<&str> = Vec::new();
    for (module, _why) in REQUIRED_MODULES {
        if !need_gender && *module == "transformers" {
            continue;
        }
        let ok = Command::new(python)
            .arg("-c")
            .arg(format!("import {module}"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            missing.push(module);
        }
    }
    if !missing.is_empty() {
        let reasons: Vec<String> = REQUIRED_MODULES
            .iter()
            .filter(|(m, _)| missing.contains(m))
            .map(|(m, why)| format!("  {m} — {why}"))
            .collect();
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
        ExportError::Failed(format!("could not stage the export scripts in {}: {e}", work.display()))
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
        .env("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1");
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

    let report = std::fs::read(&report_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<ExportReport>(&b).ok());
    let _ = std::fs::remove_dir_all(&work);

    if !status.success() {
        return Err(ExportError::Failed(format!(
            "the model export failed ({status}). Last output:\n{}",
            tail.join("\n")
        )));
    }
    Ok(report.unwrap_or_default())
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
        assert!(out.contains("end"), "must not eat the rest of the line: {out}");
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
        // /proc is a real directory that is not writable by an unprivileged process.
        let err = check_writable(Path::new("/proc/self")).unwrap_err();
        assert!(err.contains("not writable"), "{err}");
        assert!(err.contains(":ro"), "must name the read-only mount: {err}");
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

    #[test]
    fn python_resolution_prefers_the_explicit_flag() {
        assert_eq!(resolve_python(Some("/usr/bin/python3.11")), "/usr/bin/python3.11");
        std::env::remove_var("DIAR_EXPORT_PYTHON");
        assert_eq!(resolve_python(None), "python3");
        std::env::set_var("DIAR_EXPORT_PYTHON", "/opt/env/bin/python");
        assert_eq!(resolve_python(None), "/opt/env/bin/python");
        std::env::remove_var("DIAR_EXPORT_PYTHON");
    }
}
