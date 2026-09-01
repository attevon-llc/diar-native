//! The provenance marker: `<models_dir>/diar-provision.json`.
//!
//! What this file is, precisely, matters more than what it contains — overclaiming here
//! would itself be a fail-open. The marker is a **provenance record and a last-known-good
//! stamp**. It says "this directory was produced by recipe vN and passed the smoke test at
//! time T". It is NOT a live integrity check.
//!
//! Two tiers, and they are deliberately different:
//!
//! - **Startup / `/healthz` (fast, O(files) `stat`)** — marker parses, schema known,
//!   recipe version current, smoke recorded as passing, every recorded file present at its
//!   recorded byte length, and the set covers the configured mode. Deliberately NO hashing:
//!   re-reading 484 MB on every boot is not acceptable, and mtime is unusable as a proxy
//!   because `docker cp` and volume copies rewrite it.
//! - **`provision-models` / `verify-models` (deep)** — full sha256 of every file plus the
//!   whole smoke test.
//!
//! So startup answers "is this the directory that passed?", not "is this directory still
//! byte-perfect?". A file silently rewritten to the same length will pass startup and fail
//! `verify-models`, which is the honest boundary and is documented as such in
//! `docs/INSTALL_NATIVE.md`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::files::{required_files, ModelSet, EXPORT_RECIPE_VERSION, MARKER_FILE, MARKER_SCHEMA};

/// Verification state of a models directory, as reported by `/healthz` and `/readyz`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelsState {
    /// Marker present, current, and vouching for a passing smoke test.
    Verified,
    /// Provisioned, but by a different export recipe or for a narrower set than requested.
    /// The models are probably fine; they were just not produced by the code now running.
    Stale,
    /// No marker, or a marker this build cannot read. Almost every directory deployed
    /// before this feature shipped is in this state — which is exactly why it must not be
    /// fatal and must not change `/healthz`'s status code.
    Unverified,
    /// Known-bad: the marker records a failed smoke test, or files it vouches for are gone.
    Failed,
}

impl ModelsState {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelsState::Verified => "verified",
            ModelsState::Stale => "stale",
            ModelsState::Unverified => "unverified",
            ModelsState::Failed => "failed",
        }
    }

    pub fn is_verified(self) -> bool {
        self == ModelsState::Verified
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Upstream {
    pub pipeline_repo: String,
    /// `x-repo-commit` captured during preflight — the exact upstream revision the weights
    /// came from, recorded without ever needing to re-contact HuggingFace.
    pub pipeline_revision: Option<String>,
    pub gender_repo: Option<String>,
    pub gender_revision: Option<String>,
}

/// Versions of everything that can change the exported bytes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Toolchain {
    pub python: Option<String>,
    pub torch: Option<String>,
    pub torchaudio: Option<String>,
    pub onnx: Option<String>,
    pub onnxscript: Option<String>,
    pub onnxsim: Option<String>,
    pub pyannote_audio: Option<String>,
    pub transformers: Option<String>,
    /// Which constant-folder actually ran. `onnxsim` reproduces the shipped `models_folded/`
    /// segmentation graph exactly; `onnxslim` is the fallback on interpreters with no
    /// onnxsim wheel (notably CPython 3.13) and is numerically bit-exact but emits a
    /// differently-shaped graph. Recorded because "which folder" is otherwise invisible in
    /// the output and would turn into an unexplainable diff months later.
    pub folder: Option<String>,
    /// Precision the gender classifier was written at: `"fp16"` or `"fp32"`.
    ///
    /// Not decoration — it is a VRAM fact. The fp32 classifier is 378 MB on disk and costs
    /// ~500 MiB more VRAM than the fp16 one (RESULTS §7.18: 5396 -> 4890 MiB), which on a
    /// 6 GB card decides whether gender jobs run or OOM. `export_gender.py` falls back to
    /// fp32 whenever `onnxconverter_common` cannot convert the graph torch emits. Under the
    /// pinned torch 2.13 that was every run until RESULTS §7.39 elided the exporter's two
    /// no-op `Cast` nodes; fp16 is the expected outcome now, but the fallback is still live,
    /// so a directory that "has gender" says nothing about which one you got unless it is
    /// written down here.
    #[serde(default)]
    pub gender_precision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpeakrsPin {
    pub pin: Option<String>,
    pub patch_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeRecord {
    /// `"pass"` or `"fail"`. Anything else is treated as a failure.
    pub status: String,
    /// Execution mode stage 4 ran under. Stages 1-3 and 5 always run on CPU.
    pub mode: String,
    pub clip_sha256: String,
    pub num_speakers: usize,
    pub segments: usize,
    pub duration_ms: u64,
    pub checked_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SmokeRecord {
    pub fn passed(&self) -> bool {
        self.status == "pass"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub schema: u32,
    pub generated_at: String,
    pub generated_by: String,
    pub model_set: String,
    pub exporter_version: u32,
    #[serde(default)]
    pub with_gender: bool,
    #[serde(default)]
    pub upstream: Upstream,
    #[serde(default)]
    pub toolchain: Toolchain,
    #[serde(default)]
    pub speakrs: SpeakrsPin,
    #[serde(default)]
    pub files: Vec<FileRecord>,
    pub smoke: SmokeRecord,
}

/// The outcome of evaluating a models directory, including the sentence a human needs.
#[derive(Debug, Clone)]
pub struct ModelsStatus {
    pub state: ModelsState,
    pub dir: PathBuf,
    pub set: Option<String>,
    pub exporter_version: Option<u32>,
    pub pipeline_revision: Option<String>,
    pub smoke_at: Option<String>,
    /// Human-readable explanation plus the command that fixes it. Always populated for
    /// every non-verified state — a state name with no remedy is not actionable.
    pub reason: Option<String>,
}

impl Marker {
    pub fn path_in(dir: &Path) -> PathBuf {
        dir.join(MARKER_FILE)
    }

    pub fn read(dir: &Path) -> Result<Option<Self>, String> {
        let path = Self::path_in(dir);
        match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("reading {}: {e}", path.display())),
            Ok(bytes) => serde_json::from_slice::<Self>(&bytes)
                .map(Some)
                .map_err(|e| format!("parsing {}: {e}", path.display())),
        }
    }

    /// Write atomically: a marker is a claim about a directory, and a torn marker would be a
    /// claim nobody can evaluate. Write to a sibling temp file and rename.
    pub fn write(&self, dir: &Path) -> Result<(), String> {
        let path = Self::path_in(dir);
        let tmp = dir.join(format!("{MARKER_FILE}.tmp"));
        let body =
            serde_json::to_vec_pretty(self).map_err(|e| format!("serializing marker: {e}"))?;
        std::fs::write(&tmp, &body).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("renaming {} -> {}: {e}", tmp.display(), path.display()))?;
        Ok(())
    }

    pub fn model_set(&self) -> Option<ModelSet> {
        ModelSet::parse(&self.model_set)
    }
}

/// Evaluate a models directory the FAST way — parse the marker and `stat` the files it
/// vouches for. No hashing, no ORT, no VRAM. Safe to call on every startup and every
/// `/healthz` hit.
///
/// `wanted` is the set the server is configured to serve; `Fast` is a superset of `Small`.
pub fn evaluate(dir: &Path, wanted: ModelSet) -> ModelsStatus {
    let base = ModelsStatus {
        state: ModelsState::Unverified,
        dir: dir.to_path_buf(),
        set: None,
        exporter_version: None,
        pipeline_revision: None,
        smoke_at: None,
        reason: None,
    };

    let marker = match Marker::read(dir) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return ModelsStatus {
                reason: Some(format!(
                    "No provisioning marker ({MARKER_FILE}) in {}. The models may be fine — \
                     they were simply not produced by a verified `provision-models` run. \
                     Run `diar-server verify-models --models-dir {}` to check them, or \
                     `diar-server provision-models --models-dir {}` to (re)build them.",
                    dir.display(),
                    dir.display(),
                    dir.display()
                )),
                ..base
            }
        }
        Err(e) => {
            return ModelsStatus {
                reason: Some(format!(
                    "{e}. Treating the models as unverified. Run \
                     `diar-server verify-models --models-dir {}` to check them.",
                    dir.display()
                )),
                ..base
            }
        }
    };

    let known = ModelsStatus {
        set: Some(marker.model_set.clone()),
        exporter_version: Some(marker.exporter_version),
        pipeline_revision: marker.upstream.pipeline_revision.clone(),
        smoke_at: Some(marker.smoke.checked_at.clone()),
        ..base
    };

    if marker.schema > MARKER_SCHEMA {
        return ModelsStatus {
            state: ModelsState::Unverified,
            reason: Some(format!(
                "Marker schema {} is newer than this build understands (max {MARKER_SCHEMA}); \
                 cannot interpret it. Upgrade diar-server, or re-run \
                 `diar-server provision-models --models-dir {} --force`.",
                marker.schema,
                dir.display()
            )),
            ..known
        };
    }

    // Known-bad beats everything else: a recorded smoke failure is a positive statement
    // that these models did not work, which is strictly worse than "we don't know".
    if !marker.smoke.passed() {
        return ModelsStatus {
            state: ModelsState::Failed,
            reason: Some(format!(
                "The last provisioning run recorded a FAILED smoke test{}. These models are \
                 known-bad. Re-run `diar-server provision-models --models-dir {} --force`.",
                marker
                    .smoke
                    .error
                    .as_deref()
                    .map(|e| format!(": {e}"))
                    .unwrap_or_default(),
                dir.display()
            )),
            ..known
        };
    }

    // Cheap integrity: everything the marker vouches for is still there, at the length it
    // was vouched for. Catches the truncated / half-copied / partially-deleted directory,
    // which is the realistic corruption mode for a mounted volume.
    let mut bad: Vec<String> = Vec::new();
    for rec in &marker.files {
        let p = dir.join(&rec.name);
        match std::fs::metadata(&p) {
            Ok(md) if md.len() == rec.bytes => {}
            Ok(md) => bad.push(format!(
                "{} ({} bytes, expected {})",
                rec.name,
                md.len(),
                rec.bytes
            )),
            Err(_) => bad.push(format!("{} (missing)", rec.name)),
        }
    }
    if !bad.is_empty() {
        let shown: Vec<&str> = bad.iter().take(5).map(String::as_str).collect();
        return ModelsStatus {
            state: ModelsState::Failed,
            reason: Some(format!(
                "{} file(s) the marker vouches for are missing or the wrong size: {}{}. \
                 Re-run `diar-server provision-models --models-dir {} --force`.",
                bad.len(),
                shown.join(", "),
                if bad.len() > shown.len() { ", …" } else { "" },
                dir.display()
            )),
            ..known
        };
    }

    if marker.exporter_version != EXPORT_RECIPE_VERSION {
        return ModelsStatus {
            state: ModelsState::Stale,
            reason: Some(format!(
                "Models were exported by recipe version {} but this build ships recipe \
                 version {EXPORT_RECIPE_VERSION}. They will probably still work; to bring \
                 them up to date run \
                 `diar-server provision-models --models-dir {} --force`.",
                marker.exporter_version,
                dir.display()
            )),
            ..known
        };
    }

    match marker.model_set() {
        None => {
            return ModelsStatus {
                state: ModelsState::Stale,
                reason: Some(format!(
                    "Marker records an unrecognized model set {:?}. Re-run \
                     `diar-server provision-models --models-dir {} --force`.",
                    marker.model_set,
                    dir.display()
                )),
                ..known
            }
        }
        Some(have) if !have.covers(wanted) => {
            return ModelsStatus {
                state: ModelsState::Stale,
                reason: Some(format!(
                    "Models were provisioned as the '{have}' set but this server is \
                     configured for '{wanted}', which additionally needs the batch-64 \
                     graphs. Re-run \
                     `diar-server provision-models --models-dir {} --set {wanted}`.",
                    dir.display()
                )),
                ..known
            }
        }
        Some(_) => {}
    }

    ModelsStatus {
        state: ModelsState::Verified,
        reason: None,
        ..known
    }
}

/// Files that a serving process cannot start without, checked by `stat` only.
///
/// This is the startup gate's input. It is intentionally independent of the marker: a
/// directory with no marker but all its files is servable (state `unverified`), while a
/// directory missing `wespeaker-fbank.onnx` is not, marker or no marker.
pub fn missing_required(dir: &Path, set: ModelSet, with_gender: bool) -> Vec<String> {
    let mut missing = Vec::new();
    for name in required_files(set, with_gender) {
        let p = dir.join(name);
        match std::fs::metadata(&p) {
            Ok(md) if md.len() > 0 => {}
            Ok(_) => missing.push(format!("{name} (empty)")),
            Err(_) => missing.push(name.to_string()),
        }
    }
    missing
}

/// Convenience for building the `files` array.
pub fn file_records(dir: &Path, names: &[&str]) -> Result<Vec<FileRecord>, String> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let p = dir.join(name);
        let bytes = std::fs::metadata(&p)
            .map_err(|e| format!("stat {}: {e}", p.display()))?
            .len();
        out.push(FileRecord {
            name: (*name).to_string(),
            bytes,
            sha256: super::sha256_file(&p)?,
        });
    }
    Ok(out)
}

/// Map of name -> record, for verify.rs to cross-check against.
pub fn records_by_name(marker: &Marker) -> BTreeMap<&str, &FileRecord> {
    marker.files.iter().map(|r| (r.name.as_str(), r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoke_pass() -> SmokeRecord {
        SmokeRecord {
            status: "pass".into(),
            mode: "cpu".into(),
            clip_sha256: "abc".into(),
            num_speakers: 2,
            segments: 9,
            duration_ms: 1234,
            checked_at: "2026-08-31T00:00:00Z".into(),
            error: None,
        }
    }

    fn marker_for(dir: &Path, set: ModelSet, files: Vec<FileRecord>) -> Marker {
        let _ = dir;
        Marker {
            schema: MARKER_SCHEMA,
            generated_at: "2026-08-31T00:00:00Z".into(),
            generated_by: "test".into(),
            model_set: set.as_str().into(),
            exporter_version: EXPORT_RECIPE_VERSION,
            with_gender: false,
            upstream: Upstream {
                pipeline_repo: "pyannote/speaker-diarization-community-1".into(),
                pipeline_revision: Some("deadbeef".into()),
                ..Default::default()
            },
            toolchain: Toolchain::default(),
            speakrs: SpeakrsPin::default(),
            files,
            smoke: smoke_pass(),
        }
    }

    /// Build a temp dir containing `names` as files of the given size, plus a marker.
    fn scratch(names: &[(&str, usize)]) -> tempdir::Dir {
        let d = tempdir::Dir::new();
        for (n, sz) in names {
            std::fs::write(d.path().join(n), vec![7u8; *sz]).unwrap();
        }
        d
    }

    #[test]
    fn absent_marker_is_unverified_and_says_what_to_run() {
        let d = scratch(&[]);
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Unverified);
        let reason = st.reason.unwrap();
        assert!(
            reason.contains("provision-models"),
            "no remedy in: {reason}"
        );
    }

    #[test]
    fn unparseable_marker_is_unverified_not_a_hard_error() {
        let d = scratch(&[]);
        std::fs::write(d.path().join(MARKER_FILE), b"{ not json").unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Unverified);
        assert!(st.reason.unwrap().contains("verify-models"));
    }

    #[test]
    fn good_marker_with_present_files_is_verified() {
        let d = scratch(&[("a.onnx", 10), ("b.npy", 20)]);
        let files = vec![
            FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            },
            FileRecord {
                name: "b.npy".into(),
                bytes: 20,
                sha256: "y".into(),
            },
        ];
        marker_for(d.path(), ModelSet::Fast, files)
            .write(d.path())
            .unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Verified, "reason: {:?}", st.reason);
        assert!(st.reason.is_none());
        assert_eq!(st.pipeline_revision.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn recorded_smoke_failure_is_failed_and_beats_everything_else() {
        let d = scratch(&[("a.onnx", 10)]);
        let mut m = marker_for(
            d.path(),
            ModelSet::Fast,
            vec![FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        );
        m.smoke.status = "fail".into();
        m.smoke.error = Some("stage 3b disagreed".into());
        m.write(d.path()).unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Failed);
        let r = st.reason.unwrap();
        assert!(r.contains("stage 3b disagreed"));
        assert!(r.contains("--force"));
    }

    #[test]
    fn truncated_file_is_failed_and_names_the_file() {
        let d = scratch(&[("a.onnx", 5)]);
        marker_for(
            d.path(),
            ModelSet::Fast,
            vec![FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        )
        .write(d.path())
        .unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Failed);
        let r = st.reason.unwrap();
        assert!(r.contains("a.onnx"), "reason must name the file: {r}");
        assert!(r.contains("expected 10"));
    }

    #[test]
    fn missing_file_is_failed() {
        let d = scratch(&[]);
        marker_for(
            d.path(),
            ModelSet::Fast,
            vec![FileRecord {
                name: "gone.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        )
        .write(d.path())
        .unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Failed);
        assert!(st.reason.unwrap().contains("missing"));
    }

    #[test]
    fn old_recipe_version_is_stale_not_failed() {
        let d = scratch(&[("a.onnx", 10)]);
        let mut m = marker_for(
            d.path(),
            ModelSet::Fast,
            vec![FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        );
        m.exporter_version = EXPORT_RECIPE_VERSION.wrapping_sub(1);
        m.write(d.path()).unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Stale);
        assert!(st.reason.unwrap().contains("recipe version"));
    }

    #[test]
    fn small_set_serving_a_fast_request_is_stale_and_names_the_fix() {
        let d = scratch(&[("a.onnx", 10)]);
        marker_for(
            d.path(),
            ModelSet::Small,
            vec![FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        )
        .write(d.path())
        .unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Stale);
        let r = st.reason.unwrap();
        assert!(r.contains("--set fast"), "must name the fix: {r}");
    }

    #[test]
    fn fast_set_serves_a_small_request() {
        let d = scratch(&[("a.onnx", 10)]);
        marker_for(
            d.path(),
            ModelSet::Fast,
            vec![FileRecord {
                name: "a.onnx".into(),
                bytes: 10,
                sha256: "x".into(),
            }],
        )
        .write(d.path())
        .unwrap();
        assert_eq!(
            evaluate(d.path(), ModelSet::Small).state,
            ModelsState::Verified
        );
    }

    #[test]
    fn newer_schema_is_unverified_rather_than_misread() {
        let d = scratch(&[]);
        let mut m = marker_for(d.path(), ModelSet::Fast, vec![]);
        m.schema = MARKER_SCHEMA + 5;
        m.write(d.path()).unwrap();
        let st = evaluate(d.path(), ModelSet::Fast);
        assert_eq!(st.state, ModelsState::Unverified);
        assert!(st.reason.unwrap().contains("newer than this build"));
    }

    #[test]
    fn marker_round_trips_through_json() {
        let d = scratch(&[]);
        let m = marker_for(d.path(), ModelSet::Fast, vec![]);
        m.write(d.path()).unwrap();
        let back = Marker::read(d.path()).unwrap().unwrap();
        assert_eq!(back.model_set, "fast");
        assert_eq!(back.smoke.num_speakers, 2);
        // The temp file must not survive the atomic write.
        assert!(!d.path().join(format!("{MARKER_FILE}.tmp")).exists());
    }

    #[test]
    fn missing_required_reports_absent_and_empty_files() {
        let d = tempdir::Dir::new();
        // Only one real file, and one zero-length decoy.
        std::fs::write(d.path().join("segmentation-3.0.onnx"), vec![1u8; 4]).unwrap();
        std::fs::write(d.path().join("wespeaker-fbank.onnx"), b"").unwrap();
        let missing = missing_required(d.path(), ModelSet::Small, false);
        assert!(missing.contains(&"plda_lda.npy".to_string()));
        assert!(missing
            .iter()
            .any(|m| m.starts_with("wespeaker-fbank.onnx (empty)")));
        assert!(!missing
            .iter()
            .any(|m| m.starts_with("segmentation-3.0.onnx")));
    }

    /// Minimal scoped temp directory — avoids pulling a dev-dependency in for four tests.
    mod tempdir {
        use std::path::{Path, PathBuf};

        pub struct Dir(PathBuf);

        impl Dir {
            pub fn new() -> Self {
                let mut p = std::env::temp_dir();
                let n = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                p.push(format!(
                    "diar-marker-test-{n}-{:?}",
                    std::thread::current().id()
                ));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
}
