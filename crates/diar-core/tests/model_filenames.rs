//! The gender model's filename is load-bearing in FOUR places, in two languages. This pins them
//! together.
//!
//! `gender-wav2vec2.onnx` is not just a name. Two behaviours are scoped to that exact string:
//!
//!   1. **The aarch64 fp16 optimization cap** (`ort_compat::apply_workarounds`). The fp16 gender
//!      graph does not LOAD on linux/arm64 at ORT's default optimization level — see
//!      `docs/ORT_FUSION_FP16_AARCH64.md` and issue #14. The cap is applied by matching the
//!      filename, and only the filename.
//!   2. **The fp16 load gate** (`provision::verify` stage 1), which reports the gender model as
//!      the one file allowed to need that workaround.
//!
//! Both are filename-scoped, so a rename that misses a copy does not break a build or fail a
//! test — it un-scopes both behaviours, on arm64 only, and the symptom is a server that starts
//! normally, answers 200, and quietly returns no speaker genders. There is no accuracy gate that
//! can catch it, because nothing numeric is produced.
//!
//! Rust used to hold three independent copies of the literal; those are now one constant plus
//! re-exports, so THAT class of divergence is structurally impossible. What no compiler can
//! catch is the remaining seam: `scripts/provision/export_gender.py` is what actually WRITES the
//! file, and it is Python. If the exporter renames its output, every Rust site stays perfectly
//! consistent with itself and perfectly wrong about what is on disk. That seam is what the
//! `exporter_*` tests below cover.
//!
//! Deliberately NOT model-dependent: these tests read source files, never weights, so they run
//! in CI where the terms-gated artifacts must never appear.

use std::path::{Path, PathBuf};

use diar_core::gender::GENDER_MODEL_FILE;
use diar_core::provision::files::{GENDER_META, GENDER_MODEL};

/// Repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Read a `NAME = "value"` assignment out of a Python module.
///
/// A regex-free parser on purpose: it must fail loudly if the assignment is gone or reshaped
/// (e.g. into an f-string or a `Path` join), because a test that silently finds nothing is worse
/// than no test. Returns `None` only if the name is absent, which every caller treats as fatal.
fn python_str_const(source: &str, name: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let (lhs, rhs) = line.split_once('=')?;
        if lhs.trim() != name {
            return None;
        }
        // Strip a trailing comment, then require a simple single- or double-quoted literal.
        let rhs = rhs.split('#').next()?.trim();
        let quote = rhs.chars().next()?;
        if quote != '"' && quote != '\'' {
            return None;
        }
        rhs.strip_prefix(quote)?
            .split(quote)
            .next()
            .map(str::to_owned)
    })
}

fn read_repo_file(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The value itself, written out once.
///
/// This is the ONLY place in the Rust tree, outside `gender.rs` and one `ort_compat` unit test,
/// where the literal appears. Changing the constant without changing this line fails here, which
/// is the point: the rename must be deliberate, and the failure message says what else to update.
#[test]
fn the_canonical_filename_is_what_the_arm64_workarounds_key_on() {
    assert_eq!(
        GENDER_MODEL_FILE, "gender-wav2vec2.onnx",
        "The gender model filename changed. It is not just a name: the aarch64 fp16 \
         optimization cap and the verify.rs fp16 load gate are both scoped to this exact \
         string, and arm64 is the only place a mismatch shows up (silently, as missing \
         genders). If this rename is intended, update scripts/provision/export_gender.py, \
         scripts/provision/provision.py, the literal in ort_compat's unit test, and this line \
         together — then re-provision, because existing models directories still hold the old \
         filename."
    );
}

/// The re-exports are re-exports, not copies that happen to agree today.
///
/// Trivially true while `provision::files::GENDER_MODEL` is a `pub use`. It stops being trivial
/// the moment someone "simplifies" it back into its own `const`, which is exactly the state this
/// change removed.
#[test]
fn every_rust_site_resolves_to_the_one_constant() {
    assert_eq!(
        GENDER_MODEL, GENDER_MODEL_FILE,
        "provision::files::GENDER_MODEL has diverged from gender::GENDER_MODEL_FILE. It is \
         meant to be a `pub use` re-export of it, not a second definition."
    );
    assert_eq!(
        GENDER_META,
        GENDER_MODEL_FILE.replace(".onnx", ".meta.json"),
        "the meta sidecar no longer sits beside the model under a matching stem"
    );
}

/// The Python exporter writes the file; Rust reads it. Nothing but this test connects them.
#[test]
fn exporter_writes_the_filename_rust_looks_for() {
    let src = read_repo_file("scripts/provision/export_gender.py");

    let model = python_str_const(&src, "MODEL_FILE").expect(
        "MODEL_FILE is no longer a plain string literal in scripts/provision/export_gender.py. \
         Rust cannot follow it any more; keep it a literal so this cross-language check can \
         stay honest.",
    );
    assert_eq!(
        model, GENDER_MODEL_FILE,
        "the exporter writes {model:?} but Rust looks for {GENDER_MODEL_FILE:?} — the aarch64 \
         fp16 cap and the verify.rs load gate would both silently stop applying"
    );

    let meta = python_str_const(&src, "META_FILE").expect(
        "META_FILE is no longer a plain string literal in scripts/provision/export_gender.py",
    );
    assert_eq!(
        meta, GENDER_META,
        "the exporter writes the gender metadata sidecar as {meta:?}, but provisioning verifies \
         {GENDER_META:?}"
    );
}

/// `provision.py` deletes fast-only files for `--set small`. The gender meta sidecar is on that
/// list, and `files.rs` agrees it is fast-only — by name, in the other language.
#[test]
fn the_small_set_deletes_the_gender_meta_by_the_same_name() {
    let src = read_repo_file("scripts/provision/provision.py");
    let fast_only = src
        .split_once("FAST_ONLY = (")
        .expect("FAST_ONLY tuple is gone from scripts/provision/provision.py")
        .1
        .split_once(')')
        .expect("FAST_ONLY tuple is unterminated")
        .0;
    assert!(
        fast_only.contains(&format!("\"{GENDER_META}\"")),
        "provision.py's FAST_ONLY no longer names {GENDER_META:?}, so `--set small` would leave \
         it behind while files.rs still treats it as fast-only. FAST_ONLY was:\n{fast_only}"
    );
}
