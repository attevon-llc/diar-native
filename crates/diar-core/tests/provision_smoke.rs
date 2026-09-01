//! Integration tests for the provisioning smoke test, against REAL model sets.
//!
//! All `#[ignore]`d and gated on `DIAR_TEST_MODELS_DIR`, because the models are gated
//! community-1 derivatives that are never committed. Run them with:
//!
//! ```bash
//! DIAR_TEST_MODELS_DIR=/build/models_folded \
//!   cargo test --release -p diar-core --test provision_smoke -- --ignored --nocapture
//! ```
//!
//! `DIAR_TEST_SMALL_MODELS_DIR` additionally exercises the small set, and
//! `DIAR_TEST_ZEROED_DIR` supplies the weight-corruption fixture built by
//! `validation/make_corrupt_fixture.py` (which needs python+onnx, so it is prepared outside
//! the test rather than inside it).

use std::path::{Path, PathBuf};

use diar_core::provision::files::ModelSet;
use diar_core::provision::verify::{self, SmokeOptions};
use diar_core::Mode;

fn models_dir() -> Option<PathBuf> {
    std::env::var("DIAR_TEST_MODELS_DIR")
        .ok()
        .map(PathBuf::from)
}

fn clip() -> PathBuf {
    std::env::var("DIAR_TEST_CLIP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/speakrs/fixtures/test.wav")
        })
}

fn opts(dir: &Path, set: ModelSet) -> SmokeOptions {
    SmokeOptions {
        models_dir: dir.to_path_buf(),
        set,
        // Gender is present in both shipped sets; verify.rs treats it as required only
        // when asked, so mirror what is actually on disk.
        with_gender: dir.join("gender-wav2vec2.onnx").exists(),
        // CPU throughout: these tests must run on a busy box and in CI with no GPU.
        mode: Mode::Cpu,
        clip: clip(),
    }
}

/// Build a scratch models dir that hardlinks every file from `src`, so a 484 MB set costs
/// nothing to "copy". The caller then replaces individual files with real copies.
fn linked_dir(src: &Path, tag: &str) -> PathBuf {
    let dst = std::env::temp_dir().join(format!("diar-smoke-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    std::fs::create_dir_all(&dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap().filter_map(Result::ok) {
        if entry.metadata().map(|m| m.is_file()).unwrap_or(false) {
            let to = dst.join(entry.file_name());
            if std::fs::hard_link(entry.path(), &to).is_err() {
                std::fs::copy(entry.path(), &to).unwrap();
            }
        }
    }
    dst
}

/// Replace one hardlinked file with an independent, mutated copy.
fn replace_with_mutated(dir: &Path, name: &str, mutate: impl FnOnce(&mut Vec<u8>)) {
    let path = dir.join(name);
    let mut bytes = std::fs::read(&path).unwrap();
    mutate(&mut bytes);
    // Remove first: the original is a hardlink to the real model set, and writing through
    // it would corrupt models_folded/ itself.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, bytes).unwrap();
}

#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR (gated model artifacts)"]
fn smoke_passes_on_the_fast_set() {
    let Some(dir) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let report = verify::run(&opts(&dir, ModelSet::Fast)).expect("smoke test must pass");
    println!("fast set: {report:#?}");
    assert!(report.num_speakers >= 1);
    assert!(report.segments >= 1);
    assert_eq!(report.stages.len(), 5, "all five stages must run");
}

#[test]
#[ignore = "needs DIAR_TEST_SMALL_MODELS_DIR"]
fn smoke_passes_on_the_small_set() {
    let dir = PathBuf::from(
        std::env::var("DIAR_TEST_SMALL_MODELS_DIR").expect("set DIAR_TEST_SMALL_MODELS_DIR"),
    );
    let report = verify::run(&opts(&dir, ModelSet::Small)).expect("smoke test must pass");
    println!("small set: {report:#?}");
    assert!(report.num_speakers >= 1);
}

/// CORRUPTION (a): flip one byte deep inside a graph, past the protobuf header.
///
/// Stage 1 must fail and must NAME the file — an error that says only "session load failed"
/// sends the operator hunting through 24 files.
#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
fn a_flipped_byte_is_caught_by_stage_1_and_the_file_is_named() {
    let Some(src) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let target = "wespeaker-voxceleb-resnet34-tail.onnx";
    let dir = linked_dir(&src, "flip");
    replace_with_mutated(&dir, target, |b| {
        // Well past any header, and flip enough bytes that the protobuf structure breaks
        // rather than merely perturbing a weight.
        let start = b.len() / 3;
        for byte in b[start..start + 4096].iter_mut() {
            *byte = !*byte;
        }
    });

    let err = verify::run(&opts(&dir, ModelSet::Fast))
        .expect_err("a corrupted graph must not pass verification");
    let msg = format!("{err:#}");
    println!("stage 1 error: {msg}");
    assert!(
        msg.contains(target),
        "error must name the corrupted file: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// CORRUPTION (b): zero an entire initializer so the graph STILL PARSES.
///
/// This is the one that proves the smoke test is more than a protobuf parse. Stage 1 and
/// stage 2 both pass on this file — it loads, and its signature is untouched — so only the
/// cross-path numeric agreement in stage 3b can catch it. Without stage 3, a models
/// directory with silently wrong weights would be declared verified.
///
/// The fixture is built by `validation/make_corrupt_fixture.py` (needs python + onnx).
#[test]
#[ignore = "needs DIAR_TEST_ZEROED_DIR from validation/make_corrupt_fixture.py"]
fn zeroed_weights_still_parse_but_are_caught_by_stage_3() {
    let dir =
        PathBuf::from(std::env::var("DIAR_TEST_ZEROED_DIR").expect("set DIAR_TEST_ZEROED_DIR"));
    let o = opts(&dir, ModelSet::Fast);

    let err = verify::run(&o).expect_err("zeroed weights must not pass verification");
    let msg = format!("{err:#}");
    println!("stage 3 error: {msg}");
    assert!(
        msg.contains("STAGE 3"),
        "a parseable graph with wrong weights must be caught by the NUMERIC stage, not \
         the parse stage — otherwise the numeric stage is not earning its keep. Got: {msg}"
    );
}

#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
fn a_truncated_plda_file_is_caught_by_stage_5() {
    let Some(src) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let dir = linked_dir(&src, "plda");
    replace_with_mutated(&dir, "plda_tr.npy", |b| b.truncate(b.len() - 64));
    let err = verify::run(&opts(&dir, ModelSet::Fast)).expect_err("truncated PLDA must fail");
    let msg = format!("{err:#}");
    println!("stage 5 error: {msg}");
    assert!(msg.contains("plda_tr.npy"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The gender sidecar is never read at run time, so nothing else would notice a relabelling.
#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
fn a_relabelled_gender_meta_is_caught() {
    let Some(src) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    if !src.join("gender-wav2vec2.meta.json").exists() {
        return;
    }
    let dir = linked_dir(&src, "gendermeta");
    replace_with_mutated(&dir, "gender-wav2vec2.meta.json", |b| {
        // Swap the two labels — the exact upstream change that would invert every verdict.
        *b = br#"{"id2label":{"0":"male","1":"female"},"do_normalize":true,"sampling_rate":16000}"#
            .to_vec();
    });
    let err = verify::run(&opts(&dir, ModelSet::Fast)).expect_err("relabelling must fail");
    let msg = format!("{err:#}");
    println!("gender meta error: {msg}");
    assert!(
        msg.contains("invert") || msg.contains("relabelled"),
        "{msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The multimask b64 file must stay a byte copy of b32 (RESULTS §4.15). A future
/// "correction" that exports a real batch-64 graph there would crash the worker in
/// production; stage 3d is what stops it reaching production.
#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
fn a_multimask_b64_that_is_not_a_copy_of_b32_is_rejected() {
    let Some(src) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let dir = linked_dir(&src, "mm64");
    // Substitute a different (but perfectly valid) graph under the b64 name.
    let other = std::fs::read(dir.join("wespeaker-multimask-tail.onnx")).unwrap();
    let path = dir.join("wespeaker-multimask-tail-b64.onnx");
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, other).unwrap();

    let err = verify::run(&opts(&dir, ModelSet::Fast)).expect_err("must reject a non-copy");
    let msg = format!("{err:#}");
    println!("stage 3d error: {msg}");
    assert!(
        msg.contains("byte-for-byte") || msg.contains("STAGE"),
        "{msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// F1: the graph production actually executes on every batched job.
///
/// `wespeaker-multimask-tail-b32.onnx` was verified by NOTHING numeric. Stage 3c checks the
/// *batch-1* multimask; stage 3d only asserts the b64 file is a byte copy of this one. So the
/// copy relationship could be perfect and both files be wrong together.
///
/// The fixture corrupts b32 AND b64 identically, on purpose: corrupting b32 alone would be
/// caught by the sha256 equality in stage 3d, and the test would pass for the wrong reason,
/// proving nothing about numeric coverage. Built by
/// `validation/make_corrupt_fixture.py <src> <dst> wespeaker-multimask-tail-b32.onnx`.
#[test]
#[ignore = "needs DIAR_TEST_ZEROED_MULTIMASK_DIR from validation/make_corrupt_fixture.py"]
fn a_zeroed_multimask_b32_is_caught_numerically_not_by_the_hash_check() {
    let dir = PathBuf::from(
        std::env::var("DIAR_TEST_ZEROED_MULTIMASK_DIR")
            .expect("set DIAR_TEST_ZEROED_MULTIMASK_DIR"),
    );
    // Premise: the two multimask files must still agree, or stage 3d catches this first and
    // the numeric stage is never reached.
    let a = std::fs::read(dir.join("wespeaker-multimask-tail-b32.onnx")).unwrap();
    let b = std::fs::read(dir.join("wespeaker-multimask-tail-b64.onnx")).unwrap();
    assert_eq!(a, b, "fixture must keep b64 a byte copy of b32");

    let err = verify::run(&opts(&dir, ModelSet::Fast))
        .expect_err("a wrong multimask graph must not pass verification");
    let msg = format!("{err:#}");
    println!("multimask error: {msg}");
    assert!(
        msg.contains("STAGE 3e"),
        "the batch-32 multimask is the production hot path and must be caught by the NUMERIC \
         stage that was added for it, not incidentally by something else. Got: {msg}"
    );
}

/// F2: `verify-models` reported success on a directory with NO marker.
///
/// `Marker::read` returns `Ok(None)` when the file is absent, so the drift loop was skipped
/// entirely and `drift` stayed empty — indistinguishable from "every hash matched". The CLI
/// printed "OK: … verified." and exited 0 having compared zero bytes against zero recorded
/// hashes, which is the exact scenario (a `/models` directory of unknown provenance) the
/// command exists for.
#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
fn deep_verification_distinguishes_a_missing_marker_from_a_clean_one() {
    let Some(src) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let with_marker = PathBuf::from(
        std::env::var("DIAR_TEST_MARKED_MODELS_DIR")
            .expect("set DIAR_TEST_MARKED_MODELS_DIR to a provisioned dir WITH a marker"),
    );

    let marked = verify::verify_deep(&opts(&with_marker, ModelSet::Fast)).expect("must pass");
    println!("marked: hashed={} drift={:?}", marked.hashed, marked.drift);
    assert!(marked.marker_present);
    assert!(marked.hashed > 0, "a marker must cause files to be hashed");
    assert!(marked.drift.is_empty(), "{:?}", marked.drift);
    assert!(marked.fully_verified());

    // The same directory with the marker removed must NOT report as verified.
    let stripped = linked_dir(&with_marker, "nomarker");
    std::fs::remove_file(stripped.join("diar-provision.json")).unwrap();
    let bare = verify::verify_deep(&opts(&stripped, ModelSet::Fast)).expect("smoke still passes");
    println!("stripped: hashed={} drift={:?}", bare.hashed, bare.drift);
    assert!(!bare.marker_present, "there is no marker");
    assert_eq!(bare.hashed, 0, "nothing could be compared");
    assert!(
        !bare.fully_verified(),
        "a directory with no marker has NOT been verified against anything — reporting it as \
         verified is the fail-open this tier exists to prevent"
    );
    let _ = std::fs::remove_dir_all(&stripped);

    // And an unmarked directory is not simply 'clean': drift is empty in BOTH cases, so the
    // marker's presence is what carries the distinction.
    assert_eq!(bare.drift.len(), marked.drift.len());
    let _ = std::fs::remove_dir_all(src.join("__never__"));
}

/// C1: an execution device this build/machine cannot use is an ENVIRONMENT failure, and must
/// never be recorded against the models.
///
/// Reproduced without needing a GPU-less host: on a build without the `coreml` feature,
/// `Mode::CoreMl` cannot validate, exactly as `Mode::Cuda` cannot on a box with no visible
/// GPU. The classifier proves the models are fine by loading the same directory on the CPU
/// execution provider, so it is the code path, not the specific device, that is under test.
#[test]
#[ignore = "needs DIAR_TEST_MODELS_DIR"]
#[cfg(not(feature = "coreml"))]
fn an_unusable_device_is_reported_as_an_environment_failure_not_a_model_failure() {
    let Some(dir) = models_dir() else {
        panic!("set DIAR_TEST_MODELS_DIR");
    };
    let mut o = opts(&dir, ModelSet::Fast);
    o.mode = Mode::CoreMl;

    let err = verify::run(&o).expect_err("a device this build cannot serve must fail");
    let msg = format!("{err:#}");
    println!("device error: {msg}");
    assert!(
        verify::is_device_unavailable(&err),
        "an unusable device must be classified as an environment problem — otherwise \
         provision-models writes `smoke.status: fail` into the models directory and every \
         later server start exits non-zero about files that are perfectly good. Got: {msg}"
    );
    assert!(
        !diar_core::provision::should_record_failure(&err),
        "and therefore it must never be recorded as a smoke failure"
    );
    assert!(msg.contains("NOT implicated"), "{msg}");
}
