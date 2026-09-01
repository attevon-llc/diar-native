//! Regression test for issue #19 — `diar-cli` died by signal at teardown *after* writing its
//! output, so a caller checking the exit code discarded a run whose RTTM was complete on disk.
//!
//! The fault needs three things at once, which is why nothing existing caught it: an ORT
//! environment alive at process exit, a `tracing_subscriber` fmt layer installed, and
//! `ort::lifetime` enabled at TRACE. `ort` drops its process-global `Environment` from an ELF
//! `.fini_array` destructor and *logs* that drop; when `main` returns normally the Rust runtime
//! has already destroyed the main thread's thread-locals by then, so formatting the event hits
//! `AccessError` and the process aborts. See `diar_core::shutdown` for the full mechanism.
//!
//! Everything here drives the `diar-teardown-fixture` binary rather than running in-process:
//! the fault is about how `main` is left, and libtest terminates via `std::process::exit`,
//! which skips the TLS teardown and cannot arm it. On Unix a signal death gives
//! `status.code() == None`, which is precisely the distinction under test — "exited with a
//! code" versus "was killed".
//!
//! Deliberately model-free: the hazard is `ort` + `tracing-subscriber`, not diarization, so the
//! fixture creates a bare ORT environment and loads no graph. This runs in CI with none of the
//! gated model artifacts present.

use std::process::{Command, Output};

const FIXTURE: &str = env!("CARGO_BIN_EXE_diar-teardown-fixture");

/// Arbitrary, and deliberately not one of the provisioning contract codes — the point is only
/// that whatever we pass comes back out unchanged.
const ARBITRARY_CODE: i32 = 42;

/// `RUST_LOG=trace` is what arms the fault: it is the only level at which `ort::lifetime`
/// reaches the fmt layer and makes it touch its thread-local buffer.
fn fixture(args: &[&str]) -> Output {
    Command::new(FIXTURE)
        .args(args)
        .env("RUST_LOG", "trace")
        .env("DIAR_LOG_FORMAT", "text")
        .output()
        .expect("running the teardown fixture")
}

#[test]
fn exit_preserves_the_code_under_trace_logging() {
    let out = fixture(&["exit", &ARBITRARY_CODE.to_string()]);
    assert_eq!(
        out.status.code(),
        Some(ARBITRARY_CODE),
        "expected a clean exit {ARBITRARY_CODE}; `None` means death by signal in ort's \
         `.fini_array` teardown (issue #19).\nstderr tail:\n{}",
        tail(&out)
    );
}

/// The success path both binaries take. This is the acceptance criterion from the issue stated
/// as an assertion: a successful run exits 0, at trace level, every time.
#[test]
fn exit_main_reports_success_as_zero_under_trace_logging() {
    let out = fixture(&["ok"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a successful run must exit 0 — a caller trusting the exit code is the whole point of \
         issue #19.\nstderr tail:\n{}",
        tail(&out)
    );
}

/// The failure path: exit 1, with anyhow's `Error: <chain>` still on stderr. `_exit` skips
/// libc's stream flushing along with the exit handlers, so this also pins that
/// `shutdown::exit` flushes before it terminates — without that, the error message vanishes.
#[test]
fn exit_main_reports_failure_as_one_and_still_prints_the_error() {
    let out = fixture(&["err"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failed run must exit 1, not die by signal.\nstderr tail:\n{}",
        tail(&out)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Error: outer context"),
        "anyhow's `Error: <context>` line must survive `_exit`; got:\n{stderr}"
    );
    assert!(
        stderr.contains("underlying cause"),
        "the full cause chain must survive `_exit`; got:\n{stderr}"
    );
}

/// Demonstrates the defect itself: the same fixture, differing only in that `main` returns.
///
/// `#[ignore]` on purpose. This asserts the presence of an UPSTREAM bug, so it would start
/// failing the day `ort` stops logging from its `.fini_array` destructor — which is the outcome
/// we want, and not something CI should block on. Kept as an executable reproducer for the
/// upstream report and for re-checking after any `ort` change:
///
/// ```text
/// cargo test -p diar-core --test shutdown_teardown -- --ignored --nocapture
/// ```
///
/// Measured 2026-09-01 (ort 2.0.0-rc.12, glibc 2.39, x86_64): dies by SIGABRT, 5 runs of 5.
/// Recorded in `validation/RESULTS.md` §7.51.
#[test]
#[ignore = "asserts an upstream ort defect; documents the reproducer rather than gating CI"]
fn returning_from_main_still_dies_by_signal_upstream_reproducer() {
    let out = fixture(&["return"]);
    assert_eq!(
        out.status.code(),
        None,
        "expected death by signal from ort's `.fini_array` teardown. A clean exit here means \
         upstream fixed it — delete this test and note it in RESULTS.\nstderr tail:\n{}",
        tail(&out)
    );
}

fn tail(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let lines: Vec<&str> = stderr.lines().collect();
    lines[lines.len().saturating_sub(8)..].join("\n")
}
