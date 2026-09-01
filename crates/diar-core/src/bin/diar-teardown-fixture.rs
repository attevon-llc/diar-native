//! Test fixture for issue #19. NOT a shipped artifact — both Dockerfiles build only
//! `-p diar-server -p diar-cli`, so this target is never in an image.
//!
//! `tests/shutdown_teardown.rs` needs to observe how a process that looks like our binaries
//! actually *dies*, and the test harness itself cannot stand in for one: libtest terminates via
//! `std::process::exit`, which skips the Rust runtime's thread-local teardown and therefore
//! never arms the fault. Only a real `main` that returns normally does.
//!
//! Each mode arms the identical hazard — a live ORT environment plus an installed subscriber
//! with `ort::lifetime` at TRACE — and then differs only in how it leaves `main`.
//!
//!   `exit <code>`   terminate via [`diar_core::shutdown::exit`] (what the binaries now do)
//!   `ok`            hand `Ok(())` to [`diar_core::shutdown::exit_main`]
//!   `err`           hand an `Err` with a cause chain to `exit_main`
//!   `return`        return from `main` — the PRE-FIX shape, expected to die by signal

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();

    // Exactly what diar-cli does first, so the subscriber is the real one under the real
    // RUST_LOG policy rather than a hand-rolled stand-in.
    diar_core::logging::init_stderr();
    // Populates ort's private `G_ENV`, whose only release site is the `.fini_array` destructor.
    // Dropped immediately: `G_ENV` holds its own strong reference, so the hazard does not
    // depend on the caller still holding a handle. No session, and so no model file, is needed.
    drop(ort::environment::current().expect("creating an ORT environment"));

    match mode.as_str() {
        "return" => {}
        "ok" => diar_core::shutdown::exit_main(Ok(())),
        "err" => diar_core::shutdown::exit_main(Err(
            anyhow::anyhow!("underlying cause").context("outer context")
        )),
        "exit" => {
            let code = std::env::args()
                .nth(2)
                .and_then(|c| c.parse().ok())
                .expect("`exit` needs a numeric code");
            diar_core::shutdown::exit(code)
        }
        other => {
            eprintln!("unknown fixture mode {other:?}");
            diar_core::shutdown::exit(2)
        }
    }
}
