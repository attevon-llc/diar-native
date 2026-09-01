//! Process termination that does not run libc's exit handlers.
//!
//! # Why this module exists (issue #19, RESULTS §7.51)
//!
//! `ort` releases its process-global `Environment` from an ELF **`.fini_array` destructor**
//! (`release_env_on_exit`, ort 2.0.0-rc.12 `src/environment.rs:75-83`), because ONNX Runtime
//! requires `ReleaseEnv` to happen before the C++ static destructors. `G_ENV` holds a *strong*
//! `Arc`, on purpose — ORT only tolerates `CreateEnv` once per process, so the environment is
//! deliberately kept alive after the last `Session` drops.
//!
//! That destructor therefore runs from `_dl_fini`: after `main` has returned, after `exit(3)`,
//! and after the Rust runtime has destroyed the main thread's thread-locals. `Environment::drop`
//! (`src/environment.rs:240-245`) then emits a `tracing` event — `-DROP Environment`, target
//! `ort::lifetime`, level TRACE. If a `tracing_subscriber` fmt layer is installed *and* that
//! target is enabled, formatting the event reaches for the layer's thread-local buffer, gets
//! `AccessError` because TLS is already gone, and panics at a point where unwinding cannot even
//! start:
//!
//! ```text
//! thread 'main' panicked at library/std/src/thread/local.rs:428:25:
//! cannot access a Thread Local Storage value during or after destruction: AccessError
//! fatal runtime error: failed to initiate panic, error 5, aborting
//! ```
//!
//! The process dies by SIGABRT (exit 134) — or, when glibc's heap teardown is further along,
//! by SIGSEGV (exit 139) with `corrupted double-linked list` — with its work already finished
//! and its output already on disk. A caller that trusts the exit code discards a good run.
//!
//! # Why the fix is here and not at the call sites
//!
//! None of the obvious local fixes actually work. `G_ENV` is private, so dropping every
//! `Session` we own does not empty it. `set_global_default` cannot be undone, so the subscriber
//! cannot be uninstalled before returning. Clamping `ort::lifetime` in [`crate::logging`]'s
//! default filter would only hide the tracing flavour, would not survive an explicit
//! `RUST_LOG=trace`, and would do nothing about the `ReleaseEnv`-during-`_dl_fini` flavour.
//!
//! What we can do is not run the exit handlers at all. `_exit` terminates immediately, without
//! `__run_exit_handlers` or `_dl_fini`, so `release_env_on_exit` never fires. Skipping
//! `ReleaseEnv` leaks nothing that outlives the process — the kernel reclaims the address space
//! — and is strictly safer than invoking it at a point where the C++ destructor ordering is
//! already unspecified.
//!
//! This is a workaround for an upstream defect (emitting a `tracing` event from a `.fini_array`
//! destructor is unsound for any subscriber that uses thread-locals). It is reported upstream;
//! see `docs/ORT_ATEXIT_TEARDOWN.md`. When `ort` stops logging from that destructor, this module
//! stays useful anyway: it keeps the ordering hazard out of our exit path entirely.

use std::io::Write;

/// Terminate the process with `code`, skipping libc's exit handlers.
///
/// Flushes stdout and stderr first — `_exit` skips libc's stream flushing along with the
/// handlers, and stdout is a `LineWriter` carrying the CLI's JSONL and the provisioning
/// subcommands' JSON.
pub fn exit(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    terminate(code)
}

/// The `main` epilogue both binaries share: report an error the way `fn main() -> Result<()>`
/// would (`Error:` plus anyhow's full cause chain), then exit 0 or 1 via [`exit`].
///
/// Centralised so the two binaries cannot drift on either half — the error format or the
/// termination mechanism.
pub fn exit_main(result: anyhow::Result<()>) -> ! {
    match result {
        Ok(()) => exit(0),
        Err(err) => {
            eprintln!("Error: {err:?}");
            exit(1)
        }
    }
}

#[cfg(unix)]
fn terminate(code: i32) -> ! {
    // SAFETY: `_exit` never returns, touches no Rust state, and is async-signal-safe.
    unsafe { libc::_exit(code) }
}

/// Non-unix has no `.fini_array`, and we ship no non-unix target — this arm exists so the crate
/// still builds if one is ever added, not because it is known to be safe there.
#[cfg(not(unix))]
fn terminate(code: i32) -> ! {
    std::process::exit(code)
}
