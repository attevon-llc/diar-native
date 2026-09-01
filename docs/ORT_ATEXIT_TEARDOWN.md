# ort logs from a `.fini_array` destructor, which aborts the process at exit

Status: **root-caused, worked around locally** (`crates/diar-core/src/shutdown.rs`), upstream
report drafted below and **not yet filed**. Filing against `pykeio/ort` needs operator approval,
same rule as `avencera/speakrs`.

Local tracking: issue #19. Measurements: `validation/RESULTS.md` §7.51.

## The one-paragraph version

`ort` releases its process-global `Environment` from an ELF `.fini_array` destructor, and
`Environment::drop` emits a `tracing` event. `.fini_array` runs from `_dl_fini`, *after* `main`
has returned and after the Rust runtime has destroyed the main thread's thread-locals. Any
`tracing_subscriber` fmt layer formats through a thread-local buffer, so the event hits
`AccessError`, panics where unwinding cannot start, and the process dies by `SIGABRT` — with all
of its work already finished and written. Anything that binds `ort` and prints a `-DROP` line at
`TRACE` is exposed.

## Mechanism, exactly

In `ort` 2.0.0-rc.12:

- `src/environment.rs:65` — `static G_ENV: Mutex<Option<Arc<Environment>>>`. A **strong** `Arc`,
  deliberately: ORT tolerates `CreateEnv` only once per process, so the environment is kept alive
  after the last `Session` drops.
- `src/environment.rs:75-83` — `release_env_on_exit` is placed in `.fini_array` (and itself in
  `.text.exit`). It is the *only* site that empties `G_ENV`.
- `src/environment.rs:240-245` — `Environment::drop` calls `ReleaseEnv`, then
  `crate::logging::drop!(Environment, ...)`, which expands (`src/logging.rs:84-86`) to
  `trace!(target: "ort::lifetime", "-DROP {} @ {:p}", ...)`.

So at `_dl_fini` the process emits a `tracing` event. The observed backtrace, verbatim:

```
10: std::thread::local::panic_access_error
11: <std::thread::local::LocalKey<core::cell::RefCell<alloc::string::String>>>::with::<
      <tracing_subscriber::fmt::fmt_layer::Layer<...>>::on_event::{closure#0}, ()>
12: <tracing_subscriber::fmt::Subscriber<...> as tracing_core::subscriber::Subscriber>::event
13: <tracing_core::event::Event>::dispatch
14: <ort::environment::Environment as core::ops::drop::Drop>::drop
15: <alloc::sync::Arc<ort::environment::Environment>>::drop_slow
16: ort::environment::release_env_on_exit
17: _dl_call_fini
18: _dl_fini
19: __run_exit_handlers
20: __GI_exit
21: __libc_start_call_main
```

```
thread 'main' panicked at library/std/src/thread/local.rs:428:25:
cannot access a Thread Local Storage value during or after destruction: AccessError
fatal runtime error: failed to initiate panic, error 5, aborting
```

Frame 11 is the point: `tracing_subscriber`'s fmt layer formats into a thread-local
`RefCell<String>`. The main thread's TLS was destroyed by the Rust runtime between frames 21 and
20, before libc reached frame 17.

## Why it looks intermittent and log-level-dependent

Two independent gates, which is why it was first mistaken for a race:

1. **The event must be enabled.** `ort::lifetime` is `TRACE`. At `RUST_LOG=debug` or below, the
   callsite is disabled and never touches TLS. Reproduces 100% at `trace`, 0% at `debug`.
2. **`main` must return normally.** `std::process::exit` does *not* run TLS destructors, so the
   thread-local is still alive when `.fini_array` runs and the event formats fine. This is why a
   subcommand that ends in `std::process::exit` was unaffected while the CLI was not, on the same
   binary and the same log level.

Both gates together produce "sometimes, and only when verbose", which is what the bug looked like
before it was isolated.

## Minimal reproduction — no model, no session, 8 lines

```rust
// ort = "=2.0.0-rc.12"; tracing-subscriber = { version = "0.3", features = ["env-filter"] }
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("trace"))
        .with_writer(std::io::stderr)
        .init();
    let _env = ort::environment::current().expect("env");
}
```

`cargo run --release` → `fatal runtime error: failed to initiate panic, error 5, aborting`, exit
134, deterministic. Replacing the last line's implicit return with `std::process::exit(0)` exits
cleanly — the environment is still released from `.fini_array`, only TLS is still alive.

Measured on x86_64 Linux, glibc 2.39, ort 2.0.0-rc.12, rustc 1.97.1.

## Suggested upstream fixes

In rough order of preference:

1. **Do not log from `Environment::drop`.** Simplest and complete: a destructor that may run from
   `.fini_array` cannot safely call into arbitrary subscriber code. If the line is worth keeping,
   emit it from the paths that drop an environment *during* the program instead.
2. **Make the `.fini_array` path log-free.** Keep `Environment::drop` as is for normal drops, and
   have `release_env_on_exit` release the handle without the trace — e.g. a flag on the
   `Environment`, or `ReleaseEnv` directly rather than going through `Drop`.
3. **Document the hazard** if neither is acceptable, so downstreams know that returning from
   `main` with a subscriber installed is unsound and that `_exit` or `process::exit` is required.

Note that a downstream cannot fix this itself: `G_ENV` is private and holds a strong reference, so
dropping every `Session` does not empty it, and `tracing`'s global dispatcher cannot be
uninstalled once set.

## What we do about it

`diar_core::shutdown::exit` terminates via `_exit`, so libc's exit handlers — and therefore
`release_env_on_exit` — never run. Both binaries route every exit through it. This also sidesteps
the second-order hazard in the same window: `ReleaseEnv` reaching into `libonnxruntime.so` during
`_dl_fini`, where destructor ordering across shared objects is not guaranteed.

If upstream takes fix 1 or 2, `crates/diar-core/src/shutdown.rs` stays worth keeping anyway — it
keeps the whole ordering question out of our exit path — but
`returning_from_main_still_dies_by_signal_upstream_reproducer` in
`crates/diar-core/tests/shutdown_teardown.rs` will start failing and should be deleted, with the
change noted in `RESULTS.md`.
