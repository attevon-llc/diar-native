# Upstream draft — `RuntimeConfig::fbank_pool` (diar-native issue #3)

**STATUS: DRAFT. NOTHING HAS BEEN FILED.** Anything outward-facing against `avencera/speakrs`
needs the operator's explicit approval, per the ground rules in `CLAUDE.md`.

This is a small API addition that stands on its own merits, independent of the seven prepared
PR branches are listed in `docs/UPSTREAM_PRS.md`. It belongs on top of the **fbank session-pool fan-out** PR
(queue item 2, `docs/UPSTREAM_PRS.md` §fbank pool), since it modifies code that PR introduces:
if the pool PR is not accepted, this one is moot; if it is, this should be folded into it rather
than filed separately, because it changes the very lines that PR adds.

Evidence: `validation/RESULTS.md` §7.50.

---

## Draft — avencera/speakrs

**Title:** Let `RuntimeConfig` carry the fbank pool size, so embedders don't have to `setenv`

**Type:** enhancement (small, backwards-compatible)

### The problem

The fbank session pool is sized from `SPEAKRS_FBANK_POOL`, read inside
`LoadedSessions::load`. For a CLI that is fine. For a **library embedder** it is not, because the
only way to ask for a non-default pool is to mutate the process environment:

```rust
std::env::set_var("SPEAKRS_FBANK_POOL", pool.to_string());
let emb = EmbeddingModel::with_mode_and_config(path, mode, &RuntimeConfig::default())?;
```

`setenv` and `getenv` are not thread-safe in glibc — a concurrent `getenv` on another thread can
observe a freed pointer. Rust 2024 marks `std::env::set_var` `unsafe` for precisely this reason,
and speakrs itself is edition 2024, so it is already living by that rule internally.

The consequence for an embedder is not theoretical, and it is worse than "you must be careful at
startup": it means **models cannot be loaded lazily or concurrently at all**. Our server loads
one model set per execution device; loading device *N+1* runs `setenv` while device *N*'s ORT
intra-op thread pools are alive. And because speakrs also reads `SPEAKRS_ARENA_SHRINK`
(`inference.rs`) and `SPEAKRS_AHC_THREADS` (`clustering/ahc.rs`) **on the request path** rather
than only at load, any `setenv` performed while requests are in flight races those reads too.

So a single env-only knob effectively constrains the whole embedding host to "load everything
serially, before any inference starts, forever". We had to reject an otherwise obvious
optimization (lazy model loading, worth ~620 MB RSS per idle CPU model set) for this reason
alone.

### The change

Add one optional field to the existing `RuntimeConfig`, which already flows into
`LoadedSessions::load`:

```rust
pub struct RuntimeConfig {
    pub chunk_emb_workers: usize,
    /// Size of the CPU fbank session pool used for parallel per-chunk fbank.
    ///
    /// `None` (the default) auto-sizes: the `SPEAKRS_FBANK_POOL` environment override when it
    /// parses, otherwise one session per four cores clamped to `1..=8`. `Some(0)` disables the
    /// pool and falls back to the single fbank session.
    pub fbank_pool: Option<usize>,
    // ...
}
```

and consume it where the env var is read today:

```rust
let pool_size = config.fbank_pool.unwrap_or_else(auto_fbank_pool_size);
```

where `auto_fbank_pool_size()` is the existing env-then-`available_parallelism()/4` logic, moved
into a named function.

### Why this shape

- **Backwards compatible by construction.** `fbank_pool: None` is the `Default`, and `None`
  takes exactly the old code path — env var first, then the core-count heuristic. No existing
  consumer changes behaviour, and `SPEAKRS_FBANK_POOL` keeps working for CLI users.
- **Config beats environment, which is the usual precedence** and the one that makes an explicit
  caller request actually take effect. (The bug this fixed on our side was the inverse: our
  wrapper's `set_var` silently overwrote the operator's value.)
- **It reuses the plumbing that already exists.** `RuntimeConfig` is already threaded to
  `LoadedSessions::load` for `chunk_emb_compute_units`; no new parameter, no signature change,
  no new public type.
- `Some(0)` is meaningful rather than a footgun: the downstream code already treats an empty pool
  as "use the single fbank session", so `0` is a natural "disable the fan-out" and is documented
  as such.

### Also in the diff

- Removes the now-dead `#[cfg(not(feature = "coreml"))] let _ = config;`, since `config` is
  used on both cfg paths once the pool reads it.
- Adds `tracing::debug!(fbank_pool = pool_size, "fbank session pool")`. There was previously no
  way to observe the resolved pool size, which is exactly how a downstream bug (an override that
  was being silently discarded) went unnoticed for months.

### Testing

- speakrs suite green: 96 passed, 0 failed
  (`--no-default-features --features openblas-system,online`, `RUST_MIN_STACK=16777216`).
- Behaviour verified end to end on both legs of the default (unset env → pool 8 on a 48-core
  CUDA host, pool 1 in CPU mode) and with an explicit override, before and after the change.
  Defaults are unchanged; the explicit path is new. Details in `validation/RESULTS.md` §7.50.

### Open question for the maintainer

If you would prefer the environment to keep winning over an explicit `RuntimeConfig` value (i.e.
env checked first, config as fallback), that is a one-line flip and I am happy to do it — but I
think config-wins is the less surprising precedence for a library, and it is the one that makes
the field useful to an embedder who wants to be sure what they asked for is what they got.
