<!--
Conventional commits, imperative mood: <type>(<scope>): <summary>
Types: feat fix docs style refactor test build chore security perf

Never squash-merge. Merge commits (--no-ff) or rebase-then-merge — each commit here is a
documentation artifact.
-->

## What and why

<!-- What changes, and what problem it solves. Link the issue or the RESULTS.md section. -->

## Checklist

- [ ] Built **in a container** — the host cannot build this project (no OpenBLAS).
      `cargo check` on the host is iteration, not evidence.
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, if this is user-facing.

### If you touched `vendor/speakrs`

- [ ] Regenerated the patch:
      `cd vendor/speakrs && git diff HEAD > ../../patches/0001-cuda-performance-patch-set.patch`
      (`git diff HEAD`, **not** bare `git diff` — bare silently drops staged changes)
- [ ] The regenerated patch is committed alongside the change.
- [ ] Nothing was committed *inside* `vendor/` (it is gitignored and is not a submodule).
- [ ] speakrs' own tests still pass (94):
      `cargo test --release --no-default-features --features openblas-system,online`

### If this changes performance

<!-- docs/BENCHMARK_PROTOCOL.md is law. Delete this section if it does not apply. -->

- [ ] Measured on a quiet machine, one timed leg at a time (`uptime` + `docker stats` checked).
- [ ] VRAM sampled **during** the run, not after.
- [ ] The accuracy check ships with the speed claim — a number alone is not a result.
- [ ] For a pure-performance change: output identity **proved** by diffing raw records, not
      asserted.
- [ ] Appended to `validation/RESULTS.md` **with its controls**. That file is append-only; a
      superseded number is retracted explicitly, never edited away.

### Always

- [ ] No model weights, `.onnx`, `.plan` or other gated artifacts are in this diff. They are
      terms-gated pyannote community-1 derivatives and may not be redistributed.
- [ ] No secrets, tokens or private paths. `HF_TOKEN` comes from the environment, never a file.
- [ ] Did not bump `ort` (pinned `=2.0.0-rc.12`; rc.13 builds fine and then fails at session
      load, which CI cannot catch because CI has no models).

## Notes for the reviewer

<!-- Anything surprising, deliberately out of scope, or that you want a second opinion on. -->
