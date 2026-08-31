# Contributing to diar-native

Thanks for working on this. This repo has a handful of rules that are not obvious from the
source tree and that will cost you real time if you learn them the hard way. Read this page
before your first change; `CLAUDE.md` is the denser companion for day-to-day work.

---

## 1. The build only works in a container

**The host cannot build this project.** `speakrs` links OpenBLAS via its `openblas-system`
feature, and a stock developer machine does not have it. Building on the host also leaves
`target/` and `Cargo.lock` root-owned once you *do* build in a container, which then breaks the
host build in a new way.

Canonical build:

```bash
docker run --rm \
  -v "$PWD":/build -v /tmp/diar_target:/tmp/target \
  -w /build -e CARGO_TARGET_DIR=/tmp/target \
  diar-bench-builder:latest \
  cargo build --release --features cuda -p diar-server -p diar-cli
```

`diar-bench-builder:latest` is a **local, hand-built image that no Dockerfile in this repo
produces.** If you do not already have it, use the CI environment instead — it is public,
reproducible from a clean clone, and is what `.github/workflows/ci.yml` runs:

```bash
docker run --rm -it \
  -v "$PWD":/build -v /tmp/diar_target_ci:/tmp/target \
  -w /build -e CARGO_TARGET_DIR=/tmp/target \
  ubuntu:24.04 bash -c '
    apt-get update &&
    apt-get install -y --no-install-recommends \
      build-essential pkg-config cmake ca-certificates curl git \
      libssl-dev libclang-dev libopenblas-dev &&
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs |
      sh -s -- -y --default-toolchain stable --component rustfmt clippy &&
    . "$HOME/.cargo/env" &&
    cargo build --release -p diar-server -p diar-cli'
```

That package list is the single source of truth shared with
`.github/actions/setup-build-env/action.yml`. If you change one, change the other.

`cargo check` on the host works for fast iteration (point `CARGO_TARGET_DIR` at a `/tmp` dir),
but never trust it as a green light — only a container build is authoritative.

### Feature flags

| Feature   | Platform      | Notes                                                        |
|-----------|---------------|--------------------------------------------------------------|
| *(none)*  | any           | CPU via OpenBLAS. This is what CI builds and tests.           |
| `cuda`    | linux/amd64   | Purely additive. Needs the CUDA ONNX Runtime distribution — CI does **not** build it. |
| `coreml`  | macOS/aarch64 | Pulls in `objc2`; does not compile anywhere else, and is not reachable through Docker. |

`ort` is **pinned at `=2.0.0-rc.12`**. rc.13 ships a static core that mismatches the 1.24.2
provider libs and fails at session load. Do not bump it, and do not let a bot bump it — the
dependency bot is configured to ignore it.

---

## 2. `vendor/speakrs` is a required build input and is gitignored

`speakrs` is a **path dependency**, so a clean clone will not build until you populate it:

```bash
./scripts/bootstrap_vendor_speakrs.sh
```

That clones `attevon-llc/speakrs` (public, Apache-2.0, our fork of `avencera/speakrs`) at the
pinned commit and detaches. It is idempotent and needs no credentials.

**Fixture models are not in that clone.** They are gitignored upstream. Copy them into
`vendor/speakrs/fixtures/models/` from an existing checkout if you need to run the speakrs
test suite.

### The vendored-patch workflow — this is the part people get wrong

Our changes to speakrs live as the **working-tree diff** of `vendor/speakrs`, not as commits.
Never commit inside `vendor/` (it is fully gitignored and is not a submodule).

After **any** edit under `vendor/speakrs`, regenerate the patch:

```bash
cd vendor/speakrs
git diff HEAD > ../../patches/0001-cuda-performance-patch-set.patch
```

Use `git diff HEAD`, **not** bare `git diff`. Bare `git diff` silently drops anything you have
staged, and the patch will be quietly incomplete. Commit the regenerated patch alongside the
change that motivated it.

Running the speakrs test suite (94 tests):

```bash
docker run --rm -v "$PWD":/build -v /tmp/diar_target:/tmp/target \
  -w /build/vendor/speakrs -e CARGO_TARGET_DIR=/tmp/target -e RUST_MIN_STACK=16777216 \
  diar-bench-builder:latest \
  cargo test --release --no-default-features --features openblas-system,online
```

`--no-default-features` is required. Plain `--features openblas-system` fails with a duplicate
BLAS symbol.

---

## 3. Never commit model weights

`models/`, `models_folded/`, `models_small/`, and every `models*/` glob are **terms-gated
derivatives of pyannote `speaker-diarization-community-1`**. They are gitignored, and they must
stay that way.

- Regenerate them locally with `scripts/provision/` (or the server's `provision-models`
  subcommand). Never commit them, never attach them to a public PR or issue, never bake them
  into a published image.
- 254 MB of gated weights reached this repo's history once already and had to be removed with
  `git filter-repo` (see `validation/RESULTS.md` §7.8). The `.gitignore` glob covers every set
  precisely because `models/` alone missed `models_folded/`.
- `.dockerignore` is an **allowlist**, not a denylist, for the same reason: a denylist fails
  open the moment someone adds a directory.
- The pre-commit `check-added-large-files` hook caps blobs at 512 KB as a backstop. If it fires,
  the answer is essentially never `--no-verify`.
- Your Hugging Face token is yours. Pass it through `HF_TOKEN` in your environment; do not put
  it in a file, a compose YAML, a commit, or a CI workflow.

CI never downloads weights and never needs a token. The tests that require them are all
`#[ignore]`d and gated on `DIAR_TEST_MODELS_DIR` / `DIAR_TEST_SMALL_MODELS_DIR` /
`DIAR_TEST_ZEROED_DIR`; run them locally like this:

```bash
DIAR_TEST_MODELS_DIR=/build/models_folded \
  cargo test --release -p diar-core --test provision_smoke -- --ignored --nocapture
```

---

## 4. Style, hooks and commits

### Pre-commit

```bash
python3 -m venv .venv && .venv/bin/pip install pre-commit
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```

Call the venv binary directly; do not `source` the environment first. The config skips
`vendor/`, `venv/`, `venv-export/`, `target/`, and every `models*/` tree.

### Rust

- `cargo fmt -p diar-core -p diar-cli -p diar-server` — **always scope with `-p`.** A bare
  `cargo fmt --all` reformats `vendor/speakrs` too (cargo resolves path dependencies), which
  silently invalidates `patches/0001-cuda-performance-patch-set.patch`.
- `cargo clippy --workspace --all-targets` — see `clippy.toml` and the lint set in
  `.github/workflows/ci.yml` for what is enforced.
- Comments explain non-obvious **why**, never **what**.

### Python (`scripts/`, `validation/`)

`ruff check` and `ruff format --check`, configured in `.ruff.toml`. These are utility and
validation scripts, not a package — the lint set is deliberately narrow.

### Commits

Conventional commits, imperative mood:

```
<type>(<scope>): <summary>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`, `chore`, `security`, `perf`.

- **Preserve history.** Merge commits (`--no-ff`) or rebase-then-merge. **Never squash-merge** —
  each commit here is a documentation artifact.
- **Never** `--no-verify`, `--no-gpg-sign`, or `-c commit.gpgsign=false`. If a hook fails, fix
  the cause.
- `git add` explicit paths. Review `git diff --cached` before every commit.
- Annotated tags only for releases (`git tag -a`).
- Update `CHANGELOG.md` under `## [Unreleased]` in the same commit as any user-facing change.

---

## 5. Benchmarks and measurements

`docs/BENCHMARK_PROTOCOL.md` is law. The short version:

- One timed leg at a time, on a quiet machine. Check `uptime` and `docker stats` first — this
  box is routinely loaded by sibling work.
- Sample VRAM **during** a run, never after.
- Every speed claim ships with its accuracy check. For a pure-performance change, **prove**
  output identity by diffing raw records; never assert it.
- `validation/RESULTS.md` is **append-only**. Never re-run a test that is already logged except
  to compare a change against it, and never silently edit a recorded number — retract it
  explicitly with a new entry.
- Numbers to beat and corpus paths live in `docs/TEST_CORPORA_AND_BASELINES.md`.

CI deliberately runs **no** benchmarks. Shared runners cannot produce a number anyone should
trust.

---

## 6. Pull requests

- Open PRs against the upstream of the branch you are on, not an assumed `main`.
- CI must be green: format, clippy, build, test, `.dockerignore` guard, and the CPU image build.
- Include the measurement if you are claiming a performance change, and the DER check if you are
  touching anything that could move accuracy.
- Note explicitly if you changed `vendor/speakrs`, and confirm the patch was regenerated.

**`pyannote-audio-fork` is read-only** from here, and changes to `transcribe-app` go through
that repo's own branch and PR flow.

Contributions to upstream `avencera/speakrs` are coordinated separately — see
`docs/UPSTREAM_PRS.md`. Do not open PRs or issues there without explicit operator approval.

---

## 7. Reporting security issues

See [`SECURITY.md`](SECURITY.md). Do not open a public issue for a vulnerability.
