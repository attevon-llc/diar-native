# `validation/` — what lives here

Every file in this directory is either a **record** (a measurement and its verdict) or a
**harness** (the thing that produced it, kept so the number can be re-derived or a change can
be compared against it). Nothing here is built or shipped; `scripts/` holds the operator
tooling that is.

Two standing rules, both from `CLAUDE.md`:

- **`RESULTS.md` is append-only.** Add a section; never edit or silently re-number one. Retract
  a number explicitly, in a new section.
- **Never re-run a logged test** to reproduce it. Re-run only to compare a *change* against it,
  and only on a quiet machine — `docs/BENCHMARK_PROTOCOL.md` is law.

Before adding a file here, check it is not a one-off. Git history preserves anything deleted
(`git show <sha>:validation/<file>`), so the bar is "would someone reach for this again?", not
"could this ever matter?".

## Why this is not merged into `scripts/`

Asked and declined on 2026-09-01. The split is not stylistic — the two trees have opposite
relationships to the build:

- **`scripts/provision/*.py` is compiled into the product.**
  `crates/diar-core/src/provision/exporter.rs` `include_str!`s all five, so the exporter *is*
  the binary's own bytes. §7.35 records a build failing because a Dockerfile did not `COPY
  scripts/`. Editing a file there changes a shipped artifact.
- **`validation/` is deliberately excluded from the build.** `ort_fusion_probe/` is
  specifically not a workspace member so a root `cargo build` never touches it.

The second reason is harder still: **`RESULTS.md` is append-only and cites both trees by
path** — `validation/score_der.py`, `validation/b*.sh`, `validation/make_corrupt_fixture.py`,
and equally `scripts/compare_model_sets.py`, `scripts/provision/provision.py`. A move would
break citations in a document that by policy can never be corrected. Paths in both trees are
effectively frozen.

The three files that look misfiled are not:

- `score_der.py` stays — RESULTS' front matter names it as *the* scoring instrument behind
  every DER number in the log.
- `make_corrupt_fixture.py` stays — `crates/diar-core/tests/provision_smoke.rs` names it in
  four places as the builder for its `#[ignore]`d fixtures.
- `compare_model_sets.py` stays in `scripts/` — it is the acceptance check you run *after*
  provisioning, it lives next to the `scripts/provision/` recipe it checks, and §7.36 records
  a bugfix to the tool itself.

Structure inside `validation/` is handled by the sections of this file, which costs no path
churn. Prefer that to new subdirectories.

## Records

| file | what it is |
|---|---|
| `RESULTS.md` | The measurement log — every number this project has, with its controls. Append-only. |
| `REPORT.md` | Phase B narrative go/no-go report (2026-08-19): the gate table and the GO verdict on adopting speakrs. Cited by `PLAN.md` §Phase B. Historical — its T2 Triton/TensorRT recommendation was later rolled back (§7.26). |
| `TESTPLAN.md` | The test matrix: engines under test, gates, methodology, reproduction commands. Cited by `crates/diar-core/src/lib.rs`, `PLAN.md` and `docs/UPSTREAM_PRS.md`. |

## Scoring and baselines

| file | what it is | RESULTS |
|---|---|---|
| `score_der.py` | DER scorer (pyannote.metrics, collar 0.25, overlap included, per-file UEM cropping). **Every DER number in `RESULTS.md` came out of this.** Runs inside `opentranscribe-celery-worker` — stage it with `docker cp`. | §1 front matter, all of §2-§4 |
| `run_fork_baseline.py` | Engine A runner: the pinned pyannote fork in the backend image, fork bind-mounted so the code path is production's. Produces the control RTTMs every accuracy claim is measured against. | TESTPLAN §1 row A |
| `run_speakrs.sh` | Engine B runner: one speakrs process per file per run, RTTM from stdout. | §3 ("Runner:") |
| `run_e2e_baseline.sh` | T1 end-to-end leg over the benchmark corpus, 3 runs per file. Takes an `flock` — timed legs are strictly sequential. | §7.5, `docs/BENCHMARK_PROTOCOL.md` |
| `summarize_e2e_baseline.py` | Folds the per-file CSVs from `run_e2e_baseline.sh` into the RESULTS comparison table. | §7.5 |
| `task_census.sh` | Full task census for one file end-to-end, including the enrichment tail. Polls only the cheap file-status endpoint — an earlier version polled `celery inspect` and dominated the number it was reporting. | §7.15 |

## Gate harnesses

`t9a_concurrency.sh` is the concurrency gate named in `CLAUDE.md`. The `b*` scripts are the
deferred issue #5 / #14 benchmarks, each isolating a single variable. (There is no `b3` — that
slot was folded into `b4`.)

| file | question it answers | RESULTS |
|---|---|---|
| `t9a_concurrency.sh` | N concurrent `/diarize` jobs on one shared-session engine: byte-identical output, ≥2× serial throughput, no per-job VRAM scaling. | §7.25 |
| `b1_cuda_feature_cpu_cost.sh` | Does `--features cuda` slow down CPU-mode inference? (single variable: the `ort-sys` prebuilt) | §7.45 |
| `b2_cuda_both_engines.sh` | Does a resident-but-idle CPU engine cost the CUDA path anything? | §7.47 |
| `b4_mixed_device.sh` | Mixed-device concurrency: CUDA `/diarize` alongside CPU `/embed_window` on one server. | §7.44 |
| `b5_gender_fp16_vram.sh` | What does the fp16 gender model actually save in VRAM? (the CHANGELOG's "~500 MiB" was borrowed from a different basis) | §7.46 |
| `b6_gender_opt_level_cost.sh` | What does the aarch64 `Level1` cap on the gender session cost? | §7.48 |

## Model-artifact tooling

These produce **gated** community-1 derivatives: local-only, never committed or attached to a
public PR. The supported provisioning path is `scripts/provision/` (or `diar-server
provision-models`); the two `*_addendum.py` files predate it and are kept because
`scripts/provision/UPSTREAM.md` diffs against one of them and the vendored patch names both as
the producers of the b64 tail artifacts.

| file | what it is | RESULTS |
|---|---|---|
| `make_corrupt_fixture.py` | Builds the weight-corruption fixture: byte-valid ONNX everywhere, one initializer of one graph zeroed. The failure a protobuf parse cannot see, and the reason the smoke test exists. Mirrors the corruption into the b64 byte copy so verifier stage 3d still passes. | §7.35, §7.38 |
| `export_tail_b64_addendum.py` | Exports the missing `wespeaker-voxceleb-resnet34-tail-b64.onnx` — a **real** batch-64 export. Superseded by `scripts/provision/export_tail_b64.py` (step 2d), which `UPSTREAM.md` audits by `diff -u` against this file. | §7.33 |
| `convert_tail_b64_coreml_addendum.py` | The CoreML counterpart (`.mlmodelc`, Apple Silicon only). Deliberately a separate fixed-shape-64 conversion so the shipped b1/b3/b32 tails are not regenerated. Still the only CoreML tail converter in the tree. | §7.33 |

**The trap these sit next to.** Both files end in `-b64` and they need *opposite* treatment:
`wespeaker-voxceleb-resnet34-tail-b64.onnx` must be a real batch-64 export, but
`wespeaker-multimask-tail-b64.onnx` must be a **byte copy of the b32 graph** — speakrs sizes its
multimask runtime buffers for 32, so a genuine batch-64 multimask graph under that name crashes
the worker with "receiver disconnected" (§4.15). The `why` is §4.15, what ships is §7.35 step
2c, and the sha256 enforcement is verifier stage 3d (§7.38). `scripts/provision/provision.py`
does the copy; do not "fix" it into an export.

## Investigations kept as tools

| path | what it is | RESULTS |
|---|---|---|
| `ort_fusion_probe/` | Makes ORT's **load-time** graph rewriting observable — serializes the optimized graph and reports load success per configuration. Built for issue #14 (fp16 gender fails to load on linux/arm64). Not a workspace member, so a root `cargo build` never touches it. Read `docs/ORT_FUSION_FP16_AARCH64.md` before re-running. | §7.40 |
| `asr_timestamp_spike.py` | Word-timestamp accuracy vs hand labels (SequenceMatcher alignment, \|Δstart\| on matches) — faster-whisper vs parakeet. The harness for the table in `docs/ASR_TRITON_NOTES.md`; re-run protocol is to swap the model name in `run_parakeet()`. | — (`docs/ASR_TRITON_NOTES.md`) |

- **`triton_bench.py`** — Triton gRPC latency/throughput bench. **Future work, not current.** TensorRT-in-`ort` was rolled back (RESULTS §7.26), but Triton remains the intended T2 multi-user tier (2.14x throughput at 8 clients) and running TensorRT locally is a separate question. Kept so revisiting means re-running, not re-deriving.

## Removed

Deleted as spent; recover any of them with `git show <sha>:validation/<file>`.

- `export_b64_addendum.py` — exported a **real** batch-64 multimask graph, which is exactly the
  artifact that crashes the worker (§4.15). Superseded by the byte copy in
  `scripts/provision/provision.py` step 2c.
- `ort_cuda_microbench.py` — one-off re-test of a phase-6 claim about the ORT CUDA EP on an
  unfolded segmentation graph. The claim was settled and the graph is folded now.
