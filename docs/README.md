# diar-native documentation

Index of every document here, what it answers, and whether it is current. Start at the
[project README](../README.md) if you just want to run the thing.

Every fact has **one** home. If two pages seem to say the same thing, the one listed as the owner
below is authoritative and the other should be linking, not restating — please fix it if you find
otherwise.

## Start here

| document | answers | owns |
|---|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | "How do I run this properly?" | Compose files, published images and digests, the platform matrix, ports and volumes, the container user, **exit codes**, the arm64 and ubuntu-24.04 caveats |
| [CONFIGURATION.md](CONFIGURATION.md) | "What can I set, and what does it do?" | **Every environment variable** the binary reads — the authoritative list, checked in both directions |
| [API.md](API.md) | "What do I send, and what comes back?" | The four routes, request/response schemas, response headers, device selection |
| [PROVISIONING.md](PROVISIONING.md) | "How do I get the models, and can I trust them?" | The HF token, the export, the provenance marker, the five-stage smoke test, and **what verification does not prove** (issue #21) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | "Why isn't it working?" | The failure modes, in order of frequency |

## Going deeper

| document | answers | status |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | "How is it built, and why this way?" | Current. Owns the speakrs relationship, the pipeline, crate layout, the CPU+CUDA superset, and why production consumes the binary rather than the image |
| [PERFORMANCE.md](PERFORMANCE.md) | "How fast and how accurate, under what conditions?" | Current. A **bridge** to [`validation/RESULTS.md`](../validation/RESULTS.md), which is the append-only record and the real source of truth |
| [DEVELOPMENT.md](DEVELOPMENT.md) | "How do I build, test and release it?" | Current. Companion to [`CONTRIBUTING.md`](../CONTRIBUTING.md), which owns the PR flow and style rules |
| [INSTALL_NATIVE.md](INSTALL_NATIVE.md) | "How do I flip OpenTranscribe onto the native engine?" | Current. **Only** the `transcribe-app` procedure — generic deployment lives in DEPLOYMENT.md |
| [DETAILED_SPECS.md](DETAILED_SPECS.md) | "What was the design of each numbered task, including the ones that failed?" | Current, with caveats. Valuable for its **negative results** written as design guidance — read §S-T9a (shared sessions) and §S-T11 (why `SPEAKRS_TRT` is dead). Its model/effort routing matrix is dead weight |

## Not in this directory

| where | what |
|---|---|
| [`validation/README.md`](../validation/README.md) | Index of the test and benchmark harnesses, by purpose and RESULTS section — including the ones that were removed, so a stale pointer lands on an explanation rather than a 404 |
| [`validation/RESULTS.md`](../validation/RESULTS.md) | **Every measurement ever taken.** Append-only: never re-run a logged test, and retract a number explicitly rather than editing it |
| [`validation/TESTPLAN.md`](../validation/TESTPLAN.md) | The test matrix and the G1-G5 gates |
| [`../CHANGELOG.md`](../CHANGELOG.md) · [`../CONTRIBUTING.md`](../CONTRIBUTING.md) · [`../SECURITY.md`](../SECURITY.md) · [`../MODELS_SETS.md`](../MODELS_SETS.md) | Release history · contributing · vulnerability reporting · fast vs small model sets |

## Deep dives — read before touching the thing they describe

| document | answers |
|---|---|
| [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) | "How do I measure a change so the number is trustworthy?" **This is law** for anything that lands in RESULTS |
| [TEST_CORPORA_AND_BASELINES.md](TEST_CORPORA_AND_BASELINES.md) | "Where is the audio and the reference, and what number must I beat?" |
| [VRAM_AND_TIERS.md](VRAM_AND_TIERS.md) | "What holds GPU memory, why, and what fits on a 4 / 8 / 12 GB card?" |
| [ORT_FUSION_FP16_AARCH64.md](ORT_FUSION_FP16_AARCH64.md) | "Why does the fp16 gender model fail to load on linux/arm64?" Read before touching precision, session options or anything fp16 |
| [ORT_ATEXIT_TEARDOWN.md](ORT_ATEXIT_TEARDOWN.md) | "Why did the process abort *after* writing its results?" Read before changing either binary's `main` |

## Upstream contribution

| document | status |
|---|---|
| [UPSTREAM_PRS.md](UPSTREAM_PRS.md) | The speakrs contribution queue: what we are sending upstream, in what order, with what evidence. **Current for the PR bodies.** Its trailing "status at handoff" and "awaiting approval to push" sections are stale — the branches were pushed and the PRs are open |
| [upstream_drafts_fbank_pool.md](upstream_drafts_fbank_pool.md) | Draft text for the fbank-pool upstream report. **Written, deliberately unfiled** — do not file it without operator approval. Note it describes a defect that still exists *upstream* but was fixed here (RESULTS §7.50, issue #3) |
| [upstream_drafts_ort_fusion.md](upstream_drafts_ort_fusion.md) | Draft text for the ORT fusion upstream report. **Written, deliberately unfiled** — same constraint |

## Out of scope, kept for its measurements

| document | status |
|---|---|
| [ASR_TRITON_NOTES.md](ASR_TRITON_NOTES.md) | A measured ASR spike (faster-whisper vs NVIDIA Parakeet TDT: word-timestamp accuracy, timing bias and a constant-offset calibration that improves *both* engines). **diar-native does not own ASR** — this belongs in OpenTranscribe. It is kept only because the numbers are recorded nowhere else; it should be moved to that repo and deleted from here |

---

## Not here any more

These were deleted rather than archived, because git history already preserves them
(`git log --diff-filter=D -- docs/` to find the commit, `git show <sha>^:docs/FOO.md` to read
one). Each was a task list, a completed handoff, or a draft that has since been filed:
`EXECUTION_TASKS.md`, `SPEEDUP_ROADMAP.md`, `HANDOFF_T9A_SHARED_SESSIONS.md`,
`HANDOFF_DIARIZATION_SPEED.md`, `ISSUE_DRAFTS.md`, `pr_drafts.md`,
`BACKEND_EMBEDDED_OPTION.md`, `E2E_PIPELINE_MAP.md`, `NATIVE_INFERENCE_NOTES.md`,
`RUST_SERVICES_PLAN.md`. Their unique findings were folded into the pages above first.

`QUICKSTART.md` is gone too: the README's one-command install plus
[DEPLOYMENT.md](DEPLOYMENT.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) replace it, and two
quickstarts drift apart.
