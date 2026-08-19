# Agent Kickoff Prompts — start any execution session from here

Paste the master prompt (adjusted per session goal) into a fresh Claude Code session with the
model/effort from DETAILED_SPECS' matrix. Each prompt binds the agent to the plan documents so
no context is lost between sessions.

---

## MASTER PROMPT (template)

> Work in /mnt/nvm/repos/diar-native (the project of record for OpenTranscribe's native
> diarization/inference program). Before doing ANYTHING, read in this order:
> 1. PLAN.md (north star, locked decisions, accepted sequence)
> 2. docs/EXECUTION_TASKS.md (task list T1-T13 with gates)
> 3. docs/DETAILED_SPECS.md (implementation specs S-T2/T4/T5b/T9/T11 + model matrix)
> 4. docs/E2E_PIPELINE_MAP.md (file-anchored pipeline facts + levers L1-L10)
> 5. validation/RESULTS.md §headers only (what is already measured — NEVER re-run a logged test)
> 6. docs/INSTALL_NATIVE.md (if the session touches the flip)
>
> THIS SESSION'S GOAL: [e.g. "Execute T1 (flip + E2E baseline) end to end"].
>
> Ground rules (non-negotiable):
> - Every task ends with its WRITTEN GATE from EXECUTION_TASKS/DETAILED_SPECS (output-identity
>   or accuracy harness). A task without its gate passed is not done. Never relax a gate.
> - Evidence policy: measure, don't assume; quiet-machine rules for timing (RESULTS §4.11);
>   never mutate a model/data dir a running benchmark mounts; validate outputs by content.
> - Append every result to validation/RESULTS.md (numbered section) and commit with
>   conventional commits. Update PLAN.md status markers when a task closes.
> - transcribe-app: additive files + the documented hook only; NO pushes; commits there only
>   when the session goal explicitly includes them. pyannote-audio-fork: read-only, always.
> - vendor/speakrs is our patched vendored copy: after ANY change, run the clustering/pipeline
>   test suite (fixtures + RUST_MIN_STACK per RESULTS §4.23) and regenerate
>   patches/0001-cuda-performance-patch-set.patch.
> - Build facts that will bite you: ort MUST stay pinned 2.0.0-rc.12 (rc.13 breaks provider
>   pairing); ORT resolves provider libs from binary dir AND cwd; container-written dirs are
>   root-owned (chown before host renames); tokio/std threads need 16 MiB stacks (already in
>   diar-server); test fixtures need fixtures/ + fixtures/models/ mounted.
> - Escalation: a gate failing twice → stop, write up the failure in RESULTS, recommend
>   escalating one model tier. Do not thrash.
>
> Deliverables every session: (1) gates passed with numbers in RESULTS.md, (2) commits, (3) a
> closing summary listing what moved in PLAN.md and what the next session should pick up.

## SINGLE-AGENT CONFIGURATION (recommended if one agent runs the whole plan)

**Opus 5 / high effort**, using the MASTER PROMPT with the session sequence below executed in
order across sessions. Escalation rule from the prompt applies: a gate failing twice → xhigh
effort; still failing → hand that one task to Fable 5 with the failure transcript. Budget
alternative: Sonnet 5 / high (accepts more retries on T9/T11). Fable 5 reserved for: T12 design
(after the user's clustering research is located) and escalations only.

## Session sequence (recommended order + model per DETAILED_SPECS matrix)

| session | goal | model/effort |
|---|---|---|
| 1 | T1 flip + E2E baseline (needs user go post-PR) | Opus 5 / high |
| 2 | T5a priority fix + T6 telemetry + T8 VAD sweep (batchable trio) | Sonnet 5 / medium-high |
| 3 | T2 overlap (S-T2) | Sonnet 5 / high |
| 4 | T3 progressive presentation | Sonnet 5 / high |
| 5 | T4 finalize split (S-T4) | Opus 5 / high |
| 6 | T5b gender-in-sidecar (S-T5b) | Sonnet 5 / high |
| 7 | T9a shared sessions, then T9b constraints (S-T9a/b) | Opus 5 / high |
| 8 | T11 TRT EP (S-T11) | Opus 5 / high (escalate per spec) |
| 9 | T10 upstream PRs (UPSTREAM_PRS.md gameplan) | Opus 5 / high |
| 10 | T12 corpus clustering — FIRST locate user's prior research, then spec (Fable), then build | Fable 5 spec / Opus build |
| — | T13 text ladder | Sonnet 5 / medium, after baseline profiling |

Sessions 2-6 are independent enough to reorder; 3+5 both touch stages.py/pipelines.py — do not
run them concurrently in separate sessions.

## Per-session goal line examples

- "Execute T1 per EXECUTION_TASKS: apply the INSTALL_NATIVE hook (user has authorized
  transcribe-app changes for this session), flip, verify, run the E2E baseline, record."
- "Execute T2 per DETAILED_SPECS S-T2. transcribe-app edits limited to stages.py per spec.
  Gate: identical outputs + max-not-sum wall time."
- "Execute T10 Step 0-2 only (fork, rebase-check, intro issue, branch split + isolated
  validation); STOP before submitting PRs and report."
