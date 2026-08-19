# Agent Kickoff Prompts — start any execution session from here

Paste the master prompt (adjusted per session goal) into a fresh Claude Code session with the
model/effort from DETAILED_SPECS' matrix. Each prompt binds the agent to the plan documents so
no context is lost between sessions.

---

## FULL-PROGRAM PROMPT (paste this into the new session — Opus 5 / high)

> Work in /mnt/nvm/repos/diar-native — the project of record for OpenTranscribe's native
> diarization/inference program. You are the FULL-PROGRAM ORCHESTRATOR. Before doing ANYTHING,
> read in this order: PLAN.md; docs/EXECUTION_TASKS.md; docs/DETAILED_SPECS.md;
> docs/AGENT_KICKOFF.md; docs/E2E_PIPELINE_MAP.md; docs/INSTALL_NATIVE.md;
> validation/RESULTS.md section headers only (NEVER re-run a logged test).
>
> GOAL: execute the accepted sequence end to end, starting at T1 (flip + E2E baseline), then
> the session table order (T5a/T6/T8 → T2 → T3 → T4 → T5b → T9 → T11 → T10), with full
> benchmarking, output-identity/accuracy gates, and RESULTS.md documentation at every step,
> producing complete before/after speed calculations for upload→presented per hardware tier.
>
> Authorization: transcribe-app changes ARE authorized (INSTALL_NATIVE hook, staged-file
> commits, files named in DETAILED_SPECS) — but NEVER push to any remote;
> pyannote-audio-fork is read-only. Use mixed-model subagents per the delegation table
> (verifier separate from workers; Fable spawns only on double gate-failure). Timed benchmark
> legs are never co-scheduled (RESULTS §4.11). Every session ends with gates passed + numbers
> in RESULTS.md + conventional commits + a closing summary.
>
> Required user-provided config (STOP and ask if missing): ENABLE_BENCHMARK_TIMING=1 in the
> app .env; DIARIZER_ENGINE=native at flip time; DIAR_NATIVE_GPU / _MODELS_DIR / _MAX_INFLIGHT
> if non-default; gpu-split service-name adjustment in the overlay if that profile is used.
>
> Start now with T1.

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

## Delegation pattern (orchestrator + mixed-model subagents)

The orchestrating session CAN spawn subagents with different models/efforts:
- Agent tool: per-spawn `model` override (sonnet/opus/haiku/fable). Effort comes from custom
  agent definitions (`.claude/agents/*.md` frontmatter: model + reasoning effort). Suggested
  definitions to create once: `bench-runner` (sonnet, medium — runs benchmark/scoring legs),
  `doc-scribe` (sonnet, low — RESULTS/PLAN updates), `rust-surgeon` (opus xhigh — vendored-crate
  changes), `verifier` (opus high — runs gates INDEPENDENTLY of whoever did the work).
- Workflow tool: per-agent `model:` AND `effort:` overrides for deterministic fan-outs
  (requires explicit user opt-in per its rules).
- Rules: (1) subagents start with NO context — every delegation prompt must include the
  doc-reading preamble from the MASTER PROMPT; (2) the agent that does work never grades its
  own gate — orchestrator or `verifier` runs gates; (3) parallel subagents must not co-schedule
  timed benchmarks (RESULTS §4.11) or touch the same files (3+5 conflict note below);
  (4) escalation can be a `model: fable` subagent spawn with the failure transcript — no
  session switch needed.
- Economy shape: Opus orchestrator holds the plan + judgment; Sonnet subagents burn the
  mechanical tokens; Fable appears only in escalation spawns. This is the cheapest correct
  configuration for the full plan.

## Session sequence (recommended order + model per DETAILED_SPECS matrix)

| session | goal | model/effort | subagents? |
|---|---|---|---|
| 1 | T1 flip + E2E baseline (needs user go post-PR) | Opus 5 / high | YES: bench-runner for the 3-config baseline legs (run timed legs SEQUENTIALLY — §4.11); doc-scribe for RESULTS; verifier for gates |
| 2 | T5a priority fix + T6 telemetry + T8 VAD sweep | Sonnet 5 / medium-high | YES: ideal parallel trio (independent files) + one verifier pass |
| 3 | T2 overlap (S-T2) | Sonnet 5 / high | NO for the edit (single file surgery); bench-runner for the gate measurements |
| 4 | T3 progressive presentation | Sonnet 5 / high | OPTIONAL: backend-event + frontend-fetch as two subagents (disjoint files), orchestrator integrates |
| 5 | T4 finalize split (S-T4) | Opus 5 / high | NO (stages/pipelines topology change — one mind); verifier after |
| 6 | T5b gender-in-sidecar (S-T5b) | Sonnet 5 / high | YES sequential: export+parity (python) → Rust endpoint → app rewire; verifier between hops |
| 7 | T9a shared sessions, T9b constraints (S-T9a/b) | Opus 5 / high | NO (vendored-crate surgery, single rust-surgeon); verifier runs full test suite + Phase-B subset |
| 8 | T11 TRT EP (S-T11) | Opus 5 / high | NO for config; bench-runner for warm timings; escalate via fable spawn per spec |
| 9 | T10 upstream PRs | Opus 5 / high | YES: per-branch isolated validation as PARALLEL subagents with `isolation: worktree` (each branch validated on a clean tree); orchestrator submits |
| 10 | T12 corpus clustering | Fable 5 spec / Opus build | YES: research-locator + profiler subagents (parallel, read-only) feed the design |
| — | T13 text ladder | Sonnet 5 / medium | YES: per-model profiling/conversion legs parallel; verifier for parity fixtures |

Standing rule: timed benchmark legs are NEVER run in parallel with each other or with other
compute (RESULTS §4.11) — parallel subagents are for edits/research/scoring, sequential for
timing.

Sessions 2-6 are independent enough to reorder; 3+5 both touch stages.py/pipelines.py — do not
run them concurrently in separate sessions.

## Per-session goal line examples

- "Execute T1 per EXECUTION_TASKS: apply the INSTALL_NATIVE hook (user has authorized
  transcribe-app changes for this session), flip, verify, run the E2E baseline, record."
- "Execute T2 per DETAILED_SPECS S-T2. transcribe-app edits limited to stages.py per spec.
  Gate: identical outputs + max-not-sum wall time."
- "Execute T10 Step 0-2 only (fork, rebase-check, intro issue, branch split + isolated
  validation); STOP before submitting PRs and report."
