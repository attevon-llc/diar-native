# Benchmark protocol — how every speed lever gets measured

One procedure, one corpus, one anchor. Every optimization from here compares against the same
committed numbers, so gains are additive and regressions are visible immediately.

## The anchor (committed, do not re-measure)

Pre-flip PyAnnote engine, median of 3 runs, quiet machine, RTX 3080 Ti. Raw CSVs live in
`results/e2e_baseline/python/` — these are the "before" for everything.

| file | uuid | audio | **python anchor** | current |
|---|---|---|---|---|
| test_ai_video | `01a01aba-d3f9-7000-95b2-fb035b988781` | 24 s | **5.7 s** | 4.2 s |
| pyramids | `01a01a87-60a3-7000-808b-a7b9df231801` | 239 s | **11.9 s** | 7.2 s |
| warp drive | `019fd6b2-c2b7-7000-a7b1-0e6974dd62da` | 358 s | **15.0 s** | 8.6 s |
| **Karpathy** | `01a01aba-d9d2-7000-a027-9a0af248574f` | 3989 s | **108.4 s** | **54.4 s** |
| seed file | `019f2950-0f56-7000-80d9-e175004cc186` | 7558 s | **206.7 s** | 120.3 s |

**Karpathy is the primary reference.** It is the only file with both a speed anchor and
hand-labelled ground truth, so one file regresses both dimensions at once. Quote it when
reporting a lever's effect; quote the seed file for long-media behaviour.

Metric: `total_dispatch_to_postprocess` — dispatch to user-visible completion. Secondary:
`fully_indexed_duration`, and `gpu_duration` for attribution.

## Running a leg

```bash
BENCHMARK_EMAIL=admin@example.com BENCHMARK_PASSWORD=password \
  ./validation/run_e2e_baseline.sh <label>          # 5 files × 3 runs
venv/bin/python validation/summarize_e2e_baseline.py results/e2e_baseline --logs /tmp/leg_%s.log
```

The runner takes an `flock`. It will refuse to start while another leg holds it — that guard
exists because a stalled leg once resumed alongside its replacement and silently contaminated
both (RESULTS §7.13).

Existing legs: `python` (anchor), `native_fast`, `native_small`, `native_overlap`.

## Rules that make the numbers trustworthy

1. **One leg at a time.** Timed work is never co-scheduled with other timed work or any other
   compute (RESULTS §4.11). Check `nvidia-smi` and `celery inspect active` are idle first.
2. **Change one variable.** Hold the ASR model, batch size, GPU, corpus and concurrency fixed;
   vary only the lever. Record the control set alongside the result.
3. **Sample during, not after.** A VRAM or utilisation reading taken once a run finishes reports
   the idle floor and tells you nothing (RESULTS §7.14).
4. **Median of 3**, never a single run.
5. **Log what you disabled.** If something is switched off to get a clean run, it goes in RESULTS
   and gets switched back on — a benchmark accommodation is not a product decision.

## Accuracy must move with speed

No speed number is reportable on its own. Each lever also runs:

```bash
# app-level WSER, both the gate clip and the long clip
docker exec -e DIARIZER_ENGINE=native opentranscribe-celery-worker \
  python /app/scripts/benchmark_boundary.py --corpus /tmp/bench/corpus10m.json \
  --models large-v3-turbo --smoothing on --ref-dir /tmp/bench/ref --out /tmp/bench/out --cache-dir /tmp/bench/cache
```

| check | pass condition | source |
|---|---|---|
| WSER, Karpathy 10-min clip | **≤ 0.27 %** smoothed | RESULTS §7.9 |
| WSER, Karpathy 66.5-min clip | within noise of the fork (0.859 %) | §7.7 |
| DER, AMI test-16 exclusive | ≤ 17.83 % (the pyannote control) | §7.7 |
| DER, AMI test-16 full | 13.10 % ± 0.01 | §7.7 |
| output identity (for pure-perf levers) | byte-identical `diarize_records` | §7.13 |

For a lever that should not change outputs at all — overlap, threading, placement, caching —
**prove identity rather than assume it**: run with the lever off and on, and diff the raw
records. T2's shared audio buffer was exactly this risk and the diff is what settled it.

## Reporting

Append to `validation/RESULTS.md` with: the control set, the per-file table against the anchor,
the accuracy checks, and anything discarded and why. Update `PLAN.md` status markers when a task
closes. Numbers that contradict an earlier entry get an explicit retraction, not a quiet edit.
