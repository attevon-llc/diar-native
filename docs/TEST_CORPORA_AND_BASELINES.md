# Test corpora, file locations, and the numbers to beat

Everything an agent needs to reproduce or regress the diarization work: where the audio and
references live, which command produced each number, and what the number was. Nothing here needs
re-deriving — **never re-run a logged test** (RESULTS.md house rule); run it only to compare a
change against it.

Companions: `HANDOFF_T9A_SHARED_SESSIONS.md`, `HANDOFF_DIARIZATION_SPEED.md`,
`BENCHMARK_PROTOCOL.md` (the rules), `validation/RESULTS.md` (every measurement in full).

---

## 1. Accuracy corpora

| corpus | audio | references | size |
|---|---|---|---|
| **AMI test-16** | `/path/to/datasets/diarization-boundary/ami_audio/<ID>.Mix-Headset.wav` (34 files present; the 16 scored are named by the refs) | `refs/ami/<ID>.rttm` + `<ID>.uem` (16) | 2.1 GB |
| **Karpathy** (hand-labelled) | `transcribe-app/benchmark/diarization-boundary/karpathy/karpathy_kwSVtQ7dziU/` — `audio.wav` 3991 s, `karpathy_10m.wav` 600 s, `clip30.wav` 30 s | `reference.rttm` (committed), `reference.words.json`; fixed-name copy `refs/karpathy/karpathy.rttm` | 122 MB |
| **VoxConverse dev** | `/path/to/datasets/diarization-boundary/voxconverse_audio/audio/<ID>.wav` (216) | `refs/voxconverse/<ID>.rttm` (216) | 2.2 GB |

Karpathy's `reference.rttm` uses speaker names with spaces ("Sarah Guo"), which breaks standard
RTTM parsers — `refs/karpathy/` holds the fixed copy. AMI is scored **UEM-cropped**.

### Scoring command

```bash
docker exec opentranscribe-celery-worker python /tmp/score_der.py \
  --ref-dir /tmp/excl/ref --hyp-dir /tmp/excl/<variant> --uem-dir /tmp/excl/ref --collar 0.25
```
`validation/score_der.py` — pyannote.metrics DER, **collar 0.25, overlap INCLUDED**, UEM-aware.
It needs `pyannote.metrics` + `pyannote.database`, which the **app** image has and the diar-native
venv does not — run it inside `opentranscribe-celery-worker`.

## 2. Accuracy numbers to beat

DER %, collar 0.25, overlap included. "full" = overlap-aware output; "exclusive" = one speaker per
instant, which is what the app consumes for word attribution.

| corpus | representation | pyannote fork | speakrs | source |
|---|---|---|---|---|
| AMI-16 | full | **13.093** | **13.102** | §2.2, §4.26 |
| AMI-16 | exclusive | **17.828** | **17.813** | §7.7 (after the overlap fix) |
| Karpathy 66.5 min | full | **8.194** | **8.219** | §2.3, §4.21 |
| Karpathy 66.5 min | exclusive | **6.161** | **6.545** | §7.7 |
| VoxConverse dev-216 | full | 5.099 | **4.847** | §4.16d |

DER-component split on AMI-16 exclusive (the fix's evidence — the gap was *entirely* confusion):

| | DER | missed | false alarm | confusion |
|---|---|---|---|---|
| fork | 17.828 | 14.387 | 1.632 | **1.808** |
| speakrs before fix | 18.654 | 14.375 | 1.625 | **2.655** |
| speakrs after fix | **17.813** | 14.375 | 1.624 | **1.814** |

App-level WSER (word-speaker error rate, through the real pipeline, `large-v3-turbo`):

| clip | fork | speakrs | gate |
|---|---|---|---|
| Karpathy **10-min** | 0.27 % | **0.27 %** | ≤ 0.27 % — this is the clip the bar came from (§7.9) |
| Karpathy **66.5-min** | 0.859 % | **0.890 %** | parity with the fork, not an absolute bar |

## 3. Speed / E2E corpus (the app-level anchor)

Five files already ingested in the dev deployment. Median of 3 runs, quiet machine, RTX 3080 Ti,
`large-v3-turbo`, `int8_float16`, batch 8. Raw CSVs: `results/e2e_baseline/`.

| file | uuid | audio | python (pre-flip) | native + overlap |
|---|---|---|---|---|
| test_ai_video | `01a01aba-d3f9-7000-95b2-fb035b988781` | 24 s | 5.7 s | **4.2 s** |
| pyramids | `01a01a87-60a3-7000-808b-a7b9df231801` | 239 s | 11.9 s | **7.2 s** |
| warp drive | `019fd6b2-c2b7-7000-a7b1-0e6974dd62da` | 358 s | 15.0 s | **8.6 s** |
| **Karpathy** | `01a01aba-d9d2-7000-a027-9a0af248574f` | 3989 s | **108.4 s** | **54.4 s** |
| seed file | `019f2950-0f56-7000-80d9-e175004cc186` | 7558 s | 206.7 s | **120.3 s** |

**Karpathy is the primary reference** — the only file with both a speed anchor and hand-labelled
ground truth, so one run regresses speed and accuracy together.

```bash
BENCHMARK_EMAIL=admin@example.com BENCHMARK_PASSWORD=password \
  ./validation/run_e2e_baseline.sh <label>            # 5 files × 3 runs; takes an flock
venv/bin/python validation/summarize_e2e_baseline.py results/e2e_baseline --logs /tmp/leg_%s.log
./validation/task_census.sh <file-uuid> [label]        # every task on one file
```

## 4. Engine-level speed (no app involved)

Sidecar `/diarize` against a WAV, warm engine:

| file | wall | note |
|---|---|---|
| ES2004a (36.4 min) | **7.9 s ≈ 277× RT** | fast set; the headline warm number (§4.27) |
| Karpathy 10-min | ~4.9 s | diarize only |
| Karpathy 10-min + gender | ~6.0 s | gender adds ~1.5 s (§7.16) |
| AMI-16 (8 h total) | ~4 min | all 16 files, one engine load |

Model sets: `models_folded/` = fast (4.2 GB VRAM, 277× RT) · `models_small/` = laptop
(1.6 GB, 59× RT, **3.6× slower** — a VRAM floor, never a speed lever).

## 5. Concurrency baseline (what T9a must improve)

Two largest files dispatched together, spans from `diarize_request_sent`/`diarize_joined`:

| file | diarize span solo | with 2 concurrent |
|---|---|---|
| Karpathy | 37.5 s | **76.1 s (2.0×)** |
| seed 2.1 h | ~58 s | **113.8 s (2.0×)** |

Both doubling is the mutex signature. Pair wall-clock 131 s vs 174.7 s sequential. **T9a is done
when these spans stop doubling.**

## 6. VRAM baseline

GPU 1 (12 288 MiB), all warm: sidecar **4 136** + whisper **2 038** + redaction **1 346** =
**7 575 MiB** floor, 4 328 free. Marginal **~490 MiB per concurrent job** (weights are shared).
Peak under 3 concurrent jobs 9 047 MiB, settling back to exactly 7 575. Measure **during** a run —
sampling afterwards reports the idle floor (§7.14).

## 7. Traps that cost time already

- `ort` pinned `=2.0.0-rc.12`; rc.13 fails at session load.
- Tests: `cargo test --release --no-default-features --features openblas-system,online`
  with `RUST_MIN_STACK=16777216`, built in `diar-bench-builder` (host `target/` is root-owned).
- Model files are **gitignored** (`models*/`) — gated community-1 derivatives, never committed.
- Worker `/tmp` is wiped on container recreate; re-stage bench files after any restart.
- `BATCH_SIZE` is not the compose variable (`GPU_DEFAULT_BATCH_SIZE` is) — a sweep that sets the
  wrong one silently measures the same config four times (§7.17).
