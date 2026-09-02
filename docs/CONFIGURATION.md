# Configuration — environment variables

**This is the authoritative list.** Every variable below has a read site in the code; anything
not listed here is not read by anything. Kept in sync in both directions — a variable with no
read site does not belong here, and a read site with no entry here is a bug.

`DIAR_NATIVE_*` names that appear in OpenTranscribe's compose file are **not** in this table on
purpose — they are compose-level indirection that expands into these
(`DIAR_MODE=${DIAR_NATIVE_MODE:-cuda}`), and no Rust code reads them.

For running the service at all, start at the [README](../README.md).

- [Serving](#serving-diar-server-with-no-subcommand)
- [Provisioning](#provisioning-provision-models-verify-models-check-token)
- [Engine tuning](#engine-tuning-read-by-speakrs)
- [Logging](#logging)
- [Not settable, and dead](#not-settable-and-dead)
- [Known sharp edges](#known-sharp-edges)

---

## Serving (`diar-server` with no subcommand)

| var | what it does | default | notes |
|---|---|---|---|
| `DIAR_MODELS_DIR` | Directory holding the model set. | `/models` | Also read by `provision-models`/`verify-models` as the `--models-dir` default. |
| `DIAR_BIND` | Listen address. | `0.0.0.0:8701` | Not validated at parse time; a bad value fails at `bind()` and the process exits non-zero. |
| `DIAR_MAX_INFLIGHT` | Global admission gate — bounds **total** inflight across all devices, so adding an engine cannot silently double concurrency. | `2` | Unparseable → falls back to 2. **`0` is accepted and deadlocks every request** — see [Known sharp edges](#known-sharp-edges). |
| `DIAR_MAX_INFLIGHT_CPU` | Optional inner sub-gate for CPU work only. CPU requests take the global permit first, this one second — always that order. | unset (no inner gate) | Unparseable **or `0`** → treated as unset. |
| `DIAR_DEVICES` | Comma list of devices to load, e.g. `cuda,cpu`. **First entry is the default device.** Duplicates collapse, order preserved. Wins over `DIAR_MODE`. | unset → `DIAR_MODE` | Blank/whitespace-only is treated as unset (`${FOO:-}` in compose must not be fatal). An unknown or not-compiled-in name is **fatal at startup**. |
| `DIAR_MODE` | Legacy single-device knob, used when `DIAR_DEVICES` is absent. | `cuda` | Matches `cpu`, `coreml`, `coreml_fast` exactly; **unset *or unrecognized* falls through to `cuda`** — a long-standing quirk, deliberately preserved. The result is still capability-checked. |
| `DIAR_MODEL_SET` | Assert which tier the startup gate should require (`fast`\|`small`). | unset → read from the directory's own marker, else `fast` | An unparseable value is silently treated as unset. |
| `DIAR_ALLOW_UNVERIFIED_MODELS` | Downgrade the startup gate's fatal cases to warnings. | off | Accepts exactly `1`, `true`, `TRUE`, `yes`. Note `True` and `YES` do **not** work. |
| `DIAR_GENDER_MAX_SECONDS` | Cap (seconds, taken from the middle of the window) on the clip fed to the wav2vec2 gender classifier. Unbounded turns cost ~6.3 GB VRAM. | `5` | Unparseable or `0` → falls back to 5. |
| `RUST_LOG` | `tracing` filter. | `info,ort::logging=warn` | **Unset does not mean silent.** Empty is treated as unset; a malformed value warns and falls back to the default rather than starting the process blind. See [Logging](#logging). |
| `DIAR_LOG_FORMAT` | `text` (human) or `json` (one flattened object per line). | `text` | Unrecognized values warn and use `text`. |
| `RUST_MIN_STACK` | Only inspected for presence; the binary sets it to `16777216` when unset, because speakrs pipeline and ORT worker threads overflow the 2 MiB default. | effectively 16 MiB | An operator-supplied value is left untouched. The tokio runtime separately hardcodes a 16 MiB stack, so this affects non-tokio threads. |

Defaults are unchanged end to end: with neither `DIAR_DEVICES` nor `DIAR_MAX_INFLIGHT_CPU` set,
the server loads exactly one engine from `DIAR_MODE`, exactly as it always did.

> **`DIAR_MAX_INFLIGHT` is the load-safety mechanism, not the filesystem.** Backpressure at the
> admission gate is what bounds resource use, which means the **peak handoff footprint is
> `inflight × largest_file`, independent of queue depth** — a deep queue costs you nothing
> extra. Size the shared audio volume against that product, not against how many jobs are
> waiting. For reference, 16 kHz mono int16 WAV runs about 19 MB for 10 min, 58 MB for 30 min,
> 230 MB for 2 h and 540 MB for 4.7 h.

## Provisioning (`provision-models`, `verify-models`, `check-token`)

| var | what it does | default | notes |
|---|---|---|---|
| `HF_TOKEN` | Hugging Face read token. | none | Also `HUGGINGFACE_TOKEN` and `HUGGING_FACE_HUB_TOKEN`, tried in that order. `--hf-token` wins over all three. Empty values are skipped, not treated as a token. |
| `HF_ENDPOINT` | Base URL for the Hugging Face API. | `https://huggingface.co` | **The only knob that makes provisioning work against a mirror or an air-gapped proxy.** Trailing `/` is stripped. Empty is treated as unset **by the Rust side only** — see [Known sharp edges](#known-sharp-edges). |
| `HF_HOME` | Hugging Face cache directory. Forwarded to the python export child. | none (child uses its own default) | `--hf-cache` overrides. |
| `HF_HUB_OFFLINE` | Read by `huggingface_hub` **inside the python exporter**, not by Rust — set it to re-export from a warm cache with no network (the RESULTS §7.36 acceptance run did exactly this, and needed no token). | unset | **Not forwarded to the child.** `diar-core` explicitly sets only `PYTHONUNBUFFERED`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, `HF_TOKEN` and `HF_HOME` on the export subprocess, so this only works if it is already in `diar-server`'s own environment and inherited. |
| `DIAR_EXPORT_PYTHON` | Interpreter (with torch + pyannote.audio) used to run the export scripts. | `python3` | `--python` overrides. A non-working interpreter exits 6. |
| `DIAR_MODE` / `DIAR_DEVICES` | Device for the end-to-end smoke stage. | **`cpu`** | Deliberately *not* the serving default. Provisioning defaulting to a GPU is what used to brick GPU-less hosts. An unrecognized name is exit 2 here, never a silent fall-through to `cuda`. |
| `DIAR_ORT_OPT_LEVEL` | Override the ORT graph optimization level for sessions built through `ort_compat` — the gender session and the smoke test, **not** speakrs' 15 diarization graphs. | unset | `disable`\|`none`\|`0`, `basic`\|`1`, `extended`\|`2`, `all`\|`3`. Escape hatch for a platform hitting the aarch64-class problem described in [DEPLOYMENT.md](DEPLOYMENT.md#the-fp16-gender-model-on-linuxarm64). It is a **floor**: it can lower a model's optimization level but never raise it past its cap, so `=all` on aarch64 does not re-break gender. |
| `DIAR_ORT_DISABLED_OPTIMIZERS` | Pass a disable-list straight to ORT. Same scope as above. | unset | Three traps, all measured (RESULTS §7.40): the pass you probably want is `GeluFusionL2` (`GeluFusion` and `GeluFusionL1` do nothing); the separator is **`;`**, not `,`, despite the `ort` crate's doc comment (a `,` is rejected); and **a misspelled name is silently ignored** by ORT — no error, no warning, no effect, and it cannot be validated. |

## Engine tuning (read by speakrs)

| var | what it does | default |
|---|---|---|
| `SPEAKRS_LAZY_SESSIONS` | Skip building the heavy batch-64 primary and batched split-tail sessions the CUDA multimask pipeline never runs; each idle session pins its own ORT arena. Live compose sets `1`. | off (all sessions built) |
| `SPEAKRS_ARENA_SHRINK` | Shrink the device arena back to its initial chunk after each big batched run — a VRAM floor for 4 GB-tier cards, at roughly a 20% per-job cost. | off |
| `SPEAKRS_INTRA_THREADS` | Intra-op threads for the embedding sessions. | `min(cores, 6)` |
| `SPEAKRS_FBANK_THREADS` | Intra-op threads for the fbank session specifically. | `min(cores, 4)` |
| `SPEAKRS_AHC_THREADS` | Workers for the blocked pairwise-distance computation in AHC clustering. Higher oversubscribes, since each worker also drives a multi-threaded BLAS `dot`. | `min(cores, 8)` |
| `SPEAKRS_FBANK_POOL` | Size of the CPU fbank session pool fanned out per chunk. Read **once**, in `EngineConfig::new`, and passed to speakrs as a `RuntimeConfig` field — `diar-server` no longer overwrites it (RESULTS §7.50, issue #3). `0` disables the pool and keeps the single fbank session; a malformed value warns and falls back to the default. | `1` on CPU/CoreML (the pool contends with inference for cores — RESULTS §4.12), `min(cores/4, 8)` on CUDA |

## Logging

`diar-server` logs to **stdout**; fatal startup errors stay on **stderr**. See
[ARCHITECTURE.md](ARCHITECTURE.md#logging) for what a request record contains.

| knob | scope | meaning |
|---|---|---|
| `RUST_LOG` | startup | Standard `tracing` filter, e.g. `info`, `debug`, `speakrs=debug`, `diar_server=debug,speakrs=trace`. **Unset or empty ⇒ `info,ort::logging=warn`** — the container is useful out of the box. A malformed value logs a warning and falls back to that same default rather than starting the process silent. |
| `DIAR_LOG_FORMAT` | startup | `text` (default) — human-readable lines, ANSI only when stdout is a terminal. `json` — one flattened JSON object per line for log aggregation. An unrecognized value warns and uses `text`. |
| `x-request-id` | per request | Request **header**, optional. Honoured if present so a job keeps one id end to end through a larger stack; otherwise one is generated. Echoed back on the response, including on errors. Sanitized before it is logged (control characters stripped, 64 chars max) — a caller cannot forge a log record with it. |

> **Why the default is not a bare `info`.** ONNX Runtime's native log bridge (`ort::logging`)
> emits thousands of INFO lines on a CUDA startup — "Removing NodeArg …", "GraphTransformer …
> modified: 0" — against 3 lines from diar-server. Measured, not estimated (RESULTS §7.37). A
> blanket `info` buries the startup record roughly 2000:1, so the default holds that one target
> at `warn`. Its warnings are real perf diagnostics (Memcpy nodes, unassigned nodes) and are
> kept, as is `ort::ep`, which reports which execution provider actually registered. An explicit
> `RUST_LOG=ort=info` or `RUST_LOG=debug` still gets the firehose.

> **Do not set `RUST_LOG=trace`.** Enabling `ort::lifetime` at TRACE makes the process abort at
> exit, *after* its work is written. `RUST_LOG=speakrs=trace` is the safe way to get engine stage
> timings. See [ARCHITECTURE.md](ARCHITECTURE.md#both-binaries-must-never-return)
> and [ORT_ATEXIT_TEARDOWN.md](ORT_ATEXIT_TEARDOWN.md).

```bash
# human-readable, default level
docker run --rm -p 8701:8701 -v /srv/models:/models:ro davidamacey/diar-native:0.3.1-cpu

# engine stage timings, JSON for an aggregator
docker run --rm -p 8701:8701 -v /srv/models:/models:ro \
  -e RUST_LOG=speakrs=debug -e DIAR_LOG_FORMAT=json davidamacey/diar-native:0.3.1-cpu
```

## Not settable, and dead

| var | status |
|---|---|
| `SPEAKRS_TRT`, `SPEAKRS_TRT_CACHE` | **Dead.** No read sites; they do nothing. Left documented only so nobody rediscovers them in the TensorRT-era notes and assumes they work (RESULTS §7.26). |
| `ORT_DYLIB_PATH` | Not applicable to these builds — it is only read under `ort`'s `load-dynamic` feature, which is not enabled. ORT is statically linked. |

## Known sharp edges

- **`DIAR_MAX_INFLIGHT=0` deadlocks every request.** The global gate has no `> 0` guard, unlike
  `DIAR_MAX_INFLIGHT_CPU`, which explicitly treats `0` as unset for exactly this reason.
- **`DIAR_ORT_OPT_LEVEL` typos are silent.** An unrecognized level is ignored with no diagnostic
  and control falls through, whereas a bad `DIAR_ORT_DISABLED_OPTIMIZERS` is fatal.
- **`DIAR_ALLOW_UNVERIFIED_MODELS` is case-sensitive** in a way its neighbours are not: `true`
  and `TRUE` work, `True` does not.
- **Setting `HF_ENDPOINT` to the empty string breaks `provision-models`,** even though the table
  above says empty is treated as unset. That is true of the Rust side — but the variable is not
  *stripped* from the environment, and the Python export child inherits it. `huggingface_hub`
  does `os.environ.get("HF_ENDPOINT", "https://huggingface.co")`, which returns the **empty
  string** when the key exists and is blank, so the download URL loses its scheme and the export
  dies with `httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://'
  protocol`. It reads like a network fault and is not one. This is easy to hit from compose,
  where `HF_ENDPOINT: ${HF_ENDPOINT:-}` is the natural way to make a variable optional; the
  bundled `docker-compose.yml` defaults it to the literal URL instead (RESULTS §7.43).

---

See also: [DEPLOYMENT.md](DEPLOYMENT.md) · [PROVISIONING.md](PROVISIONING.md) ·
[API.md](API.md) · `.env.example` (every setting as shipped) · [README](../README.md)
