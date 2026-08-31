# Provenance of the scripts in this directory

These are **adapted copies**, not the originals. Licensing is unchanged: the copied code is
Apache-2.0 from [`avencera/speakrs`](https://github.com/avencera/speakrs), and the repo-root
`LICENSE` is the same Apache-2.0.

| file | copied from | relationship |
|---|---|---|
| `export_models.py` | `vendor/speakrs/scripts/export_models.py` | near-verbatim; `export_plda()` rewritten, `main()` takes an argument |
| `export_tail_b64.py` | `validation/export_tail_b64_addendum.py` | wrapper class verbatim; packaged as a function |
| `fold_segmentation.py` | — | new; the step that was never written down |
| `export_gender.py` | — | new; replaces the `optimum-cli` line in `docs/DETAILED_SPECS.md` |
| `provision.py` | — | new; the driver |

Vendor pin: **`b0756b1`** (see `CLAUDE.md`; `vendor/` is a gitignored working-tree clone).

To review what diverged from upstream:

```bash
diff -u vendor/speakrs/scripts/export_models.py scripts/provision/export_models.py
diff -u validation/export_tail_b64_addendum.py scripts/provision/export_tail_b64.py
```

## Why copies rather than editing `vendor/`

1. **The patch file is load-bearing.** `CLAUDE.md` requires regenerating
   `patches/0001-cuda-performance-patch-set.patch` after *any* vendored edit, and that patch
   feeds seven upstream PR-prep branches whose purpose is clean review of CUDA **performance**
   work. Export-tooling changes do not belong in that diff.
2. **There is no single vendored file to edit anyway.** The real recipe spans
   `vendor/speakrs/scripts/export_models.py` + `validation/export_tail_b64_addendum.py` + a
   constant-folding pass + a `cp` + a gender export that did not previously exist. Two of the
   five already lived outside `vendor/`.
3. **Two steps must never go upstream.** The multimask-b64 byte copy (step 2c) and the
   segmentation fold-under-the-plain-name (step 2b) are diar-native-specific workarounds for
   loader/exporter mismatches. Upstreaming them as-is would be wrong; the correct upstream fix
   is to make exporter and loader agree, which is tracked separately.
4. **Precedent exists.** `validation/export_b64_addendum.py` and
   `validation/export_tail_b64_addendum.py` already copy the same wrappers verbatim with
   Apache-2.0 attribution.

`vendor/speakrs` is **not modified by this work** — verified with `git -C vendor/speakrs diff
HEAD --stat` before and after (unchanged at 23 files / 1359 insertions / 256 deletions).

## Embedded in the binary

`crates/diar-core/src/provision/exporter.rs` `include_str!`s every `.py` here into
`diar-server` (~60 KB on a 33 MB binary). At run time they are written to a private temp
directory (mode 0700) and executed with `DIAR_EXPORT_PYTHON` (default `python3`).

Consequence worth stating plainly: **the exporter is the binary's own bytes**, so
`marker.exporter_version` can never disagree with the server that reads it. Editing a file
here without rebuilding changes nothing.

Bump `EXPORT_RECIPE_VERSION` in `crates/diar-core/src/provision/files.rs` on **any** change
to these scripts or to the pins in `requirements.txt` that alters exported bytes.
