//! The authoritative list of what a provisioned models directory must contain.
//!
//! Deliberately NOT reusing speakrs' `ModelBundle::required_files()`
//! (`vendor/speakrs/src/models.rs`): it is `#[cfg(feature = "online")]` and so is not even
//! compiled into our build, it omits the `-tail*` and `-b64` families entirely, and it
//! demands `wespeaker-voxceleb-resnet34.onnx.data` — an external-data sidecar our exports
//! never produce because every `torch.onnx.export` call passes `external_data=False`.
//! Pointing provisioning at that list would have declared a correct directory broken and a
//! broken one correct.

use std::fmt;

/// Bump on ANY change to the export recipe (`scripts/provision/`), including a pin change
/// that alters graph bytes. A marker whose `exporter_version` differs from this is treated
/// as `stale` — the models still work, but they were not produced by the code now shipping.
pub const EXPORT_RECIPE_VERSION: u32 = 1;

/// Marker schema version. Separate from the recipe version: the schema describes the JSON
/// shape, the recipe describes the bytes it vouches for. A reader that understands schema 1
/// can still say something useful about a directory built by a newer recipe.
pub const MARKER_SCHEMA: u32 = 1;

/// Filename of the provenance marker inside the models dir.
///
/// A visible name, not a dotfile: it is meant to be found. Safe to add because
/// `ModelBundle::from_dir` (`vendor/speakrs/src/models.rs`) joins fixed filenames and never
/// globs the directory, so an extra file cannot be mistaken for a model.
pub const MARKER_FILE: &str = "diar-provision.json";

/// Optional gender classifier. Absent => `GenderModel::load_optional` returns `Ok(None)` and
/// the feature is silently off, which is why provisioning treats it as a first-class artifact
/// rather than an afterthought.
pub const GENDER_MODEL: &str = "gender-wav2vec2.onnx";
/// Documentation-only sidecar for the gender model. `gender.rs` hardcodes its `ID2LABEL`, so
/// this file is never read at runtime — verify.rs cross-checks it against that constant,
/// which turns a decorative file into a guard against upstream relabelling.
pub const GENDER_META: &str = "gender-wav2vec2.meta.json";

/// Which tier of model set to produce. `Small` is `Fast` minus the batch-64 graphs; the
/// shared files are byte-identical across the two sets by construction (one export, then a
/// delete), which is why `models_folded/` and `models_small/` agree on all 19 today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSet {
    /// Default tier, >=6 GB GPUs. Adds the b64 graphs used by the batched embedding path.
    Fast,
    /// Laptop tier. No batch-64 graphs.
    Small,
}

impl ModelSet {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelSet::Fast => "fast",
            ModelSet::Small => "small",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fast" | "folded" => Some(ModelSet::Fast),
            "small" => Some(ModelSet::Small),
            _ => None,
        }
    }

    /// Does a directory provisioned as `self` satisfy a server asking for `wanted`?
    ///
    /// `Fast` is a strict superset of `Small`, so a fast dir serves a small request. The
    /// reverse is not true: a small dir is missing the b64 graphs, and with
    /// `SPEAKRS_LAZY_SESSIONS=1` their absence would not surface until the first batched
    /// job, long after startup declared success.
    pub fn covers(self, wanted: ModelSet) -> bool {
        matches!(
            (self, wanted),
            (ModelSet::Fast, _) | (ModelSet::Small, ModelSet::Small)
        )
    }
}

impl fmt::Display for ModelSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Files every set must contain, gender excluded.
pub const SHARED_REQUIRED: &[&str] = &[
    "plda_lda.npy",
    "plda_mean1.npy",
    "plda_mean2.npy",
    "plda_mu.npy",
    "plda_psi.npy",
    "plda_tr.npy",
    "wespeaker-voxceleb-resnet34.min_num_samples.txt",
    "segmentation-3.0.onnx",
    "segmentation-3.0-b32.onnx",
    "wespeaker-fbank.onnx",
    "wespeaker-fbank-b32.onnx",
    "wespeaker-multimask-tail.onnx",
    "wespeaker-multimask-tail-b32.onnx",
    "wespeaker-voxceleb-resnet34.onnx",
    "wespeaker-voxceleb-resnet34-b32.onnx",
    "wespeaker-voxceleb-resnet34-tail.onnx",
    "wespeaker-voxceleb-resnet34-tail-b3.onnx",
    "wespeaker-voxceleb-resnet34-tail-b32.onnx",
];

/// The batch-64 graphs that make a set `fast`.
pub const FAST_ONLY_REQUIRED: &[&str] = &[
    "segmentation-3.0-b64.onnx",
    "wespeaker-voxceleb-resnet34-b64.onnx",
    "wespeaker-multimask-tail-b64.onnx",
    "wespeaker-voxceleb-resnet34-tail-b64.onnx",
];

/// Every file a provisioned dir of this set must contain.
///
/// `with_gender` folds in the two gender artifacts. `GENDER_META` is fast-only, matching
/// the shipped `models_folded/` vs `models_small/` difference exactly.
pub fn required_files(set: ModelSet, with_gender: bool) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = SHARED_REQUIRED.to_vec();
    if set == ModelSet::Fast {
        out.extend_from_slice(FAST_ONLY_REQUIRED);
    }
    if with_gender {
        out.push(GENDER_MODEL);
        if set == ModelSet::Fast {
            out.push(GENDER_META);
        }
    }
    out.sort_unstable();
    out
}

/// Files present in `Fast` but not `Small`. `--set small` runs the full export then deletes
/// exactly these, which is cheaper than a second export path and is what guarantees the
/// shared files stay byte-identical between the two sets.
pub fn fast_only_files(with_gender: bool) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = FAST_ONLY_REQUIRED.to_vec();
    if with_gender {
        out.push(GENDER_META);
    }
    out.sort_unstable();
    out
}

/// Every `.onnx` in the set — the graphs verify.rs stage 1 must be able to parse.
pub fn onnx_files(set: ModelSet, with_gender: bool) -> Vec<&'static str> {
    required_files(set, with_gender)
        .into_iter()
        .filter(|f| f.ends_with(".onnx"))
        .collect()
}

/// numpy dtype of a PLDA parameter file, as it actually appears on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F64,
}

impl NpyDtype {
    /// The `descr` field numpy writes in the header.
    pub fn descr(self) -> &'static str {
        match self {
            NpyDtype::F32 => "<f4",
            NpyDtype::F64 => "<f8",
        }
    }

    pub fn size(self) -> usize {
        match self {
            NpyDtype::F32 => 4,
            NpyDtype::F64 => 8,
        }
    }
}

/// Expected header of each PLDA array.
///
/// READ THE DTYPES BEFORE TRUSTING A SIZE. These were confirmed by parsing the real headers
/// in `models_folded/`, not inferred from file size — and inference would have been wrong
/// for four of the six. `plda_tr.npy` and `plda_lda.npy` are both 131200 bytes but are
/// (128,128) f64 and (256,128) f32 respectively; `plda_mu`/`plda_psi` are 1152 bytes as
/// (128,) f64, not (256,) f32. Byte size alone cannot distinguish those, which is precisely
/// why the check parses the header instead.
pub const PLDA_SPECS: &[(&str, NpyDtype, &[usize])] = &[
    ("plda_lda.npy", NpyDtype::F32, &[256, 128]),
    ("plda_mean1.npy", NpyDtype::F64, &[256]),
    ("plda_mean2.npy", NpyDtype::F32, &[128]),
    ("plda_mu.npy", NpyDtype::F64, &[128]),
    ("plda_psi.npy", NpyDtype::F64, &[128]),
    ("plda_tr.npy", NpyDtype::F64, &[128, 128]),
];

/// A graph's expected input/output names and shapes.
///
/// Names are load-bearing beyond documentation: `gender.rs` feeds `"input_values"` and reads
/// `"logits"` by name, and speakrs binds the diarization graphs the same way. A model with
/// the right filename but the wrong signature fails here rather than at the first request.
/// `None` in a dimension means "dynamic or batch — do not assert".
pub struct IoSpec {
    pub file: &'static str,
    pub inputs: &'static [(&'static str, &'static [Option<i64>])],
    pub outputs: &'static [(&'static str, &'static [Option<i64>])],
}

const B: Option<i64> = None;

pub const IO_SPECS: &[IoSpec] = &[
    // Batch is FIXED at 1 here and the sample axis is the dynamic one — the opposite of
    // the obvious guess. Read off the shipped graphs, not assumed.
    IoSpec {
        file: "segmentation-3.0.onnx",
        inputs: &[("input", &[Some(1), Some(1), B])],
        outputs: &[("output", &[Some(1), B, Some(7)])],
    },
    IoSpec {
        file: "segmentation-3.0-b32.onnx",
        inputs: &[("input", &[Some(32), Some(1), Some(160000)])],
        outputs: &[("output", &[Some(32), Some(589), Some(7)])],
    },
    IoSpec {
        file: "segmentation-3.0-b64.onnx",
        inputs: &[("input", &[Some(64), Some(1), Some(160000)])],
        outputs: &[("output", &[Some(64), Some(589), Some(7)])],
    },
    IoSpec {
        file: "wespeaker-fbank.onnx",
        inputs: &[("waveform", &[Some(1), Some(1), Some(160000)])],
        outputs: &[("fbank", &[Some(1), Some(998), Some(80)])],
    },
    IoSpec {
        file: "wespeaker-fbank-b32.onnx",
        inputs: &[("waveform", &[Some(32), Some(1), Some(160000)])],
        outputs: &[("fbank", &[Some(32), Some(998), Some(80)])],
    },
    IoSpec {
        file: "wespeaker-multimask-tail.onnx",
        inputs: &[
            ("fbank", &[Some(1), Some(998), Some(80)]),
            ("masks", &[Some(3), Some(589)]),
        ],
        outputs: &[("output", &[Some(3), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-multimask-tail-b32.onnx",
        inputs: &[
            ("fbank", &[Some(32), Some(998), Some(80)]),
            ("masks", &[Some(96), Some(589)]),
        ],
        outputs: &[("output", &[Some(96), Some(256)])],
    },
    // NOTE: no entry for wespeaker-multimask-tail-b64.onnx. It is a byte COPY of the b32
    // graph (RESULTS §4.15 — a genuine batch-64 graph under that name crashes the worker,
    // whose buffers are sized 32), so it declares batch 32 and asserting 64 here would fail
    // a correct directory. Stage 3d checks the copy relationship directly instead.
    IoSpec {
        file: "wespeaker-voxceleb-resnet34.onnx",
        inputs: &[
            ("waveform", &[Some(1), Some(1), Some(160000)]),
            ("weights", &[Some(1), Some(589)]),
        ],
        outputs: &[("output", &[Some(1), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-b32.onnx",
        inputs: &[
            ("waveform", &[Some(32), Some(1), Some(160000)]),
            ("weights", &[Some(32), Some(589)]),
        ],
        outputs: &[("output", &[Some(32), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-b64.onnx",
        inputs: &[
            ("waveform", &[Some(64), Some(1), Some(160000)]),
            ("weights", &[Some(64), Some(589)]),
        ],
        outputs: &[("output", &[Some(64), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-tail.onnx",
        inputs: &[
            ("fbank", &[Some(1), Some(998), Some(80)]),
            ("weights", &[Some(1), Some(589)]),
        ],
        outputs: &[("output", &[Some(1), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-tail-b3.onnx",
        inputs: &[
            ("fbank", &[Some(3), Some(998), Some(80)]),
            ("weights", &[Some(3), Some(589)]),
        ],
        outputs: &[("output", &[Some(3), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-tail-b32.onnx",
        inputs: &[
            ("fbank", &[Some(32), Some(998), Some(80)]),
            ("weights", &[Some(32), Some(589)]),
        ],
        outputs: &[("output", &[Some(32), Some(256)])],
    },
    IoSpec {
        file: "wespeaker-voxceleb-resnet34-tail-b64.onnx",
        inputs: &[
            ("fbank", &[Some(64), Some(998), Some(80)]),
            ("weights", &[Some(64), Some(589)]),
        ],
        outputs: &[("output", &[Some(64), Some(256)])],
    },
    // Both batch AND sample count are dynamic: `gender.rs` feeds one variable-length clip
    // at a time. Only the 2-way class axis is fixed, and that is the axis that matters —
    // it is what `ID2LABEL` indexes into.
    IoSpec {
        file: GENDER_MODEL,
        inputs: &[("input_values", &[B, B])],
        outputs: &[("logits", &[B, Some(2)])],
    },
];

pub fn io_spec(file: &str) -> Option<&'static IoSpec> {
    IO_SPECS.iter().find(|s| s.file == file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_is_a_strict_superset_of_small() {
        let fast = required_files(ModelSet::Fast, true);
        let small = required_files(ModelSet::Small, true);
        for f in &small {
            assert!(fast.contains(f), "fast set is missing {f} which small requires");
        }
        assert!(small.len() < fast.len());
    }

    #[test]
    fn set_counts_match_the_shipped_directories() {
        // models_folded/ has 24 files, models_small/ has 19 (marker excluded).
        assert_eq!(required_files(ModelSet::Fast, true).len(), 24);
        assert_eq!(required_files(ModelSet::Small, true).len(), 19);
    }

    #[test]
    fn small_is_fast_minus_exactly_the_fast_only_files() {
        let fast = required_files(ModelSet::Fast, true);
        let small = required_files(ModelSet::Small, true);
        let mut diff: Vec<&str> = fast
            .iter()
            .filter(|f| !small.contains(f))
            .copied()
            .collect();
        diff.sort_unstable();
        // The real difference, measured against the shipped dirs — MODELS_SETS.md used to
        // claim it was a single file.
        assert_eq!(
            diff,
            vec![
                "gender-wav2vec2.meta.json",
                "segmentation-3.0-b64.onnx",
                "wespeaker-multimask-tail-b64.onnx",
                "wespeaker-voxceleb-resnet34-b64.onnx",
                "wespeaker-voxceleb-resnet34-tail-b64.onnx",
            ]
        );
        assert_eq!(diff, fast_only_files(true));
    }

    #[test]
    fn skipping_gender_drops_both_gender_artifacts() {
        let files = required_files(ModelSet::Fast, false);
        assert!(!files.contains(&GENDER_MODEL));
        assert!(!files.contains(&GENDER_META));
        assert_eq!(files.len(), 22);
    }

    #[test]
    fn coverage_is_one_directional() {
        assert!(ModelSet::Fast.covers(ModelSet::Small));
        assert!(ModelSet::Fast.covers(ModelSet::Fast));
        assert!(ModelSet::Small.covers(ModelSet::Small));
        // A small dir cannot serve a fast request: the b64 graphs are simply absent, and
        // SPEAKRS_LAZY_SESSIONS=1 would hide that until the first batched job.
        assert!(!ModelSet::Small.covers(ModelSet::Fast));
    }

    #[test]
    fn every_required_onnx_has_an_io_spec_except_the_documented_copy() {
        for f in onnx_files(ModelSet::Fast, true) {
            if f == "wespeaker-multimask-tail-b64.onnx" {
                assert!(io_spec(f).is_none(), "the b64 multimask copy must NOT be spec'd");
                continue;
            }
            assert!(io_spec(f).is_some(), "no I/O spec for {f}");
        }
    }

    #[test]
    fn plda_specs_match_the_byte_sizes_on_disk() {
        // 128-byte header + payload. Confirms the table is self-consistent with the sizes
        // observed in models_folded/, which is the check that caught the wrong dtypes.
        let expected: &[(&str, usize)] = &[
            ("plda_lda.npy", 131200),
            ("plda_mean1.npy", 2176),
            ("plda_mean2.npy", 640),
            ("plda_mu.npy", 1152),
            ("plda_psi.npy", 1152),
            ("plda_tr.npy", 131200),
        ];
        for (name, dtype, shape) in PLDA_SPECS {
            let elems: usize = shape.iter().product();
            let bytes = 128 + elems * dtype.size();
            let want = expected.iter().find(|(n, _)| n == name).unwrap().1;
            assert_eq!(bytes, want, "{name} size mismatch");
        }
    }

    #[test]
    fn set_parsing_round_trips() {
        assert_eq!(ModelSet::parse("fast"), Some(ModelSet::Fast));
        assert_eq!(ModelSet::parse("SMALL"), Some(ModelSet::Small));
        assert_eq!(ModelSet::parse("nonsense"), None);
        assert_eq!(ModelSet::parse(ModelSet::Fast.as_str()), Some(ModelSet::Fast));
    }
}
