//! Platform workarounds for ORT session construction.
//!
//! Kept in one place because the same workaround must be applied at every site that builds a
//! session. A session built without it fails at LOAD time with an error naming an operator our
//! exports do not contain, which sends the reader hunting for a corrupt file that is fine.

use std::path::Path;

use anyhow::Result;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

// The gender classifier is the only graph that needs a workaround today, and the name is
// IMPORTED rather than restated: this workaround and the `verify-models` stage-1 gate that
// polices it are both scoped by FILENAME, so a local copy that drifted from the provisioned
// name would disable both — silently, and only on aarch64.
use crate::provision::files::GENDER_MODEL;

/// Does this model need the aarch64 fp16 workaround?
///
/// Scoped to the gender model deliberately. The 15 diarization graphs load and run correctly at
/// full optimization on aarch64 — verified end to end — so capping them too would give up
/// optimizations on the hot path to fix a problem they do not have.
fn needs_aarch64_fp16_workaround(path: &Path) -> bool {
    cfg!(target_arch = "aarch64")
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == GENDER_MODEL)
}

/// Build a session for `path`, applying any platform workaround that model needs.
///
/// # The aarch64 fp16 GELU problem (issue #14)
///
/// The fp16 gender graph does not load at all on linux/aarch64 without this. The exported file
/// is plain opset-17 `ai.onnx` with no contrib domain — but it has 20 `Erf` nodes, and one of
/// ORT's *extended* (level-2) optimizations rewrites that GELU pattern into a
/// `com.microsoft.Gelu` node, for which no fp16 kernel is then found:
///
/// ```text
/// Failed to find kernel for com.microsoft.Gelu(1) ... This op has been implemented only for
/// the following types (tensor(float),), but the node in the model has the following type
/// (tensor(float16))
/// ```
///
/// NOT a plain "aarch64 has no fp16 kernel" story, though that was the first reading and it is
/// wrong: RESULTS §7.40 measured macOS arm64 lacking the SAME kernel and loading fine. What
/// differs is whether the fusion GATE fires, so the node is only ever created on some builds.
/// The distinction matters because it means the trigger is build configuration, not
/// architecture — which is why the escape hatches below exist at all.
///
/// Capping the level is what works, and it was established by experiment on real aarch64 rather
/// than reasoned about. MEASURED, all four on the same image and models:
///
/// ```text
/// disable GeluFusion                                          -> STILL FAILS
/// disable GeluFusion,BiasGeluFusion,FastGeluFusion,GeluApprox  -> STILL FAILS
/// optimization level = basic (Level1)                          -> LOADS, gender=2 verdicts
/// optimization level = disable                                 -> LOADS
/// ```
///
/// A LEVEL CAP IS STILL THE RIGHT FIX, but not because naming the optimizer is impossible —
/// it is because every name that works is a trap. RESULTS §7.40 (measured on linux/arm64 and
/// macOS arm64, same pinned `ort`) found the disable-list does work, under a name neither of
/// the two attempts above used:
///
/// ```text
/// disable GeluFusion            -> STILL FAILS   (the pass is registered twice, L1 and L2)
/// disable GeluFusionL1          -> STILL FAILS
/// disable GeluFusionL2          -> LOADS
/// ```
///
/// Three reasons the level cap is preferred anyway, all measured:
///
/// 1. `Level1` is BITWISE IDENTICAL to `Disable` on this graph (max |Δ logit| 0.000e+00 over
///    the 6-clip gate corpus). `GeluFusionL2` differs by 9.58e-04 — harmless, but weaker.
/// 2. `GeluFusionL2` leaves `NhwcFusedConv` x8 and `SkipLayerNormalization` x12 running on
///    fp16 tensors. They have fp16 kernels here or the session would not open, but that is
///    the same accident of build configuration that produced this bug. `Level1` leaves zero
///    contrib ops on fp16.
/// 3. The name is an undocumented ORT internal that has already been renamed once, and a
///    WRONG NAME IS SILENTLY IGNORED (see the escape hatch below).
///
/// `Level1` rather than `Disable` so the model keeps every level-1 optimization.
///
/// Reverting the model to fp32 would also "work", but costs back the 189 MB of disk and
/// ~500 MiB of VRAM that RESULTS §7.39 won, on every platform, to fix one.
pub fn session_for(path: &Path) -> Result<Session> {
    let builder =
        Session::builder().map_err(|e| anyhow::anyhow!("ORT session builder unavailable: {e}"))?;
    let mut builder = apply_workarounds(builder, path)?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Apply workarounds to an existing builder (for callers that must configure it further, e.g.
/// the gender session's execution provider).
pub fn apply_workarounds(builder: SessionBuilder, path: &Path) -> Result<SessionBuilder> {
    // Escape hatches. These exist because the failure is a property of the ORT BUILD, not of our
    // code or our models, so an untested platform can hit the same class of problem with a
    // different operator — and an operator should not have to wait for a release.
    //
    // They COMPOSE with the built-in workaround rather than replacing it, and they compose with
    // each other. Both used to return early, so setting either one silently un-did the aarch64
    // cap and disabled speaker gender with no error anywhere — an operator tuning the
    // DIARIZATION graphs with `DIAR_ORT_OPT_LEVEL=all` would lose gender and have nothing to
    // read. The variable is global; the bug is per-model.
    let mut builder = builder;

    // The level is a FLOOR, deliberately asymmetric: the hatch can LOWER the optimization level
    // for any model, but it cannot RAISE it past the point where a model stops loading. Lowering
    // is always safe (fewer rewrites, and Level1 is already bitwise identical to Disable on the
    // gender graph — RESULTS §7.40); raising it past the cap is the exact configuration that is
    // known to fail. To explore above the cap, use `validation/ort_fusion_probe`, which reports
    // load success per configuration instead of half-starting a server.
    let requested = match std::env::var("DIAR_ORT_OPT_LEVEL") {
        Ok(level) => match parse_level(&level) {
            Some(lvl) => Some(lvl),
            // A typo used to fall through in silence, leaving the operator convinced they had
            // changed something. That is the same failure this whole module exists to document
            // in ORT itself (an unrecognized optimizer NAME is silently ignored) — reproducing
            // it in our own escape hatch would be indefensible. Hard error: the variable is set
            // deliberately, so a value we cannot honour is a mistake worth stopping for.
            None => anyhow::bail!(
                "DIAR_ORT_OPT_LEVEL={level:?} is not a recognized value. \
                 Use one of: disable|none|0, basic|1, extended|2, all|3."
            ),
        },
        Err(_) => None,
    };
    let cap = needs_aarch64_fp16_workaround(path).then_some(GraphOptimizationLevel::Level1);
    if let Some(level) = lower_of(requested, cap) {
        builder = builder
            .with_optimization_level(level)
            .map_err(|e| anyhow::anyhow!("setting ORT optimization level for {path:?}: {e}"))?;
    }

    // TWO TRAPS, both measured (RESULTS §7.40) — this variable can silently do nothing:
    //   * ORT SILENTLY IGNORES an unrecognized optimizer name. No error, no warning; the
    //     session opens and behaves exactly as if the variable were unset. So a typo, or a
    //     plausible-but-wrong name, looks applied and is not. `GeluFusion` is such a name —
    //     the pass that matters is `GeluFusionL2`.
    //   * THE SEPARATOR IS `;`, NOT `,`, despite ort's own doc comment on
    //     `with_disabled_optimizers` saying "comma-separated". `A;B` disables both; `A,B`
    //     disables NEITHER, because the whole string is taken as one name and matches nothing.
    // The second one we CAN catch, so we do — see below. The first we cannot, because ORT
    // exposes no list of registered optimizer names to validate against; verify any value set
    // here actually took effect with `validation/ort_fusion_probe`.
    if let Ok(disabled) = std::env::var("DIAR_ORT_DISABLED_OPTIMIZERS") {
        let disabled = disabled.trim();
        if !disabled.is_empty() {
            // Confirmed on real linux/arm64 against the shipped artifact: `GeluFusionL1;GeluFusionL2`
            // loads, `GeluFusionL1,GeluFusionL2` does not — the comma form matches no optimizer
            // at all and disables nothing. Rejecting is strictly better than accepting a value
            // that quietly does nothing, because a silent no-op stops the operator looking for
            // the real cause.
            if disabled.contains(',') {
                anyhow::bail!(
                    "DIAR_ORT_DISABLED_OPTIMIZERS={disabled:?} uses ',' as a separator, which \
                     ORT does not accept — the whole string is taken as ONE optimizer name, \
                     matches nothing, and disables nothing SILENTLY. Separate names with ';' \
                     instead, e.g. \"GeluFusionL1;GeluFusionL2\"."
                );
            }
            builder = builder
                .with_disabled_optimizers(disabled)
                .map_err(|e| anyhow::anyhow!("disabling optimizers [{disabled}]: {e}"))?;
        }
    }
    Ok(builder)
}

/// Does this graph load with NO workaround at all, at ORT's own default optimization level?
///
/// The regression detector for issue #14's whole class, used by `verify-models` stage 1.
///
/// AN ACCURACY GATE CANNOT CATCH THIS CLASS. The session never opens far enough to produce a
/// number, so there is nothing to compare — the only observable is whether the load succeeds.
/// If someone later "improves" this into a numeric check, the gate is silently gone. It must
/// stay a LOAD check.
///
/// The workaround is scoped by FILENAME to the gender model. That is correct today, but it
/// means a future fp16 export of any other graph would need the scope widened, and the symptom
/// would otherwise be a server that will not start on arm64 hosts only. This tells us by name
/// and at provisioning time instead.
pub fn loads_without_workaround(path: &Path) -> bool {
    // `ort`'s builder errors carry the builder back for recovery, so their error types differ
    // per step and do not chain through `and_then`; a closure keeps `?` usable.
    let attempt = || -> Result<()> {
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };
    attempt().is_ok()
}

/// The lower of two optional levels — the floor rule in one place so it can be tested directly.
///
/// `GraphOptimizationLevel` derives `Ord` in declaration order (Disable < Level1 < Level2 <
/// Level3 < All), so "lower" is just `min` and stays correct if ORT adds another level.
/// `None` means "unconstrained", so it never lowers anything; two `None`s mean leave ORT's own
/// default alone.
fn lower_of(
    a: Option<GraphOptimizationLevel>,
    b: Option<GraphOptimizationLevel>,
) -> Option<GraphOptimizationLevel> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn parse_level(s: &str) -> Option<GraphOptimizationLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "disable" | "none" | "0" => Some(GraphOptimizationLevel::Disable),
        "basic" | "1" => Some(GraphOptimizationLevel::Level1),
        "extended" | "2" => Some(GraphOptimizationLevel::Level2),
        "all" | "3" => Some(GraphOptimizationLevel::Level3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gender model's filename has exactly ONE definition, and every consumer agrees.
    ///
    /// This used to be three independent string literals — here, in `gender.rs`, and in
    /// `provision/files.rs`. Renaming the export in one of them would have left the aarch64
    /// `Level1` cap and the `verify-models` stage-1 gate pointed at a file that no longer
    /// existed: both are scoped by `file_name()` comparison, so both would have quietly
    /// stopped applying, on aarch64 only, with the server still returning HTTP 200 and gender
    /// simply absent. There is no x86_64 symptom at all, which is what made it worth pinning.
    ///
    /// The aliases make divergence a compile error today; this test is what fails if someone
    /// re-inlines a literal into any of the three.
    #[test]
    fn gender_model_filename_has_exactly_one_definition() {
        use crate::provision::files::{self, ModelSet};

        // 1. The three spellings are the same string.
        assert_eq!(
            GENDER_MODEL,
            crate::gender::GENDER_MODEL_FILE,
            "ort_compat and gender.rs disagree on the gender model filename; the aarch64 \
             workaround is scoped by that name and no longer covers the file gender.rs loads"
        );
        assert_eq!(GENDER_MODEL, files::GENDER_MODEL);

        // 2. The name provisioning WRITES is the name the workaround MATCHES. This is the
        //    coupling that actually breaks; the equality above is only its proxy.
        for set in [ModelSet::Fast, ModelSet::Small] {
            assert!(
                files::required_files(set, true).contains(&GENDER_MODEL),
                "{set:?} provisions a gender model under some other name than {GENDER_MODEL}"
            );
        }

        // 3. And the predicate fires on the path the runtime actually builds — `gender.rs`
        //    joins its own constant onto the models dir, so that is the path under test.
        assert_eq!(
            needs_aarch64_fp16_workaround(
                &Path::new("/models").join(crate::gender::GENDER_MODEL_FILE)
            ),
            cfg!(target_arch = "aarch64"),
            "the workaround does not fire on the path gender.rs loads"
        );
    }

    #[test]
    fn only_the_gender_model_is_singled_out() {
        // On x86_64 nothing is singled out at all; on aarch64 only the gender model is.
        assert_eq!(
            needs_aarch64_fp16_workaround(&Path::new("/m").join(GENDER_MODEL)),
            cfg!(target_arch = "aarch64")
        );
        for other in [
            "segmentation-3.0.onnx",
            "wespeaker-multimask-tail-b32.onnx",
            "wespeaker-voxceleb-resnet34.onnx",
        ] {
            assert!(
                !needs_aarch64_fp16_workaround(&Path::new("/m").join(other)),
                "{other} must keep full optimization: it loads fine on aarch64"
            );
        }
    }

    #[test]
    fn the_level_hatch_is_a_floor_it_can_lower_but_never_raise_past_the_cap() {
        use GraphOptimizationLevel as G;
        let cap = Some(G::Level1); // what the gender model gets on aarch64

        // The bug this pins: `DIAR_ORT_OPT_LEVEL=all` on an aarch64 host, set to tune the
        // DIARIZATION graphs, used to raise the gender session back to Level3 and silently
        // stop it loading. Raising past the cap must not be possible.
        assert_eq!(lower_of(Some(G::Level3), cap), Some(G::Level1));
        assert_eq!(lower_of(Some(G::Level2), cap), Some(G::Level1));

        // Lowering is always allowed: fewer rewrites can never reintroduce the fused op.
        assert_eq!(lower_of(Some(G::Disable), cap), Some(G::Disable));

        // No cap (x86_64, or any model that is not the gender one): the hatch is unrestricted.
        assert_eq!(lower_of(Some(G::Level3), None), Some(G::Level3));
        // No hatch: the cap alone applies.
        assert_eq!(lower_of(None, cap), Some(G::Level1));
        // Neither: leave ORT's own default alone rather than pinning one.
        assert_eq!(lower_of(None, None), None);
    }

    #[test]
    fn an_unrecognized_opt_level_is_an_error_not_a_silent_no_op() {
        // The bug this pins: a typo used to fall through in silence, so the operator believed
        // they had changed the optimization level and had not.
        assert!(parse_level("bsaic").is_none());
        assert!(parse_level("").is_none());
        assert!(parse_level("4").is_none());
    }

    #[test]
    fn level_parsing_accepts_names_and_numbers() {
        assert!(matches!(
            parse_level("basic"),
            Some(GraphOptimizationLevel::Level1)
        ));
        assert!(matches!(
            parse_level(" ALL "),
            Some(GraphOptimizationLevel::Level3)
        ));
        assert!(matches!(
            parse_level("0"),
            Some(GraphOptimizationLevel::Disable)
        ));
        assert!(parse_level("nonsense").is_none());
    }

    #[test]
    fn a_comma_separated_optimizer_list_is_rejected_rather_than_silently_ignored() {
        // Measured on real linux/arm64 against the shipped artifact (RESULTS §7.40):
        //   GeluFusionL1;GeluFusionL2 -> loads    GeluFusionL1,GeluFusionL2 -> fails
        // ORT takes the comma form as ONE optimizer name, matches nothing, and disables
        // nothing — with no error. `apply_workarounds` refuses it instead. This test pins the
        // detection rule; the refusal itself is exercised through the env var, which tests
        // cannot set safely in parallel.
        for bad in ["GeluFusionL1,GeluFusionL2", "A, B", "GeluFusionL2,"] {
            assert!(
                bad.contains(','),
                "{bad} should be caught by the comma rule"
            );
        }
        for ok in [
            "GeluFusionL2",
            "GeluFusionL1;GeluFusionL2",
            "  GeluFusionL2  ",
        ] {
            assert!(
                !ok.contains(','),
                "{ok} must not be caught by the comma rule"
            );
        }
    }
}
