//! Platform workarounds for ORT session construction.
//!
//! Kept in one place because the same workaround must be applied at every site that builds a
//! session. A session built without it fails at LOAD time with an error naming an operator our
//! exports do not contain, which sends the reader hunting for a corrupt file that is fine.

use std::path::Path;

use anyhow::Result;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

/// The gender classifier is the only graph that needs a workaround today.
const GENDER_MODEL_FILE: &str = "gender-wav2vec2.onnx";

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
            .is_some_and(|n| n == GENDER_MODEL_FILE)
}

/// Build a session for `path`, applying any platform workaround that model needs.
///
/// # The aarch64 fp16 GELU problem (issue #14)
///
/// The fp16 gender graph does not load at all on aarch64 without this. The exported file is
/// plain opset-17 `ai.onnx` with no contrib domain — but it has 20 `Erf` nodes, and one of ORT's
/// *extended* (level-2) optimizations rewrites that GELU pattern into a `com.microsoft.Gelu`
/// node. The x86_64 ORT build ships an fp16 kernel for that fused contrib op; the aarch64 build
/// ships fp32 only. So the optimizer synthesizes a node the very same runtime then refuses to
/// execute:
///
/// ```text
/// Failed to find kernel for com.microsoft.Gelu(1) ... This op has been implemented only for
/// the following types (tensor(float),), but the node in the model has the following type
/// (tensor(float16))
/// ```
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
/// Naming the optimizer does NOT suppress the rewrite; only capping below level 2 does. That is
/// why this sets a level rather than a disable-list, despite the disable-list being the more
/// surgical-looking option. `Level1` rather than `Disable` so the model keeps every level-1
/// optimization.
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
    if let Ok(level) = std::env::var("DIAR_ORT_OPT_LEVEL") {
        if let Some(lvl) = parse_level(&level) {
            return builder
                .with_optimization_level(lvl)
                .map_err(|e| anyhow::anyhow!("setting ORT optimization level to {level}: {e}"));
        }
    }
    if let Ok(disabled) = std::env::var("DIAR_ORT_DISABLED_OPTIMIZERS") {
        if !disabled.trim().is_empty() {
            return builder
                .with_disabled_optimizers(&disabled)
                .map_err(|e| anyhow::anyhow!("disabling optimizers [{disabled}]: {e}"));
        }
    }
    if needs_aarch64_fp16_workaround(path) {
        return builder
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("capping ORT optimization level for {path:?}: {e}"));
    }
    Ok(builder)
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

    #[test]
    fn only_the_gender_model_is_singled_out() {
        // On x86_64 nothing is singled out at all; on aarch64 only the gender model is.
        assert_eq!(
            needs_aarch64_fp16_workaround(Path::new("/m/gender-wav2vec2.onnx")),
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
}
