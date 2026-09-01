#!/usr/bin/env bash
# Reproduce issue #14 end to end from an Apple Silicon Mac: the NATIVE macOS arm64 result and
# the linux/arm64 result (via Docker, which runs arm64 containers natively here), side by side
# against the same model files.
#
#   validation/ort_fusion_probe/run_probe.sh <models-dir> [out-dir]
#
# Read docs/ORT_FUSION_FP16_AARCH64.md first — this is for verifying a FIX or asking the same
# question of a new graph, not for repeating the 2026-09-01 investigation. Its findings are in
# validation/RESULTS.md §7.40 and must not be silently re-derived or overwritten.
#
# Not a timed leg: every line of output is a load/no-load, graph-shape or output-identity
# fact. None of it is a benchmark, so BENCHMARK_PROTOCOL's quiet-machine rule does not apply.
set -euo pipefail

MODELS="${1:?usage: run_probe.sh <models-dir> [out-dir]}"
OUT="${2:-/tmp/ort_fusion_probe}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$REPO/validation/ort_fusion_probe/Cargo.toml"

[ -d "$MODELS" ] || { echo "no such models dir: $MODELS" >&2; exit 1; }
MODELS="$(cd "$MODELS" && pwd)"
mkdir -p "$OUT"/{dumps_mac,dumps_linux,bin}

# numpy + onnx. Any venv with them works; provisioning's environment already has both.
PY="${PROBE_PYTHON:-python3}"
"$PY" "$REPO/validation/ort_fusion_probe/make_clips.py" "$OUT/clips.bin"

echo
echo "=============================================================="
echo " NATIVE macOS arm64 (the platform where this does NOT fail)"
echo "=============================================================="
# Homebrew openblas is not on the default linker path; speakrs needs it and so does anything
# built alongside it.
export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:${LIBRARY_PATH:-}"
cargo build --release --manifest-path "$MANIFEST" --target-dir "$OUT/bin" >/dev/null
PROBE="$OUT/bin/release/ort-fusion-probe"

echo "--- every graph, load + optimizer rewrite ---"
"$PROBE" load "$OUT/dumps_mac" "$MODELS"/*.onnx
echo
"$PY" "$REPO/validation/ort_fusion_probe/inspect_dumps.py" "$OUT/dumps_mac"
echo
echo "--- gender model, per configuration (L0 = the unoptimized reference) ---"
"$PROBE" run "$MODELS/gender-wav2vec2.onnx" "$OUT/clips.bin" "$OUT/dumps_mac" \
    L3 L3:GeluFusionL2 L1 L0

echo
echo "=============================================================="
echo " linux/arm64 via Docker (the platform where this DOES fail)"
echo "=============================================================="
if ! docker version >/dev/null 2>&1; then
    echo "SKIPPED: docker is not available. The macOS half above still stands."
    exit 0
fi
# rust:1-trixie, NOT bookworm: this ORT needs glibc >= 2.38 (__isoc23_strtol) and bookworm's
# 2.36 dies at link. -lstdc++ is needed or the static ORT lib fails on __cxa_call_terminate.
docker run --rm --platform linux/arm64 \
    -v "$REPO":/repo:ro -v "$MODELS":/models:ro -v "$OUT":/out \
    -w /out rust:1-trixie bash -euc '
        export RUSTFLAGS="-C link-arg=-lstdc++"
        cargo build --release --manifest-path /repo/validation/ort_fusion_probe/Cargo.toml \
            --target-dir /out/bin_linux >/dev/null 2>&1
        P=/out/bin_linux/release/ort-fusion-probe
        echo "--- arch: $(uname -m), glibc: $(ldd --version | head -1) ---"
        echo
        echo "--- every graph, load + optimizer rewrite ---"
        $P load /out/dumps_linux /models/*.onnx
        echo
        echo "--- gender model, per configuration (L0 = the unoptimized reference) ---"
        $P run /models/gender-wav2vec2.onnx /out/clips.bin /out/dumps_linux \
            L3 L3:GeluFusion L3:GeluFusionL2 L1 L0
    ' 2>&1 | grep -v "cpuid_info warning"

echo
"$PY" "$REPO/validation/ort_fusion_probe/inspect_dumps.py" "$OUT/dumps_linux" || true
echo
echo "Expected (RESULTS §7.40): gender L3 and L3:GeluFusion FAIL on linux/arm64;"
echo "L3:GeluFusionL2, L1 and L0 load. L1 is bitwise identical to L0. macOS: all load."
