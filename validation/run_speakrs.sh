#!/usr/bin/env bash
# Run speakrs (diar-bench image) over audio files, one process per file per run,
# emitting harness-layout RTTMs (<label>_run<N>.rttm) plus a timing JSONL.
# Exit codes are NOT trusted (known ORT-CUDA teardown crash after results flush);
# output validity is judged by RTTM content downstream.
#
# Usage: run_speakrs.sh <mode> <gpu|none> <out_dir> <runs> <label:path> [label:path ...]
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=$1; GPU=$2; OUT=$3; RUNS=$4; shift 4
mkdir -p "$OUT"
GPU_ARGS=()
if [ "$GPU" != "none" ]; then
  GPU_ARGS=(--gpus "\"device=$GPU\"")
fi
for spec in "$@"; do
  label="${spec%%:*}"; path="${spec#*:}"
  dir=$(dirname "$path"); file=$(basename "$path")
  for run in $(seq 0 $((RUNS - 1))); do
    start=$(date +%s.%N)
    docker run --rm "${GPU_ARGS[@]}" \
      -v "$REPO_ROOT/models":/models:ro \
      -v "$dir":/audio:ro \
      diar-bench:latest diarize --mode "$MODE" --models-dir /models "/audio/$file" \
      > "$OUT/${label}_run${run}.rttm" 2>> "$OUT/${label}.log"
    end=$(date +%s.%N)
    lines=$(wc -l < "$OUT/${label}_run${run}.rttm")
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.2f", b-a}')
    echo "{\"label\":\"$label\",\"run\":$run,\"mode\":\"$MODE\",\"elapsed_s\":$elapsed,\"rttm_lines\":$lines}" \
      | tee -a "$OUT/timing.jsonl"
  done
done
