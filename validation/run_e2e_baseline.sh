#!/usr/bin/env bash
# T1 E2E baseline: one engine configuration over the benchmark corpus, 3 runs per file.
# Timed legs are strictly sequential (RESULTS §4.11) — never run two of these at once.
#
#   usage: run_e2e_baseline.sh <label>
#   env:   BENCHMARK_EMAIL / BENCHMARK_PASSWORD
set -euo pipefail

LABEL="${1:?usage: run_e2e_baseline.sh <label>}"
APP=/mnt/nvm/repos/transcribe-app
PY=/mnt/nvm/repos/diar-native/venv/bin/python
OUT=/mnt/nvm/repos/diar-native/results/e2e_baseline/$LABEL
mkdir -p "$OUT"

# file_uuid:short_name — ordered shortest first so a failure surfaces cheaply
FILES=(
  "01a01aba-d3f9-7000-95b2-fb035b988781:test_ai_video_24s"
  "01a01a87-60a3-7000-808b-a7b9df231801:pyramids_239s"
  "019fd6b2-c2b7-7000-a7b1-0e6974dd62da:warpdrive_358s"
  "01a01aba-d9d2-7000-a027-9a0af248574f:karpathy_3989s"
  "019f2950-0f56-7000-80d9-e175004cc186:seed_7558s"
)

cd "$APP"
for entry in "${FILES[@]}"; do
  uuid="${entry%%:*}"; name="${entry##*:}"
  echo "=== [$LABEL] $name ==="
  "$PY" scripts/benchmark_e2e.py --file-uuid "$uuid" --iterations 3 --timeout 3600 \
      --output "$OUT/$name.csv" 2>&1 | tail -25
done
echo "=== [$LABEL] complete -> $OUT ==="
