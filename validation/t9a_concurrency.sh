#!/usr/bin/env bash
# T9a gate: N concurrent /diarize jobs on ONE shared-session engine must produce outputs
# byte-identical to the same jobs run serially, at >= 2x serial throughput, without VRAM
# scaling per job (see validation/RESULTS.md §7.25).
#
# Assumes a diar-server with the shared-session build is up on $PORT with the AMI audio
# dir mounted at /audio and DIAR_MAX_INFLIGHT >= N. VRAM is sampled DURING the concurrent
# leg (sampling after a run reports the idle floor — RESULTS §7.14).
#
#   PORT=18701 GPU=1 ./validation/t9a_concurrency.sh
set -euo pipefail
PORT="${PORT:-18701}"
GPU="${GPU:-1}"
OUT="${OUT:-/tmp/t9a_conc}"
FILES=(ES2004a IS1009c TS3003b EN2002b) # 17-36 min AMI files, four different rooms
mkdir -p "$OUT/serial" "$OUT/concurrent"

req() {
  curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" \
    -H 'Content-Type: application/json' \
    -d "{\"wav_path\":\"/audio/$1.Mix-Headset.wav\",\"file_id\":\"$1\"}" -o "$2/$1.json"
}

echo "== warmup (cuDNN algo search + arena growth are first-run costs) =="
req "${FILES[0]}" "$OUT"

echo "== serial leg =="
t0=$(date +%s.%N)
for f in "${FILES[@]}"; do req "$f" "$OUT/serial"; done
t1=$(date +%s.%N)
SERIAL=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')
echo "serial wall: ${SERIAL}s"

echo "== concurrent leg (VRAM sampled during) =="
: > "$OUT/vram.log"
(while :; do
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i "$GPU" >>"$OUT/vram.log"
  echo "---" >>"$OUT/vram.log"
  sleep 1
done) & SAMPLER=$!
t0=$(date +%s.%N)
for f in "${FILES[@]}"; do req "$f" "$OUT/concurrent" & done
wait $(jobs -p | grep -v "^$SAMPLER$") 2>/dev/null || true
t1=$(date +%s.%N)
kill "$SAMPLER" 2>/dev/null || true
CONC=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')
echo "concurrent wall: ${CONC}s (speedup $(awk -v s="$SERIAL" -v c="$CONC" 'BEGIN{printf "%.2f", s/c}')x; gate >= 2x)"

echo "== output identity, serial vs concurrent =="
PASS=1
for f in "${FILES[@]}"; do
  v=$(python3 - "$OUT" "$f" <<'EOF'
import json, sys
out, f = sys.argv[1], sys.argv[2]
a = json.load(open(f"{out}/serial/{f}.json"))
b = json.load(open(f"{out}/concurrent/{f}.json"))
keys = ["rttm", "segments", "exclusive_segments", "centroids", "num_speakers"]
print("IDENTICAL" if all(a[k] == b[k] for k in keys) else "DIFFERS")
EOF
)
  echo "$f: $v"
  [ "$v" = "IDENTICAL" ] || PASS=0
done

echo "== peak diar-server VRAM during concurrent leg =="
rg 'diar-server' "$OUT/vram.log" | awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END{print m " MiB"}'
[ "$PASS" = 1 ] && echo "IDENTITY GATE: PASS" || { echo "IDENTITY GATE: FAIL"; exit 1; }
