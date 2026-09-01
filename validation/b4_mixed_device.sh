#!/usr/bin/env bash
# B4 gate (RESULTS §7.34 "NOT measured here"): mixed-device concurrency.
#
# N concurrent CUDA /diarize alongside M concurrent CPU /embed_window on ONE server with
# DIAR_DEVICES=cuda,cpu. PASS = the CUDA leg's wall time stays within noise of a CUDA-ONLY
# leg. This is where §4.15 (CPU fbank is up to 76% of CUDA E2E wall time) and §7.32 (CPU
# embedding sessions take up to 6 intra-op threads) collide.
#
# The two legs are INTERLEAVED (cuda_only, mixed, cuda_only, mixed, ...) rather than run as
# all-A-then-all-B, so slow drift in background host load cannot masquerade as an effect.
# VRAM is sampled DURING each leg (sampling after a run reports the idle floor — §7.14).
#
# Assumes a diar-server is up on $PORT with the AMI audio dir mounted at /audio and
# DIAR_MAX_INFLIGHT >= N+M (otherwise the outer admission gate serialises the mix and there
# is no contention left to measure — that gate is the shipped default and is itself the
# answer at DIAR_MAX_INFLIGHT=2).
#
#   PORT=18711 GPU=0 ROUNDS=3 ./validation/b4_mixed_device.sh
set -euo pipefail
PORT="${PORT:-18711}"
GPU="${GPU:-0}"
OUT="${OUT:-/tmp/b4_mixed}"
ROUNDS="${ROUNDS:-3}"
# CUDA leg: four 17-36 min AMI files, four different rooms (same set as t9a_concurrency.sh)
FILES=(ES2004a IS1009c TS3003b EN2002b)
# CPU load generators: M concurrent /embed_window loops, run for the whole CUDA leg
M="${M:-4}"
CPU_CLIP="${CPU_CLIP:-/audio/EN2002c_360.wav}"

mkdir -p "$OUT"

cuda_req() { # $1=file $2=outdir
  curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" \
    -H 'Content-Type: application/json' \
    -d "{\"wav_path\":\"/audio/$1.Mix-Headset.wav\",\"file_id\":\"$1\",\"device\":\"cuda\"}" \
    -o "$2/$1.json"
}

# One CPU /embed_window. Windows are staggered per worker so the M loops are not all
# embedding byte-identical audio (which a cache could collapse).
cpu_req() { # $1=worker index  $2=counter file
  local s=$(( 10 + ($1 * 7) % 300 ))
  curl -s -m 600 -X POST "http://localhost:$PORT/embed_window" \
    -H 'Content-Type: application/json' \
    -d "{\"wav_path\":\"$CPU_CLIP\",\"start_s\":$s,\"end_s\":$((s+4)),\"device\":\"cpu\"}" \
    -o /dev/null && echo x >> "$2"
}

# NOTE: the sampler subshell's stdout MUST be redirected away. cuda_leg is called inside a
# command substitution, and a background job that inherits that substitution's stdout holds
# the pipe open forever — the substitution then never returns and the whole harness hangs
# before it launches a single request. (Cost 12 min of wall clock to find; do not "simplify"
# this back.) SAMPLER is a global for the same reason: `x=$(start_sampler)` reintroduces it.
SAMPLER=""
start_sampler() { # $1=logfile
  : > "$1"
  (while :; do
     echo "T $(date +%s.%N) $(uptime | sed 's/.*load average: //')" >> "$1"
     nvidia-smi --query-compute-apps=pid,process_name,used_memory \
       --format=csv,noheader -i "$GPU" >> "$1"
     echo "---" >> "$1"
     sleep 1
   done) >/dev/null 2>&1 &
  SAMPLER=$!
}

# Runs the N CUDA jobs concurrently and prints their wall time.
# $1 = outdir, $2 = vram log
cuda_leg() {
  start_sampler "$2"; local sampler=$SAMPLER
  local t0 t1 pids=()
  t0=$(date +%s.%N)
  for f in "${FILES[@]}"; do cuda_req "$f" "$1" & pids+=($!); done
  wait "${pids[@]}"
  t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null || true
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}'
}

echo "== warmup (cuDNN algo search + arena growth are first-run costs) =="
mkdir -p "$OUT/warmup"; cuda_req "${FILES[0]}" "$OUT/warmup"
curl -s -m 600 -X POST "http://localhost:$PORT/embed_window" -H 'Content-Type: application/json' \
  -d "{\"wav_path\":\"$CPU_CLIP\",\"start_s\":10,\"end_s\":14,\"device\":\"cpu\"}" -o /dev/null

for r in $(seq 1 "$ROUNDS"); do
  # ---- leg A: CUDA only (control) ----
  mkdir -p "$OUT/cuda_only_r$r"
  A=$(cuda_leg "$OUT/cuda_only_r$r" "$OUT/vram_cuda_only_r$r.log")
  echo "round $r  cuda_only : ${A}s   la=$(uptime | sed 's/.*load average: //')"
  sleep 5

  # ---- leg B: CUDA + M CPU embed loops ----
  mkdir -p "$OUT/mixed_r$r"
  CNT="$OUT/cpu_count_r$r"; : > "$CNT"
  STOP="$OUT/stop_r$r"; rm -f "$STOP"
  loops=()
  for m in $(seq 1 "$M"); do
    (while [ ! -f "$STOP" ]; do cpu_req "$m" "$CNT" || true; done) & loops+=($!)
  done
  sleep 2   # let the CPU engines get busy before the CUDA leg starts
  B=$(cuda_leg "$OUT/mixed_r$r" "$OUT/vram_mixed_r$r.log")
  touch "$STOP"; wait "${loops[@]}" 2>/dev/null || true
  echo "round $r  mixed     : ${B}s   cpu_reqs=$(wc -l < "$CNT")   la=$(uptime | sed 's/.*load average: //')"
  echo "$r $A $B" >> "$OUT/walls.txt"
  sleep 5
done

echo
echo "== wall times (s) =="
printf 'round  cuda_only  mixed  ratio\n'
awk '{printf "%-6s %-10s %-6s %.3f\n", $1, $2, $3, $3/$2}' "$OUT/walls.txt"
echo "medians:"
python3 -c "
import statistics
rows=[l.split() for l in open('$OUT/walls.txt')]
a=[float(r[1]) for r in rows]; b=[float(r[2]) for r in rows]
ma,mb=statistics.median(a),statistics.median(b)
print(f'  cuda_only {ma:.2f}   mixed {mb:.2f}   ratio {mb/ma:.3f}')"

echo
echo "== CUDA output identity across every leg (the accuracy check) =="
python3 - "$OUT" "$ROUNDS" "${FILES[@]}" <<'EOF'
import json, sys, hashlib
out, rounds = sys.argv[1], int(sys.argv[2]); files = sys.argv[3:]
keys = ["rttm", "segments", "exclusive_segments", "centroids", "num_speakers"]
ok = True
for f in files:
    hs = set()
    for r in range(1, rounds + 1):
        for leg in ("cuda_only", "mixed"):
            d = json.load(open(f"{out}/{leg}_r{r}/{f}.json"))
            hs.add(hashlib.sha256(
                json.dumps([d[k] for k in keys], sort_keys=True).encode()).hexdigest())
    print(f"{f}: {'IDENTICAL' if len(hs)==1 else 'DIFFERS'}  {sorted(hs)[0][:16]}")
    ok &= len(hs) == 1
print("IDENTITY GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF

echo
echo "== peak diar-server VRAM during each leg =="
for f in "$OUT"/vram_*.log; do
  printf '%-34s %s MiB\n' "$(basename "$f")" \
    "$(rg 'diar-server' "$f" | awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END{print m+0}')"
done
