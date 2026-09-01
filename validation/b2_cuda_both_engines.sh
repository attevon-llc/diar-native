#!/usr/bin/env bash
# B2 gate (RESULTS §7.34): does keeping a CPU engine resident alongside the CUDA one cost
# CUDA-mode latency?
#
# Single variable: DIAR_DEVICES=cuda vs DIAR_DEVICES=cuda,cpu. Same image, same clip, same
# request, every request explicitly device=cuda so the second engine is RESIDENT BUT IDLE —
# which is exactly the deployment shape the superset claim (§7.34) invites operators into.
# §7.34 B3 already showed the VRAM side is zero at idle and at peak on a quiet card; this
# leg adds the latency side and re-checks VRAM DURING the run, per-PID.
#
# Legs are INTERLEAVED (cuda, cuda+cpu, cuda, ...) rather than all-A-then-all-B.
#
#   GPU=0 ROUNDS=3 ./validation/b2_cuda_both_engines.sh
set -euo pipefail
GPU="${GPU:-0}"
PORT="${PORT:-18713}"
IMAGE="${IMAGE:-diar-server:bench}"
MODELS="${MODELS:-/mnt/nvm/repos/diar-native/models_folded}"
AUDIO="${AUDIO:-/tmp/bench_audio}"
CLIP="${CLIP:-EN2002c_360.wav}"
OUT="${OUT:-/tmp/b2_cuda}"
ROUNDS="${ROUNDS:-3}"
REQS="${REQS:-5}"     # timed requests per leg, after a warmup
NAME=diar-b2

mkdir -p "$OUT"; : > "$OUT/times.txt"

SAMPLER=""
# stdout redirected away — a background job inside a command substitution hangs the harness
# (§7.44). Kept as a global for the same reason.
start_sampler() { # $1=logfile  $2=host pid
  : > "$1"
  (while :; do
     nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" \
       | grep "^$2," >> "$1" || true
     sleep 0.5
   done) >/dev/null 2>&1 &
  SAMPLER=$!
}

run_variant() { # $1=devices  $2=label  $3=round
  local d="$OUT/$2_r$3"; mkdir -p "$d"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "\"device=$GPU\"" \
    -e DIAR_DEVICES="$1" -e DIAR_MAX_INFLIGHT=2 -e SPEAKRS_LAZY_SESSIONS=1 \
    -e RUST_LOG=info,ort::logging=warn \
    -v "$MODELS":/models:ro -v "$AUDIO":/audio:ro -p "$PORT":8701 "$IMAGE" >/dev/null

  for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/healthz" >"$d/healthz.json" 2>/dev/null && break
    sleep 1
  done
  local hostpid; hostpid=$(docker inspect -f '{{.State.Pid}}' "$NAME")

  # warmup: cuDNN algo search + arena growth are first-run costs, not steady-state latency
  curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" -H 'Content-Type: application/json' \
    -d "{\"wav_path\":\"/audio/$CLIP\",\"file_id\":\"w\",\"device\":\"cuda\"}" -o "$d/warmup.json"

  start_sampler "$d/vram.log" "$hostpid"
  local ts=()
  for i in $(seq 1 "$REQS"); do
    local t0 t1
    t0=$(date +%s.%N)
    curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" -H 'Content-Type: application/json' \
      -d "{\"wav_path\":\"/audio/$CLIP\",\"file_id\":\"b2\",\"device\":\"cuda\"}" -o "$d/out$i.json"
    t1=$(date +%s.%N)
    ts+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')")
  done
  kill "$SAMPLER" 2>/dev/null || true

  local peak; peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "$d/vram.log")
  local med; med=$(printf '%s\n' "${ts[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "$3 $2 $med $peak" >> "$OUT/times.txt"
  printf 'round %s  %-8s median=%ss  peak=%s MiB  [%s]  la=%s\n' \
    "$3" "$2" "$med" "$peak" "${ts[*]}" "$(uptime | sed 's/.*load average: //')"
}

echo "== B2: CUDA latency with a CPU engine resident vs not =="
echo "clip: $CLIP  image: $IMAGE  gpu: $GPU  reqs/leg: $REQS"
echo

for r in $(seq 1 "$ROUNDS"); do
  run_variant cuda      cuda     "$r"
  run_variant cuda,cpu  cuda_cpu "$r"
done

echo
echo "== results =="
python3 -c "
import statistics
rows=[l.split() for l in open('$OUT/times.txt')]
a=[float(r[2]) for r in rows if r[1]=='cuda']; b=[float(r[2]) for r in rows if r[1]=='cuda_cpu']
pa=[int(r[3]) for r in rows if r[1]=='cuda']; pb=[int(r[3]) for r in rows if r[1]=='cuda_cpu']
ma,mb=statistics.median(a),statistics.median(b)
print(f'  cuda      {a}  median {ma:.3f}s  peak VRAM {pa} median {statistics.median(pa):.0f} MiB')
print(f'  cuda,cpu  {b}  median {mb:.3f}s  peak VRAM {pb} median {statistics.median(pb):.0f} MiB')
print(f'  latency ratio cuda_cpu/cuda = {mb/ma:.4f}  ({mb-ma:+.3f}s)')
print(f'  VRAM delta for the resident CPU engine = {statistics.median(pb)-statistics.median(pa):+.0f} MiB')
print(f'  spread: cuda {max(a)-min(a):.3f}s, cuda_cpu {max(b)-min(b):.3f}s  <- compare the delta against THIS')"

echo
echo "== ACCURACY CHECK: CUDA output identity across both configurations =="
python3 - "$OUT" "$ROUNDS" "$REQS" <<'EOF'
import json, sys, hashlib
out, rounds, reqs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
keys = ["rttm", "segments", "exclusive_segments", "centroids", "num_speakers"]
hs = set()
for lab in ("cuda", "cuda_cpu"):
    for r in range(1, rounds + 1):
        for i in range(1, reqs + 1):
            d = json.load(open(f"{out}/{lab}_r{r}/out{i}.json"))
            hs.add(hashlib.sha256(json.dumps([d[k] for k in keys], sort_keys=True).encode()).hexdigest())
print(f"  distinct record hashes across every request of both configs: {len(hs)}")
print(f"  {sorted(h[:16] for h in hs)}")
print("IDENTITY GATE:", "PASS" if len(hs) == 1 else "FAIL")
sys.exit(0 if len(hs) == 1 else 1)
EOF
