#!/usr/bin/env bash
# B5 gate (RESULTS §7.34 / §7.39): what does the fp16 gender model actually save in VRAM?
#
# §7.39 restored fp16 gender (378.5 -> 189.5 MB on disk) and claimed "~500 MiB of VRAM" by
# CITING §7.18's measurement of the same pair rather than taking a fresh one. That number is
# asserted in the CHANGELOG, so it deserves its own measurement. Control: §7.18's 5396 MiB
# (fp32) -> 4890 MiB (fp16), i.e. -506 MiB.
#
# Single variable: a bind-mount of gender-wav2vec2.onnx over an otherwise IDENTICAL
# models_folded (the same technique §7.39 used for its end-to-end check). Same image, same
# clip, same request, `{"gender": true}`.
#
# VRAM is sampled DURING the run (§7.14 — sampling after reports the idle floor) and filtered
# to THIS container's PID: the box routinely carries other diar-server and python processes on
# the same card, and whole-GPU sampling silently attributes their memory to us (the exact
# contamination §7.34 had to retract a measurement for).
#
# Legs are INTERLEAVED (fp32, fp16, fp32, fp16, ...) rather than all-A-then-all-B.
#
#   GPU=0 ROUNDS=3 ./validation/b5_gender_fp16_vram.sh
set -euo pipefail
GPU="${GPU:-0}"
PORT="${PORT:-18712}"
IMAGE="${IMAGE:-diar-server:bench}"
MODELS="${MODELS:-/mnt/nvm/repos/diar-native/models_folded}"
FP32="${FP32:-/tmp/bench_models/gender-fp32.onnx}"
AUDIO="${AUDIO:-/tmp/bench_audio}"
CLIP="${CLIP:-karpathy_10m.wav}"
OUT="${OUT:-/tmp/b5_gender}"
ROUNDS="${ROUNDS:-3}"
NAME=diar-b5

mkdir -p "$OUT"; : > "$OUT/peaks.txt"

SAMPLER=""
# stdout MUST be redirected away — a background job inside a command substitution holds the
# substitution's stdout open forever and hangs the harness (see §7.44).
start_sampler() { # $1=logfile  $2=host pid of the container
  : > "$1"
  (while :; do
     echo "T $(date +%s.%N) la=$(uptime | sed 's/.*load average: //')" >> "$1"
     nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" \
       | grep "^$2," >> "$1" || true
     sleep 0.5
   done) >/dev/null 2>&1 &
  SAMPLER=$!
}

run_variant() { # $1=fp32|fp16  $2=round
  local d="$OUT/$1_r$2"; mkdir -p "$d"
  local mounts=(-v "$MODELS":/models:ro)
  [ "$1" = fp32 ] && mounts+=(-v "$FP32":/models/gender-wav2vec2.onnx:ro)

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "\"device=$GPU\"" \
    -e DIAR_DEVICES=cuda -e DIAR_MAX_INFLIGHT=2 -e SPEAKRS_LAZY_SESSIONS=1 \
    -e RUST_LOG=info,ort::logging=warn \
    "${mounts[@]}" -v "$AUDIO":/audio:ro -p "$PORT":8701 "$IMAGE" >/dev/null

  # wait for readiness (gender session is committed eagerly at engine load, so a 200 here
  # already proves the graph loaded under this precision)
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/healthz" >"$d/healthz.json" 2>/dev/null && break
    sleep 1
  done
  local hostpid; hostpid=$(docker inspect -f '{{.State.Pid}}' "$NAME")

  start_sampler "$d/vram.log" "$hostpid"
  # two gender runs: the first grows the arena, the second confirms the peak is stable
  for i in 1 2; do
    curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" \
      -H 'Content-Type: application/json' \
      -d "{\"wav_path\":\"/audio/$CLIP\",\"file_id\":\"k10m\",\"gender\":true}" \
      -o "$d/out$i.json"
  done
  kill "$SAMPLER" 2>/dev/null || true

  local peak; peak=$(awk -F', ' '/MiB/{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "$d/vram.log")
  local n; n=$(grep -c '^T ' "$d/vram.log")
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "$2 $1 $peak $n" >> "$OUT/peaks.txt"
  printf 'round %s  %-5s peak=%s MiB  samples=%s  la=%s\n' \
    "$2" "$1" "$peak" "$n" "$(uptime | sed 's/.*load average: //')"
}

echo "== B5: gender fp16 vs fp32 VRAM, sampled DURING, per-PID =="
echo "clip: $CLIP  image: $IMAGE  gpu: $GPU"
ls -l "$MODELS/gender-wav2vec2.onnx" "$FP32" | awk '{print "  " $5 "  " $9}'
echo

for r in $(seq 1 "$ROUNDS"); do
  run_variant fp32 "$r"
  run_variant fp16 "$r"
done

echo
echo "== peak VRAM (MiB) =="
python3 -c "
import statistics
rows=[l.split() for l in open('$OUT/peaks.txt')]
a=[int(r[2]) for r in rows if r[1]=='fp32']; b=[int(r[2]) for r in rows if r[1]=='fp16']
print('  fp32 ', a, ' median', statistics.median(a))
print('  fp16 ', b, ' median', statistics.median(b))
print(f'  saving = {statistics.median(a)-statistics.median(b):.0f} MiB'
      f'   (control: §7.18 5396 -> 4890 = 506 MiB)')"

echo
echo "== ACCURACY CHECK: gender verdicts and diarization records, fp32 vs fp16 =="
python3 - "$OUT" "$ROUNDS" <<'EOF'
import json, sys, hashlib
out, rounds = sys.argv[1], int(sys.argv[2])
keys = ["rttm", "segments", "exclusive_segments", "centroids", "num_speakers"]
recs, genders = {}, {}
for v in ("fp32", "fp16"):
    rs, gs = set(), []
    for r in range(1, rounds + 1):
        d = json.load(open(f"{out}/{v}_r{r}/out2.json"))
        rs.add(hashlib.sha256(json.dumps([d[k] for k in keys], sort_keys=True).encode()).hexdigest())
        gs.append(d.get("speaker_gender") or {})
    recs[v], genders[v] = rs, gs
    print(f"  {v}: {len(rs)} distinct record hash(es) across {rounds} rounds -> {sorted(rs)[0][:16]}")

# Gender does not feed clustering, so swapping its precision must leave the diarization
# records untouched. Anything else would be a real bug, not a VRAM result.
same = recs["fp32"] == recs["fp16"] and len(recs["fp32"]) == 1
print(f"  diarization records fp32 == fp16 : {'IDENTICAL' if same else 'DIFFER  <-- INVESTIGATE'}")

a, b = genders["fp32"][0], genders["fp16"][0]
print(f"  speakers: fp32 {sorted(a)}  fp16 {sorted(b)}")
ok = sorted(a) == sorted(b)
for k in sorted(a):
    la, ca = a[k]["label"], a[k]["confidence"]
    lb, cb = b.get(k, {}).get("label"), b.get(k, {}).get("confidence")
    agree = la == lb
    ok &= agree
    print(f"    {k}: fp32 {la} {ca:.6f} | fp16 {lb} {cb:.6f} | "
          f"delta {abs(ca-cb):.2e} | label {'AGREE' if agree else 'DISAGREE'}")
print("GENDER LABEL GATE:", "PASS" if ok else "FAIL")
EOF
