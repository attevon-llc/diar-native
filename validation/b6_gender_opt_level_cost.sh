#!/usr/bin/env bash
# B6 (issue #14 follow-up): what does the aarch64 `GraphOptimizationLevel::Level1` cap on the
# GENDER session actually cost?
#
# The cap ships on aarch64 only (`crates/diar-core/src/ort_compat.rs`), where the uncapped
# alternative does not load at all — so the cost cannot be measured there: there is no
# comparison to make. It CAN be measured on x86_64, where both levels load, via
# `DIAR_ORT_OPT_LEVEL` (a FLOOR — it can lower a model's level, never raise it past its cap).
#
#   leg "all"   = DIAR_ORT_OPT_LEVEL unset  -> ORT default (Level3), what x86_64 ships
#   leg "basic" = DIAR_ORT_OPT_LEVEL=basic  -> Level1, what aarch64 is pinned to
#
# ISOLATING GENDER. `DIAR_ORT_OPT_LEVEL` reaches only sessions built through
# `diar_core::ort_compat` — the gender model and the smoke test — NOT speakrs' 15 diarization
# graphs. So each leg measures BOTH `gender:false` and `gender:true`, and the reported figure
# is the MARGINAL cost of gender (true - false) within the same container. That subtracts the
# diarization time, which the knob cannot touch, instead of hunting a ~1.5 s effect inside a
# ~5 s wall time. `gender:false` also doubles as a null control: it must NOT move between legs.
#
#   GPU=0 ROUNDS=3 ./validation/b6_gender_opt_level_cost.sh
set -euo pipefail
GPU="${GPU:-0}"
PORT="${PORT:-18714}"
IMAGE="${IMAGE:-diar-server:bench}"
MODELS="${MODELS:-/mnt/nvm/repos/diar-native/models_folded}"
AUDIO="${AUDIO:-/tmp/bench_audio}"
CLIP="${CLIP:-karpathy_10m.wav}"
OUT="${OUT:-/tmp/b6_optlevel}"
ROUNDS="${ROUNDS:-3}"
REQS="${REQS:-5}"
NAME=diar-b6

mkdir -p "$OUT"; : > "$OUT/times.txt"

# $1=outfile prefix  $2=gender bool  -> prints median seconds
timed_reqs() {
  local d=$1 g=$2 ts=() t0 t1 i
  for i in $(seq 1 "$REQS"); do
    t0=$(date +%s.%N)
    curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" -H 'Content-Type: application/json' \
      -d "{\"wav_path\":\"/audio/$CLIP\",\"file_id\":\"b6\",\"gender\":$g}" \
      -o "${d}_g${g}_$i.json"
    t1=$(date +%s.%N)
    ts+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')")
  done
  printf '%s\n' "${ts[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

run_variant() { # $1=label  $2=round
  local d="$OUT/$1_r$2"; mkdir -p "$d"
  local env=()
  [ "$1" = basic ] && env=(-e DIAR_ORT_OPT_LEVEL=basic)

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "\"device=$GPU\"" \
    -e DIAR_DEVICES=cuda -e DIAR_MAX_INFLIGHT=2 -e SPEAKRS_LAZY_SESSIONS=1 \
    -e RUST_LOG=info,ort::logging=warn "${env[@]}" \
    -v "$MODELS":/models:ro -v "$AUDIO":/audio:ro -p "$PORT":8701 "$IMAGE" >/dev/null

  for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/healthz" >"$d/healthz.json" 2>/dev/null && break
    sleep 1
  done
  # a 200 here already proves the gender graph LOADED at this optimization level
  curl -s -m 3600 -X POST "http://localhost:$PORT/diarize" -H 'Content-Type: application/json' \
    -d "{\"wav_path\":\"/audio/$CLIP\",\"file_id\":\"w\",\"gender\":true}" -o "$d/warmup.json"

  local nog; nog=$(timed_reqs "$d/r" false)
  local wig; wig=$(timed_reqs "$d/r" true)
  local marg; marg=$(awk -v a="$wig" -v b="$nog" 'BEGIN{printf "%.3f", a-b}')
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "$2 $1 $nog $wig $marg" >> "$OUT/times.txt"
  printf 'round %s  %-5s  no_gender=%ss  gender=%ss  MARGINAL=%ss  la=%s\n' \
    "$2" "$1" "$nog" "$wig" "$marg" "$(uptime | sed 's/.*load average: //')"
}

echo "== B6: cost of the aarch64 Level1 gender cap, measured on x86_64 =="
echo "clip: $CLIP  image: $IMAGE  gpu: $GPU  reqs/cell: $REQS"
echo

for r in $(seq 1 "$ROUNDS"); do
  run_variant all   "$r"
  run_variant basic "$r"
done

echo
echo "== results =="
python3 -c "
import statistics
rows=[l.split() for l in open('$OUT/times.txt')]
def col(lab,i): return [float(r[i]) for r in rows if r[1]==lab]
for lab in ('all','basic'):
    print(f'  {lab:6s} no_gender {col(lab,2)} median {statistics.median(col(lab,2)):.3f}s')
    print(f'  {lab:6s} gender    {col(lab,3)} median {statistics.median(col(lab,3)):.3f}s')
    print(f'  {lab:6s} MARGINAL  {col(lab,4)} median {statistics.median(col(lab,4)):.3f}s')
ma,mb=statistics.median(col('all',4)),statistics.median(col('basic',4))
print()
print(f'  gender marginal: all(Level3) {ma:.3f}s  vs  basic(Level1) {mb:.3f}s')
print(f'  cost of the cap = {mb-ma:+.3f}s  ({(mb/ma-1)*100:+.1f}%)')
na,nb=statistics.median(col('all',2)),statistics.median(col('basic',2))
print(f'  NULL CONTROL (no_gender, knob must not matter): {na:.3f}s vs {nb:.3f}s  delta {nb-na:+.3f}s')
print(f'  marginal spread: all {max(col(\"all\",4))-min(col(\"all\",4)):.3f}s, '
      f'basic {max(col(\"basic\",4))-min(col(\"basic\",4)):.3f}s  <- compare the cost against THIS')"

echo
echo "== ACCURACY CHECK: gender verdicts and records, Level3 vs Level1 =="
python3 - "$OUT" "$ROUNDS" <<'EOF'
import json, sys, hashlib
out, rounds = sys.argv[1], int(sys.argv[2])
keys = ["rttm", "segments", "exclusive_segments", "centroids", "num_speakers"]
recs, gen = {}, {}
for lab in ("all", "basic"):
    rs, g = set(), None
    for r in range(1, rounds + 1):
        d = json.load(open(f"{out}/{lab}_r{r}/r_gtrue_1.json"))
        rs.add(hashlib.sha256(json.dumps([d[k] for k in keys], sort_keys=True).encode()).hexdigest())
        g = g or d.get("speaker_gender") or {}
    recs[lab], gen[lab] = rs, g
    print(f"  {lab}: {len(rs)} distinct record hash(es) -> {sorted(rs)[0][:16]}")
print(f"  diarization records all == basic : "
      f"{'IDENTICAL' if recs['all']==recs['basic'] else 'DIFFER  <-- expected, knob is gender-only'}")
a, b = gen["all"], gen["basic"]
ok = sorted(a) == sorted(b)
for k in sorted(a):
    la, ca = a[k]["label"], a[k]["confidence"]
    lb, cb = b.get(k, {}).get("label"), b.get(k, {}).get("confidence")
    ok &= la == lb
    print(f"    {k}: Level3 {la} {ca:.8f} | Level1 {lb} {cb:.8f} | delta {abs(ca-cb):.2e}"
          f" | {'AGREE' if la==lb else 'DISAGREE'}")
print("GENDER LABEL GATE:", "PASS" if ok else "FAIL")
EOF
