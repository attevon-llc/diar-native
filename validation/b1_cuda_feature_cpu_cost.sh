#!/usr/bin/env bash
# B1 gate (RESULTS §7.34 "NOT measured here"): does `--features cuda` slow down CPU-mode
# inference?
#
# THE SINGLE VARIABLE is the `ort-sys` prebuilt distribution. `ort/cuda` selects a different
# prebuilt tarball, whose statically-linked MLAS (the ORT CPU EP kernel library) may have been
# compiled with different flags. Everything else is held fixed: same source tree, same
# toolchain, same container, same models, same audio, same `--mode cpu`.
#
# Both binaries are built from ONE tree by ONE toolchain:
#   docker run --rm -v $PWD:/build -v /tmp/diar_target_bench/def:/tmp/target -w /build \
#     -e CARGO_TARGET_DIR=/tmp/target diar-native-builder:bench cargo build --release -p diar-cli
#   docker run --rm -v $PWD:/build -v /tmp/diar_target_bench/cuda:/tmp/target -w /build \
#     -e CARGO_TARGET_DIR=/tmp/target diar-native-builder:bench \
#     cargo build --release --features cuda -p diar-cli -p diar-server
#
# Legs are INTERLEAVED (def, cuda, def, cuda, ...) rather than all-A-then-all-B, so slow drift
# in background host load cannot masquerade as an effect.
#
# ACCURACY CHECK = OUTPUT IDENTITY, proven not asserted: the RTTM MD5 and the full JSON record
# (segments / exclusive_segments / centroids / num_speakers) must be identical across every
# run of both builds. A different MLAS build that changed results would be a correctness bug,
# not a speed result.
#
#   ROUNDS=3 ./validation/b1_cuda_feature_cpu_cost.sh
set -euo pipefail
IMAGE="${IMAGE:-diar-native-builder:bench}"
DEF_BIN="${DEF_BIN:-/tmp/diar_target_bench/def/release/diar-cli}"
CUDA_BIN="${CUDA_BIN:-/tmp/diar_target_bench/cuda/release/diar-cli}"
MODELS="${MODELS:-/mnt/nvm/repos/diar-native/models_folded}"
AUDIO="${AUDIO:-/tmp/bench_audio}"
CLIP="${CLIP:-EN2002c_360.wav}"
OUT="${OUT:-/tmp/b1_cpu}"
ROUNDS="${ROUNDS:-3}"

mkdir -p "$OUT"
: > "$OUT/walls.txt"

run_leg() { # $1=label  $2=binary  $3=round
  local d="$OUT/$1_r$3"; mkdir -p "$d"
  local t0 t1
  t0=$(date +%s.%N)
  docker run --rm \
    -v "$2":/usr/local/bin/diar-cli:ro \
    -v "$MODELS":/models:ro \
    -v "$AUDIO":/audio:ro \
    -v "$d":/out \
    "$IMAGE" diar-cli --mode cpu --models-dir /models --out-dir /out \
      --label EN2002c --json "/audio/$CLIP" \
    > "$d/stdout.jsonl" 2> "$d/stderr.log"
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}'
}

echo "== B1: CPU-mode inference, default-features vs --features cuda =="
echo "clip: $CLIP   models: $MODELS   image: $IMAGE"
md5sum "$AUDIO/$CLIP"
echo

for r in $(seq 1 "$ROUNDS"); do
  for leg in def cuda; do
    bin=$DEF_BIN; [ "$leg" = cuda ] && bin=$CUDA_BIN
    la_pre=$(uptime | sed 's/.*load average: //')
    w=$(run_leg "$leg" "$bin" "$r")
    # engine-reported elapsed (excludes process start + model load); stdout is JSONL
    e=$(python3 -c "import json,sys; print(json.loads(open('$OUT/${leg}_r$r/stdout.jsonl').read().strip().splitlines()[-1])['elapsed_s'])")
    m=$(md5sum "$OUT/${leg}_r$r/EN2002c_run0.rttm" | cut -d' ' -f1)
    echo "round $r  $leg  engine=${e}s  wall=${w}s  md5=${m:0:12}  la_pre=[$la_pre]"
    echo "$r $leg $e $w $m" >> "$OUT/walls.txt"
    sleep 5
  done
done

echo
echo "== medians (engine elapsed_s, the metric §7.32 reported) =="
python3 -c "
import statistics
rows=[l.split() for l in open('$OUT/walls.txt')]
for leg in ('def','cuda'):
    v=[float(r[2]) for r in rows if r[1]==leg]
    w=[float(r[3]) for r in rows if r[1]==leg]
    print(f'  {leg:5s} engine {\" / \".join(f\"{x:.2f}\" for x in v)}  median {statistics.median(v):.2f}s'
          f'   | wall median {statistics.median(w):.2f}s')
d=[float(r[2]) for r in rows if r[1]=='def']; c=[float(r[2]) for r in rows if r[1]=='cuda']
md,mc=statistics.median(d),statistics.median(c)
print(f'  ratio cuda/def = {mc/md:.4f}  ({(mc-md):+.2f}s)')
print(f'  spread: def {max(d)-min(d):.2f}s, cuda {max(c)-min(c):.2f}s  <- compare the delta against THIS')
"

echo
echo "== ACCURACY CHECK: output identity across all runs of both builds =="
python3 - "$OUT" "$ROUNDS" <<'EOF'
import json, sys, hashlib, glob, os
out, rounds = sys.argv[1], int(sys.argv[2])
keys = ["segments", "exclusive_segments", "centroids", "num_speakers"]
rttm, rec = set(), set()
for leg in ("def", "cuda"):
    for r in range(1, rounds + 1):
        d = f"{out}/{leg}_r{r}"
        rttm.add(hashlib.md5(open(f"{d}/EN2002c_run0.rttm", "rb").read()).hexdigest())
        j = json.load(open(f"{d}/EN2002c_run0.json"))
        rec.add(hashlib.sha256(
            json.dumps([j[k] for k in keys], sort_keys=True).encode()).hexdigest())
print(f"  distinct RTTM md5s        : {len(rttm)}  {sorted(rttm)}")
print(f"  distinct record sha256s   : {len(rec)}  {[h[:16] for h in sorted(rec)]}")
ok = len(rttm) == 1 and len(rec) == 1
print("IDENTITY GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
