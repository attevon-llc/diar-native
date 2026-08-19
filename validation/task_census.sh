#!/usr/bin/env bash
# Full task census for ONE file: where the end-to-end time goes, including the enrichment
# tail that runs after the user sees the transcript.
#
# Measurement rule learned the hard way: do NOT poll `celery inspect` in a loop. Each call
# costs seconds per worker, so the harness ends up dominating the very number it reports —
# an early version spent minutes "measuring" a 54 s job. Poll only the cheap file-status
# endpoint, then read durations out of the DB and worker logs after the fact.
#
#   ./validation/task_census.sh <file-uuid> [label]
set -u

UUID="${1:?usage: task_census.sh <file-uuid> [label]}"
LABEL="${2:-census}"
API=http://localhost:5174
SETTLE=25          # seconds of quiet after user-visible completion for the tail to finish
OUT=/mnt/nvm/repos/diar-native/results/task_census
mkdir -p "$OUT"

WORKERS=(celery-worker celery-cpu-worker celery-nlp-worker celery-embedding-worker
         celery-redaction)

TOK=$(python3 -c "import json;print(json.load(open('/tmp/tok.json'))['access_token'])")

START=$(date +%s)
code=$(curl -s -m 30 -o /dev/null -w '%{http_code}' \
        -X POST "$API/api/files/$UUID/reprocess" -H "Authorization: Bearer $TOK")
[ "$code" = "200" ] || { echo "reprocess failed: HTTP $code" >&2; exit 1; }
echo "dispatched (HTTP $code)"

VISIBLE=""
for _ in $(seq 1 600); do          # cheap poll: ~50 ms per check
  s=$(curl -s -m 8 "$API/api/files/$UUID" -H "Authorization: Bearer $TOK" \
       | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))" 2>/dev/null)
  if [ "$s" = "completed" ] || [ "$s" = "error" ]; then
    VISIBLE=$(( $(date +%s) - START )); echo "user-visible at t=${VISIBLE}s ($s)"; break
  fi
  sleep 2
done
[ -n "$VISIBLE" ] || { echo "never reached a terminal status" >&2; exit 1; }

echo "letting the enrichment tail drain (${SETTLE}s)…"
sleep "$SETTLE"
WINDOW=$(( $(date +%s) - START + 30 ))

echo
echo "=== every task that ran, slowest first ==="
for w in "${WORKERS[@]}"; do
  docker logs "opentranscribe-$w" --since "${WINDOW}s" 2>&1 \
    | grep -oE 'Task [a-zA-Z_.]+\[[^]]+\] succeeded in [0-9.]+s' \
    | sed -E "s/Task ([a-zA-Z_.]+)\[[^]]+\] succeeded in ([0-9.]+)s/\1 \2 ${w#celery-}/"
done | sort -k2 -gr | awk '
  {printf "%-40s %8.2fs  %s\n", $1, $2, $3; total+=$2; n++}
  END {printf "\n%-40s %8.2fs  across %d tasks\n", "SUM OF TASK TIME", total, n}'

echo
echo "=== pipeline stages (authoritative, from file_pipeline_timing) ==="
docker exec opentranscribe-postgres sh -lc "psql -U \$POSTGRES_USER -d \$POSTGRES_DB -P pager=off -c \"
select round((preprocess_end_ms-preprocess_task_prerun_ms)/1000.0,1) prep_s,
       round((gpu_end_ms-gpu_task_prerun_ms)/1000.0,1) gpu_s,
       round((postprocess_end_ms-postprocess_task_prerun_ms)/1000.0,1) post_s,
       round(user_perceived_duration_ms/1000.0,1) user_visible_s
from file_pipeline_timing
where file_id=(select id from media_file where uuid='$UUID') order by created_at desc limit 1;\""

printf '%s,%s,%s\n' "$LABEL" "$UUID" "$VISIBLE" >> "$OUT/summary.csv"
