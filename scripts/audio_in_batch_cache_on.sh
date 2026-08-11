#!/bin/bash
# Run 10 sequential audio-in benchmarks with the client-authoritative
# conversation cache ENABLED. Sends conversation_id and
# conversation_require_cache=true (from turn 2 onward).
#
# Usage:
#   scripts/audio_in_batch_cache_on.sh [BASE_URL] [LABEL_PREFIX]
# Defaults:
#   BASE_URL      http://127.0.0.1:8000/v1
#   LABEL_PREFIX  cache-on
#
# Examples:
#   scripts/audio_in_batch_cache_on.sh
#   scripts/audio_in_batch_cache_on.sh http://192.168.7.228:8010/v1 bf16-cache-on
#
# Outputs:
#   /tmp/audio-in-stat-batch/results_${LABEL_PREFIX}.tsv
#   /tmp/audio-in-stat-batch/${LABEL_PREFIX}_on??.log
#   trace dirs under /tmp/audio-in-${LABEL_PREFIX}-on??-*/

set -u

BASE_URL="${1:-http://127.0.0.1:8000/v1}"
LABEL_PREFIX="${2:-cache-on}"

OUT=/tmp/audio-in-stat-batch
mkdir -p "$OUT"
RESULTS="$OUT/results_${LABEL_PREFIX}.tsv"
echo -e "label\trun_dir\texit_status\tturns\ttrace_dir" > "$RESULTS"

export MTE_NEMOTRON_AUDIO_IN_BASE_URL="$BASE_URL"
export MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1
unset MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY  # deprecated in vendored service

echo "=== ${LABEL_PREFIX} ==="
echo "endpoint: $BASE_URL"
echo "config:   MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1"
echo "results:  $RESULTS"
echo

cd /home/khkramer/src/aiewf-eval

for n in 01 02 03 04 05 06 07 08 09 10; do
  label="${LABEL_PREFIX}-on${n}"
  trace_dir=$(mktemp -d -t audio-in-${LABEL_PREFIX}-on${n}-XXXX)
  logfile="$OUT/${LABEL_PREFIX}_on${n}.log"
  echo "[$(date +%H:%M:%S)] START $label (trace=$trace_dir)"
  NEMOTRON_OMNI_TRACE_DIR="$trace_dir" \
    uv run multi-turn-eval run aiwf_medium_context \
      --model nemotron_3_nano_omni \
      --service nemotron-audio-in \
      --pipeline audio-in > "$logfile" 2>&1
  exit_status=$?
  run_dir=$(grep -oE 'runs/aiwf_medium_context/[^[:space:]]+' "$logfile" | head -1)
  turns=$(grep -c 'Recorded turn' "$logfile" || echo 0)
  echo -e "${label}\t${run_dir}\t${exit_status}\t${turns}\t${trace_dir}" >> "$RESULTS"
  echo "[$(date +%H:%M:%S)] END   $label exit=$exit_status turns=$turns"
done

echo
echo "=== ${LABEL_PREFIX} summary ==="
cat "$RESULTS"
