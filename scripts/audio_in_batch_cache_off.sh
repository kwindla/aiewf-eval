#!/bin/bash
# Run 10 sequential audio-in benchmarks with the client-authoritative
# conversation cache DISABLED. No conversation_id, no
# conversation_require_cache. Full context sent every turn.
#
# Usage:
#   scripts/audio_in_batch_cache_off.sh [BASE_URL] [LABEL_PREFIX]
# Defaults:
#   BASE_URL      http://127.0.0.1:8000/v1
#   LABEL_PREFIX  cache-off
#
# Examples:
#   scripts/audio_in_batch_cache_off.sh
#   scripts/audio_in_batch_cache_off.sh http://192.168.7.228:8010/v1 bf16-cache-off
#
# Outputs:
#   /tmp/audio-in-stat-batch/results_${LABEL_PREFIX}.tsv
#   /tmp/audio-in-stat-batch/${LABEL_PREFIX}_off??.log
#   trace dirs under /tmp/audio-in-${LABEL_PREFIX}-off??-*/

set -u

BASE_URL="${1:-http://127.0.0.1:8000/v1}"
LABEL_PREFIX="${2:-cache-off}"

OUT=/tmp/audio-in-stat-batch
mkdir -p "$OUT"
RESULTS="$OUT/results_${LABEL_PREFIX}.tsv"
echo -e "label\trun_dir\texit_status\tturns\ttrace_dir" > "$RESULTS"

export MTE_NEMOTRON_AUDIO_IN_BASE_URL="$BASE_URL"
export MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0
unset MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY

echo "=== ${LABEL_PREFIX} ==="
echo "endpoint: $BASE_URL"
echo "config:   MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0"
echo "results:  $RESULTS"
echo

cd /home/khkramer/src/aiewf-eval

for n in 01 02 03 04 05 06 07 08 09 10; do
  label="${LABEL_PREFIX}-off${n}"
  trace_dir=$(mktemp -d -t audio-in-${LABEL_PREFIX}-off${n}-XXXX)
  logfile="$OUT/${LABEL_PREFIX}_off${n}.log"
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
