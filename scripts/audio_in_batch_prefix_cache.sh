#!/bin/bash
# Run 10 sequential audio-in benchmarks with the client-authoritative
# conversation cache DISABLED, expecting the SERVER to have vLLM's
# standard prefix caching ENABLED.
#
# IMPORTANT: this script does not change server config. Before running,
# restart the vLLM server with prefix caching enabled (e.g. remove
# --no-enable-prefix-caching or pass --enable-prefix-caching). After the
# batch, restart the server with prefix caching disabled to return to
# the baseline.
#
# Usage:
#   scripts/audio_in_batch_prefix_cache.sh [BASE_URL] [LABEL_PREFIX]
# Defaults:
#   BASE_URL      http://127.0.0.1:8000/v1
#   LABEL_PREFIX  prefix-cache
#
# Examples:
#   scripts/audio_in_batch_prefix_cache.sh
#   scripts/audio_in_batch_prefix_cache.sh http://192.168.7.228:8010/v1 bf16-prefix-cache
#
# Outputs:
#   /tmp/audio-in-stat-batch/results_${LABEL_PREFIX}.tsv
#   /tmp/audio-in-stat-batch/${LABEL_PREFIX}_pc??.log
#   trace dirs under /tmp/audio-in-${LABEL_PREFIX}-pc??-*/

set -u

BASE_URL="${1:-http://127.0.0.1:8000/v1}"
LABEL_PREFIX="${2:-prefix-cache}"

OUT=/tmp/audio-in-stat-batch
mkdir -p "$OUT"
RESULTS="$OUT/results_${LABEL_PREFIX}.tsv"
echo -e "label\trun_dir\texit_status\tturns\ttrace_dir" > "$RESULTS"

export MTE_NEMOTRON_AUDIO_IN_BASE_URL="$BASE_URL"
# Conversation cache disabled (client side) — only vLLM's standard
# prefix caching contributes any reuse.
export MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0
unset MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY

echo "=== ${LABEL_PREFIX} ==="
echo "endpoint: $BASE_URL"
echo "config:   MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0 (client)"
echo "          vLLM standard prefix caching expected ENABLED on server"
echo "results:  $RESULTS"
echo

cd /home/khkramer/src/aiewf-eval

for n in 01 02 03 04 05 06 07 08 09 10; do
  label="${LABEL_PREFIX}-pc${n}"
  trace_dir=$(mktemp -d -t audio-in-${LABEL_PREFIX}-pc${n}-XXXX)
  logfile="$OUT/${LABEL_PREFIX}_pc${n}.log"
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
