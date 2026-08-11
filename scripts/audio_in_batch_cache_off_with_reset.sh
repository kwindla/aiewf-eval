#!/bin/bash
# Run 10 sequential audio-in benchmarks with client conversation cache
# DISABLED, and reset the server's prefix cache between runs via
# POST <host>/reset_prefix_cache so each run starts from a comparable
# cold state.
#
# Usage:
#   scripts/audio_in_batch_cache_off_with_reset.sh [BASE_URL] [LABEL_PREFIX]
# Defaults:
#   BASE_URL      http://127.0.0.1:8000/v1
#   LABEL_PREFIX  cache-off-reset
#
# The reset URL is derived from BASE_URL by stripping the trailing /v1
# (so http://host:port/v1 -> http://host:port/reset_prefix_cache).

set -u

BASE_URL="${1:-http://127.0.0.1:8000/v1}"
LABEL_PREFIX="${2:-cache-off-reset}"

# Derive the reset endpoint from the base URL.
ROOT_URL="${BASE_URL%/v1}"
RESET_URL="$ROOT_URL/reset_prefix_cache"

OUT=/tmp/audio-in-stat-batch
mkdir -p "$OUT"
RESULTS="$OUT/results_${LABEL_PREFIX}.tsv"
echo -e "label\trun_dir\texit_status\tturns\ttrace_dir\treset_http_status" > "$RESULTS"

export MTE_NEMOTRON_AUDIO_IN_BASE_URL="$BASE_URL"
export MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0
unset MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY

reset_prefix_cache() {
  curl -s -X POST -o /dev/null -w "%{http_code}" -m 10 "$RESET_URL"
}

echo "=== ${LABEL_PREFIX} ==="
echo "endpoint:    $BASE_URL"
echo "reset URL:   $RESET_URL"
echo "config:      MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0"
echo "results:     $RESULTS"
echo

# Reset cache once at the very start so the first run is also cold.
echo "[$(date +%H:%M:%S)] PRE-RESET prefix cache"
status=$(reset_prefix_cache)
echo "  reset_prefix_cache returned HTTP $status"
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
  echo "[$(date +%H:%M:%S)] END   $label exit=$exit_status turns=$turns"

  # Reset cache BEFORE writing the row, so the row reflects what the
  # NEXT run will start with. The first-run pre-reset above plus this
  # post-run reset gives every run a freshly-reset prefix cache.
  reset_status=$(reset_prefix_cache)
  echo "[$(date +%H:%M:%S)] RESET  $label -> HTTP $reset_status"

  echo -e "${label}\t${run_dir}\t${exit_status}\t${turns}\t${trace_dir}\t${reset_status}" >> "$RESULTS"
done

echo
echo "=== ${LABEL_PREFIX} summary ==="
cat "$RESULTS"
