#!/bin/bash
# Run 10 sequential TEXT-MODE benchmarks against a vLLM OpenAI-compatible
# endpoint, resetting the server's prefix cache between runs via
# POST <host>/reset_prefix_cache so each run starts from a comparable
# cold state.
#
# This is the text-mode companion to scripts/audio_in_batch_cache_off_with_reset.sh:
# same scenario, same model, same server, but the user side sends plain text
# instead of audio. Designed for direct comparison against the cache-off
# audio-in numbers.
#
# Usage:
#   scripts/text_batch_vllm_with_reset.sh [BASE_URL] [LABEL_PREFIX]
# Defaults:
#   BASE_URL      http://192.168.7.228:8000/v1
#   LABEL_PREFIX  text-vllm-reset

set -u

BASE_URL="${1:-http://192.168.7.228:8000/v1}"
LABEL_PREFIX="${2:-text-vllm-reset}"

# Derive the reset endpoint from the base URL (strip trailing /v1).
ROOT_URL="${BASE_URL%/v1}"
RESET_URL="$ROOT_URL/reset_prefix_cache"

OUT=/tmp/audio-in-stat-batch
mkdir -p "$OUT"
RESULTS="$OUT/results_${LABEL_PREFIX}.tsv"
echo -e "label\trun_dir\texit_status\tturns\treset_http_status" > "$RESULTS"

export VLLM_BASE_URL="$BASE_URL"
# api key unused by vLLM but pipecat OpenAILLMService requires *some* value.
export VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-placeholder}"
# Default to thinking OFF for apples-to-apples match with the audio-in
# pipeline (vendored Nemotron service also uses enable_thinking=False).
# Bump idle timeout modestly to absorb cold-turn variance.
export MTE_VLLM_THINKING="${MTE_VLLM_THINKING:-0}"
export MTE_TEXT_IDLE_TIMEOUT_SECS="${MTE_TEXT_IDLE_TIMEOUT_SECS:-90}"

reset_prefix_cache() {
  curl -s -X POST -o /dev/null -w "%{http_code}" -m 10 "$RESET_URL"
}

echo "=== ${LABEL_PREFIX} ==="
echo "endpoint:        $BASE_URL"
echo "reset URL:       $RESET_URL"
echo "service:         vllm-openai  (pipeline: text)"
echo "thinking:        $MTE_VLLM_THINKING"
echo "idle timeout:    ${MTE_TEXT_IDLE_TIMEOUT_SECS}s"
echo "results:         $RESULTS"
echo

# Reset cache once at the very start so the first run is also cold.
echo "[$(date +%H:%M:%S)] PRE-RESET prefix cache"
status=$(reset_prefix_cache)
echo "  reset_prefix_cache returned HTTP $status"
echo

cd /home/khkramer/src/aiewf-eval

for n in 01 02 03 04 05 06 07 08 09 10; do
  label="${LABEL_PREFIX}-t${n}"
  logfile="$OUT/${LABEL_PREFIX}_t${n}.log"
  echo "[$(date +%H:%M:%S)] START $label"
  uv run multi-turn-eval run aiwf_medium_context \
      --model nemotron_3_nano_omni \
      --service vllm-openai \
      --pipeline text > "$logfile" 2>&1
  exit_status=$?
  run_dir=$(grep -oE 'runs/aiwf_medium_context/[^[:space:]]+' "$logfile" | head -1)
  turns=$(grep -c 'Recorded turn' "$logfile" || echo 0)
  echo "[$(date +%H:%M:%S)] END   $label exit=$exit_status turns=$turns"

  reset_status=$(reset_prefix_cache)
  echo "[$(date +%H:%M:%S)] RESET  $label -> HTTP $reset_status"

  echo -e "${label}\t${run_dir}\t${exit_status}\t${turns}\t${reset_status}" >> "$RESULTS"
done

echo
echo "=== ${LABEL_PREFIX} summary ==="
cat "$RESULTS"
