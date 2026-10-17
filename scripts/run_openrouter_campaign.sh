#!/bin/bash
# Sequential aiwf_medium_context text campaign against an OpenRouter-served
# model, judging each conversation as it completes and appending accepted run
# dirs to an allowlist for later aggregation. Mirrors run_zai_glm53_campaign.sh
# so cohorts are methodologically comparable.
#
# Usage:
#   scripts/run_openrouter_campaign.sh <model-slug> <label> <effort|none> <count> [suffix]
#
# Launch detached (a 30-conversation campaign runs for hours):
#   setsid nohup scripts/run_openrouter_campaign.sh qwen/qwen3-32b qwen3-32b low 30 n30 \
#       > /tmp/qwen3-aiewf-low.log 2>&1 &
set -u

MODEL="${1:?openrouter model slug required}"
NAME="${2:?short label required}"
EFFORT="${3:?effort required: low|medium|high|none}"
COUNT="${4:?conversation count required}"
SUFFIX="${5:-n${4}}"

cd "$(dirname "$0")/.." || exit 1

LABEL="${NAME}-${EFFORT}-${SUFFIX}"
OUT="ops/openrouter-${NAME}/aiewf-medium-${EFFORT}-${SUFFIX}-$(date -u +%Y%m%d)"
ALLOWLIST="$OUT/run-dirs.txt"
RESULTS="$OUT/results.tsv"
mkdir -p "$OUT/logs"
[ -f "$RESULTS" ] || printf 'label\trun_dir\texit_status\tturns\tjudged\n' > "$RESULTS"

# Pin every knob this route reads so a stale export cannot silently change the
# configuration being measured.
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"
unset MTE_OPENROUTER_REASONING_OFF
if [ "$EFFORT" = "none" ]; then
  unset MTE_OPENROUTER_REASONING_EFFORT
else
  export MTE_OPENROUTER_REASONING_EFFORT="$EFFORT"
fi
export MTE_OPENROUTER_MAX_TOKENS="${MTE_OPENROUTER_MAX_TOKENS:-16384}"
# Some OpenRouter providers queue cold requests behind long keep-alive periods;
# 45s idle-timeouts then kill otherwise-healthy conversations.
export MTE_TEXT_IDLE_TIMEOUT_SECS="${MTE_TEXT_IDLE_TIMEOUT_SECS:-120}"
# Bound each request's first attempt and retry once on stall.
export MTE_OPENROUTER_RETRY_TIMEOUT_SECS="${MTE_OPENROUTER_RETRY_TIMEOUT_SECS:-45}"

echo "=== $LABEL ==="
echo "model:       $MODEL"
echo "effort:      ${MTE_OPENROUTER_REASONING_EFFORT:-(default)}"
echo "max_tokens:  $MTE_OPENROUTER_MAX_TOKENS"
echo "count:       $COUNT"
echo "out:         $OUT"
echo

for n in $(seq -w 1 "$COUNT"); do
  logfile="$OUT/logs/${LABEL}-c${n}.log"
  echo "[$(date -u +%FT%TZ)] START ${LABEL}-c${n}"
  uv run multi-turn-eval run aiwf_medium_context \
      --model "$MODEL" \
      --service openrouter \
      --pipeline text > "$logfile" 2>&1
  exit_status=$?
  run_dir=$(grep -oE 'runs/aiwf_medium_context/[^[:space:]]+' "$logfile" \
            | sed -e 's|/transcript\.jsonl$||' | tail -1)
  turns=$(grep -c 'Recorded turn' "$logfile" || echo 0)

  judged=0
  if [ "$exit_status" -eq 0 ] && [ -n "$run_dir" ] && [ -f "$run_dir/transcript.jsonl" ]; then
    echo "$run_dir" >> "$ALLOWLIST"
    if uv run multi-turn-eval judge "$run_dir" > "$run_dir/judge.log" 2>&1; then
      judged=1
    else
      echo "[$(date -u +%FT%TZ)] JUDGE FAILED for $run_dir — see $run_dir/judge.log"
    fi
  else
    echo "[$(date -u +%FT%TZ)] RUN FAILED exit=$exit_status dir='$run_dir' — see $logfile"
  fi

  echo "[$(date -u +%FT%TZ)] END   ${LABEL}-c${n} exit=$exit_status turns=$turns judged=$judged"
  printf '%s\t%s\t%s\t%s\t%s\n' "${LABEL}-c${n}" "$run_dir" "$exit_status" "$turns" "$judged" >> "$RESULTS"
done

echo
echo "=== $LABEL complete $(date -u +%FT%TZ) ==="
cat "$RESULTS"
