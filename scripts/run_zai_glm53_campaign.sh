#!/bin/bash
# Sequential aiwf_medium_context text campaign against Z.AI's hosted GLM-5.3,
# judging each conversation as it completes and appending accepted run dirs to
# an allowlist for later aggregation.
#
# GLM-5.3 always reasons, so there is no thinking-off arm: MTE_ZAI_REASONING_EFFORT
# accepts only low|high|max (anything else is HTTP 400 code 1210).
#
# Usage:
#   scripts/run_zai_glm53_campaign.sh <effort:low|high|max> <count> [label-suffix]
#
# Launch detached (a 30-conversation campaign runs many hours):
#   setsid nohup scripts/run_zai_glm53_campaign.sh low 30 n30 > /tmp/glm53-aiewf-low.log 2>&1 &
set -u

EFFORT="${1:?effort required: low|high|max}"
COUNT="${2:?conversation count required}"
SUFFIX="${3:-n${2}}"

case "$EFFORT" in
  low|high|max) ;;
  *) echo "invalid effort '$EFFORT' (GLM-5.3 supports low|high|max only)" >&2; exit 2 ;;
esac

cd /home/khkramer/src/aiewf-eval || exit 1

LABEL="zai-glm53-${EFFORT}-${SUFFIX}"
OUT="ops/zai-glm-5.3/aiewf-medium-${EFFORT}-${SUFFIX}-$(date -u +%Y%m%d)"
ALLOWLIST="$OUT/run-dirs.txt"
RESULTS="$OUT/results.tsv"
mkdir -p "$OUT/logs"
[ -f "$RESULTS" ] || printf 'label\trun_dir\texit_status\tturns\tjudged\n' > "$RESULTS"

# Pin every knob this route reads so a stale export cannot silently change the
# configuration being measured.
export ZAI_API_KEY="$(grep '^ZAI_API_KEY=' .env | cut -d= -f2-)"
export MTE_ZAI_REASONING_EFFORT="$EFFORT"
export MTE_ZAI_MAX_TOKENS="${MTE_ZAI_MAX_TOKENS:-16384}"
export MTE_ZAI_TEMPERATURE="${MTE_ZAI_TEMPERATURE:-1.0}"

echo "=== $LABEL ==="
echo "effort:      $MTE_ZAI_REASONING_EFFORT"
echo "max_tokens:  $MTE_ZAI_MAX_TOKENS"
echo "count:       $COUNT"
echo "out:         $OUT"
echo

for n in $(seq -w 1 "$COUNT"); do
  logfile="$OUT/logs/${LABEL}-c${n}.log"
  echo "[$(date -u +%FT%TZ)] START ${LABEL}-c${n}"
  uv run multi-turn-eval run aiwf_medium_context \
      --model glm-5.3 \
      --service zai \
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
