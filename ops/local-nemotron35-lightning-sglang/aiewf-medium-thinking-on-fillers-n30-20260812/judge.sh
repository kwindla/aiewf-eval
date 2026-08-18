#!/usr/bin/env bash

set -u -o pipefail

ROOT="/home/khkramer/src/aiewf-eval"
CAMPAIGN="$ROOT/ops/local-nemotron35-lightning-sglang/aiewf-medium-thinking-on-fillers-n30-20260812/artifacts"
CANONICAL="$CAMPAIGN/canonical.tsv"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  ANTHROPIC_KEY="$ANTHROPIC_API_KEY"
else
  ANTHROPIC_KEY="$(rg --no-line-number '^ANTHROPIC_API_KEY=' "$ROOT/.env" | cut -d= -f2-)"
fi

cd "$ROOT" || exit 1
mkdir -p "$CAMPAIGN/judging/logs" "$CAMPAIGN/judging/parts"

canonical_count=$(awk 'END { print NR - 1 }' "$CANONICAL")
if [[ "$canonical_count" -ne 60 ]]; then
  echo "JUDGING_BLOCKED canonical=$canonical_count expected=60"
  exit 2
fi

judge_one() {
  local slot="$1" arm="$2" run_dir="$3"
  local log="$CAMPAIGN/judging/logs/slot$(printf '%03d' "$slot")-${arm}.log"
  local rc=1 strict="" scored=""

  if [[ -f "$run_dir/claude_summary.json" && -f "$run_dir/claude_judged.jsonl" ]]; then
    rc=0
  else
    for attempt in 1 2 3; do
      echo "JUDGE_START slot=$slot arm=$arm attempt=$attempt utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      ANTHROPIC_API_KEY="$ANTHROPIC_KEY" \
        uv run multi-turn-eval judge "$run_dir" --judge-model claude-opus-4-5 \
        > "$log" 2>&1
      rc=$?
      if [[ "$rc" -eq 0 && -f "$run_dir/claude_summary.json" && -f "$run_dir/claude_judged.jsonl" ]]; then
        break
      fi
    done
  fi

  if [[ "$rc" -eq 0 ]]; then
    read -r strict scored < <(python3 - "$run_dir/claude_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["turn_pass"]["count"], summary["turns_scored"])
PY
)
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$slot" "$arm" "$run_dir" "$rc" "$strict" "$scored" \
    > "$CAMPAIGN/judging/parts/$(printf '%03d' "$slot").tsv"
  echo "JUDGE_EXIT slot=$slot arm=$arm rc=$rc strict=$strict/$scored utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

export -f judge_one
export ROOT CAMPAIGN ANTHROPIC_KEY

rm -f "$CAMPAIGN/judging/parts/"*.tsv
while IFS=$'\t' read -r slot arm _attempt run_dir _turns _response_turns _tool_calls _classification; do
  [[ "$slot" == "slot" ]] && continue
  judge_one "$slot" "$arm" "$run_dir" &
  while (( $(jobs -rp | wc -l) >= JUDGE_CONCURRENCY )); do
    wait -n || true
  done
done < "$CANONICAL"
wait || true

printf 'slot\tarm\trun_dir\tjudge_rc\tstrict_pass\tturns_scored\n' \
  > "$CAMPAIGN/judging/judged.tsv"
for part in "$CAMPAIGN/judging/parts/"*.tsv; do
  cat "$part"
done >> "$CAMPAIGN/judging/judged.tsv"

failures=$(awk -F '\t' 'NR > 1 && $4 != 0 { count++ } END { print count + 0 }' \
  "$CAMPAIGN/judging/judged.tsv")
rows=$(awk 'END { print NR - 1 }' "$CAMPAIGN/judging/judged.tsv")
echo "JUDGING_COMPLETE rows=$rows failures=$failures utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$rows" -eq 60 && "$failures" -eq 0 ]]
