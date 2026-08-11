#!/usr/bin/env bash

set -u

CAMPAIGN_ROOT="runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z"
ANTHROPIC_API_KEY="$(rg --no-line-number '^ANTHROPIC_API_KEY=' .env | cut -d= -f2-)"

: > "$CAMPAIGN_ROOT/judged-runs.txt"
printf 'cohort_index\trun_dir\tjudge_rc\tstrict_pass\tturns_scored\tstarted_utc\tfinished_utc\n' \
  > "$CAMPAIGN_ROOT/judging.tsv"

cohort_index=0
while IFS= read -r run_dir; do
  cohort_index=$((cohort_index + 1))
  judge_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  judge_rc=1
  for judge_attempt in 1 2; do
    judge_log="$CAMPAIGN_ROOT/logs/judge-$(printf '%02d' "$cohort_index")-attempt-$judge_attempt.log"
    echo "JUDGE_START cohort_index=$cohort_index attempt=$judge_attempt run_dir=$run_dir utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
      uv run multi-turn-eval judge "$run_dir" > "$judge_log" 2>&1
    judge_rc=$?
    if [[ "$judge_rc" -eq 0 && -f "$run_dir/claude_summary.json" ]]; then
      break
    fi
  done

  strict_pass=""
  turns_scored=""
  if [[ "$judge_rc" -eq 0 && -f "$run_dir/claude_summary.json" ]]; then
    read -r strict_pass turns_scored < <(python3 - "$run_dir/claude_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
print(summary["turn_pass"]["count"], summary["turns_scored"])
PY
)
    printf '%s\n' "$run_dir" >> "$CAMPAIGN_ROOT/judged-runs.txt"
  fi
  judge_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cohort_index" "$run_dir" "$judge_rc" "$strict_pass" "$turns_scored" \
    "$judge_started" "$judge_finished" >> "$CAMPAIGN_ROOT/judging.tsv"
  echo "JUDGE_EXIT cohort_index=$cohort_index rc=$judge_rc strict=$strict_pass/$turns_scored utc=$judge_finished"
done < "$CAMPAIGN_ROOT/included-runs.txt"

mapfile -t judged_runs < "$CAMPAIGN_ROOT/judged-runs.txt"
if [[ "${#judged_runs[@]}" -gt 0 ]]; then
  uv run python scripts/benchmark_summary.py "${judged_runs[@]}" \
    > "$CAMPAIGN_ROOT/aggregate.txt" 2> "$CAMPAIGN_ROOT/aggregate.stderr.log"
  aggregate_rc=$?
  uv run python scripts/benchmark_summary.py "${judged_runs[@]}" --json \
    > "$CAMPAIGN_ROOT/aggregate.json" 2> "$CAMPAIGN_ROOT/aggregate-json.stderr.log"
  aggregate_json_rc=$?
else
  aggregate_rc=1
  aggregate_json_rc=1
fi

echo "JUDGING_COMPLETE judged=${#judged_runs[@]} aggregate_rc=$aggregate_rc aggregate_json_rc=$aggregate_json_rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
