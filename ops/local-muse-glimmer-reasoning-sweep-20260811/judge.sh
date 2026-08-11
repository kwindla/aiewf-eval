#!/usr/bin/env bash

set -uo pipefail

CAMPAIGN_ROOT="runs/muse-glimmer-reasoning-strength-n30-20260811"
ANTHROPIC_API_KEY="$(rg --no-line-number '^ANTHROPIC_API_KEY=' .env | cut -d= -f2-)"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"

mkdir -p "$CAMPAIGN_ROOT/logs"

judge_one() {
  local ordinal="$1" arm="$2" arm_index="$3" run_dir="$4"
  local judge_started judge_finished judge_rc judge_attempt judge_log
  local strict_pass="" turns_scored=""
  judge_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  judge_rc=1
  if [[ -f "$run_dir/claude_summary.json" && -f "$run_dir/claude_judged.jsonl" ]]; then
    judge_rc=0
  else
    for judge_attempt in 1 2; do
      judge_log="$CAMPAIGN_ROOT/logs/judge-$(printf '%03d' "$ordinal")-$arm-attempt-$judge_attempt.log"
      echo "JUDGE_START ordinal=$ordinal arm=$arm attempt=$judge_attempt utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        uv run multi-turn-eval judge "$run_dir" --judge-model claude-opus-4-5 \
        > "$judge_log" 2>&1
      judge_rc=$?
      if [[ "$judge_rc" -eq 0 && -f "$run_dir/claude_summary.json" && -f "$run_dir/claude_judged.jsonl" ]]; then
        break
      fi
    done
  fi

  if [[ "$judge_rc" -eq 0 && -f "$run_dir/claude_summary.json" ]]; then
    read -r strict_pass turns_scored < <(python3 - "$run_dir/claude_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)
print(summary["turn_pass"]["count"], summary["turns_scored"])
PY
)
  fi
  judge_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ordinal" "$arm" "$arm_index" "$run_dir" "$judge_rc" "$strict_pass" \
    "$turns_scored" "$judge_started" "$judge_finished" \
    > "$CAMPAIGN_ROOT/judging-parts/$(printf '%03d' "$ordinal").tsv"
  echo "JUDGE_EXIT ordinal=$ordinal arm=$arm rc=$judge_rc strict=$strict_pass/$turns_scored utc=$judge_finished"
  return "$judge_rc"
}

if [[ "${1:-}" == "--one" ]]; then
  shift
  judge_one "$@"
  exit $?
fi

rm -rf "$CAMPAIGN_ROOT/judging-parts"
mkdir -p "$CAMPAIGN_ROOT/judging-parts"
printf 'ordinal\tarm\tarm_index\trun_dir\tjudge_rc\tstrict_pass\tturns_scored\tstarted_utc\tfinished_utc\n' \
  > "$CAMPAIGN_ROOT/judging.tsv"
for arm in low medium high xhigh; do
  : > "$CAMPAIGN_ROOT/judged-$arm.txt"
done

while IFS=$'\t' read -r ordinal _block _position arm arm_index run_dir; do
  [[ "$ordinal" == "ordinal" ]] && continue
  "$0" --one "$ordinal" "$arm" "$arm_index" "$run_dir" &
  while (( $(jobs -rp | wc -l) >= JUDGE_CONCURRENCY )); do
    wait -n || true
  done
done < "$CAMPAIGN_ROOT/included.tsv"
wait || true

for part in "$CAMPAIGN_ROOT"/judging-parts/*.tsv; do
  cat "$part"
done >> "$CAMPAIGN_ROOT/judging.tsv"

while IFS=$'\t' read -r ordinal arm _arm_index run_dir judge_rc _strict_pass turns_scored _started _finished; do
  [[ "$ordinal" == "ordinal" ]] && continue
  if [[ "$judge_rc" -eq 0 && "$turns_scored" -eq 30 ]]; then
    printf '%s\n' "$run_dir" >> "$CAMPAIGN_ROOT/judged-$arm.txt"
  fi
done < "$CAMPAIGN_ROOT/judging.tsv"

all_complete=1
for arm in low medium high xhigh; do
  mapfile -t judged_runs < "$CAMPAIGN_ROOT/judged-$arm.txt"
  if [[ "${#judged_runs[@]}" -ne 30 ]]; then
    all_complete=0
    continue
  fi
  uv run python scripts/benchmark_summary.py "${judged_runs[@]}" \
    > "$CAMPAIGN_ROOT/aggregate-$arm.txt" \
    2> "$CAMPAIGN_ROOT/logs/aggregate-$arm.stderr.log"
  uv run python scripts/benchmark_summary.py "${judged_runs[@]}" --json \
    > "$CAMPAIGN_ROOT/aggregate-$arm.json" \
    2> "$CAMPAIGN_ROOT/logs/aggregate-$arm-json.stderr.log"
done

echo "JUDGING_COMPLETE all_complete=$all_complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$all_complete" -eq 1 ]]
