#!/usr/bin/env bash
# Judge included historical attempts that were omitted by survivor-only ledgers.
set -uo pipefail

ROOT=/home/khkramer/src/aiewf-eval
DATA="$ROOT/docs/filler-study-data/dot-stability-n30-2026-07-20"
LEDGER="$DATA/existing-included.tsv"
STATE="$DATA/state/existing-judge"
ATTEMPTS="$STATE/judge-attempts.tsv"
LOG="$STATE/driver.log"
LOCK="$STATE/driver.lock"
JUDGE_LOCK="$DATA/state/judge.lock"

mkdir -p "$STATE" "$DATA/state"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another existing-judge driver is active" >&2
  exit 2
fi
exec >>"$LOG" 2>&1
cd "$ROOT" || exit 1

if [ ! -f "$ATTEMPTS" ]; then
  printf 'model\tarm\trun_dir\tstart_utc\tend_utc\tjudge_rc\tlog\n' > "$ATTEMPTS"
fi

while IFS=$'\t' read -r model arm run_dir transcript_rows es_turn classification judged provenance source; do
  [ "$model" = model ] && continue
  if [ -s "$run_dir/claude_judged.jsonl" ]; then
    continue
  fi
  start=$(date -u --iso-8601=seconds)
  judge_log="$run_dir/judge-dot-n30-existing.log"
  printf '[%s] judge model=%s arm=%s run=%s\n' "$(date --iso-8601=seconds)" "$model" "$arm" "$run_dir"
  (
    flock 8
    uv run multi-turn-eval judge "$run_dir" > "$judge_log" 2>&1
  ) 8>"$JUDGE_LOCK"
  rc=$?
  end=$(date -u --iso-8601=seconds)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$arm" "$run_dir" "$start" "$end" "$rc" "$judge_log" >> "$ATTEMPTS"
done < "$LEDGER"

missing=0
while IFS=$'\t' read -r model arm run_dir transcript_rows es_turn classification judged provenance source; do
  [ "$model" = model ] && continue
  if [ ! -s "$run_dir/claude_judged.jsonl" ]; then
    printf '[%s] missing judgment run=%s\n' "$(date --iso-8601=seconds)" "$run_dir"
    missing=$((missing + 1))
  fi
done < "$LEDGER"
if [ "$missing" -ne 0 ]; then
  exit 9
fi
touch "$STATE/COMPLETE"
printf '[%s] existing judgments complete\n' "$(date --iso-8601=seconds)"
