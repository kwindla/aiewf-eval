#!/usr/bin/env bash
# Resumable fixed-N driver for the frozen GPT-5.4 dash confirmation protocol.
set -uo pipefail

ROOT=/home/khkramer/src/aiewf-eval
DATA="$ROOT/docs/filler-study-data/gpt54-dash-confirmation-2026-07-20"
SCHEDULE="$DATA/schedule.tsv"
ATTEMPTS="$DATA/attempts.tsv"
COUNTED="$DATA/counted.tsv"
MANIFEST="$DATA/manifest.tsv"
JUDGE_ATTEMPTS="$DATA/judge-attempts.tsv"
LOG_DIR="$DATA/logs"
DRIVER_LOG="$DATA/driver.log"
LOCK="$DATA/driver.lock"
MODEL=gpt-5.4-2026-03-05

mkdir -p "$LOG_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another confirmation driver holds $LOCK" >&2
  exit 2
fi
exec >>"$DRIVER_LOG" 2>&1
cd "$ROOT" || exit 1

if [ ! -f "$ATTEMPTS" ]; then
  printf 'slot\tpair\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir\ttranscript_rows\tend_session_turn\tclassification\tlog\n' > "$ATTEMPTS"
fi
if [ ! -f "$COUNTED" ]; then
  printf 'slot\tpair\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir\ttranscript_rows\tend_session_turn\tclassification\tjudge_rc\n' > "$COUNTED"
fi
if [ ! -f "$MANIFEST" ]; then
  printf 'config\trun_dir\n' > "$MANIFEST"
fi
if [ ! -f "$JUDGE_ATTEMPTS" ]; then
  printf 'slot\trun_dir\tstart_utc\tend_utc\tjudge_rc\tlog\n' > "$JUDGE_ATTEMPTS"
fi

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

verify_file() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    log "INTEGRITY FAILURE $path expected=$expected actual=$actual"
    exit 3
  fi
}

verify_integrity() {
  verify_file ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164 benchmarks/aiwf_medium_context/config.py
  verify_file 6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6 benchmarks/aiwf_medium_context/prompts/system.py
  verify_file c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b benchmarks/_shared/turns.py
  verify_file d0d3ea02d69797c56e7d1395b752b04d132f003b3148b3d8e847f69067bf0d15 src/multi_turn_eval/services/filler.py
  verify_file 863b58d390fefb84d237f4382039f89ad77af12ab70f006274925a32d8cdfb80 src/multi_turn_eval/services/openai_responses.py
  verify_file 2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c src/multi_turn_eval/pipelines/base.py
  verify_file 3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18 src/multi_turn_eval/judging/claude_judge.py
}

is_counted() {
  awk -F '\t' -v slot="$1" 'NR > 1 && $1 == slot { found=1 } END { exit !found }' "$COUNTED"
}

attempt_count() {
  awk -F '\t' -v slot="$1" 'NR > 1 && $1 == slot { n++ } END { print n+0 }' "$ATTEMPTS"
}

run_dir_from_log() {
  local path=$1
  sed -n 's/^Output directory: //p' "$path" | tail -1
}

end_session_turn() {
  python3 - "$1" <<'PY'
import json
import sys

best = -1
for line in open(sys.argv[1]):
    row = json.loads(line)
    if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
        best = max(best, int(row.get("turn", -1)))
print(best)
PY
}

objective_infrastructure_failure() {
  local path=$1
  grep -qiE 'Pipeline failed|timed out|timeout|connection|rate.?limit|HTTP[/ ]+[45][0-9][0-9]|(^|[^0-9])429([^0-9]|$)|APIError|InternalServerError|ServiceUnavailable|OPENAI_API_KEY.*required|Traceback' "$path"
}

judge_run() {
  local slot=$1 run_dir=$2
  local judge_log="$run_dir/judge-confirmation.log"
  local start end rc
  start=$(date -u --iso-8601=seconds)
  uv run multi-turn-eval judge "$run_dir" > "$judge_log" 2>&1
  rc=$?
  end=$(date -u --iso-8601=seconds)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$slot" "$run_dir" "$start" "$end" "$rc" "$judge_log" >> "$JUDGE_ATTEMPTS"
  return "$rc"
}

run_model_attempt() {
  local arm=$1 log_path=$2
  if [ "$arm" = dash96 ]; then
    env -u MTE_FILLER_DOTS -u MTE_FILLER_TOKEN -u MTE_FILLER_POSITION \
      MTE_OPENAI_RESPONSES_REASONING_EFFORT=none \
      MTE_FILLER_DOTS=96 \
      MTE_FILLER_TOKEN=- \
      MTE_FILLER_POSITION=suffix \
      MTE_ENABLE_RECOVERY=1 \
      MTE_DEDUPE_TOOL_CALLS=1 \
      MTE_TOOL_RESULT_RUN_LLM=0 \
      MTE_TEXT_IDLE_TIMEOUT_SECS=45 \
      uv run multi-turn-eval run aiwf_medium_context \
        --model "$MODEL" --service openai --pipeline text > "$log_path" 2>&1
  else
    env -u MTE_FILLER_DOTS -u MTE_FILLER_TOKEN -u MTE_FILLER_POSITION \
      MTE_OPENAI_RESPONSES_REASONING_EFFORT=none \
      MTE_ENABLE_RECOVERY=1 \
      MTE_DEDUPE_TOOL_CALLS=1 \
      MTE_TOOL_RESULT_RUN_LLM=0 \
      MTE_TEXT_IDLE_TIMEOUT_SECS=45 \
      uv run multi-turn-eval run aiwf_medium_context \
        --model "$MODEL" --service openai --pipeline text > "$log_path" 2>&1
  fi
}

log "driver start model=$MODEL schedule_sha=$(sha256sum "$SCHEDULE" | cut -d' ' -f1)"
verify_file d851c4fd3906492118d775ddfecb7f2e95cd7963687cb153cfb7cec44429a624 "$SCHEDULE"
verify_integrity
if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -qE '^OPENAI_API_KEY=..+' .env 2>/dev/null; then
  log "OPENAI_API_KEY not available"
  exit 5
fi

while IFS=$'\t' read -r slot pair pair_position arm; do
  [ "$slot" = slot ] && continue
  if is_counted "$slot"; then
    log "slot=$slot pair=$pair arm=$arm already counted; skip"
    continue
  fi

  replacements=0
  while ! is_counted "$slot"; do
    verify_integrity
    attempt=$(( $(attempt_count "$slot") + 1 ))
    start=$(date -u --iso-8601=seconds)
    log_path="$LOG_DIR/slot-$(printf '%03d' "$slot")-${arm}-attempt-${attempt}.log"
    log "slot=$slot pair=$pair position=$pair_position arm=$arm attempt=$attempt start"
    run_model_attempt "$arm" "$log_path"
    run_rc=$?
    end=$(date -u --iso-8601=seconds)
    run_dir=$(run_dir_from_log "$log_path")
    transcript_rows=0
    es_turn=-1
    if [ -n "$run_dir" ] && [ -f "$run_dir/transcript.jsonl" ]; then
      transcript_rows=$(wc -l < "$run_dir/transcript.jsonl")
      if [ "$transcript_rows" -gt 0 ]; then
        es_turn=$(end_session_turn "$run_dir/transcript.jsonl")
      fi
    fi

    if [ "$transcript_rows" -eq 0 ]; then
      if objective_infrastructure_failure "$log_path"; then
        classification=infra_zero_response_replaced
        replacements=$((replacements + 1))
      else
        classification=zero_response_unclassified
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$slot" "$pair" "$arm" "$attempt" "$start" "$end" "$run_rc" "${run_dir:-NA}" \
        "$transcript_rows" "$es_turn" "$classification" "$log_path" >> "$ATTEMPTS"
      log "slot=$slot arm=$arm attempt=$attempt classification=$classification rc=$run_rc"
      if [ "$classification" = zero_response_unclassified ]; then
        log "manual arm-blind classification required; stopping"
        exit 6
      fi
      if [ "$replacements" -ge 3 ]; then
        log "slot=$slot exhausted three objective infrastructure replacements; stopping"
        exit 7
      fi
      continue
    fi

    if [ "$es_turn" -eq 29 ]; then
      classification=strict_complete
    elif [ "$es_turn" -ge 0 ]; then
      classification=model_abort
    else
      classification=incomplete_no_end_session
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$slot" "$pair" "$arm" "$attempt" "$start" "$end" "$run_rc" "$run_dir" \
      "$transcript_rows" "$es_turn" "$classification" "$log_path" >> "$ATTEMPTS"

    judge_rc=0
    if ! judge_run "$slot" "$run_dir"; then
      judge_rc=$?
      # The negated command's status is zero here; recover the actual latest row.
      judge_rc=$(tail -1 "$JUDGE_ATTEMPTS" | cut -f5)
      log "slot=$slot judge pending rc=$judge_rc"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$slot" "$pair" "$arm" "$attempt" "$start" "$end" "$run_rc" "$run_dir" \
      "$transcript_rows" "$es_turn" "$classification" "$judge_rc" >> "$COUNTED"
    printf '%s\t%s\n' "$arm" "$run_dir" >> "$MANIFEST"
    log "slot=$slot arm=$arm counted classification=$classification rows=$transcript_rows es=$es_turn judge_rc=$judge_rc"
  done
done < "$SCHEDULE"

# One non-adaptive retry pass for judge failures/missing judge files. Model allocation
# is already complete and is never changed by this pass.
while IFS=$'\t' read -r slot pair arm attempt start end run_rc run_dir transcript_rows es_turn classification judge_rc; do
  [ "$slot" = slot ] && continue
  if [ ! -s "$run_dir/claude_judged.jsonl" ]; then
    log "slot=$slot retrying missing judgment"
    judge_run "$slot" "$run_dir" || log "slot=$slot judgment still pending"
  fi
done < "$COUNTED"

counted_total=$(awk 'END { print NR-1 }' "$COUNTED")
control_total=$(awk -F '\t' 'NR > 1 && $3 == "nofiller" { n++ } END { print n+0 }' "$COUNTED")
dash_total=$(awk -F '\t' 'NR > 1 && $3 == "dash96" { n++ } END { print n+0 }' "$COUNTED")
missing_judges=0
while IFS=$'\t' read -r slot pair arm attempt start end run_rc run_dir rest; do
  [ "$slot" = slot ] && continue
  [ -s "$run_dir/claude_judged.jsonl" ] || missing_judges=$((missing_judges + 1))
done < "$COUNTED"
log "driver complete counted=$counted_total control=$control_total dash=$dash_total missing_judges=$missing_judges"
if [ "$counted_total" -ne 82 ] || [ "$control_total" -ne 41 ] || [ "$dash_total" -ne 41 ]; then
  exit 8
fi
if [ "$missing_judges" -ne 0 ]; then
  exit 9
fi
