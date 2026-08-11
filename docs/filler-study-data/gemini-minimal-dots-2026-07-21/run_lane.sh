#!/usr/bin/env bash
# Resumable per-model driver for the staged Gemini campaign.
set -uo pipefail

ROOT=/home/khkramer/src/aiewf-eval
DATA="$ROOT/docs/filler-study-data/gemini-minimal-dots-2026-07-21"
LANE=${1:-}
case "$LANE" in
  g35|g35control|g35lite|g36|g35topup|g35litetopup|g36topup|g35focused|g35litefocused|g36focused) ;;
  *) echo "usage: $0 {g35|g35control|g35lite|g36|g35topup|g35litetopup|g36topup|g35focused|g35litefocused|g36focused}" >&2; exit 2 ;;
esac

SCHEDULE="$DATA/schedule-$LANE.tsv"
STATE="$DATA/state/$LANE"
ATTEMPTS="$STATE/attempts.tsv"
COUNTED="$STATE/counted.tsv"
MANIFEST="$STATE/manifest.tsv"
LOG_DIR="$STATE/logs"
DRIVER_LOG="$STATE/driver.log"
JUDGE_LOG="$STATE/judge-attempts.tsv"
JUDGE_LOCK="$DATA/state/judge.lock"

mkdir -p "$LOG_DIR" "$DATA/state"
exec 9>"$STATE/driver.lock"
if ! flock -n 9; then
  echo "another driver owns lane $LANE" >&2
  exit 2
fi
exec >>"$DRIVER_LOG" 2>&1
cd "$ROOT" || exit 2

if [ ! -f "$ATTEMPTS" ]; then
  printf 'slot\tmodel\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir\ttranscript_rows\tend_session_turn\tclassification\tlog\n' > "$ATTEMPTS"
fi
if [ ! -f "$COUNTED" ]; then
  printf 'slot\tmodel\tarm\tattempt\trun_dir\tclassification\tjudge_rc\n' > "$COUNTED"
fi
if [ ! -f "$MANIFEST" ]; then
  printf 'model\tarm\trun_dir\n' > "$MANIFEST"
fi
if [ ! -f "$JUDGE_LOG" ]; then
  printf 'slot\trun_dir\tattempt\tstart_utc\tend_utc\tjudge_rc\tlog\n' > "$JUDGE_LOG"
fi

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

verify_file() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    log "INTEGRITY FAILURE path=$path expected=$expected actual=$actual"
    exit 3
  fi
}

verify_integrity() {
  verify_file ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164 benchmarks/aiwf_medium_context/config.py
  verify_file 6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6 benchmarks/aiwf_medium_context/prompts/system.py
  verify_file c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b benchmarks/_shared/turns.py
  verify_file 2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c src/multi_turn_eval/pipelines/base.py
  verify_file 97294f5a086d9516ff501c638aa14d525e67cceb11e8df692f50c8f0d1c227c3 src/multi_turn_eval/services/google_logged.py
  verify_file 3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18 src/multi_turn_eval/judging/claude_judge.py
  case "$LANE" in
    g35) verify_file b7ee9062fb28b4bbddb2ce716774040ead57293fb15beb555005cc709e7156d7 "$SCHEDULE" ;;
    g35control) verify_file 8a1173fb531f01a0206652310bc54282a59ab7f4f69c71c8e1672b7394268d68 "$SCHEDULE" ;;
    g35lite) verify_file 054277eb9fbee1c054836f71aed88a099c3d4c59bd89442628e698246bb0f9c5 "$SCHEDULE" ;;
    g36) verify_file ecb1d145ce270beecaafba364670c753997854e37f4137289a9851f0f8765b66 "$SCHEDULE" ;;
    g35topup) verify_file 2eba63aff34d9426afdc9a436325d145bdf30e61cc39fb1ada52d8f248482b15 "$SCHEDULE" ;;
    g35litetopup) verify_file 2a936ef1f5f3deb52a5f7d87f2d955cd4591ea353d15d5887372b8ad53436c69 "$SCHEDULE" ;;
    g36topup) verify_file bdc055458eaa90e9a739f1d9d21783786ef85a32c884b4770e3c4ce8c145ea97 "$SCHEDULE" ;;
    g35focused) verify_file 6123f3679e5de4697cb521bb9fd72138db8aa99f57ef10e556a167db323ba839 "$SCHEDULE" ;;
    g35litefocused) verify_file c43a8d76b744d67572c081fd813a55095cc5a6b0a6c9535a0ed32e54b175bc99 "$SCHEDULE" ;;
    g36focused) verify_file ca570799585424aa79f48d32d5ec1966f4c374ea493bf12bed85519d8f5213fa "$SCHEDULE" ;;
  esac
}

allowed_model() {
  case "$1|$2" in
    gemini35flash\|gemini-3.5-flash|gemini35flashlite\|gemini-3.5-flash-lite|gemini36flash\|gemini-3.6-flash) return 0 ;;
    *) return 1 ;;
  esac
}

is_counted() {
  awk -F '\t' -v slot="$1" 'NR > 1 && $1 == slot { found=1 } END { exit !found }' "$COUNTED"
}

attempt_count() {
  awk -F '\t' -v slot="$1" 'NR > 1 && $1 == slot { n++ } END { print n+0 }' "$ATTEMPTS"
}

run_dir_from_log() {
  sed -n 's/^Output directory: //p' "$1" | tail -1
}

end_session_turn() {
  uv run python - "$1" <<'PY'
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
  grep -qiE 'DeadlineExceeded|ResourceExhausted|ReadTimeout|ConnectTimeout|Connection(Error|Reset|Refused)|rate.?limit|HTTP[/ ]+5[0-9][0-9]|(^|[^0-9])429([^0-9]|$)|InternalServerError|ServiceUnavailable|Upstream error' "$1"
}

classify_attempt() {
  local rows=$1 es_turn=$2 run_log=$3
  if [ "$es_turn" -lt 0 ] && objective_infrastructure_failure "$run_log"; then
    if [ "$rows" -eq 0 ]; then
      printf 'infra_zero_response_replaced\n'
    else
      printf 'infra_partial_response_replaced\n'
    fi
  elif [ "$rows" -eq 0 ]; then
    printf 'zero_response_unclassified\n'
  elif [ "$es_turn" -eq 29 ]; then
    printf 'strict_complete\n'
  elif [ "$es_turn" -ge 0 ]; then
    printf 'model_abort\n'
  else
    printf 'incomplete_no_end_session\n'
  fi
}

judge_run() {
  local slot=$1 run_dir=$2 judge_attempt start end rc judge_file
  judge_file="$run_dir/judge-gemini-minimal-dots.log"
  verify_integrity
  if valid_judgment "$run_dir"; then
    return 0
  fi
  for judge_attempt in 1 2 3; do
    start=$(date -u --iso-8601=seconds)
    (
      flock 8
      uv run multi-turn-eval judge "$run_dir" > "$judge_file" 2>&1
    ) 8>"$JUDGE_LOCK"
    rc=$?
    end=$(date -u --iso-8601=seconds)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$slot" "$run_dir" "$judge_attempt" "$start" "$end" "$rc" "$judge_file" >> "$JUDGE_LOG"
    if [ "$rc" -eq 0 ] && valid_judgment "$run_dir"; then
      return 0
    fi
    log "judge retry slot=$slot attempt=$judge_attempt rc=$rc"
  done
  return 1
}

valid_judgment() {
  local run_dir=$1
  uv run python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
transcript = run / "transcript.jsonl"
judged = run / "claude_judged.jsonl"
summary = run / "claude_summary.json"
if not transcript.is_file() or not judged.is_file() or not summary.is_file():
    raise SystemExit(1)
observed = set()
for line in transcript.read_text().splitlines():
    row = json.loads(line)
    turn = row.get("turn")
    if isinstance(turn, int) and 0 <= turn < 30 and row.get("recovery_turn") is not True:
        observed.add(turn)
final = {}
for line in judged.read_text().splitlines():
    row = json.loads(line)
    turn = row.get("turn")
    if isinstance(turn, int) and turn in observed:
        final[turn] = row
if set(final) != observed:
    raise SystemExit(1)
for row in final.values():
    scores = row.get("scores") or {}
    if not all(isinstance(scores.get(key), bool) for key in ("tool_use_correct", "instruction_following", "kb_grounding")):
        raise SystemExit(1)
meta = json.loads(summary.read_text())
if meta.get("turns_scored") != len(observed) or not meta.get("judge_model") or not meta.get("judge_version"):
    raise SystemExit(1)
PY
}

commit_attempt() {
  local slot=$1 model=$2 arm=$3 attempt=$4 run_dir=$5 classification=$6
  if ! judge_run "$slot" "$run_dir"; then
    log "judge failed after retries slot=$slot run=$run_dir"
    return 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t0\n' \
    "$slot" "$model" "$arm" "$attempt" "$run_dir" "$classification" >> "$COUNTED"
  printf '%s\t%s\t%s\n' "$model" "$arm" "$run_dir" >> "$MANIFEST"
  log "counted slot=$slot model=$model arm=$arm classification=$classification run=$run_dir"
}

adopt_uncommitted() {
  local slot=$1 expected_model=$2 expected_arm=$3 row
  local got_slot model arm attempt start end rc run_dir rows es_turn classification run_log
  row=$(awk -F '\t' -v slot="$slot" 'NR > 1 && $1 == slot && $11 !~ /^infra_/ { row=$0 } END { print row }' "$ATTEMPTS")
  [ -n "$row" ] || return 1
  IFS=$'\t' read -r got_slot model arm attempt start end rc run_dir rows es_turn classification run_log <<< "$row"
  if [ "$model" != "$expected_model" ] || [ "$arm" != "$expected_arm" ]; then
    log "resume mismatch slot=$slot got=$model/$arm expected=$expected_model/$expected_arm"
    exit 3
  fi
  if [ "$run_dir" = NA ] || [ ! -s "$run_dir/transcript.jsonl" ]; then
    log "cannot adopt slot=$slot run=$run_dir"
    exit 6
  fi
  log "adopting uncommitted slot=$slot attempt=$attempt run=$run_dir"
  if ! commit_attempt "$slot" "$model" "$arm" "$attempt" "$run_dir" "$classification"; then
    exit 8
  fi
  return 0
}

run_model_attempt() {
  local arm=$1 requested_model=$2 run_log=$3
  local -a clean filler common
  clean=(-u MTE_FILLER_DOTS -u MTE_FILLER_TOKEN -u MTE_FILLER_POSITION)
  filler=()
  if [ "$arm" = dots96 ]; then
    filler=(MTE_FILLER_DOTS=96 MTE_FILLER_TOKEN=. MTE_FILLER_POSITION=suffix)
  fi
  common=(
    MTE_GOOGLE_THINKING_MODE=minimal
    MTE_ENABLE_RECOVERY=1
    MTE_DEDUPE_TOOL_CALLS=1
    MTE_TOOL_RESULT_RUN_LLM=0
    MTE_TEXT_IDLE_TIMEOUT_SECS=45
  )
  env "${clean[@]}" "${common[@]}" "${filler[@]}" \
    uv run multi-turn-eval run aiwf_medium_context \
      --model "$requested_model" --service google --pipeline text > "$run_log" 2>&1
}

log "driver start lane=$LANE schedule=$(basename "$SCHEDULE")"
verify_integrity

while IFS=$'\t' read -r slot model arm requested_model; do
  [ -z "$slot" ] && continue
  [ "$slot" = slot ] && continue
  if ! allowed_model "$model" "$requested_model"; then
    log "MODEL POLICY FAILURE slot=$slot model=$model requested=$requested_model"
    exit 4
  fi
  if is_counted "$slot"; then
    continue
  fi
  if adopt_uncommitted "$slot" "$model" "$arm"; then
    continue
  fi

  while :; do
    verify_integrity
    attempt=$(( $(attempt_count "$slot") + 1 ))
    if [ "$attempt" -gt 4 ]; then
      log "replacement limit reached slot=$slot"
      exit 7
    fi
    run_log="$LOG_DIR/${slot}-attempt-${attempt}.log"
    start=$(date -u --iso-8601=seconds)
    log "run slot=$slot attempt=$attempt model=$requested_model arm=$arm"
    run_model_attempt "$arm" "$requested_model" "$run_log"
    rc=$?
    end=$(date -u --iso-8601=seconds)
    run_dir=$(run_dir_from_log "$run_log")
    [ -n "$run_dir" ] || run_dir=NA
    rows=0
    es_turn=-1
    if [ "$run_dir" != NA ] && [ -s "$run_dir/transcript.jsonl" ]; then
      rows=$(wc -l < "$run_dir/transcript.jsonl")
      es_turn=$(end_session_turn "$run_dir/transcript.jsonl")
    fi
    classification=$(classify_attempt "$rows" "$es_turn" "$run_log")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$slot" "$model" "$arm" "$attempt" "$start" "$end" "$rc" "$run_dir" \
      "$rows" "$es_turn" "$classification" "$run_log" >> "$ATTEMPTS"
    log "attempt slot=$slot attempt=$attempt rc=$rc rows=$rows es=$es_turn classification=$classification"
    if [[ "$classification" == infra_*_replaced ]]; then
      continue
    fi
    if [ "$classification" = zero_response_unclassified ]; then
      log "unclassified zero-response slot=$slot; stopping"
      exit 6
    fi
    if ! commit_attempt "$slot" "$model" "$arm" "$attempt" "$run_dir" "$classification"; then
      exit 8
    fi
    break
  done
done < "$SCHEDULE"

touch "$STATE/COMPLETE"
log "COMPLETE lane=$LANE"
