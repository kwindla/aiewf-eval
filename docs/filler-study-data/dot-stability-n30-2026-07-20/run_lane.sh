#!/usr/bin/env bash
# Resumable provider-lane driver for the frozen n=30 dot campaign.
set -uo pipefail

ROOT=/home/khkramer/src/aiewf-eval
DATA="$ROOT/docs/filler-study-data/dot-stability-n30-2026-07-20"
LANE=${1:-}
SCHEDULE="$DATA/schedule-$LANE.tsv"
STATE="$DATA/state/$LANE"
ATTEMPTS="$STATE/attempts.tsv"
COUNTED="$STATE/counted.tsv"
MANIFEST="$STATE/manifest.tsv"
JUDGE_ATTEMPTS="$STATE/judge-attempts.tsv"
LOG_DIR="$STATE/logs"
DRIVER_LOG="$STATE/driver.log"
LOCK="$STATE/driver.lock"
JUDGE_LOCK="$DATA/state/judge.lock"
INVALIDATED="$DATA/invalidated.tsv"
MAX_INFRA_REPLACEMENTS=${MTE_N30_MAX_INFRA_REPLACEMENTS:-3}

case "$LANE" in
  openai-a|openai-b|lilac|baseten|openrouter|baseten-qwen) ;;
  *) echo "usage: $0 {openai-a|openai-b|lilac|baseten|openrouter|baseten-qwen}" >&2; exit 2 ;;
esac
if [ ! -f "$SCHEDULE" ]; then
  echo "missing schedule: $SCHEDULE" >&2
  exit 2
fi
if ! [[ "$MAX_INFRA_REPLACEMENTS" =~ ^[0-9]+$ ]] || [ "$MAX_INFRA_REPLACEMENTS" -lt 3 ] || [ "$MAX_INFRA_REPLACEMENTS" -gt 12 ]; then
  echo "MTE_N30_MAX_INFRA_REPLACEMENTS must be an integer from 3 through 12" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$DATA/state"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another driver holds $LOCK" >&2
  exit 2
fi
exec >>"$DRIVER_LOG" 2>&1
cd "$ROOT" || exit 1

if [ ! -f "$ATTEMPTS" ]; then
  printf 'slot\tmodel\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir\ttranscript_rows\tend_session_turn\tclassification\tlog\n' > "$ATTEMPTS"
fi
if [ ! -f "$COUNTED" ]; then
  printf 'slot\tmodel\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir\ttranscript_rows\tend_session_turn\tclassification\tjudge_rc\n' > "$COUNTED"
fi
if [ ! -f "$MANIFEST" ]; then
  printf 'model\tarm\trun_dir\n' > "$MANIFEST"
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
  verify_file dd79b7227cc2c3578eb113b879e4efbd2b08af1283e93d5ce3226a55007a936d src/multi_turn_eval/services/lilac_logged.py
  verify_file 6b605a2c065ba35f561cf57621e5d2d2dd7f6df24d78cc82e37a1993a1e7fb08 src/multi_turn_eval/services/vllm_openai.py
  verify_file 2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c src/multi_turn_eval/pipelines/base.py
  verify_file 3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18 src/multi_turn_eval/judging/claude_judge.py
  verify_file e3bbb32a1a9fc279c928006fe614627cedd70e59f8defbbf2cddd303906f349f src/multi_turn_eval/model_policy.py
  verify_file 7a8eeb0b14c43f7a0d770fd7019d40d255b39718d97c6a1a4071815230c9597d ops/baseten-qwen3-8b-vllm/config.yaml
}

verify_schedule() {
  local expected actual
  expected=$(awk -F '\t' -v f="$(basename "$SCHEDULE")" '$1==f {print $2}' "$DATA/schedule-hashes.tsv")
  actual=$(sha256sum "$SCHEDULE" | cut -d' ' -f1)
  if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    log "SCHEDULE INTEGRITY FAILURE expected=$expected actual=$actual schedule=$SCHEDULE"
    exit 3
  fi
}

allowed_model() {
  case "$1|$2" in
    gpt-5.4\|openai|gpt-5.5\|openai|gpt-5.6-terra\|openai|gpt-5.6-sol\|openai|lilac/gemma-4-31b-it\|lilac|thinkingmachines/inkling\|baseten|qwen/qwen3-8b\|openrouter|qwen/qwen3-8b\|vllm-openai|zai-org/GLM-5.2\|baseten) return 0 ;;
    *) return 1 ;;
  esac
}

is_counted() {
  awk -F '\t' -v lane="$LANE" -v slot="$1" '
    FNR == NR { if (FNR > 1 && $1 == lane && $2 == slot) invalid[$3] = 1; next }
    FNR > 1 && $1 == slot && !invalid[$4] { found=1 }
    END { exit !found }
  ' "$INVALIDATED" "$COUNTED"
}

is_invalidated() {
  awk -F '\t' -v lane="$LANE" -v slot="$1" -v attempt="$2" '
    NR > 1 && $1 == lane && $2 == slot && $3 == attempt { found=1 }
    END { exit !found }
  ' "$INVALIDATED"
}

attempt_count() {
  awk -F '\t' -v slot="$1" 'NR > 1 && $1 == slot { n++ } END { print n+0 }' "$ATTEMPTS"
}

run_dir_from_log() {
  sed -n 's/^Output directory: //p' "$1" | tail -1
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
  grep -viE 'idle_timeout_secs|MTE_TEXT_IDLE_TIMEOUT' "$1" | \
    grep -qiE 'Pipeline failed|Idle timeout detected|timed out|ReadTimeout|ConnectTimeout|Connection(Error|Reset|Refused)|rate.?limit|HTTP[/ ]+[45][0-9][0-9]|(^|[^0-9])429([^0-9]|$)|APIError|InternalServerError|ServiceUnavailable|EngineCore|Upstream error|Traceback'
}

load_baseten_key() {
  if [ -n "${BASETEN_API_KEY:-}" ]; then
    return
  fi
  local key_file=/home/khkramer/src/gb-benchmarks/.env
  if [ ! -f "$key_file" ]; then
    log "BASETEN_API_KEY source missing: $key_file"
    exit 5
  fi
  BASETEN_API_KEY=$(uv run python - "$key_file" <<'PY'
from dotenv import dotenv_values
import sys

print(dotenv_values(sys.argv[1]).get("BASETEN_API_KEY", ""))
PY
  )
  export BASETEN_API_KEY
  if [ -z "${BASETEN_API_KEY:-}" ]; then
    log "BASETEN_API_KEY unavailable after loading sibling .env"
    exit 5
  fi
}

judge_run() {
  local slot=$1 run_dir=$2 start end rc
  local judge_log="$run_dir/judge-dot-n30.log"
  start=$(date -u --iso-8601=seconds)
  (
    flock 8
    uv run multi-turn-eval judge "$run_dir" > "$judge_log" 2>&1
  ) 8>"$JUDGE_LOCK"
  rc=$?
  end=$(date -u --iso-8601=seconds)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$slot" "$run_dir" "$start" "$end" "$rc" "$judge_log" >> "$JUDGE_ATTEMPTS"
  return "$rc"
}

classify_attempt() {
  local transcript_rows=$1 es_turn=$2 log_path=$3
  if [ "$es_turn" -lt 0 ] && objective_infrastructure_failure "$log_path"; then
    if [ "$transcript_rows" -eq 0 ]; then
      printf 'infra_zero_response_replaced\n'
    else
      printf 'infra_partial_response_replaced\n'
    fi
  elif [ "$transcript_rows" -eq 0 ]; then
    printf 'zero_response_unclassified\n'
  elif [ "$es_turn" -eq 29 ]; then
    printf 'strict_complete\n'
  elif [ "$es_turn" -ge 0 ]; then
    printf 'model_abort\n'
  else
    printf 'incomplete_no_end_session\n'
  fi
}

replacement_count() {
  awk -F '\t' -v lane="$LANE" -v slot="$1" '
    FNR == NR { if (FNR > 1 && $1 == lane && $2 == slot) seen[$3] = 1; next }
    FNR > 1 && $1 == slot && $11 ~ /^infra_.*_replaced$/ { seen[$4] = 1 }
    END { for (attempt in seen) n++; print n+0 }
  ' "$INVALIDATED" "$ATTEMPTS"
}

commit_attempt() {
  local slot=$1 model_label=$2 arm=$3 attempt=$4 start=$5 end=$6 run_rc=$7
  local run_dir=$8 transcript_rows=$9 es_turn=${10} classification=${11}
  local judge_rc
  judge_run "$slot" "$run_dir"
  judge_rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$slot" "$model_label" "$arm" "$attempt" "$start" "$end" "$run_rc" "$run_dir" \
    "$transcript_rows" "$es_turn" "$classification" "$judge_rc" >> "$COUNTED"
  printf '%s\t%s\t%s\n' "$model_label" "$arm" "$run_dir" >> "$MANIFEST"
  log "slot=$slot counted classification=$classification rows=$transcript_rows es=$es_turn judge_rc=$judge_rc"
}

# Adopt a terminal model attempt that was durably logged before an interrupted
# judging step. Return 0 when adopted, 1 when no adoptable attempt exists.
adopt_uncommitted_attempt() {
  local slot=$1 expected_model=$2 expected_arm=$3 row
  local got_slot model_label arm attempt start end run_rc run_dir transcript_rows es_turn classification log_path
  row=$(awk -F '\t' -v slot="$slot" 'NR > 1 && $1 == slot { row=$0 } END { print row }' "$ATTEMPTS")
  [ -n "$row" ] || return 1
  IFS=$'\t' read -r got_slot model_label arm attempt start end run_rc run_dir transcript_rows es_turn classification log_path <<< "$row"
  if [ "$got_slot" != "$slot" ]; then
    log "RESUME INTEGRITY FAILURE requested slot=$slot found slot=$got_slot"
    exit 3
  fi
  if is_invalidated "$slot" "$attempt" || [[ "$classification" == infra_*_replaced ]]; then
    return 1
  fi
  if [ "$model_label" != "$expected_model" ] || [ "$arm" != "$expected_arm" ]; then
    log "RESUME INTEGRITY FAILURE slot=$slot logged=$model_label/$arm expected=$expected_model/$expected_arm"
    exit 3
  fi
  if [ "$classification" = zero_response_unclassified ]; then
    log "slot=$slot has an unresolved zero-response attempt; stopping"
    exit 6
  fi
  if [ "$run_dir" = NA ] || [ ! -s "$run_dir/transcript.jsonl" ]; then
    log "slot=$slot cannot adopt attempt=$attempt: transcript unavailable"
    exit 6
  fi
  # Re-apply the amended objective rule to attempts written by an older driver.
  classification=$(classify_attempt "$transcript_rows" "$es_turn" "$log_path")
  if [[ "$classification" == infra_*_replaced ]]; then
    log "slot=$slot attempt=$attempt is objective infrastructure failure under amended rule; invalidation required"
    exit 6
  fi
  log "slot=$slot adopting uncommitted attempt=$attempt before judging"
  commit_attempt "$slot" "$model_label" "$arm" "$attempt" "$start" "$end" "$run_rc" \
    "$run_dir" "$transcript_rows" "$es_turn" "$classification"
  return 0
}

run_model_attempt() {
  local model_label=$1 arm=$2 requested_model=$3 service=$4 log_path=$5
  local -a clean filler common provider
  clean=(-u MTE_FILLER_DOTS -u MTE_FILLER_TOKEN -u MTE_FILLER_POSITION)
  filler=()
  if [ "$arm" = dots96 ]; then
    filler=(MTE_FILLER_DOTS=96 MTE_FILLER_TOKEN=. MTE_FILLER_POSITION=suffix)
  fi
  common=(MTE_ENABLE_RECOVERY=1 MTE_DEDUPE_TOOL_CALLS=1 MTE_TOOL_RESULT_RUN_LLM=0 MTE_TEXT_IDLE_TIMEOUT_SECS=45)
  provider=()
  case "$model_label" in
    gpt54|gpt55|terra|sol)
      provider=(MTE_OPENAI_RESPONSES_REASONING_EFFORT=none)
      ;;
    gemma431)
      provider=(MTE_LILAC_THINKING=0)
      ;;
    inkling|glm52)
      load_baseten_key
      clean+=(-u MTE_BASETEN_ENABLE_THINKING)
      provider=(MTE_BASETEN_REASONING_EFFORT=none MTE_BASETEN_MAX_TOKENS=8192 MTE_BASETEN_TEMPERATURE=1.0)
      ;;
    qwen3_8b)
      if [ "$service" = vllm-openai ]; then
        load_baseten_key
        clean+=(
          -u MTE_VLLM_THINKING_BUDGET
          -u MTE_VLLM_NATIVE_BUDGET
          -u MTE_VLLM_GRACE
        )
        provider=(
          VLLM_API_KEY="$BASETEN_API_KEY"
          VLLM_BASE_URL="${MTE_QWEN_BASETEN_BASE_URL:-https://model-wnp6rky3.api.baseten.co/deployment/wgvnndv/sync/v1}"
          MTE_VLLM_THINKING=0
          MTE_VLLM_TEMPERATURE=0.7
          MTE_VLLM_TOP_P=0.8
          MTE_VLLM_TOP_K=20
          MTE_VLLM_MAX_TOKENS=8192
        )
      else
        provider=(MTE_OPENROUTER_REASONING_OFF=1 MTE_OPENROUTER_MAX_TOKENS=8192)
      fi
      ;;
    *)
      log "unrecognized model label: $model_label"
      return 8
      ;;
  esac
  env "${clean[@]}" "${common[@]}" "${provider[@]}" "${filler[@]}" \
    uv run multi-turn-eval run aiwf_medium_context \
      --model "$requested_model" --service "$service" --pipeline text > "$log_path" 2>&1
}

log "driver start lane=$LANE schedule=$(basename "$SCHEDULE")"
verify_schedule
verify_integrity

while IFS=$'\t' read -r slot model_label arm requested_model service; do
  [ "$slot" = slot ] && continue
  if ! allowed_model "$requested_model" "$service"; then
    log "MODEL POLICY FAILURE slot=$slot model=$requested_model service=$service"
    exit 4
  fi
  case "${requested_model,,}" in *-pro|*-pro-*) log "PRO MODEL POLICY FAILURE $requested_model"; exit 4 ;; esac
  if is_counted "$slot"; then
    log "slot=$slot already counted; skip"
    continue
  fi

  if adopt_uncommitted_attempt "$slot" "$model_label" "$arm"; then
    continue
  fi

  while ! is_counted "$slot"; do
    verify_schedule
    verify_integrity
    if [ "$(replacement_count "$slot")" -ge "$MAX_INFRA_REPLACEMENTS" ]; then
      log "slot=$slot already has $MAX_INFRA_REPLACEMENTS objective infrastructure replacements; stopping"
      exit 7
    fi
    attempt=$(( $(attempt_count "$slot") + 1 ))
    start=$(date -u --iso-8601=seconds)
    log_path="$LOG_DIR/${slot}-${model_label}-${arm}-attempt-${attempt}.log"
    log "slot=$slot model=$model_label arm=$arm attempt=$attempt start"
    run_model_attempt "$model_label" "$arm" "$requested_model" "$service" "$log_path"
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

    classification=$(classify_attempt "$transcript_rows" "$es_turn" "$log_path")
    if [[ "$classification" == infra_*_replaced ]] || [ "$classification" = zero_response_unclassified ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$slot" "$model_label" "$arm" "$attempt" "$start" "$end" "$run_rc" "${run_dir:-NA}" \
        "$transcript_rows" "$es_turn" "$classification" "$log_path" >> "$ATTEMPTS"
      log "slot=$slot classification=$classification rc=$run_rc"
      if [ "$classification" = zero_response_unclassified ]; then
        log "manual arm-blind classification required; stopping"
        exit 6
      fi
      if [ "$(replacement_count "$slot")" -ge "$MAX_INFRA_REPLACEMENTS" ]; then
        log "slot=$slot exhausted $MAX_INFRA_REPLACEMENTS objective infrastructure replacements; stopping"
        exit 7
      fi
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$slot" "$model_label" "$arm" "$attempt" "$start" "$end" "$run_rc" "$run_dir" \
      "$transcript_rows" "$es_turn" "$classification" "$log_path" >> "$ATTEMPTS"

    commit_attempt "$slot" "$model_label" "$arm" "$attempt" "$start" "$end" "$run_rc" \
      "$run_dir" "$transcript_rows" "$es_turn" "$classification"
  done
done < "$SCHEDULE"

# Non-adaptive judge retry pass. It never launches another model attempt.
while IFS=$'\t' read -r slot model_label arm attempt start end run_rc run_dir transcript_rows es_turn classification judge_rc; do
  [ "$slot" = slot ] && continue
  if [ ! -s "$run_dir/claude_judged.jsonl" ]; then
    log "slot=$slot retrying missing judgment"
    judge_run "$slot" "$run_dir" || true
  fi
done < "$COUNTED"

missing=0
while IFS=$'\t' read -r slot model_label arm requested_model service; do
  [ "$slot" = slot ] && continue
  if ! is_counted "$slot"; then
    log "missing counted slot=$slot"
    missing=$((missing + 1))
  fi
done < "$SCHEDULE"
if [ "$missing" -ne 0 ]; then
  exit 9
fi
touch "$STATE/COMPLETE"
log "driver complete lane=$LANE"
