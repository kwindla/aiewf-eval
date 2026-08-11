#!/usr/bin/env bash

set -uo pipefail

PROTOCOL_ROOT="ops/local-muse-glimmer-reasoning-sweep-20260811"
CAMPAIGN_ROOT="runs/muse-glimmer-reasoning-strength-n30-20260811"
SCHEDULE="$PROTOCOL_ROOT/schedule.tsv"
SERVER_BIN="/home/khkramer/src/llama-cpp/llama.cpp/build-muse-glimmer/bin/llama-server"
MAIN_MODEL="/home/khkramer/src/llama-cpp/models/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-dynamic.gguf"
DRAFT_MODEL="/home/khkramer/src/llama-cpp/models/Muse-Glimmer-30B-GGUF/dflash-kquant.gguf"
MAX_ATTEMPTS_PER_SLOT=3
SERVER_PID=""

mkdir -p "$CAMPAIGN_ROOT/logs" "$CAMPAIGN_ROOT/slot-cache"
if [[ ! -f "$SCHEDULE" ]]; then
  python3 "$PROTOCOL_ROOT/make_schedule.py"
fi
if [[ ! -f "$CAMPAIGN_ROOT/attempts.tsv" ]]; then
  printf 'ordinal\tblock\tposition\tarm\tarm_index\tattempt\trun_dir\trun_rc\trecords\tscripted_turns\trequest_verified\tstatus\tstarted_utc\tfinished_utc\n' \
    > "$CAMPAIGN_ROOT/attempts.tsv"
fi
if [[ ! -f "$CAMPAIGN_ROOT/included.tsv" ]]; then
  printf 'ordinal\tblock\tposition\tarm\tarm_index\trun_dir\n' \
    > "$CAMPAIGN_ROOT/included.tsv"
fi

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID"
    for _stop_i in $(seq 1 30); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID"
    fi
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

server_stamp=$(date -u +%Y%m%dT%H%M%SZ)
server_log="$CAMPAIGN_ROOT/logs/server-$server_stamp.log"
echo "CAMPAIGN_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) schedule=$SCHEDULE"

# No server-wide reasoning-strength default is supplied. Each request carries
# the scheduled value through chat_template_kwargs. The slot is erased before
# each conversation below, while normal safe prefix reuse remains available
# between turns of that conversation.
"$SERVER_BIN" \
  --model "$MAIN_MODEL" \
  --alias muse-glimmer-30b \
  --gpu-layers all \
  --ctx-size 32768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --parallel 1 \
  --jinja \
  --reasoning on \
  --reasoning-budget -1 \
  --slot-save-path "$CAMPAIGN_ROOT/slot-cache" \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --min-p 0.0 \
  --spec-draft-model "$DRAFT_MODEL" \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  --gpu-layers-draft all \
  --metrics \
  --host 127.0.0.1 \
  --port 8080 \
  > "$server_log" 2>&1 &
SERVER_PID=$!

ready=0
for _ready_i in $(seq 1 180); do
  if curl --silent --fail http://127.0.0.1:8080/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "SERVER_FAILED log=$server_log utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi

nvidia-smi \
  --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$CAMPAIGN_ROOT/gpu-ready-$server_stamp.csv"

uv run python "$PROTOCOL_ROOT/audit_template.py" \
  http://127.0.0.1:8080 "$CAMPAIGN_ROOT/template-render-audit.json" \
  > "$CAMPAIGN_ROOT/logs/template-audit-$server_stamp.log" 2>&1
audit_rc=$?
if [[ "$audit_rc" -ne 0 ]]; then
  echo "TEMPLATE_AUDIT_FAILED rc=$audit_rc"
  exit 1
fi

sha256sum \
  "$PROTOCOL_ROOT/README.md" \
  "$PROTOCOL_ROOT/make_schedule.py" \
  "$PROTOCOL_ROOT/audit_template.py" \
  "$PROTOCOL_ROOT/worker.sh" \
  "$PROTOCOL_ROOT/judge.sh" \
  "$PROTOCOL_ROOT/analyze.py" \
  "$SCHEDULE" \
  benchmarks/aiwf_medium_context/config.py \
  > "$CAMPAIGN_ROOT/source-sha256.txt"

while IFS=$'\t' read -r ordinal block position arm arm_index; do
  [[ "$ordinal" == "ordinal" ]] && continue
  if awk -F $'\t' -v wanted="$ordinal" \
      'NR > 1 && $1 == wanted { found=1 } END { exit !found }' \
      "$CAMPAIGN_ROOT/included.tsv"; then
    echo "RUN_SKIP ordinal=$ordinal arm=$arm reason=already_included"
    continue
  fi

  included=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS_PER_SLOT"); do
    started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    run_log="$CAMPAIGN_ROOT/logs/run-$(printf '%03d' "$ordinal")-$arm-$server_stamp-attempt-$attempt.log"
    echo "RUN_START ordinal=$ordinal arm=$arm arm_index=$arm_index attempt=$attempt utc=$started_utc"

    # One slot is used. Erasing it here prevents cross-trajectory and cross-arm
    # KV provenance while retaining prefix caching during the conversation.
    erase_result=$(curl --silent --show-error --fail -X POST \
      'http://127.0.0.1:8080/slots/0?action=erase')
    erase_rc=$?
    if [[ "$erase_rc" -ne 0 ]]; then
      echo "SLOT_ERASE_FAILED ordinal=$ordinal arm=$arm rc=$erase_rc"
      exit 3
    fi
    printf '%s\n' "$erase_result" \
      > "$CAMPAIGN_ROOT/logs/slot-erase-$(printf '%03d' "$ordinal")-$arm-$server_stamp-attempt-$attempt.json"

    BASETEN_API_KEY=dummy \
    BASETEN_BASE_URL=http://127.0.0.1:8080/v1 \
    MTE_BASETEN_REASONING_EFFORT=omit \
    MTE_BASETEN_REASONING_STRENGTH="$arm" \
    MTE_BASETEN_MAX_TOKENS="" \
    MTE_BASETEN_TEMPERATURE=1.0 \
    MTE_BASETEN_TOP_P=0.95 \
    MTE_TEXT_IDLE_TIMEOUT_SECS=180 \
    uv run multi-turn-eval run aiwf_medium_context \
      --model muse-glimmer-30b \
      --service baseten \
      --pipeline text \
      > "$run_log" 2>&1
    run_rc=$?

    run_dir=$(rg -o 'Transcript: runs/[^[:space:]]+/transcript\.jsonl' "$run_log" \
      | tail -1 \
      | sed -e 's/^Transcript: //' -e 's|/transcript\.jsonl$||')
    records=0
    scripted_turns=0
    request_verified=0
    status="missing_transcript"
    if rg -q "reasoning_strength=$arm" "$run_log"; then
      request_verified=1
    fi
    if [[ -n "$run_dir" && -f "$run_dir/transcript.jsonl" ]]; then
      records=$(wc -l < "$run_dir/transcript.jsonl")
      scripted_turns=$(python3 - "$run_dir/transcript.jsonl" <<'PY'
import json
import sys

count = 0
with open(sys.argv[1]) as handle:
    for line in handle:
        record = json.loads(line)
        if not record.get("recovery_turn") and 0 <= int(record["turn"]) < 30:
            count += 1
print(count)
PY
)
      status="incomplete"
      if [[ "$run_rc" -eq 0 && "$scripted_turns" -eq 30 && "$request_verified" -eq 1 ]]; then
        status="included"
        included=1
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$ordinal" "$block" "$position" "$arm" "$arm_index" "$run_dir" \
          >> "$CAMPAIGN_ROOT/included.tsv"
      fi
    fi
    finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ordinal" "$block" "$position" "$arm" "$arm_index" "$attempt" \
      "$run_dir" "$run_rc" "$records" "$scripted_turns" "$request_verified" \
      "$status" "$started_utc" "$finished_utc" \
      >> "$CAMPAIGN_ROOT/attempts.tsv"
    echo "RUN_EXIT ordinal=$ordinal arm=$arm rc=$run_rc scripted=$scripted_turns request_verified=$request_verified status=$status utc=$finished_utc"
    [[ "$included" -eq 1 ]] && break
  done
  if [[ "$included" -ne 1 ]]; then
    echo "COLLECTION_BLOCKED ordinal=$ordinal arm=$arm"
    exit 2
  fi
done < "$SCHEDULE"

stop_server
trap - EXIT INT TERM

read -r total low medium high xhigh < <(awk -F $'\t' '
  NR > 1 { count[$4]++; total++ }
  END { print total+0, count["low"]+0, count["medium"]+0, count["high"]+0, count["xhigh"]+0 }
' "$CAMPAIGN_ROOT/included.tsv")
echo "COLLECTION_COMPLETE total=$total low=$low medium=$medium high=$high xhigh=$xhigh utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$total" -eq 120 && "$low" -eq 30 && "$medium" -eq 30 && "$high" -eq 30 && "$xhigh" -eq 30 ]]
