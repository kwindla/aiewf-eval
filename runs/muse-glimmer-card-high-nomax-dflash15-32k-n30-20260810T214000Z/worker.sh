#!/usr/bin/env bash

set -u

CAMPAIGN_ROOT="runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z"
SERVER_BIN="/home/khkramer/src/llama-cpp/llama.cpp/build-muse-glimmer/bin/llama-server"
MAIN_MODEL="/home/khkramer/src/llama-cpp/models/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-dynamic.gguf"
DRAFT_MODEL="/home/khkramer/src/llama-cpp/models/Muse-Glimmer-30B-GGUF/dflash-kquant.gguf"
TARGET_RUNS=30
MAX_ATTEMPTS=45
SERVER_PID=""

mkdir -p "$CAMPAIGN_ROOT/logs"
: > "$CAMPAIGN_ROOT/included-runs.txt"
printf 'attempt\tcohort_index\trun_dir\trun_rc\trecords\tscripted_turns\tstatus\tstarted_utc\tfinished_utc\n' \
  > "$CAMPAIGN_ROOT/attempts.tsv"

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

echo "CAMPAIGN_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) target=$TARGET_RUNS"

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
  --chat-template-kwargs '{"reasoning_strength":"high"}' \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --min-p 0.0 \
  --spec-draft-model "$DRAFT_MODEL" \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  --gpu-layers-draft all \
  --host 127.0.0.1 \
  --port 8080 \
  > "$CAMPAIGN_ROOT/server.log" 2>&1 &
SERVER_PID=$!

READY=0
for _ready_i in $(seq 1 180); do
  if curl --silent --fail http://127.0.0.1:8080/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "SERVER_FAILED utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi

nvidia-smi \
  --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader > "$CAMPAIGN_ROOT/gpu-ready.csv"
echo "SERVER_READY utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

attempt=0
included=0
while [[ "$included" -lt "$TARGET_RUNS" && "$attempt" -lt "$MAX_ATTEMPTS" ]]; do
  attempt=$((attempt + 1))
  cohort_index=$((included + 1))
  started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  run_log="$CAMPAIGN_ROOT/logs/run-attempt-$(printf '%02d' "$attempt").log"
  echo "RUN_START attempt=$attempt cohort_index=$cohort_index utc=$started_utc"

  BASETEN_API_KEY=dummy \
  BASETEN_BASE_URL=http://127.0.0.1:8080/v1 \
  MTE_BASETEN_REASONING_EFFORT=omit \
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
  status="missing_transcript"
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
    if [[ "$run_rc" -eq 0 && "$scripted_turns" -eq 30 ]]; then
      status="included"
      included=$((included + 1))
      printf '%s\n' "$run_dir" >> "$CAMPAIGN_ROOT/included-runs.txt"
    fi
  fi
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$attempt" "$cohort_index" "$run_dir" "$run_rc" "$records" \
    "$scripted_turns" "$status" "$started_utc" "$finished_utc" \
    >> "$CAMPAIGN_ROOT/attempts.tsv"
  echo "RUN_EXIT attempt=$attempt included=$included/$TARGET_RUNS rc=$run_rc records=$records scripted=$scripted_turns status=$status utc=$finished_utc"
done

stop_server
trap - EXIT INT TERM
echo "COLLECTION_COMPLETE included=$included attempts=$attempt utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$included" -ne "$TARGET_RUNS" ]]; then
  echo "COLLECTION_INCOMPLETE expected=$TARGET_RUNS actual=$included"
  exit 2
fi
