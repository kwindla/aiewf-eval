#!/usr/bin/env bash
set -euo pipefail

MODAL=/home/khkramer/.pyenv/versions/3.12.10/bin/modal
PY=/home/khkramer/.pyenv/versions/3.12.10/bin/python
APP_ID=ap-dIotBPY2GHeT2tlEpSLNB1
ENDPOINT=https://daily--nemotron-super-bf16-030326-no-mtp-serve.modal.run
RESULT_DIR=/home/khkramer/src/aiewf-eval/docs/nemotron-super-modal-optimization-data/bf16-030326-no-mtp

exec > >(tee -a "$RESULT_DIR/orchestration.log") 2>&1

CLEANED=0
cleanup() {
  local state_tasks
  if [[ "$CLEANED" == 1 ]]; then
    return 0
  fi
  CLEANED=1
  "$MODAL" app stop "$APP_ID" || true
  for _ in $(seq 1 30); do
    "$MODAL" app list --json > "$RESULT_DIR/app-list-cleanup-latest.json" || true
    state_tasks=$(jq -r --arg id "$APP_ID" \
      '.[] | select(."App ID"==$id) | [.State,.Tasks] | @tsv' \
      "$RESULT_DIR/app-list-cleanup-latest.json") || state_tasks=missing
    if [[ "$state_tasks" == $'stopped\t0' ]]; then
      printf 'Cleanup verified: %s %s\n' "$APP_ID" "$state_tasks"
      return 0
    fi
    sleep 2
  done
  printf 'Modal app did not drain: %s %s\n' "$APP_ID" "$state_tasks" >&2
  return 1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Workflow start: %s\n' "$(date --iso-8601=seconds)"

curl --fail-with-body -sSL --max-time 2700 \
  "$ENDPOINT/v1/models" | tee "$RESULT_DIR/models.json"

"$MODAL" app list --json > "$RESULT_DIR/app-list-after-ready.json"
test "$(jq -r --arg id "$APP_ID" \
  '.[] | select(."App ID"==$id) | .Tasks' \
  "$RESULT_DIR/app-list-after-ready.json")" = 1

"$PY" /home/khkramer/src/modal-super/probe_modal_super_endpoint.py \
  "$ENDPOINT" --output "$RESULT_DIR/probes.json"

cd /home/khkramer/src/aiewf-eval
env VLLM_BASE_URL="$ENDPOINT/v1" MTE_VLLM_THINKING=1 \
  MTE_VLLM_NATIVE_BUDGET=1 MTE_VLLM_THINKING_BUDGET=64 \
  MTE_VLLM_MAX_TOKENS=8192 uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron-3-super-120b --service vllm-openai \
  --only-turns 0,1,2,3 > "$RESULT_DIR/four-turn-smoke.log" 2>&1

/home/khkramer/src/modal-super/run_modal_quality_cell.sh \
  bf16_030326_no_mtp "$ENDPOINT" "$RESULT_DIR"

printf 'Workflow complete: %s\n' "$(date --iso-8601=seconds)"
