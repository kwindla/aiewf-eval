#!/usr/bin/env bash

set -u -o pipefail

ROOT="/home/khkramer/src/aiewf-eval"
PROTOCOL_ROOT="$ROOT/ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811"
STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$ROOT" || exit 1
mkdir -p "$PROTOCOL_ROOT/smoke-logs"

for arm in off on-unbounded; do
  if [[ "$arm" == "off" ]]; then
    enable_thinking=false
  else
    enable_thinking=true
  fi
  log="$PROTOCOL_ROOT/smoke-logs/${STAMP}-${arm}.log"

  echo "SMOKE_START arm=$arm utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  curl -fsS -X POST http://127.0.0.1:8000/flush_cache | tee -a "$log"
  TOGETHER_API_KEY=dummy \
  TOGETHER_BASE_URL=http://127.0.0.1:8000/v1 \
  MTE_TOGETHER_REASONING_EFFORT=omit \
  MTE_TOGETHER_ENABLE_THINKING="$enable_thinking" \
  MTE_TOGETHER_MAX_TOKENS="" \
  MTE_TOGETHER_TEMPERATURE=1.0 \
  MTE_TOGETHER_TOP_P=0.95 \
  MTE_TEXT_IDLE_TIMEOUT_SECS=900 \
  MTE_ENABLE_RECOVERY=1 \
  MTE_DEDUPE_TOOL_CALLS=1 \
  MTE_TOOL_RESULT_RUN_LLM=0 \
  uv run multi-turn-eval run aiwf_medium_context \
    --model nemotron-3.5-lightning \
    --service together \
    --pipeline text \
    2>&1 | tee -a "$log"
  run_rc=${PIPESTATUS[0]}

  run_dir=$(rg -o 'Transcript: runs/[^[:space:]]+/transcript\.jsonl' "$log" \
    | tail -1 \
    | sed -e 's/^Transcript: //' -e 's|/transcript\.jsonl$||')
  if [[ -n "$run_dir" && -f "$run_dir/transcript.jsonl" ]]; then
    uv run python - "$arm" "$run_dir/transcript.jsonl" <<'PY'
import json
import sys

arm, path = sys.argv[1:]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
scripted = [row for row in rows if not row.get("recovery_turn") and 0 <= int(row["turn"]) < 30]
raw = [row["raw_ttfb_ms"] for row in scripted if row.get("raw_ttfb_ms") is not None]
answer = [row["ttfb_ms"] for row in scripted if row.get("ttfb_ms") is not None]
print(
    f"SMOKE_METRICS arm={arm} scripted={len(scripted)} raw_ttft={len(raw)} "
    f"ttfat={len(answer)} raw_p50={sorted(raw)[len(raw)//2] if raw else 'missing'} "
    f"ttfat_p50={sorted(answer)[len(answer)//2] if answer else 'missing'}"
)
if len(scripted) != 30 or len(raw) != 30 or len(answer) != 30:
    raise SystemExit(2)
PY
    metrics_rc=$?
  else
    metrics_rc=2
  fi
  echo "SMOKE_EXIT arm=$arm run_rc=$run_rc metrics_rc=$metrics_rc run_dir=$run_dir utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

