#!/bin/bash
# Lilac gemma-4-31b-it: 10 runs thinking-OFF then 10 runs thinking-ON, judged,
# into two allowlists. Runs SEQUENTIALLY (no concurrency) so TTFT measurements
# stay clean — accurate first-content vs raw TTFT is the whole point here.
#
# Launch detached:
#   setsid nohup scripts/run_lilac_thinking_sweep.sh > /tmp/lilac_sweep.log 2>&1 < /dev/null &
#
# Aggregate afterwards:
#   uv run python scripts/benchmark_summary.py $(cat docs/ten-run-allowlists/lilac-gemma-4-31b-it-off-2026-06-15.txt)
#   uv run python scripts/benchmark_summary.py $(cat docs/ten-run-allowlists/lilac-gemma-4-31b-it-thinking-2026-06-15.txt)
set -u
cd /home/khkramer/src/aiewf-eval || exit 1

ALLOW_DIR=docs/ten-run-allowlists
OFF_LIST="${ALLOW_DIR}/lilac-gemma-4-31b-it-off-2026-06-15.txt"
ON_LIST="${ALLOW_DIR}/lilac-gemma-4-31b-it-thinking-2026-06-15.txt"
LOG_DIR=/tmp/lilac-sweep-logs
mkdir -p "$LOG_DIR" "$ALLOW_DIR"

run_dir_from_log() {
  grep -oE 'Transcript: runs/[^ ]+/transcript\.jsonl' "$1" | tail -1 \
    | sed -e 's/^Transcript: //' -e 's|/transcript\.jsonl$||'
}

# finish_run <label> <rc> <log> <allowlist>
finish_run() {
  local label=$1 rc=$2 log=$3 list=$4
  local dir turns
  dir=$(run_dir_from_log "$log")
  if [ "$rc" -ne 0 ] || [ -z "$dir" ] || [ ! -f "$dir/transcript.jsonl" ]; then
    echo "[$label] FAILED (rc=$rc, dir='$dir') — see $log"; return 1
  fi
  if grep -qE "Something went wrong|Unknown error occurred|Idle timeout detected|RESOURCE_EXHAUSTED" "$log"; then
    echo "[$label] SUSPECT (pipeline error/429 in log) — $dir NOT added; see $log"; return 1
  fi
  turns=$(wc -l < "$dir/transcript.jsonl")
  [ "$turns" -ne 30 ] && echo "[$label] SHORT: $turns/30 turns in $dir (kept)"
  echo "$dir" >> "$list"
  echo "[$label] ok: $dir ($turns turns), judging"
  if uv run multi-turn-eval judge "$dir" > "$dir/judge.log" 2>&1; then
    echo "[$label] judged"
  else
    echo "[$label] JUDGE FAILED — see $dir/judge.log"
  fi
}

echo "=== Lilac thinking-OFF: 10 runs ==="
for i in $(seq 1 10); do
  echo "[off $i] starting"
  env -u MTE_LILAC_THINKING uv run multi-turn-eval run aiwf_medium_context \
    --model lilac/gemma-4-31b-it --service lilac > "$LOG_DIR/off_$i.log" 2>&1
  finish_run "off $i" "$?" "$LOG_DIR/off_$i.log" "$OFF_LIST"
done

echo "=== Lilac thinking-ON: 10 runs ==="
for i in $(seq 1 10); do
  echo "[on $i] starting"
  MTE_LILAC_THINKING=1 uv run multi-turn-eval run aiwf_medium_context \
    --model lilac/gemma-4-31b-it --service lilac > "$LOG_DIR/on_$i.log" 2>&1
  finish_run "on $i" "$?" "$LOG_DIR/on_$i.log" "$ON_LIST"
done

echo "Sweep complete: off=$(wc -l < "$OFF_LIST" 2>/dev/null || echo 0) on=$(wc -l < "$ON_LIST" 2>/dev/null || echo 0)"
