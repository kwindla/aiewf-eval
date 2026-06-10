#!/bin/bash
# claude-fable-5 sweep: 10 runs at MTE_ANTHROPIC_EFFORT=low plus 10
# default-config runs (effort unset = adaptive thinking at the model's default
# effort, high — claude-fable-5 cannot run without thinking), one of each per
# batch, judged in parallel as each batch completes. Run dirs are appended to
# ten-run allowlists for later aggregation.
#
# Launch detached so it survives the launching shell (total ~2.5-3 hours):
#   setsid nohup scripts/run_fable5_sweep.sh > /tmp/fable5_sweep.log 2>&1 &
#
# Aggregate afterwards:
#   uv run python scripts/benchmark_summary.py $(cat docs/ten-run-allowlists/claude-fable-5-low-2026-06-09.txt)
#   uv run python scripts/benchmark_summary.py $(cat docs/ten-run-allowlists/claude-fable-5-default-2026-06-09.txt)
set -u
cd /home/khkramer/src/aiewf-eval

ALLOW_DIR=docs/ten-run-allowlists
LOW_LIST="${ALLOW_DIR}/claude-fable-5-low-2026-06-09.txt"
BASE_LIST="${ALLOW_DIR}/claude-fable-5-default-2026-06-09.txt"
LOG_DIR=/tmp/fable5-sweep-logs
mkdir -p "$LOG_DIR" "$ALLOW_DIR"

# Pin every MTE_ANTHROPIC_* knob the harness reads so a leftover export from
# an earlier experiment (e.g. the voice-optimized probe's THINKING_DISPLAY /
# VOICE_STEERING) can't silently change the configuration being measured.
PIN_ENV=(env -u MTE_ANTHROPIC_THINKING_DISPLAY -u MTE_ANTHROPIC_VOICE_STEERING
         -u MTE_ANTHROPIC_MAX_TOKENS)

run_dir_from_log() {
  grep -oE 'Transcript: runs/[^ ]+/transcript\.jsonl' "$1" | tail -1 \
    | sed -e 's/^Transcript: //' -e 's|/transcript\.jsonl$||'
}

# finish_run <batch> <label> <rc> <log> <allowlist>
# Validates a completed run, appends it to the allowlist, and judges it.
finish_run() {
  local batch=$1 label=$2 rc=$3 log=$4 list=$5
  local dir turns
  dir=$(run_dir_from_log "$log")
  if [ "$rc" -ne 0 ] || [ -z "$dir" ] || [ ! -f "$dir/transcript.jsonl" ]; then
    echo "[batch $batch] $label FAILED (rc=$rc, dir='$dir') — see $log"
    return 1
  fi
  # Pipeline errors don't set a nonzero exit code; detect them from the log.
  if grep -qE "Something went wrong|Unknown error occurred|Idle timeout detected" "$log"; then
    echo "[batch $batch] $label SUSPECT (pipeline error in log) — $dir NOT added; see $log"
    return 1
  fi
  turns=$(wc -l < "$dir/transcript.jsonl")
  if [ "$turns" -ne 30 ]; then
    # Early end_session is legitimate model behavior (denominators below 300
    # appear in the README for other models), so keep the run but flag it.
    echo "[batch $batch] $label SHORT: $turns/30 turns in $dir (kept)"
  fi
  echo "$dir" >> "$list"
  echo "[batch $batch] $label ok: $dir ($turns turns), judging"
  if uv run multi-turn-eval judge "$dir" > "$dir/judge.log" 2>&1; then
    echo "[batch $batch] $label judged"
  else
    echo "[batch $batch] $label JUDGE FAILED — see $dir/judge.log"
  fi
}

for i in $(seq 1 10); do
  echo "[batch $i] starting low + default runs"

  "${PIN_ENV[@]}" MTE_ANTHROPIC_EFFORT=low uv run multi-turn-eval run aiwf_medium_context \
    --model claude-fable-5 --service anthropic \
    > "$LOG_DIR/low_$i.log" 2>&1 &
  low_pid=$!

  "${PIN_ENV[@]}" -u MTE_ANTHROPIC_EFFORT uv run multi-turn-eval run aiwf_medium_context \
    --model claude-fable-5 --service anthropic \
    > "$LOG_DIR/base_$i.log" 2>&1 &
  base_pid=$!

  wait "$low_pid"; low_rc=$?
  wait "$base_pid"; base_rc=$?

  finish_run "$i" low "$low_rc" "$LOG_DIR/low_$i.log" "$LOW_LIST" &
  finish_run "$i" default "$base_rc" "$LOG_DIR/base_$i.log" "$BASE_LIST" &
  wait
done

echo "Sweep complete: $(wc -l < "$LOW_LIST" 2>/dev/null || echo 0) low runs, $(wc -l < "$BASE_LIST" 2>/dev/null || echo 0) default runs"
