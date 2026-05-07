#!/bin/bash
# Judge all dolphin runs that don't yet have a claude_summary.json.
# Runs N at a time in parallel via xargs.
set -u
cd /home/khkramer/src/aiewf-eval

PARALLEL="${PARALLEL:-4}"

mapfile -t TODO < <(
  for d in runs/aiwf_medium_context/*dolphin*; do
    [ -f "$d/claude_summary.json" ] && continue
    [ ! -f "$d/transcript.jsonl" ] && continue
    lines=$(wc -l < "$d/transcript.jsonl")
    [ "$lines" -lt 30 ] && continue
    echo "$d"
  done
)

echo "judging ${#TODO[@]} runs (parallel=$PARALLEL)..."
printf '%s\n' "${TODO[@]}" | xargs -n1 -P "$PARALLEL" -I{} bash -c \
  'd="$1"; echo "START $d"; uv run multi-turn-eval judge "$d" 2>&1 | tail -7 | sed "s|^|  $(basename $d) |"; echo "DONE $d"' _ {}

echo "judge sweep complete"
