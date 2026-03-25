#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <2.5|3.1> [extra args...]"
    echo "  2.5  → gemini-2.5-flash-native-audio-preview-12-2025"
    echo "  3.1  → gemini-3.1-flash-audio-eap"
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    2.5) model="gemini-2.5-flash-native-audio-preview-12-2025" ;;
    3.1) model="gemini-3.1-flash-audio-eap" ;;
    *)   usage ;;
esac
shift

benchmark="aiwf_medium_context"
runs_dir="runs/${benchmark}"

# Run the benchmark
uv run multi-turn-eval run "$benchmark" \
    --model "$model" \
    --service gemini-live \
    --pipeline realtime \
    "$@"

# Find the most recent run directory matching this model
safe_model="${model//\//_}"
safe_model="${safe_model//:/_}"
run_dir=$(ls -dt "${runs_dir}/"*"${safe_model}"* 2>/dev/null | head -1)

if [[ -z "$run_dir" ]]; then
    echo "ERROR: Could not find run directory for ${model} in ${runs_dir}"
    exit 1
fi

echo ""
echo "=== Metrics ${run_dir} ==="
echo ""
uv run python scripts/analyze_turn_metrics.py "$run_dir" -v

echo ""
echo "=== Judging ${run_dir} ==="
echo ""
uv run multi-turn-eval judge "$run_dir"
