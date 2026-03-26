#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <2.5|3.1> [-n NUM_RUNS] [-t THINKING] [extra args...]"
    echo "  2.5  → gemini-2.5-flash-native-audio-preview-12-2025"
    echo "  3.1  → gemini-3.1-flash-audio-eap"
    echo ""
    echo "Options:"
    echo "  -n NUM_RUNS   Number of runs (default: 1)"
    echo "  -t THINKING   Thinking level: disabled, minimal, low, medium, high, default"
    exit 1
}

[[ $# -lt 1 ]] && usage

case "$1" in
    2.5) model="gemini-2.5-flash-native-audio-preview-12-2025" ;;
    3.1) model="gemini-3.1-flash-live-preview" ;;
    *)   usage ;;
esac
shift

num_runs=1
thinking=""

while getopts "n:t:" opt; do
    case "$opt" in
        n) num_runs="$OPTARG" ;;
        t) thinking="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

thinking_args=()
if [[ -n "$thinking" ]]; then
    thinking_args=(--thinking "$thinking")
    echo "Thinking level: $thinking"
fi

benchmark="aiwf_medium_context"
# benchmark="aiwf_long_context"
runs_dir="runs/${benchmark}"

# If multiple runs, create a group subdirectory
group_dir=""
if (( num_runs > 1 )); then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    group_dir="${runs_dir}/${timestamp}_${model//\//_}_x${num_runs}"
    mkdir -p "$group_dir"
    echo "Group directory: $group_dir"
fi

for (( i = 1; i <= num_runs; i++ )); do
    if (( num_runs > 1 )); then
        echo ""
        echo "=== Run $i of $num_runs ==="
    fi

    # Run the benchmark
    uv run multi-turn-eval run "$benchmark" \
        --model "$model" \
        --service gemini-live \
        --pipeline realtime \
        "${thinking_args[@]+"${thinking_args[@]}"}" \
        "$@"

    # Find the most recent run directory matching this model
    safe_model="${model//\//_}"
    safe_model="${safe_model//:/_}"
    run_dir=$(ls -dt "${runs_dir}/"*"${safe_model}"* 2>/dev/null | head -1)

    if [[ -z "$run_dir" ]]; then
        echo "ERROR: Could not find run directory for ${model} in ${runs_dir}"
        exit 1
    fi

    # Move into group directory if doing multiple runs
    if [[ -n "$group_dir" ]]; then
        mv "$run_dir" "$group_dir/"
        run_dir="$group_dir/$(basename "$run_dir")"
    fi

    echo ""
    echo "=== Metrics ${run_dir} ==="
    echo ""
    uv run python scripts/analyze_turn_metrics.py "$run_dir" -v 2>&1 | tee "$run_dir/metrics.txt"
    uv run python scripts/analyze_turn_metrics.py "$run_dir" --json > "$run_dir/turn_metrics.json"

    echo ""
    echo "=== Judging ${run_dir} ==="
    echo ""
    uv run multi-turn-eval judge "$run_dir"
done

# Generate comparison report for multi-run groups
if [[ -n "$group_dir" ]]; then
    echo ""
    echo "=== Generating comparison report ==="
    echo ""
    claude -p "$(cat <<'EOF'
Read all of the turn_metrics.json and judge results in each subdirectory of the
directory I specify below.  Then create a summary comparison table with these columns:
- Model
- Tool Use (X/30)
- Instruction (X/30)
- KB Ground (X/30)
- Turn Ok (X/30)
- Pass Rate
- Non-Tool V2V Median
- Non-Tool V2V Max
- Tool V2V Mean
- Silence Pad Mean

Separate metrics for tool-call turns vs non-tool-call turns in the analysis.

Directory:
EOF
    ) ${group_dir}" > "$group_dir/REPORT.md"
    echo "Report saved to $group_dir/REPORT.md"
fi
