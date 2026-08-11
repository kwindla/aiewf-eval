#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
campaign_dir="$repo_dir/docs/async-gemini-live-comparison-2026-08-04"
log_dir="$campaign_dir/logs"
runner="$repo_dir/.venv/bin/multi-turn-eval"

mkdir -p "$log_dir"

run_lane() {
    local lane="$1"
    local target=5
    local valid=0
    local attempt=0
    local max_attempts=8
    local allowlist="$campaign_dir/${lane}-valid-runs.txt"
    local attempts="$campaign_dir/${lane}-attempts.tsv"

    : > "$allowlist"
    printf 'attempt\texit_code\tturns\trun_dir\tlog\n' > "$attempts"

    while (( valid < target && attempt < max_attempts )); do
        attempt=$((attempt + 1))
        local log="$log_dir/${lane}-attempt-$(printf '%02d' "$attempt").log"
        local exit_code=0
        MTE_OPENAI_REALTIME_REASONING_EFFORT=low \
            "$runner" run aiwf_medium_context \
                --model gpt-realtime-2.1 --service openai-realtime \
                > "$log" 2>&1 || exit_code=$?

        local run_dir
        run_dir="$(sed -n 's/^Output directory: //p' "$log" | tail -1)"
        local turns=0
        if [[ -n "$run_dir" && -f "$repo_dir/$run_dir/transcript.jsonl" ]]; then
            turns="$(wc -l < "$repo_dir/$run_dir/transcript.jsonl")"
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$attempt" "$exit_code" "$turns" "$run_dir" "${log#$repo_dir/}" >> "$attempts"

        if [[ "$exit_code" -eq 0 && "$turns" -eq 30 ]]; then
            printf '%s\n' "$run_dir" >> "$allowlist"
            valid=$((valid + 1))
        fi
    done

    if (( valid != target )); then
        printf '%s: obtained %d/%d valid runs after %d attempts\n' \
            "$lane" "$valid" "$target" "$attempt" >&2
        return 1
    fi
}

cd "$repo_dir"
status=0
pids=()
for suffix in a b c d; do
    run_lane "openai-topup-$suffix" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
