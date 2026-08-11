#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
campaign_dir="$repo_dir/docs/async-gemini-live-comparison-2026-08-04"
log_dir="$campaign_dir/logs"
runner="$repo_dir/.venv/bin/multi-turn-eval"
async_model="${MTE_ASYNC_GEMINI_LIVE_MODEL:?Set MTE_ASYNC_GEMINI_LIVE_MODEL}"
lane="${1:-clever-c}"
if [[ "$lane" != "clever-c" && "$lane" != "clever-d" ]]; then
    printf 'unsupported rescue lane: %s\n' "$lane" >&2
    exit 2
fi
allowlist="$campaign_dir/${lane}-valid-runs.txt"
attempts="$campaign_dir/${lane}-attempts.tsv"
target_total=14
max_attempts=30

selected_count() {
    {
        for path in "$campaign_dir"/clever-?-valid-runs.txt; do
            [[ -f "$path" ]] && sed '/^[[:space:]]*$/d' "$path"
        done
    } | sort -u | wc -l
}

mkdir -p "$log_dir"
: > "$allowlist"
printf 'attempt\texit_code\tturns\trun_dir\tlog\n' > "$attempts"

cd "$repo_dir"
attempt=0
while (( attempt < max_attempts )); do
    if (( $(selected_count) >= target_total )); then
        exit 0
    fi

    attempt=$((attempt + 1))
    log="$log_dir/${lane}-attempt-$(printf '%02d' "$attempt").log"
    exit_code=0
    "$runner" run aiwf_medium_context --model "$async_model" --service gemini-live \
        --gemini-3-protocol --gemini-require-interaction-status \
        --gemini-explicit-audio-activity --no-turn-replay \
        > "$log" 2>&1 || exit_code=$?

    run_dir="$(sed -n 's/^Output directory: //p' "$log" | tail -1)"
    turns=0
    if [[ -n "$run_dir" && -f "$repo_dir/$run_dir/transcript.jsonl" ]]; then
        turns="$(wc -l < "$repo_dir/$run_dir/transcript.jsonl")"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$attempt" "$exit_code" "$turns" "$run_dir" "${log#$repo_dir/}" >> "$attempts"

    if [[ "$exit_code" -eq 0 && "$turns" -eq 30 ]] \
        && (( $(selected_count) < target_total )); then
        printf '%s\n' "$run_dir" >> "$allowlist"
    fi
done

printf '%s: campaign still below %d valid additions after %d attempts\n' \
    "$lane" "$target_total" "$max_attempts" >&2
exit 1
