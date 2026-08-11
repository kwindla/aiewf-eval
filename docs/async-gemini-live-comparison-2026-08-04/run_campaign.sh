#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
campaign_dir="$repo_dir/docs/async-gemini-live-comparison-2026-08-04"
log_dir="$campaign_dir/logs"
runner="$repo_dir/.venv/bin/multi-turn-eval"
async_model="${MTE_ASYNC_GEMINI_LIVE_MODEL:?Set MTE_ASYNC_GEMINI_LIVE_MODEL}"

mkdir -p "$log_dir"

run_lane() {
    local lane="$1"
    local model="$2"
    local service="$3"
    local target="$4"
    local reasoning_effort="${5:-}"
    local behavior="${6:-}"
    local -a behavior_args=()
    if [[ "$behavior" == "async-gemini-live" ]]; then
        behavior_args=(
            --gemini-3-protocol
            --gemini-require-interaction-status
            --gemini-explicit-audio-activity
            --no-turn-replay
        )
    fi
    local valid=0
    local attempt=0
    local max_attempts=$((target + 8))
    local allowlist="$campaign_dir/${lane}-valid-runs.txt"
    local attempts="$campaign_dir/${lane}-attempts.tsv"

    : > "$allowlist"
    printf 'attempt\texit_code\tturns\trun_dir\tlog\n' > "$attempts"

    while (( valid < target && attempt < max_attempts )); do
        attempt=$((attempt + 1))
        local log="$log_dir/${lane}-attempt-$(printf '%02d' "$attempt").log"
        local exit_code=0

        if [[ -n "$reasoning_effort" ]]; then
            MTE_OPENAI_REALTIME_REASONING_EFFORT="$reasoning_effort" \
                "$runner" run aiwf_medium_context --model "$model" --service "$service" \
                "${behavior_args[@]}" \
                > "$log" 2>&1 || exit_code=$?
        else
            "$runner" run aiwf_medium_context --model "$model" --service "$service" \
                "${behavior_args[@]}" \
                > "$log" 2>&1 || exit_code=$?
        fi

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

run_lane clever-a "$async_model" gemini-live 7 "" async-gemini-live &
pid_clever_a=$!
run_lane clever-b "$async_model" gemini-live 7 "" async-gemini-live &
pid_clever_b=$!
run_lane openai-a gpt-realtime-2.1 openai-realtime 5 low &
pid_openai_a=$!
run_lane openai-b gpt-realtime-2.1 openai-realtime 5 low &
pid_openai_b=$!

status=0
wait "$pid_clever_a" || status=1
wait "$pid_clever_b" || status=1
wait "$pid_openai_a" || status=1
wait "$pid_openai_b" || status=1
exit "$status"
