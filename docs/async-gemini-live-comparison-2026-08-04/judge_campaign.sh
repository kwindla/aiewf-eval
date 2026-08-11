#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
campaign_dir="$repo_dir/docs/async-gemini-live-comparison-2026-08-04"
log_dir="$campaign_dir/logs"
judge="$repo_dir/.venv/bin/multi-turn-eval"

judge_lane() {
    local lane="$1"
    local allowlist="$2"
    local status_file="$campaign_dir/${lane}-judging.tsv"

    printf 'run_dir\tattempt\texit_code\tjudged_turns\tlog\n' > "$status_file"
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        local attempt=0
        local judged_turns=0

        if [[ -f "$repo_dir/$run_dir/claude_summary.json" ]]; then
            judged_turns="$(
                "$repo_dir/.venv/bin/python" -c \
                    'import json,sys; print(json.load(open(sys.argv[1])).get("turns_scored", 0))' \
                    "$repo_dir/$run_dir/claude_summary.json"
            )"
        fi

        if (( judged_turns == 30 )); then
            printf '%s\t0\t0\t30\t%s\n' "$run_dir" 'already-judged' >> "$status_file"
            continue
        fi

        while (( judged_turns != 30 && attempt < 3 )); do
            attempt=$((attempt + 1))
            local run_name="${run_dir##*/}"
            local log="$log_dir/judge-${lane}-${run_name}-attempt-${attempt}.log"
            local exit_code=0

            "$judge" judge "$run_dir" > "$log" 2>&1 || exit_code=$?
            judged_turns=0
            if [[ -f "$repo_dir/$run_dir/claude_summary.json" ]]; then
                judged_turns="$(
                    "$repo_dir/.venv/bin/python" -c \
                        'import json,sys; print(json.load(open(sys.argv[1])).get("turns_scored", 0))' \
                        "$repo_dir/$run_dir/claude_summary.json"
                )"
            fi
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$run_dir" "$attempt" "$exit_code" "$judged_turns" "${log#$repo_dir/}" \
                >> "$status_file"
        done

        if (( judged_turns != 30 )); then
            printf '%s: failed to judge %s after %d attempts\n' \
                "$lane" "$run_dir" "$attempt" >&2
            return 1
        fi
    done < "$allowlist"
}

cd "$repo_dir"

status=0
judge_lane clever-a "$campaign_dir/clever-a-valid-runs.txt" &
pid_clever_a=$!
judge_lane clever-b "$campaign_dir/clever-b-valid-runs.txt" &
pid_clever_b=$!
judge_lane clever-c "$campaign_dir/clever-c-valid-runs.txt" &
pid_clever_c=$!
judge_lane clever-d "$campaign_dir/clever-d-valid-runs.txt" &
pid_clever_d=$!
judge_lane openai-a "$campaign_dir/openai-a-valid-runs.txt" &
pid_openai_a=$!
judge_lane openai-b "$campaign_dir/openai-b-valid-runs.txt" &
pid_openai_b=$!
judge_lane openai-topup-a "$campaign_dir/openai-topup-a-valid-runs.txt" &
pid_openai_topup_a=$!
judge_lane openai-topup-b "$campaign_dir/openai-topup-b-valid-runs.txt" &
pid_openai_topup_b=$!
judge_lane openai-topup-c "$campaign_dir/openai-topup-c-valid-runs.txt" &
pid_openai_topup_c=$!
judge_lane openai-topup-d "$campaign_dir/openai-topup-d-valid-runs.txt" &
pid_openai_topup_d=$!

wait "$pid_clever_a" || status=1
wait "$pid_clever_b" || status=1
wait "$pid_clever_c" || status=1
wait "$pid_clever_d" || status=1
wait "$pid_openai_a" || status=1
wait "$pid_openai_b" || status=1
wait "$pid_openai_topup_a" || status=1
wait "$pid_openai_topup_b" || status=1
wait "$pid_openai_topup_c" || status=1
wait "$pid_openai_topup_d" || status=1
exit "$status"
