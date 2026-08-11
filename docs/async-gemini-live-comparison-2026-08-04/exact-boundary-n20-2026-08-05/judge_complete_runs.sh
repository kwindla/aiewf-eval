#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
campaign_dir="${CAMPAIGN_DIR:-$repo_dir/docs/async-gemini-live-comparison-2026-08-04/exact-boundary-n20-2026-08-05}"
manifest="$campaign_dir/complete-runs.txt"
status_file="$campaign_dir/judging.tsv"
log_dir="$campaign_dir/logs"
judge="$repo_dir/.venv/bin/multi-turn-eval"
python="$repo_dir/.venv/bin/python"

mkdir -p "$log_dir"
if [[ ! -f "$status_file" ]]; then
    printf 'run_dir\tattempt\texit_code\tjudged_turns\tlog\n' > "$status_file"
fi

cd "$repo_dir"
while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] || continue
    judged_turns=0
    summary="$repo_dir/$run_dir/claude_summary.json"
    if [[ -f "$summary" ]]; then
        judged_turns="$($python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("turns_scored", 0))' "$summary")"
    fi
    (( judged_turns == 30 )) && continue

    for attempt in 1 2 3; do
        run_name="${run_dir##*/}"
        log="$log_dir/judge-${run_name}-attempt-${attempt}.log"
        exit_code=0
        "$judge" judge "$run_dir" > "$log" 2>&1 || exit_code=$?
        judged_turns=0
        if [[ -f "$summary" ]]; then
            judged_turns="$($python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("turns_scored", 0))' "$summary")"
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$run_dir" "$attempt" "$exit_code" "$judged_turns" "${log#$repo_dir/}" \
            >> "$status_file"
        (( judged_turns == 30 )) && break
    done
    if (( judged_turns != 30 )); then
        printf 'failed to judge %s after 3 attempts\n' "$run_dir" >&2
        exit 1
    fi
done < "$manifest"
