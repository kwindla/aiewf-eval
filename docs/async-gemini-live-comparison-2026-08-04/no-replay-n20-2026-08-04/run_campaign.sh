#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
campaign_dir="$repo_dir/docs/async-gemini-live-comparison-2026-08-04/no-replay-n20-2026-08-04"
log_dir="$campaign_dir/logs"
overlay="/home/khkramer/.cache/aiewf-eval/interaction-status-overlay-private-214"
launcher="$repo_dir/docs/async-gemini-live-comparison-2026-08-04/probe_interaction_status.py"
python="$repo_dir/.venv/bin/python"
attempts="$campaign_dir/attempts.tsv"
run_manifest="$campaign_dir/run-dirs.txt"
completed_manifest="$campaign_dir/complete-runs.txt"
failure_marker="$campaign_dir/UNEXPECTED_ERROR.txt"
target=20
async_model="${MTE_ASYNC_GEMINI_LIVE_MODEL:?Set MTE_ASYNC_GEMINI_LIVE_MODEL}"

mkdir -p "$log_dir"
printf 'attempt\texit_code\tclassification\tturns\tfailure_turn\traw_events\tin_progress\tterminal\taudio_bytes\trun_dir\tlog\n' > "$attempts"
: > "$run_manifest"
: > "$completed_manifest"
: > "$failure_marker"

json_string() {
    local key="$1"
    local file="$2"
    sed -n "s/.*\"$key\": \"\([^\"]*\)\".*/\1/p" "$file" | head -1
}

json_number() {
    local key="$1"
    local file="$2"
    sed -n "s/.*\"$key\": \([0-9][0-9.]*\).*/\1/p" "$file" | head -1
}

stop_unexpected() {
    local message="$1"
    printf '%s\n' "$message" | tee -a "$failure_marker" >&2
    exit 2
}

cd "$repo_dir"
for attempt in $(seq 1 "$target"); do
    attempt_label="$(printf '%02d' "$attempt")"
    log="$log_dir/attempt-$attempt_label.log"
    exit_code=0

    PYTHONPATH="$overlay" "$python" "$launcher" run aiwf_medium_context \
        --model "$async_model" --service gemini-live --thinking minimal \
        --gemini-3-protocol --gemini-require-interaction-status \
        --gemini-explicit-audio-activity --no-turn-replay \
        > "$log" 2>&1 || exit_code=$?

    run_dir="$(sed -n 's/^Output directory: //p' "$log" | tail -1)"
    [[ -n "$run_dir" ]] || stop_unexpected "attempt $attempt: no run directory in launcher output"
    runtime="$repo_dir/$run_dir/runtime.json"
    transcript="$repo_dir/$run_dir/transcript.jsonl"
    audio="$repo_dir/$run_dir/conversation.wav"
    run_log="$repo_dir/$run_dir/run.log"
    [[ -f "$runtime" ]] || stop_unexpected "attempt $attempt: missing runtime.json ($run_dir)"
    [[ -f "$transcript" ]] || stop_unexpected "attempt $attempt: missing transcript.jsonl ($run_dir)"
    [[ -s "$audio" ]] || stop_unexpected "attempt $attempt: missing/empty conversation.wav ($run_dir)"
    [[ -f "$run_log" ]] || stop_unexpected "attempt $attempt: missing run.log ($run_dir)"

    status="$(json_string status "$runtime")"
    reason="$(json_string reason "$runtime")"
    failure_turn="$(json_number turn "$runtime")"
    turns="$(wc -l < "$transcript")"
    raw_events="$(rg -c '\[GEMINI_RAW_EVENT\]' "$run_log" || true)"
    in_progress="$(rg -c 'interaction_status=IN_PROGRESS' "$run_log" || true)"
    terminal="$(rg -c 'interaction_status=REQUIRES_ACTION' "$run_log" || true)"
    audio_bytes="$(stat -c '%s' "$audio")"
    classification=""

    [[ "$raw_events" -gt 0 ]] || stop_unexpected "attempt $attempt: raw event trace is empty ($run_dir)"
    if rg -q 'RuntimeWarning|coroutine .* was never awaited|Task was destroyed|GEMINI_RAW_DECODE_ERROR' "$log" "$run_log"; then
        stop_unexpected "attempt $attempt: runtime/decode warning detected ($run_dir; $log)"
    fi

    if [[ "$status" == "failed" ]]; then
        case "$reason" in
            no_audio_timeout|empty_audio_response|gemini_connection_reconnect)
                classification="$reason"
                [[ "$exit_code" -ne 0 ]] || stop_unexpected "attempt $attempt: failed runtime returned exit 0 ($run_dir)"
                if ! rg -q '"replayed": false' "$runtime"; then
                    stop_unexpected "attempt $attempt: failed run was replayed ($run_dir)"
                fi
                if rg -q 'Re-queuing audio|Successfully re-queued audio' "$run_log"; then
                    stop_unexpected "attempt $attempt: replay log found despite no-replay policy ($run_dir)"
                fi
                ;;
            *)
                stop_unexpected "attempt $attempt: unexpected failure reason '$reason' ($run_dir)"
                ;;
        esac
    elif [[ "$status" == "completed" && "$exit_code" -eq 0 && "$turns" -eq 30 ]]; then
        classification="complete"
        printf '%s\n' "$run_dir" >> "$completed_manifest"
    elif [[ "$status" == "completed" && "$exit_code" -eq 0 && "$turns" -lt 30 ]]; then
        classification="model_ended_early"
    else
        stop_unexpected "attempt $attempt: inconsistent status=$status exit=$exit_code turns=$turns ($run_dir)"
    fi

    printf '%s\n' "$run_dir" >> "$run_manifest"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$attempt" "$exit_code" "$classification" "$turns" "${failure_turn:-}" \
        "$raw_events" "${in_progress:-0}" "${terminal:-0}" "$audio_bytes" \
        "$run_dir" "${log#$repo_dir/}" >> "$attempts"
    printf 'attempt %d/%d: %s, turns=%s, raw_events=%s, run=%s\n' \
        "$attempt" "$target" "$classification" "$turns" "$raw_events" "$run_dir"
done

: > "$failure_marker"
