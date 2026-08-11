#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
campaign_dir="${CAMPAIGN_DIR:-$repo_dir/docs/async-gemini-live-comparison-2026-08-04/exact-boundary-n20-2026-08-05}"
log_dir="$campaign_dir/logs"
overlay="/home/khkramer/.cache/aiewf-eval/interaction-status-overlay-private-214"
launcher="$repo_dir/docs/async-gemini-live-comparison-2026-08-04/probe_interaction_status.py"
event_flow_validator="$repo_dir/docs/async-gemini-live-comparison-2026-08-04/root-cause-probes/validate_exact_run.py"
python="$repo_dir/.venv/bin/python"
attempts="$campaign_dir/attempts.tsv"
run_manifest="$campaign_dir/run-dirs.txt"
completed_manifest="$campaign_dir/complete-runs.txt"
failure_marker="$campaign_dir/UNEXPECTED_ERROR.txt"
progress="$campaign_dir/PROGRESS.txt"
target="${TARGET_RUNS:-20}"
async_model="${MTE_ASYNC_GEMINI_LIVE_MODEL:?Set MTE_ASYNC_GEMINI_LIVE_MODEL}"

mkdir -p "$log_dir"
if [[ ! -f "$attempts" ]]; then
    printf 'attempt\texit_code\tclassification\tturns\tfailure_turn\traw_events\tin_progress\tterminal\tactivity_starts\tactivity_ends\tmodel_audio_during_input\tterminal_during_input\tinterruption_during_input\tmissing_ttfb\taudio_bytes\trun_dir\tlog\n' > "$attempts"
    : > "$run_manifest"
    : > "$completed_manifest"
    : > "$failure_marker"
fi

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
    printf '%s\n' "$message" | tee -a "$failure_marker" "$progress" >&2
    exit 2
}

completed_attempts=$(($(wc -l < "$attempts") - 1))
if ((completed_attempts >= target)); then
    printf 'campaign already complete: %d/%d attempts\n' "$completed_attempts" "$target" | tee "$progress"
    exit 0
fi

cd "$repo_dir"
for attempt in $(seq $((completed_attempts + 1)) "$target"); do
    attempt_label="$(printf '%02d' "$attempt")"
    log="$log_dir/attempt-$attempt_label.log"
    exit_code=0

    printf 'attempt %d/%d running\n' "$attempt" "$target" | tee "$progress"
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
    in_progress="$(rg -c 'interaction_status.:.IN_PROGRESS' "$run_log" || true)"
    terminal="$(rg -c 'interaction_status.:.REQUIRES_ACTION' "$run_log" || true)"
    activity_starts="$(rg -c 'activity_start.:true' "$run_log" || true)"
    activity_ends="$(rg -c 'activity_end.:true' "$run_log" || true)"
    audio_bytes="$(stat -c '%s' "$audio")"
    event_flow="$log_dir/attempt-$attempt_label-event-flow.json"
    classification=""

    raw_events="${raw_events:-0}"
    in_progress="${in_progress:-0}"
    terminal="${terminal:-0}"
    activity_starts="${activity_starts:-0}"
    activity_ends="${activity_ends:-0}"

    [[ "$raw_events" -gt 0 ]] || stop_unexpected "attempt $attempt: raw event trace is empty ($run_dir)"
    if rg -q 'RuntimeWarning|coroutine .* was never awaited|Task was destroyed|GEMINI_RAW_DECODE_ERROR|GEMINI_RAW_SEND_ERROR' "$log" "$run_log"; then
        stop_unexpected "attempt $attempt: runtime/decode/send warning detected ($run_dir; $log)"
    fi
    [[ "$activity_starts" -eq "$activity_ends" ]] || \
        stop_unexpected "attempt $attempt: unbalanced activity boundaries starts=$activity_starts ends=$activity_ends ($run_dir)"
    "$python" "$event_flow_validator" "$repo_dir/$run_dir" --output "$event_flow" || \
        stop_unexpected "attempt $attempt: exact-boundary event-flow violation ($run_dir; ${event_flow#$repo_dir/})"
    model_audio_during_input="$(jq '.model_audio_during_input | length' "$event_flow")"
    terminal_during_input="$(jq '.terminal_during_input | length' "$event_flow")"
    interruption_during_input="$(jq '.interruption_during_input | length' "$event_flow")"
    missing_ttfb="$(jq '.nonempty_transcript_missing_ttfb_turns | length' "$event_flow")"

    if [[ "$status" == "failed" ]]; then
        case "$reason" in
            no_audio_timeout|empty_audio_response|gemini_connection_reconnect)
                classification="$reason"
                [[ "$exit_code" -ne 0 ]] || stop_unexpected "attempt $attempt: failed runtime returned exit 0 ($run_dir)"
                rg -q '"replayed": false' "$runtime" || \
                    stop_unexpected "attempt $attempt: failed run was replayed ($run_dir)"
                if rg -q 'Re-queuing audio|Successfully re-queued audio' "$run_log"; then
                    stop_unexpected "attempt $attempt: replay log found despite no-replay policy ($run_dir)"
                fi
                [[ "$failure_turn" == "$turns" ]] || \
                    stop_unexpected "attempt $attempt: failure turn=$failure_turn does not match $turns recorded turns ($run_dir)"
                [[ "$activity_starts" -eq $((turns + 1)) ]] || \
                    stop_unexpected "attempt $attempt: failed-run activity=$activity_starts, expected $((turns + 1)) ($run_dir)"
                ;;
            *)
                stop_unexpected "attempt $attempt: unexpected failure reason '$reason' ($run_dir)"
                ;;
        esac
    elif [[ "$status" == "completed" && "$exit_code" -eq 0 && "$turns" -eq 30 ]]; then
        classification="complete"
        [[ "$activity_starts" -eq "$turns" ]] || \
            stop_unexpected "attempt $attempt: completed-run activity=$activity_starts turns=$turns ($run_dir)"
        printf '%s\n' "$run_dir" >> "$completed_manifest"
    elif [[ "$status" == "completed" && "$exit_code" -eq 0 && "$turns" -lt 30 ]]; then
        classification="model_ended_early"
        [[ "$activity_starts" -eq "$turns" ]] || \
            stop_unexpected "attempt $attempt: early-ended activity=$activity_starts turns=$turns ($run_dir)"
    else
        stop_unexpected "attempt $attempt: inconsistent status=$status exit=$exit_code turns=$turns ($run_dir)"
    fi

    printf '%s\n' "$run_dir" >> "$run_manifest"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$attempt" "$exit_code" "$classification" "$turns" "${failure_turn:-}" \
        "$raw_events" "${in_progress:-0}" "${terminal:-0}" \
        "$activity_starts" "$activity_ends" \
        "$model_audio_during_input" "$terminal_during_input" \
        "$interruption_during_input" "$missing_ttfb" "$audio_bytes" \
        "$run_dir" "${log#$repo_dir/}" >> "$attempts"
    printf 'attempt %d/%d: %s, turns=%s, in_progress=%s, run=%s\n' \
        "$attempt" "$target" "$classification" "$turns" "${in_progress:-0}" "$run_dir" | tee "$progress"
done

: > "$failure_marker"
printf 'campaign complete: %d/%d attempts\n' "$target" "$target" | tee "$progress"
