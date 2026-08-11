#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/../../.." && pwd)"
campaign_dir="${CAMPAIGN_DIR:-$repo_dir/docs/async-gemini-live-comparison-2026-08-04/gemini31-topup-n5-2026-08-05}"
log_dir="$campaign_dir/logs"
attempts="$campaign_dir/attempts.tsv"
run_manifest="$campaign_dir/run-dirs.txt"
complete_manifest="$campaign_dir/complete-runs.txt"
progress="$campaign_dir/PROGRESS.txt"
failure_marker="$campaign_dir/UNEXPECTED_ERROR.txt"
runner="$repo_dir/.venv/bin/multi-turn-eval"
target_valid="${TARGET_VALID:-5}"
target_attempts="${TARGET_ATTEMPTS:-}"
max_attempts="${MAX_ATTEMPTS:-10}"

mkdir -p "$log_dir"
if [[ ! -f "$attempts" ]]; then
    printf 'attempt\texit_code\tclassification\tturns\tretry_events\taudio_bytes\trun_dir\tlog\n' > "$attempts"
    : > "$run_manifest"
    : > "$complete_manifest"
    : > "$failure_marker"
fi

stop_unexpected() {
    local message="$1"
    printf '%s\n' "$message" | tee -a "$failure_marker" "$progress" >&2
    exit 2
}

attempt=$(($(wc -l < "$attempts") - 1))
valid=$(wc -l < "$complete_manifest")
if [[ -n "$target_attempts" ]] && ((attempt >= target_attempts)); then
    printf 'campaign already complete: %d/%d attempts\n' "$attempt" "$target_attempts" | tee "$progress"
    exit 0
elif [[ -z "$target_attempts" ]] && ((valid >= target_valid)); then
    printf 'campaign already complete: %d/%d valid runs\n' "$valid" "$target_valid" | tee "$progress"
    exit 0
fi

cd "$repo_dir"
while ((attempt < max_attempts)); do
    if [[ -n "$target_attempts" ]]; then
        ((attempt >= target_attempts)) && break
    else
        ((valid >= target_valid)) && break
    fi
    attempt=$((attempt + 1))
    attempt_label="$(printf '%02d' "$attempt")"
    log="$log_dir/attempt-$attempt_label.log"
    exit_code=0

    printf 'attempt %d running; valid %d/%d\n' "$attempt" "$valid" "$target_valid" | tee "$progress"
    "$runner" run aiwf_medium_context \
        --model gemini-3.1-flash-live-preview \
        --service gemini-live \
        --thinking minimal \
        > "$log" 2>&1 || exit_code=$?

    run_dir="$(sed -n 's/^Output directory: //p' "$log" | tail -1)"
    [[ -n "$run_dir" ]] || stop_unexpected "attempt $attempt: no run directory in launcher output"
    transcript="$repo_dir/$run_dir/transcript.jsonl"
    audio="$repo_dir/$run_dir/conversation.wav"
    run_log="$repo_dir/$run_dir/run.log"
    runtime="$repo_dir/$run_dir/runtime.json"
    [[ -f "$transcript" ]] || stop_unexpected "attempt $attempt: missing transcript.jsonl ($run_dir)"
    [[ -s "$audio" ]] || stop_unexpected "attempt $attempt: missing/empty conversation.wav ($run_dir)"
    [[ -f "$run_log" ]] || stop_unexpected "attempt $attempt: missing run.log ($run_dir)"
    [[ -f "$runtime" ]] || stop_unexpected "attempt $attempt: missing runtime.json ($run_dir)"

    if rg -q 'RuntimeWarning|coroutine .* was never awaited|Task was destroyed|GEMINI_RAW_DECODE_ERROR|GEMINI_RAW_SEND_ERROR' "$log" "$run_log"; then
        stop_unexpected "attempt $attempt: runtime/decode/send warning detected ($run_dir)"
    fi

    turns="$(wc -l < "$transcript")"
    retry_events="$(rg -c 'Re-queuing audio|Successfully re-queued audio|scheduling turn .* retry|\[EMPTY_RESPONSE\]|\[NO_RESPONSE\]' "$run_log" || true)"
    retry_events="${retry_events:-0}"
    audio_bytes="$(stat -c '%s' "$audio")"
    classification="incomplete"

    if [[ "$exit_code" -eq 0 && "$turns" -eq 30 ]] && \
       jq -e 'select(.status == "completed" and .valid == true and .turns == 30)' "$runtime" >/dev/null; then
        classification="complete"
        valid=$((valid + 1))
        printf '%s\n' "$run_dir" >> "$complete_manifest"
    elif [[ "$exit_code" -ne 0 ]]; then
        classification="failed"
    fi

    printf '%s\n' "$run_dir" >> "$run_manifest"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$attempt" "$exit_code" "$classification" "$turns" "$retry_events" \
        "$audio_bytes" "$run_dir" "${log#$repo_dir/}" >> "$attempts"
    printf 'attempt %d: %s, turns=%s, retries=%s; valid %d/%d\n' \
        "$attempt" "$classification" "$turns" "$retry_events" "$valid" "$target_valid" | tee "$progress"
done

if [[ -n "$target_attempts" ]] && ((attempt < target_attempts)); then
    stop_unexpected "campaign stopped at max attempts: $attempt/$target_attempts attempts"
elif [[ -z "$target_attempts" ]] && ((valid < target_valid)); then
    stop_unexpected "campaign stopped at max attempts: $valid/$target_valid valid runs"
fi

: > "$failure_marker"
if [[ -n "$target_attempts" ]]; then
    printf 'campaign complete: %d/%d attempts; %d complete runs\n' \
        "$attempt" "$target_attempts" "$valid" | tee "$progress"
else
    printf 'campaign complete: %d/%d valid runs in %d attempts\n' \
        "$valid" "$target_valid" "$attempt" | tee "$progress"
fi
