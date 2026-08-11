#!/usr/bin/env python3
"""Analyze the Async Gemini Live no-replay campaign and its raw receive trace."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import wave
from collections import Counter
from pathlib import Path


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_DIR = CAMPAIGN_DIR.parents[2]
RAW_MARKER = "[GEMINI_RAW_EVENT] "
SEND_MARKER = "[GEMINI_RAW_SEND] "
OUTBOUND_VALIDATION_RUN = REPO_DIR / (
    "runs/aiwf_medium_context/20260804T210138_async-gemini-live_0b5a6d58"
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 1.0]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - margin, center + margin]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def raw_event_from_line(line: str) -> dict | None:
    if RAW_MARKER not in line:
        return None
    return json.loads(line.split(RAW_MARKER, 1)[1])


def analyze_outbound_validation() -> dict | None:
    log_path = OUTBOUND_VALIDATION_RUN / "run.log"
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    timeout_index = next(
        index for index, line in enumerate(lines) if "[NO_RESPONSE] No TTS response" in line
    )
    queued_index = max(
        index
        for index, line in enumerate(lines[:timeout_index])
        if "[USER_AUDIO_QUEUED]" in line
    )
    finished_index = max(
        index
        for index, line in enumerate(lines[:timeout_index])
        if "FINISHED SENDING AUDIO" in line
    )
    sends = [
        (index, json.loads(line.split(SEND_MARKER, 1)[1]))
        for index, line in enumerate(lines)
        if SEND_MARKER in line
    ]
    receives = [
        (index, raw_event_from_line(line))
        for index, line in enumerate(lines)
        if RAW_MARKER in line
    ]
    real_audio_sends = [
        event for index, event in sends if queued_index < index < finished_index
    ]
    post_audio_sends = [
        event for index, event in sends if finished_index < index < timeout_index
    ]
    post_audio_receives = [
        event for index, event in receives if finished_index < index < timeout_index
    ]
    cancellation_line = next(
        line for line in lines if "[GEMINI_RAW_RECV_CANCELLED]" in line
    )
    cancellation_wait = re.search(r"wait_ms=([0-9.]+)", cancellation_line)
    runtime = json.loads((OUTBOUND_VALIDATION_RUN / "runtime.json").read_text())
    return {
        "run_dir": str(OUTBOUND_VALIDATION_RUN.relative_to(REPO_DIR)),
        "benchmark_sample": False,
        "classification": runtime.get("failure", {}).get("reason"),
        "failure_turn": runtime.get("failure", {}).get("turn"),
        "successful_real_audio_sends": len(real_audio_sends),
        "successful_real_audio_bytes": sum(event.get("audio_bytes") or 0 for event in real_audio_sends),
        "successful_post_audio_sends": len(post_audio_sends),
        "successful_post_audio_bytes": sum(event.get("audio_bytes") or 0 for event in post_audio_sends),
        "maximum_send_wait_ms": max(event["send_wait_ms"] for _, event in sends),
        "raw_receives_after_audio_finished": len(post_audio_receives),
        "send_errors": sum("[GEMINI_RAW_SEND_ERROR]" in line for line in lines),
        "send_cancellations": sum("[GEMINI_RAW_SEND_CANCELLED]" in line for line in lines),
        "receive_cancellations": sum("[GEMINI_RAW_RECV_CANCELLED]" in line for line in lines),
        "receive_wait_at_cancellation_ms": (
            float(cancellation_wait.group(1)) if cancellation_wait else None
        ),
        "last_successful_send_log_timestamp": next(
            line[:23]
            for line in reversed(lines[:timeout_index])
            if SEND_MARKER in line
        ),
        "timeout_log_timestamp": lines[timeout_index][:23],
    }


def summarize() -> dict:
    with (CAMPAIGN_DIR / "attempts.tsv").open(newline="") as handle:
        attempts = list(csv.DictReader(handle, delimiter="\t"))

    classifications = Counter(row["classification"] for row in attempts)
    failure_turns = [
        int(row["failure_turn"])
        for row in attempts
        if row["classification"] == "no_audio_timeout"
    ]
    top_level_keys: Counter[str] = Counter()
    server_content_keys: Counter[str] = Counter()
    part_types: Counter[str] = Counter()
    interaction_statuses: Counter[str] = Counter()
    raw_events_total = 0
    raw_empty_events = 0
    tool_call_events = 0
    error_events = 0
    go_away_events = 0
    sequence_gaps: list[dict] = []
    timeout_boundaries: list[dict] = []
    raw_close_logs = 0
    raw_decode_errors = 0
    replay_logs = 0
    unexpected_runtime_logs = 0
    transcript_turns = 0
    ttfb_ms: list[float] = []
    latency_ms: list[float] = []
    audio: list[dict] = []

    for row in attempts:
        attempt = int(row["attempt"])
        run_dir = REPO_DIR / row["run_dir"]
        log_path = REPO_DIR / row["log"]
        lines = log_path.read_text(errors="replace").splitlines()

        events: list[tuple[int, dict]] = []
        for index, line in enumerate(lines):
            event = raw_event_from_line(line)
            if event is None:
                continue
            events.append((index, event))
            raw_events_total += 1
            if not event.get("top_level_keys"):
                raw_empty_events += 1
            top_level_keys.update(event.get("top_level_keys") or [])
            server_content_keys.update(event.get("server_content_keys") or [])
            for part in event.get("model_turn_parts") or []:
                part_types[part.get("type", "unknown")] += 1
            if event.get("interaction_status"):
                interaction_statuses[str(event["interaction_status"])] += 1
            if event.get("tool_calls"):
                tool_call_events += 1
            if event.get("error"):
                error_events += 1
            if event.get("go_away"):
                go_away_events += 1

        sequences = [event["sequence"] for _, event in events]
        expected = list(range(1, len(sequences) + 1))
        if sequences != expected:
            sequence_gaps.append(
                {"attempt": attempt, "observed": sequences, "expected_count": len(expected)}
            )

        raw_close_logs += sum("[GEMINI_RAW_CLOSE]" in line for line in lines)
        raw_decode_errors += sum("[GEMINI_RAW_DECODE_ERROR]" in line for line in lines)
        replay_logs += sum(
            "Re-queuing audio" in line or "Successfully re-queued audio" in line
            for line in lines
        )
        unexpected_runtime_logs += sum(
            marker in line
            for line in lines
            for marker in (
                "RuntimeWarning",
                "was never awaited",
                "Task was destroyed",
            )
        )

        if row["classification"] == "no_audio_timeout":
            timeout_index = next(
                index for index, line in enumerate(lines) if "[NO_RESPONSE] No TTS response" in line
            )
            sent_indices = [
                index
                for index, line in enumerate(lines[:timeout_index])
                if "FINISHED SENDING AUDIO" in line
            ]
            vad_indices = [
                index
                for index, line in enumerate(lines[:timeout_index])
                if "[VAD] UserStoppedSpeaking" in line
            ]
            last_send = max(sent_indices)
            last_vad = max(vad_indices)
            after_send = [event for index, event in events if last_send < index < timeout_index]
            after_vad = [event for index, event in events if last_vad < index < timeout_index]

            def boundary_counts(boundary_events: list[dict]) -> dict:
                return {
                    "all_raw_events": len(boundary_events),
                    "empty_json_events": sum(not event.get("top_level_keys") for event in boundary_events),
                    "structured_events": sum(bool(event.get("top_level_keys")) for event in boundary_events),
                    "audio_events": sum(
                        any(part.get("type") == "inline_data" for part in event.get("model_turn_parts") or [])
                        for event in boundary_events
                    ),
                    "text_events": sum(
                        any(part.get("type") == "text" for part in event.get("model_turn_parts") or [])
                        or bool(event.get("output_transcription"))
                        for event in boundary_events
                    ),
                    "tool_call_events": sum(bool(event.get("tool_calls")) for event in boundary_events),
                    "status_events": sum(bool(event.get("interaction_status")) for event in boundary_events),
                    "error_events": sum(bool(event.get("error")) for event in boundary_events),
                    "go_away_events": sum(bool(event.get("go_away")) for event in boundary_events),
                }

            timeout_boundaries.append(
                {
                    "attempt": attempt,
                    "turn": int(row["failure_turn"]),
                    "after_audio_send_finished": boundary_counts(after_send),
                    "after_last_local_vad_stop": boundary_counts(after_vad),
                    "last_raw_event_before_timeout": events[-1][1] if events else None,
                }
            )

        transcript_path = run_dir / "transcript.jsonl"
        for line in transcript_path.read_text().splitlines():
            turn = json.loads(line)
            transcript_turns += 1
            if turn.get("ttfb_ms") is not None:
                ttfb_ms.append(float(turn["ttfb_ms"]))
            if turn.get("latency_ms") is not None:
                latency_ms.append(float(turn["latency_ms"]))

        audio_path = run_dir / "conversation.wav"
        with wave.open(str(audio_path), "rb") as wav:
            audio.append(
                {
                    "attempt": attempt,
                    "channels": wav.getnchannels(),
                    "sample_rate": wav.getframerate(),
                    "duration_seconds": wav.getnframes() / wav.getframerate(),
                    "bytes": audio_path.stat().st_size,
                }
            )

    completion_interval = wilson_interval(classifications["complete"], len(attempts))
    return {
        "attempts": len(attempts),
        "classifications": dict(sorted(classifications.items())),
        "complete_conversations": classifications["complete"],
        "completion_rate": classifications["complete"] / len(attempts),
        "completion_rate_wilson_95": completion_interval,
        "recorded_partial_turns": transcript_turns,
        "timeout_turns": {
            "values": failure_turns,
            "counts": {str(key): value for key, value in sorted(Counter(failure_turns).items())},
            "minimum": min(failure_turns),
            "median": statistics.median(failure_turns),
            "maximum": max(failure_turns),
        },
        "raw_trace": {
            "events": raw_events_total,
            "empty_json_events": raw_empty_events,
            "top_level_key_occurrences": dict(sorted(top_level_keys.items())),
            "server_content_key_occurrences": dict(sorted(server_content_keys.items())),
            "model_part_type_occurrences": dict(sorted(part_types.items())),
            "interaction_status_occurrences": dict(sorted(interaction_statuses.items())),
            "tool_call_events": tool_call_events,
            "error_events": error_events,
            "go_away_events": go_away_events,
            "socket_close_logs": raw_close_logs,
            "decode_errors": raw_decode_errors,
            "sequence_gaps": sequence_gaps,
        },
        "timeout_boundaries": timeout_boundaries,
        "no_replay_validation": {
            "runtime_rows_with_replayed_false": sum(
                row["classification"] != "no_audio_timeout"
                or json.loads((REPO_DIR / row["run_dir"] / "runtime.json").read_text())
                .get("failure", {})
                .get("replayed")
                is False
                for row in attempts
            ),
            "replay_log_lines": replay_logs,
        },
        "runtime_warning_lines": unexpected_runtime_logs,
        "partial_turn_timing": {
            "ttfb_samples": len(ttfb_ms),
            "ttfb_median_ms": percentile(ttfb_ms, 0.5),
            "ttfb_p90_ms": percentile(ttfb_ms, 0.9),
            "response_latency_samples": len(latency_ms),
            "response_latency_median_ms": percentile(latency_ms, 0.5),
            "response_latency_p90_ms": percentile(latency_ms, 0.9),
        },
        "audio": {
            "files": len(audio),
            "all_stereo": all(item["channels"] == 2 for item in audio),
            "all_24khz": all(item["sample_rate"] == 24000 for item in audio),
            "total_bytes": sum(item["bytes"] for item in audio),
            "duration_seconds": {
                "minimum": min(item["duration_seconds"] for item in audio),
                "median": statistics.median(item["duration_seconds"] for item in audio),
                "maximum": max(item["duration_seconds"] for item in audio),
            },
            "files_detail": audio,
        },
        "outbound_validation": analyze_outbound_validation(),
    }


def write_markdown(analysis: dict) -> None:
    timeout_after_send = [
        row["after_audio_send_finished"] for row in analysis["timeout_boundaries"]
    ]
    timeout_after_vad = [
        row["after_last_local_vad_stop"] for row in analysis["timeout_boundaries"]
    ]
    lines = [
        "# Async Gemini Live no-replay campaign",
        "",
        "Date: 2026-08-04",
        "",
        "## Result",
        "",
        f"- Attempts: {analysis['attempts']}",
        f"- Full 30-turn completions: {analysis['complete_conversations']}",
        f"- No-audio timeouts: {analysis['classifications'].get('no_audio_timeout', 0)}",
        f"- Model-ended-early runs: {analysis['classifications'].get('model_ended_early', 0)}",
        f"- Partial turns recorded before termination: {analysis['recorded_partial_turns']}",
        "- No complete runs were available to judge.",
        "",
        "The observed completion rate was 0/20. The two-sided 95% Wilson interval is "
        f"{analysis['completion_rate_wilson_95'][0] * 100:.1f}% to "
        f"{analysis['completion_rate_wilson_95'][1] * 100:.1f}%.",
        "",
        "## Timeout boundary audit",
        "",
        "The trace hooks the private SDK immediately after `WebSocket.recv()` and before "
        "JSON-to-SDK conversion. Across the 19 timeout runs:",
        "",
        f"- {sum(row['all_raw_events'] == 0 for row in timeout_after_send)}/19 had no raw "
        "server event after the final user-audio send completed.",
        f"- {sum(row['all_raw_events'] == 0 for row in timeout_after_vad)}/19 had no raw "
        "server event after the last local `UserStoppedSpeaking` signal.",
        f"- Raw events captured across all attempts: {analysis['raw_trace']['events']:,}.",
        f"- Raw receive sequence gaps: {len(analysis['raw_trace']['sequence_gaps'])}.",
        f"- Raw JSON decode errors: {analysis['raw_trace']['decode_errors']}.",
        f"- Raw socket-close events: {analysis['raw_trace']['socket_close_logs']}.",
        f"- Server error events: {analysis['raw_trace']['error_events']}.",
        f"- GoAway events: {analysis['raw_trace']['go_away_events']}.",
        "- No unrecognized model-part shape was observed; the raw trace inventory is in "
        "`analysis.json`.",
        "",
        "Because the receive loop continuously calls the SDK's `session.receive()` and the "
        "trace executes before conversion, these data rule out a server message arriving "
        "during any observed 15-second wait and then being dropped by the SDK or wrapper. "
        "They cannot prove that the server would not have responded just after the watchdog "
        "cancelled the run, nor can any client-side logging prove what was never delivered "
        "over the socket.",
        "",
        "## Other validation",
        "",
        f"- Runtime records with either a non-timeout result or `replayed=false`: "
        f"{analysis['no_replay_validation']['runtime_rows_with_replayed_false']}/20.",
        f"- Replay log lines: {analysis['no_replay_validation']['replay_log_lines']}.",
        f"- Runtime warning lines: {analysis['runtime_warning_lines']}.",
        f"- Recordings: {analysis['audio']['files']}/20; all stereo: "
        f"{str(analysis['audio']['all_stereo']).lower()}; all 24 kHz: "
        f"{str(analysis['audio']['all_24khz']).lower()}.",
        "",
        "Timeout zero-based turn counts: "
        + ", ".join(
            f"{turn}: {count}" for turn, count in analysis["timeout_turns"]["counts"].items()
        )
        + ".",
        "",
        "The timeout turn median was "
        f"{analysis['timeout_turns']['median']:.0f} (zero-based), with range "
        f"{analysis['timeout_turns']['minimum']}-{analysis['timeout_turns']['maximum']}.",
    ]
    outbound = analysis.get("outbound_validation")
    if outbound:
        lines.extend(
            [
                "",
                "## Outbound API-boundary validation",
                "",
                "A separate diagnostic attempt—not included in the 20-run benchmark sample—"
                "traced successful `send_realtime_input()` returns as well as raw receives. "
                f"It timed out at zero-based turn {outbound['failure_turn']}.",
                "",
                f"- The complete real WAV produced {outbound['successful_real_audio_sends']:,} "
                f"successful SDK sends totaling {outbound['successful_real_audio_bytes']:,} bytes.",
                f"- After the WAV ended, {outbound['successful_post_audio_sends']:,} further "
                f"continuous-audio sends totaling {outbound['successful_post_audio_bytes']:,} "
                "bytes succeeded before the timeout.",
                f"- Maximum SDK send wait: {outbound['maximum_send_wait_ms']:.3f} ms; "
                f"send errors: {outbound['send_errors']}; send cancellations: "
                f"{outbound['send_cancellations']}.",
                f"- Raw receives after the WAV ended: "
                f"{outbound['raw_receives_after_audio_finished']}.",
                f"- The outstanding raw receive was explicitly cancelled after waiting "
                f"{outbound['receive_wait_at_cancellation_ms']:,.1f} ms.",
                f"- Last successful send: `{outbound['last_successful_send_log_timestamp']}`; "
                f"timeout: `{outbound['timeout_log_timestamp']}`.",
                "",
                f"Run: `{outbound['run_dir']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "./.venv/bin/python docs/async-gemini-live-comparison-2026-08-04/"
            "no-replay-n20-2026-08-04/analyze_campaign.py",
            "```",
        ]
    )
    (CAMPAIGN_DIR / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = summarize()
    (CAMPAIGN_DIR / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(result)
