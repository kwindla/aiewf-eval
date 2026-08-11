#!/usr/bin/env python3
"""Reject event-flow violations in an exact-audio-boundary Live run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RAW_SEND = re.compile(r"\[GEMINI_RAW_SEND\] (\{.*\})$")
RAW_EVENT = re.compile(r"\[GEMINI_RAW_EVENT\] (\{.*\})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activity_open = False
    starts = 0
    ends = 0
    model_audio_during_input = []
    terminal_during_input = []
    interruption_during_input = []
    tool_calls_during_input = []
    boundary_errors = []

    for line_number, line in enumerate(
        (args.run_dir / "run.log").read_text(errors="replace").splitlines(), start=1
    ):
        send_match = RAW_SEND.search(line)
        if send_match:
            event = json.loads(send_match.group(1))
            if event.get("activity_start") is True:
                starts += 1
                if activity_open:
                    boundary_errors.append(
                        {"line": line_number, "reason": "nested_activity_start"}
                    )
                activity_open = True
            if event.get("activity_end") is True:
                ends += 1
                if not activity_open:
                    boundary_errors.append(
                        {"line": line_number, "reason": "activity_end_without_start"}
                    )
                activity_open = False
            continue

        event_match = RAW_EVENT.search(line)
        if not event_match or not activity_open:
            continue
        event = json.loads(event_match.group(1))
        if any(
            part.get("type") == "inline_data"
            and str(part.get("mime_type", "")).startswith("audio/")
            for part in event.get("model_turn_parts") or []
        ):
            model_audio_during_input.append(
                {"line": line_number, "sequence": event.get("sequence")}
            )
        if event.get("interaction_status") == "REQUIRES_ACTION":
            terminal_during_input.append(
                {"line": line_number, "sequence": event.get("sequence")}
            )
        if event.get("interrupted") is True:
            interruption_during_input.append(
                {"line": line_number, "sequence": event.get("sequence")}
            )
        if event.get("tool_calls"):
            tool_calls_during_input.append(
                {
                    "line": line_number,
                    "sequence": event.get("sequence"),
                    "names": [call.get("name") for call in event["tool_calls"]],
                }
            )

    transcript = [
        json.loads(line)
        for line in (args.run_dir / "transcript.jsonl").read_text().splitlines()
        if line
    ]
    missing_ttfb = [
        row.get("turn")
        for row in transcript
        if row.get("assistant_text", "").strip() and row.get("ttfb_ms") is None
    ]
    result = {
        "activity_starts": starts,
        "activity_ends": ends,
        "activity_open_at_eof": activity_open,
        "boundary_errors": boundary_errors,
        "model_audio_during_input": model_audio_during_input,
        "terminal_during_input": terminal_during_input,
        "interruption_during_input": interruption_during_input,
        "tool_calls_during_input": tool_calls_during_input,
        "nonempty_transcript_missing_ttfb_turns": missing_ttfb,
        "valid": not (
            activity_open
            or boundary_errors
            or model_audio_during_input
            or tool_calls_during_input
            or missing_ttfb
        ),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
