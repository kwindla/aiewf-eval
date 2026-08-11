#!/usr/bin/env python3
"""Compare clean non-tool spoken-output rates for Clever and Gemini Preview."""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
QUEUE_RE = re.compile(r"Queued paced audio for (?:first )?turn(?: (\d+))?:")
TURN_END_RE = re.compile(r"on_turn_end start: turn_idx=(\d+)")
WORD_RE = re.compile(r"\b[\w']+\b")
REQUIRED_TOOL_TURNS = {11, 12, 15, 17, 24, 29}


def timestamp(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()


def turn_output_rows(run: dict) -> list[dict]:
    run_dir = Path(run["run_dir"])
    lines = (run_dir / "run.log").read_text(errors="replace").splitlines()
    events = []
    for line in lines:
        event_time = timestamp(line)
        if event_time is None:
            continue
        if match := QUEUE_RE.search(line):
            events.append((event_time, "queue", int(match.group(1) or 0)))
        elif "[LLM→] TTSStartedFrame" in line:
            events.append((event_time, "start", None))
        elif "[LLM→] TTSStoppedFrame" in line:
            events.append((event_time, "stop", None))
        elif match := TURN_END_RE.search(line):
            events.append((event_time, "end", int(match.group(1))))

    transcripts = {
        row["turn"]: row
        for row in map(
            json.loads, (run_dir / "transcript.jsonl").read_text().splitlines()
        )
        if not row.get("recovery_turn") and isinstance(row.get("turn"), int)
    }
    affected = {event["turn"] for event in run["timeout_events"]}
    rows = []
    for queue_time, _, turn in (event for event in events if event[1] == "queue"):
        ends = [
            event_time
            for event_time, event_type, event_turn in events
            if event_type == "end" and event_turn == turn and event_time > queue_time
        ]
        if not ends:
            continue
        end_time = min(ends)
        active_start = None
        spans = []
        for event_time, event_type, _ in events:
            if not queue_time <= event_time <= end_time:
                continue
            if event_type == "start" and active_start is None:
                active_start = event_time
            elif event_type == "stop" and active_start is not None:
                spans.append(event_time - active_start)
                active_start = None

        transcript = transcripts.get(turn)
        words = len(WORD_RE.findall(transcript.get("assistant_text", ""))) if transcript else 0
        output_seconds = sum(spans)
        if output_seconds and words:
            rows.append(
                {
                    "turn": turn,
                    "timeout_affected": turn in affected,
                    "output_seconds": output_seconds,
                    "words": words,
                    "words_per_minute": 60 * words / output_seconds,
                    "output_segments": len(spans),
                }
            )
    return rows


def summarize(name: str, runs: list[dict]) -> dict:
    rows = [row for run in runs for row in turn_output_rows(run)]
    rows = [row for row in rows if row["turn"] not in REQUIRED_TOOL_TURNS]
    if name == "Async Gemini Live":
        rows = [row for row in rows if not row["timeout_affected"]]
        selection = "non-tool turns excluding timeout-affected turns"
    else:
        selection = "non-tool turns"
    return {
        "name": name,
        "selection": selection,
        "turn_count": len(rows),
        "median_output_seconds": statistics.median(
            row["output_seconds"] for row in rows
        ),
        "median_words": statistics.median(row["words"] for row in rows),
        "median_words_per_minute": statistics.median(
            row["words_per_minute"] for row in rows
        ),
        "multi_segment_turn_count": sum(row["output_segments"] > 1 for row in rows),
    }


def main() -> None:
    timeout_analysis = json.loads((HERE / "timeout-replay-analysis.json").read_text())
    result = {
        "analysis_date": "2026-08-04",
        "method": (
            "Assistant transcript word count divided by summed TTSStarted-to-TTSStopped "
            "wall-clock spans within each benchmark turn."
        ),
        "cohorts": [
            summarize(cohort["name"], cohort["runs"])
            for cohort in timeout_analysis["cohorts"]
        ],
    }
    output = HERE / "speaking-rate-analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("wrote", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
