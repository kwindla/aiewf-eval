#!/usr/bin/env python3
"""Measure premature Async Gemini Live endpointing in the no-replay campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from benchmarks._shared.turns import turns as benchmark_turns


TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
TRANSCRIPTION = re.compile(r"\[Transcription:user\] \[(.*)\]")


def timestamp(line: str) -> float | None:
    match = TIMESTAMP.match(line)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1)).timestamp()


def normalized_similarity(expected: str, actual: str) -> float:
    clean = lambda text: " ".join(text.lower().split())
    return SequenceMatcher(None, clean(expected), clean(actual)).ratio()


def parse_log(path: Path) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    awaiting_transcription: list[dict[str, Any]] = []

    for order, line in enumerate(path.read_text(errors="replace").splitlines()):
        at = timestamp(line)
        if "SENDING REAL AUDIO:" in line:
            active = {
                "turn": len(observed),
                "send_started": at,
                "send_started_order": order,
                "send_finished": None,
                "send_finished_order": None,
                "first_model_output": None,
                "first_model_output_order": None,
                "input_transcriptions": [],
            }
            observed.append(active)
            continue
        if "FINISHED SENDING AUDIO" in line and active is not None:
            active["send_finished"] = at
            active["send_finished_order"] = order
            awaiting_transcription.append(active)
            active = None
            continue
        if active is not None and "[GEMINI_RAW_EVENT]" in line:
            if '"model_turn_parts":[]' not in line and '"model_turn_parts":[' in line:
                active["first_model_output"] = active["first_model_output"] or at
                active["first_model_output_order"] = (
                    active["first_model_output_order"] or order
                )
        match = TRANSCRIPTION.search(line)
        if match:
            target = active or (awaiting_transcription[-1] if awaiting_transcription else None)
            if target is not None:
                target["input_transcriptions"].append(match.group(1))

    for row in observed:
        expected = benchmark_turns[row["turn"]]["input"] if row["turn"] < len(benchmark_turns) else ""
        actual = " ".join(row.pop("input_transcriptions"))
        row["expected_input"] = expected
        row["server_input_transcription"] = actual
        row["transcription_similarity"] = (
            round(normalized_similarity(expected, actual), 3) if actual else None
        )
        row["model_output_before_audio_finished"] = bool(
            row["first_model_output_order"] is not None
            and row["send_finished_order"] is not None
            and row["first_model_output_order"] < row["send_finished_order"]
        )
        if (
            row["model_output_before_audio_finished"]
            and row["first_model_output"] is not None
            and row["send_finished"] is not None
        ):
            row["output_lead_ms"] = round(
                (row["send_finished"] - row["first_model_output"]) * 1000
            )
        else:
            row["output_lead_ms"] = None
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts: dict[int, dict[str, str]] = {}
    with (args.campaign / "attempts.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            attempts[int(row["attempt"])] = row

    rows: list[dict[str, Any]] = []
    attempt_summaries = []
    for attempt, metadata in attempts.items():
        log = args.campaign / "logs" / f"attempt-{attempt:02d}.log"
        turns = parse_log(log)
        for row in turns:
            row["attempt"] = attempt
            rows.append(row)
        early = [row for row in turns if row["model_output_before_audio_finished"]]
        poor_asr = [
            row
            for row in turns
            if row["transcription_similarity"] is not None
            and row["transcription_similarity"] < 0.75
        ]
        attempt_summaries.append(
            {
                "attempt": attempt,
                "classification": metadata["classification"],
                "failure_turn": int(metadata["failure_turn"])
                if metadata["failure_turn"]
                else None,
                "turns_sent": len(turns),
                "premature_output_turns": [row["turn"] for row in early],
                "poor_transcription_turns": [row["turn"] for row in poor_asr],
            }
        )

    failed = [row for row in attempt_summaries if row["classification"] == "no_audio_timeout"]
    result = {
        "campaign": str(args.campaign),
        "attempts": len(attempt_summaries),
        "turns_sent": len(rows),
        "turns_with_model_output_before_audio_finished": sum(
            row["model_output_before_audio_finished"] for row in rows
        ),
        "turns_with_transcription_similarity_below_0_75": sum(
            row["transcription_similarity"] is not None
            and row["transcription_similarity"] < 0.75
            for row in rows
        ),
        "timeout_attempts_with_prior_premature_output": sum(
            bool(row["premature_output_turns"]) for row in failed
        ),
        "timeout_attempts": len(failed),
        "attempt_summaries": attempt_summaries,
        "turns": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"turns", "attempt_summaries"}}, indent=2))


if __name__ == "__main__":
    main()
