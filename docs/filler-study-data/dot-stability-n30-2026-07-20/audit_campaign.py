#!/usr/bin/env python3
"""Arm-blind operational audit for the focused dot campaign."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LANES = ("openai-a", "openai-b", "lilac", "baseten", "openrouter", "baseten-qwen")
PRIMARY_LANES = {"openai-a", "openai-b", "lilac", "baseten", "baseten-qwen"}
OBJECTIVE_ERROR = re.compile(
    r"Pipeline failed|Idle timeout detected|timed out|ReadTimeout|ConnectTimeout|"
    r"Connection(?:Error|Reset|Refused)|rate.?limit|HTTP[/ ]+[45][0-9][0-9]|"
    r"(?:^|[^0-9])429(?:[^0-9]|$)|APIError|InternalServerError|ServiceUnavailable|"
    r"EngineCore|Upstream error|Traceback",
    re.IGNORECASE | re.MULTILINE,
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def objective_failure(log: Path) -> bool:
    text = "\n".join(
        line for line in log.read_text(errors="replace").splitlines()
        if "idle_timeout_secs" not in line and "MTE_TEXT_IDLE_TIMEOUT" not in line
    )
    return bool(OBJECTIVE_ERROR.search(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    invalid_rows = rows(HERE / "invalidated.tsv")
    invalid = {(row["lane"], row["slot"], row["attempt"]) for row in invalid_rows}
    problems: list[str] = []
    effective: list[tuple[str, dict[str, str]]] = []
    all_attempts: dict[tuple[str, str, str], dict[str, str]] = {}

    for lane in LANES:
        schedule = rows(HERE / f"schedule-{lane}.tsv")
        scheduled = {row["slot"]: row for row in schedule}
        for row in schedule:
            requested = row["requested_model"].lower()
            if row["service"] == "openai" and (requested.endswith("-pro") or "-pro-" in requested):
                problems.append(f"prohibited OpenAI pro model in schedule: {requested}")
        attempts = rows(HERE / "state" / lane / "attempts.tsv")
        counted = rows(HERE / "state" / lane / "counted.tsv")
        for row in attempts:
            key = (lane, row["slot"], row["attempt"])
            if key in all_attempts:
                problems.append(f"duplicate attempt key: {key}")
            all_attempts[key] = row
            log = Path(row["log"])
            has_objective_error = log.is_file() and objective_failure(log)
            should_replace = int(row["end_session_turn"]) < 0 and has_objective_error
            classified_replace = row["classification"].startswith("infra_") and row["classification"].endswith("_replaced")
            if should_replace and not (classified_replace or key in invalid):
                problems.append(f"objective failure needs invalidation: {key} {log}")
            if classified_replace and int(row["end_session_turn"]) >= 0:
                problems.append(f"replacement called end_session: {key}")
        by_slot: Counter[str] = Counter()
        for row in counted:
            key = (lane, row["slot"], row["attempt"])
            if row["slot"] not in scheduled:
                problems.append(f"counted unscheduled slot: {key}")
            if key in invalid:
                continue
            by_slot[row["slot"]] += 1
            effective.append((lane, row))
            run_dir = Path(row["run_dir"])
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            if not (run_dir / "transcript.jsonl").is_file():
                problems.append(f"effective attempt lacks transcript: {key}")
            judged = run_dir / "claude_judged.jsonl"
            if args.require_complete and (not judged.is_file() or not judged.stat().st_size):
                problems.append(f"effective attempt lacks judgment: {key}")
        duplicate_slots = sorted(slot for slot, count in by_slot.items() if count != 1)
        if duplicate_slots:
            problems.append(f"lane {lane} has duplicate effective counted slots: {duplicate_slots}")
        if args.require_complete and lane in PRIMARY_LANES:
            missing = sorted(set(scheduled) - set(by_slot))
            if missing:
                problems.append(f"lane {lane} missing {len(missing)} effective slots")
            if not (HERE / "state" / lane / "COMPLETE").is_file():
                problems.append(f"lane {lane} lacks COMPLETE marker")

    for key in sorted(invalid - set(all_attempts)):
        problems.append(f"invalidated attempt not found: {key}")

    existing = rows(HERE / "existing-included.tsv")
    if args.require_complete:
        for row in existing:
            judged = ROOT / row["run_dir"] / "claude_judged.jsonl"
            if not judged.is_file() or not judged.stat().st_size:
                problems.append(f"historical attempt lacks judgment: {row['run_dir']}")
        if not (HERE / "state" / "existing-judge" / "COMPLETE").is_file():
            problems.append("historical judge lacks COMPLETE marker")

    primary_effective = [(lane, row) for lane, row in effective if lane in PRIMARY_LANES]
    superseded_openrouter = [(lane, row) for lane, row in effective if lane == "openrouter"]
    primary_existing = [row for row in existing if row["model"] != "qwen3_8b"]
    models = Counter((row["model"], row["arm"]) for _lane, row in primary_effective)
    print(f"primary effective new attempts: {len(primary_effective)} / 306")
    print(f"primary historical attempts: {len(primary_existing)} / 174")
    print(f"superseded OpenRouter effective attempts retained for audit: {len(superseded_openrouter)}")
    print(f"superseded historical Qwen attempts retained for audit: {len(existing) - len(primary_existing)}")
    print(f"invalidated attempts: {len(invalid)}")
    if models:
        print("effective new cells:")
        for cell, count in sorted(models.items()):
            print(f"  {cell[0]}/{cell[1]}: {count}")
    if problems:
        print("audit problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("campaign audit passed")


if __name__ == "__main__":
    main()
