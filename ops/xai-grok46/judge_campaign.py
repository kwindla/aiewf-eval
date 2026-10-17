#!/usr/bin/env python3
"""Resumably judge the frozen Grok 4.6 N=30 campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "artifacts/canonical.tsv"
JUDGING = HERE / "judging"
LOGS = JUDGING / "logs"
RESULTS = JUDGING / "results.tsv"
MODEL = "grok-4.6"
JUDGE_MODEL = "claude-opus-4-5"
TARGET = 30


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def valid_output(run_dir: Path) -> tuple[bool, str]:
    try:
        transcript = read_jsonl(run_dir / "transcript.jsonl")
        scheduled_turns = [
            int(row["turn"])
            for row in transcript
            if row.get("recovery_turn") is not True
        ]
        judged = read_jsonl(run_dir / "claude_judged.jsonl")
        judged_turns = [int(row["turn"]) for row in judged]
        if judged_turns != scheduled_turns:
            raise RuntimeError(
                f"judged turns {judged_turns} do not match scheduled turns {scheduled_turns}"
            )
        for row in judged:
            scores = row.get("scores") or {}
            if not all(
                isinstance(scores.get(name), bool)
                for name in (
                    "tool_use_correct",
                    "instruction_following",
                    "kb_grounding",
                )
            ):
                raise RuntimeError("judgment lacks boolean component scores")
        summary = json.loads((run_dir / "claude_summary.json").read_text(encoding="utf-8"))
        if summary.get("model_name") != MODEL:
            raise RuntimeError("judge summary model mismatch")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def judge_one(entry: dict[str, str], max_attempts: int) -> dict[str, Any]:
    slot = int(entry["slot"])
    run_dir = (ROOT / entry["run_dir"]).resolve()
    run_dir.relative_to(ROOT)
    valid, error = valid_output(run_dir)
    if valid:
        return {"slot": slot, "run_dir": entry["run_dir"], "attempts": 0, "valid": 1, "error": ""}

    LOGS.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        started = datetime.now(UTC).isoformat(timespec="seconds")
        process = subprocess.run(
            [
                str(ROOT / ".venv/bin/multi-turn-eval"),
                "judge",
                str(run_dir),
                "--judge-model",
                JUDGE_MODEL,
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = LOGS / f"slot{slot:03d}-attempt{attempt:02d}.log"
        log_path.write_text(
            f"started_at={started}\nexit_code={process.returncode}\n{process.stdout}",
            encoding="utf-8",
        )
        valid, error = valid_output(run_dir)
        if valid:
            return {
                "slot": slot,
                "run_dir": entry["run_dir"],
                "attempts": attempt,
                "valid": 1,
                "error": "",
            }
    return {
        "slot": slot,
        "run_dir": entry["run_dir"],
        "attempts": max_attempts,
        "valid": 0,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    with CANONICAL.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    if [int(row["slot"]) for row in entries] != list(range(1, TARGET + 1)):
        raise RuntimeError("canonical manifest must contain slots 1 through 30")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda row: judge_one(row, args.max_attempts), entries))
    results.sort(key=lambda row: int(row["slot"]))
    JUDGING.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("slot", "run_dir", "attempts", "valid", "error"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    failures = [row for row in results if not row["valid"]]
    print(f"judged={TARGET - len(failures)}/{TARGET} results={RESULTS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
