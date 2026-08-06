#!/usr/bin/env python3
"""Recover a repeatedly malformed full-transcript judgment using overlapping windows."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
from pathlib import Path

from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from multi_turn_eval.judging import claude_judge as judge_module
from multi_turn_eval.judging.claude_judge import (
    judge_with_claude,
    load_transcript,
    write_outputs,
)


_STRICT_JSON_LOADS = json.loads


def _literal_tolerant_loads(value: str, *args: object, **kwargs: object) -> object:
    """Accept strict JSON or a literal mapping, ignoring brace-bearing preamble."""
    starts = [0] + [index for index, char in enumerate(value) if char == "{" and index]
    last_error: Exception | None = None
    for start in reversed(starts):
        candidate = value[start:].strip()
        try:
            parsed = _STRICT_JSON_LOADS(candidate, *args, **kwargs)
            if start == 0 or (isinstance(parsed, dict) and "final_judgments" in parsed):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict) and "final_judgments" in parsed:
                return parsed
        except (SyntaxError, ValueError) as exc:
            last_error = exc
    raise ValueError("no parseable JSON or literal mapping suffix") from last_error


# Recovery-local parser fallback; the production judge source remains unchanged.
judge_module.json.loads = _literal_tolerant_loads


WINDOWS = (
    (set(range(0, 15)), set(range(0, 10))),
    (set(range(10, 25)), set(range(10, 20))),
    (set(range(15, 30)), set(range(20, 30))),
)


async def recover(run_dir: Path) -> None:
    merged: dict[int, dict] = {}
    notes: list[str] = []
    tracking: dict[str, object] = {}
    model_name: str | None = None
    for window_index, (window, retained) in enumerate(WINDOWS, start=1):
        cache_path = run_dir / f"judge-recovery-window-{window_index}.json"
        if cache_path.is_file():
            cached = _STRICT_JSON_LOADS(cache_path.read_text())
            judgments = {int(turn): row for turn, row in cached["judgments"].items()}
            if set(judgments) != window:
                raise ValueError(f"cached window {window_index} coverage mismatch")
            merged.update({turn: judgments[turn] for turn in retained})
            notes.extend(cached.get("notes", []))
            tracking.update(cached.get("tracking", {}))
            model_name = cached["model_name"]
            print(f"window={window_index} cached valid")
            continue
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = await judge_with_claude(
                    run_dir,
                    only_turns=window,
                    debug=False,
                    expected_turns=BenchmarkConfig.turns,
                )
                judgments = result["judgments"]
                if set(judgments) != window:
                    raise ValueError(
                        f"window {window_index} coverage mismatch: "
                        f"expected={sorted(window)} got={sorted(judgments)}"
                    )
                merged.update({turn: judgments[turn] for turn in retained})
                note = result.get("realignment_notes", "")
                if note:
                    notes.append(f"Window {window_index}: {note}")
                tracking.update(result.get("function_tracking", {}))
                model_name = result["model_name"]
                cache_path.write_text(
                    json.dumps(
                        {
                            "judgments": judgments,
                            "notes": [f"Window {window_index}: {note}"] if note else [],
                            "tracking": result.get("function_tracking", {}),
                            "model_name": model_name,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"window={window_index} attempt={attempt} valid")
                break
            except Exception as exc:
                last_error = exc
                print(f"window={window_index} attempt={attempt} invalid: {exc}")
        else:
            raise RuntimeError(f"window {window_index} failed after retries") from last_error

    if set(merged) != set(range(30)):
        raise ValueError(f"merged coverage mismatch: {sorted(merged)}")
    records = load_transcript(run_dir)
    write_outputs(
        run_dir,
        records,
        merged,
        summary="",
        model_name=model_name or "unknown",
        realignment_notes="\n\n".join(notes),
        function_tracking=tracking,
        turn_taking_analysis=None,
    )
    print("merged judgment valid for turns 0-29")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    asyncio.run(recover(args.run_dir))


if __name__ == "__main__":
    main()
