#!/usr/bin/env python3
"""Preflight or resumably judge the frozen Gemma 4 31B N=30 cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANONICAL = HERE / "canonical.tsv"
JUDGING = HERE / "judging"
INPUTS = JUDGING / "canonical-inputs.tsv"
ATTEMPTS = JUDGING / "judge-attempts.tsv"
LOGS = JUDGING / "logs"
COMPLETE = JUDGING / "COMPLETE.json"
MODEL = "google/gemma-4-31B-it"
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
TARGET = 30
N_TURNS = 30
INPUT_FIELDS = ("slot", "run_dir", "transcript_sha256", "scheduled_turns")
ATTEMPT_FIELDS = (
    "slot",
    "run_dir",
    "attempt",
    "started_at",
    "finished_at",
    "exit_code",
    "valid",
    "transcript_sha256",
    "judge_model",
    "judge_version",
    "log",
    "error",
)
_ledger_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with _ledger_lock:
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            if new:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def scheduled_turns(run_dir: Path) -> list[int]:
    rows = read_jsonl(run_dir / "transcript.jsonl")
    turns = [
        row.get("turn") for row in rows if row.get("recovery_turn") is not True
    ]
    if turns != list(range(len(turns))) or not turns:
        raise RuntimeError(f"invalid scheduled-turn sequence: {run_dir}")
    if any(row.get("model_name") != MODEL for row in rows):
        raise RuntimeError(f"model identity mismatch: {run_dir}")
    return turns


def load_inputs() -> list[dict[str, Any]]:
    canonical = read_tsv(CANONICAL)
    if len(canonical) != TARGET:
        raise RuntimeError(f"judging requires {TARGET} canonical runs, found {len(canonical)}")
    if [int(row["slot"]) for row in canonical] != list(range(1, TARGET + 1)):
        raise RuntimeError("canonical slots are not exactly 1..30")
    entries = []
    seen: set[Path] = set()
    for row in canonical:
        run_dir = (ROOT / row["run_dir"]).resolve()
        run_dir.relative_to(ROOT.resolve())
        if run_dir in seen:
            raise RuntimeError(f"duplicate run directory: {run_dir}")
        seen.add(run_dir)
        turns = scheduled_turns(run_dir)
        transcript = run_dir / "transcript.jsonl"
        entries.append(
            {
                "slot": int(row["slot"]),
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": sha256(transcript),
                "turns": turns,
            }
        )
    return entries


def freeze_inputs(entries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "slot": entry["slot"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": len(entry["turns"]),
        }
        for entry in entries
    ]
    if INPUTS.exists():
        expected = [{key: str(row[key]) for key in INPUT_FIELDS} for row in rows]
        if read_tsv(INPUTS) != expected:
            raise RuntimeError("frozen judge inputs changed")
    else:
        write_tsv(INPUTS, INPUT_FIELDS, rows)


def validate_judgment(entry: dict[str, Any]) -> tuple[bool, str]:
    run_dir = entry["run_dir"]
    try:
        judged = read_jsonl(run_dir / "claude_judged.jsonl")
        summary = json.loads((run_dir / "claude_summary.json").read_text())
        if [row.get("turn") for row in judged] != entry["turns"]:
            raise RuntimeError("judged turns do not match the transcript")
        for row in judged:
            scores = row.get("scores") or {}
            for key in (
                "tool_use_correct",
                "instruction_following",
                "kb_grounding",
            ):
                if not isinstance(scores.get(key), bool):
                    raise RuntimeError(f"missing boolean score {key}")
        if summary.get("turns_scored") != len(entry["turns"]):
            raise RuntimeError("summary turn count mismatch")
        if summary.get("model_name") != MODEL:
            raise RuntimeError("summary model mismatch")
        if summary.get("judge_model") != JUDGE_MODEL:
            raise RuntimeError("judge model mismatch")
        if summary.get("judge_version") != JUDGE_VERSION:
            raise RuntimeError("judge version mismatch")
        if sha256(entry["transcript"]) != entry["transcript_sha256"]:
            raise RuntimeError("transcript changed")
        return True, ""
    except Exception as error:
        return False, str(error)


def previous_attempts(slot: int) -> int:
    return sum(int(row["slot"]) == slot for row in read_tsv(ATTEMPTS))


def judge_one(entry: dict[str, Any], max_attempts: int) -> tuple[int, bool, str]:
    valid, error = validate_judgment(entry)
    if valid:
        return entry["slot"], True, "existing valid judgment"
    for attempt in range(previous_attempts(entry["slot"]) + 1, max_attempts + 1):
        log = LOGS / f"slot{entry['slot']:02d}-attempt{attempt:02d}.log"
        started = now()
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                [
                    str(ROOT / ".venv/bin/multi-turn-eval"),
                    "judge",
                    str(entry["run_dir"]),
                ],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        valid, error = validate_judgment(entry)
        append_tsv(
            ATTEMPTS,
            ATTEMPT_FIELDS,
            {
                "slot": entry["slot"],
                "run_dir": entry["run_dir_text"],
                "attempt": attempt,
                "started_at": started,
                "finished_at": now(),
                "exit_code": result.returncode,
                "valid": int(valid),
                "transcript_sha256": entry["transcript_sha256"],
                "judge_model": JUDGE_MODEL if valid else "",
                "judge_version": JUDGE_VERSION if valid else "",
                "log": str(log.relative_to(ROOT)),
                "error": error,
            },
        )
        if valid:
            return entry["slot"], True, f"valid attempt {attempt}"
    return entry["slot"], False, error or "attempt cap exhausted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    entries = load_inputs()
    existing = sum(validate_judgment(entry)[0] for entry in entries)
    print(f"preflight: canonical={len(entries)}/{TARGET}, valid={existing}/{TARGET}")
    if not args.execute:
        return 0
    JUDGING.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    freeze_inputs(entries)
    with (JUDGING / ".judge.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another judge campaign is active") from error
        failures = []
        pending = [entry for entry in entries if not validate_judgment(entry)[0]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_one, entry, args.max_attempts) for entry in pending]
            for future in concurrent.futures.as_completed(futures):
                slot, valid, detail = future.result()
                print(f"slot={slot:02d} valid={int(valid)} {detail}", flush=True)
                if not valid:
                    failures.append((slot, detail))
        if failures:
            raise RuntimeError(f"judging incomplete: {failures}")
        final = sum(validate_judgment(entry)[0] for entry in entries)
        if final != TARGET:
            raise RuntimeError(f"judging validation found {final}/{TARGET}")
        payload = {
            "completed_at": now(),
            "canonical_runs": TARGET,
            "scripted_turns": sum(len(entry["turns"]) for entry in entries),
            "judge_model": JUDGE_MODEL,
            "judge_version": JUDGE_VERSION,
            "canonical_inputs_sha256": sha256(INPUTS),
        }
        COMPLETE.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"judging complete: {final}/{TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
