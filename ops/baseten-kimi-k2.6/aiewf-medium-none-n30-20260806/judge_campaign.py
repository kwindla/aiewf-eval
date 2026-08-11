#!/usr/bin/env python3
"""Resumably judge the frozen BaseTen Kimi K2.6 N=30 cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
CANONICAL = HERE / "canonical.tsv"
CAMPAIGN_LOG = HERE / "campaign.log"
JUDGING = HERE / "judging"
INPUTS = JUDGING / "canonical-inputs.tsv"
SOURCE_HASHES = JUDGING / "judge-source-sha256.txt"
ATTEMPTS = JUDGING / "judge-attempts.tsv"
LOGS = JUDGING / "logs"
INVALID = JUDGING / "invalid-output-snapshots"
COMPLETE = JUDGING / "COMPLETE.json"
LOCK = JUDGING / ".judge.lock"

MODEL = "moonshotai/Kimi-K2.6"
TARGET = 30
N_TURNS = 30
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
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
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        add_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            if add_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object row {path}:{number}")
        rows.append(row)
    return rows


def slot_number(slot: str) -> int:
    match = re.fullmatch(r"K26-(\d{2})", slot)
    if not match:
        raise RuntimeError(f"invalid slot: {slot}")
    return int(match.group(1))


def load_entries(*, require_complete: bool) -> list[dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("model") != MODEL or config.get("target_complete_conversations") != TARGET:
        raise RuntimeError("frozen campaign configuration mismatch")
    rows = sorted(read_tsv(CANONICAL), key=lambda row: slot_number(row["slot"]))
    if require_complete and [slot_number(row["slot"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"judging requires exactly K26-01 through K26-30; found {len(rows)}")
    seen: set[Path] = set()
    result = []
    for row in rows:
        run_dir = (ROOT / row["run_dir"]).resolve()
        run_dir.relative_to(ROOT)
        if run_dir in seen:
            raise RuntimeError(f"duplicate canonical run: {run_dir}")
        seen.add(run_dir)
        transcript = run_dir / "transcript.jsonl"
        runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        source_rows = read_jsonl(transcript)
        scheduled = [item for item in source_rows if item.get("recovery_turn") is not True]
        if [item.get("turn") for item in scheduled] != list(range(N_TURNS)):
            raise RuntimeError(f"canonical transcript is not 30 scripted turns: {run_dir}")
        if any(item.get("model_name") != MODEL for item in source_rows):
            raise RuntimeError(f"model mismatch: {run_dir}")
        if runtime.get("status") != "completed" or runtime.get("valid") is not True:
            raise RuntimeError(f"runtime is not complete/valid: {run_dir}")
        result.append(
            {
                "slot": row["slot"],
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": sha256(transcript),
            }
        )
    return result


def validate_judge_identity() -> None:
    text = (ROOT / "src/multi_turn_eval/judging/claude_judge.py").read_text(encoding="utf-8")
    model = re.search(r'^JUDGE_MODEL\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version = re.search(r'^JUDGE_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not model or not version or model.group(1) != JUDGE_MODEL or version.group(1) != JUDGE_VERSION:
        raise RuntimeError("judge identity changed")


def validate_output(entry: dict[str, Any]) -> tuple[bool, str]:
    try:
        judged = read_jsonl(entry["run_dir"] / "claude_judged.jsonl")
        summary = json.loads((entry["run_dir"] / "claude_summary.json").read_text(encoding="utf-8"))
        analysis = entry["run_dir"] / "claude_analysis.md"
        if not analysis.is_file() or not analysis.stat().st_size:
            raise RuntimeError("missing claude_analysis.md")
        if [row.get("turn") for row in judged] != list(range(N_TURNS)):
            raise RuntimeError("judged output is not exactly scripted turns 0-29")
        for row in judged:
            scores = row.get("scores")
            if not isinstance(scores, dict) or not all(
                isinstance(scores.get(name), bool)
                for name in ("tool_use_correct", "instruction_following", "kb_grounding")
            ):
                raise RuntimeError("judged row lacks boolean component scores")
        if summary.get("judge_model") != JUDGE_MODEL or summary.get("judge_version") != JUDGE_VERSION:
            raise RuntimeError("judge identity mismatch in summary")
        if summary.get("model_name") != MODEL or summary.get("turns_scored") != N_TURNS:
            raise RuntimeError("model/turn count mismatch in judge summary")
        if sha256(entry["transcript"]) != entry["transcript_sha256"]:
            raise RuntimeError("judging changed canonical transcript")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def source_hash_text() -> str:
    paths = (
        Path(__file__).resolve(),
        CONFIG,
        CANONICAL,
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/judging/claude_judge.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
        ROOT / "benchmarks/aiwf_medium_context/prompts/system.py",
        ROOT / "benchmarks/_shared/turns.py",
    )
    return "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in paths)


def freeze_inputs(entries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "slot": entry["slot"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": N_TURNS,
        }
        for entry in entries
    ]
    if INPUTS.exists() and read_tsv(INPUTS) != [{key: str(row[key]) for key in INPUT_FIELDS} for row in rows]:
        raise RuntimeError("frozen judging inputs changed")
    if not INPUTS.exists():
        write_tsv(INPUTS, INPUT_FIELDS, rows)
    expected = source_hash_text()
    if SOURCE_HASHES.exists() and SOURCE_HASHES.read_text(encoding="utf-8") != expected:
        raise RuntimeError("judge/campaign source changed after judging began")
    if not SOURCE_HASHES.exists():
        SOURCE_HASHES.write_text(expected, encoding="utf-8")


def attempt_count(slot: str) -> int:
    if not ATTEMPTS.exists():
        return 0
    return max((int(row["attempt"]) for row in read_tsv(ATTEMPTS) if row["slot"] == slot), default=0)


def snapshot_invalid(entry: dict[str, Any], attempt: int) -> None:
    paths = [entry["run_dir"] / name for name in ("claude_judged.jsonl", "claude_summary.json", "claude_analysis.md")]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    destination = INVALID / f"{entry['slot']}-before-attempt{attempt:02d}-{int(time.time())}"
    destination.mkdir(parents=True)
    for path in existing:
        shutil.copy2(path, destination / path.name)
        path.unlink()


def judge_one(entry: dict[str, Any], *, max_attempts: int) -> tuple[str, bool, str]:
    valid, error = validate_output(entry)
    if valid:
        return entry["slot"], True, "already valid"
    while True:
        attempt = attempt_count(entry["slot"]) + 1
        if attempt > max_attempts:
            return entry["slot"], False, f"attempt cap exhausted: {error}"
        snapshot_invalid(entry, attempt)
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / f"{entry['slot']}-attempt{attempt:02d}.log"
        started = now()
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.run(
                [str(ROOT / ".venv/bin/multi-turn-eval"), "judge", entry["run_dir_text"]],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        valid, error = validate_output(entry)
        append_tsv(
            ATTEMPTS,
            ATTEMPT_FIELDS,
            {
                "slot": entry["slot"],
                "run_dir": entry["run_dir_text"],
                "attempt": attempt,
                "started_at": started,
                "finished_at": now(),
                "exit_code": process.returncode,
                "valid": int(valid),
                "transcript_sha256": entry["transcript_sha256"],
                "judge_model": JUDGE_MODEL,
                "judge_version": JUDGE_VERSION,
                "log": str(log_path.relative_to(ROOT)),
                "error": error.replace("\t", " ").replace("\n", " "),
            },
        )
        if valid:
            return entry["slot"], True, f"valid after attempt {attempt}"
        if attempt >= max_attempts:
            return entry["slot"], False, error
        time.sleep(5 * 2 ** (attempt - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise RuntimeError("--workers must be 1..4")
    validate_judge_identity()
    entries = load_entries(require_complete=args.execute)
    valid = sum(validate_output(entry)[0] for entry in entries)
    print(f"preflight: canonical={len(entries)}/{TARGET} valid_judgments={valid}/{len(entries)}")
    if not args.execute:
        return 0
    if "campaign collection complete canonical=30/30" not in CAMPAIGN_LOG.read_text(encoding="utf-8"):
        raise RuntimeError("campaign completion marker is missing")
    JUDGING.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another judge collector holds the lock") from exc
        freeze_inputs(entries)
        pending = [entry for entry in entries if not validate_output(entry)[0]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_one, entry, max_attempts=args.max_attempts) for entry in pending]
            for future in concurrent.futures.as_completed(futures):
                slot, ok, note = future.result()
                print(f"{slot}: {'OK' if ok else 'FAILED'} {note}", flush=True)
        valid = sum(validate_output(entry)[0] for entry in entries)
        if valid != TARGET:
            raise RuntimeError(f"judging incomplete: {valid}/{TARGET}")
        COMPLETE.write_text(
            json.dumps(
                {
                    "campaign_id": json.loads(CONFIG.read_text(encoding="utf-8"))["campaign_id"],
                    "completed_at": now(),
                    "canonical_runs": TARGET,
                    "scripted_turns": TARGET * N_TURNS,
                    "judge_model": JUDGE_MODEL,
                    "judge_version": JUDGE_VERSION,
                    "canonical_inputs_sha256": sha256(INPUTS),
                    "judge_source_sha256": sha256(SOURCE_HASHES),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"judging complete: {valid}/{TARGET}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
