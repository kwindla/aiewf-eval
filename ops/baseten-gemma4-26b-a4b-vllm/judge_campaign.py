#!/usr/bin/env python3
"""Preflight or resumably judge the frozen Gemma 4 AIEWF N=30 cohort.

The default invocation is read-only. ``--execute`` requires all 30 canonical
runs, freezes transcript and judge-source hashes, and uses at most four Claude
workers. It never sends requests to the BaseTen deployment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "configuration.json"
FROZEN_PATH = HERE / "frozen-order.tsv"
CANONICAL_PATH = HERE / "canonical.tsv"
CAMPAIGN_LOG = HERE / "campaign.log"
JUDGING_DIR = HERE / "judging"
INPUTS_PATH = JUDGING_DIR / "canonical-inputs.tsv"
SOURCE_HASH_PATH = JUDGING_DIR / "judge-source-sha256.txt"
ATTEMPTS_PATH = JUDGING_DIR / "judge-attempts.tsv"
LOG_DIR = JUDGING_DIR / "logs"
INVALID_DIR = JUDGING_DIR / "invalid-output-snapshots"
COMPLETE_PATH = JUDGING_DIR / "COMPLETE.json"

EXPECTED_MODEL = "google/gemma-4-26B-A4B-it"
EXPECTED_JUDGE_MODEL = "claude-opus-4-5"
EXPECTED_JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run path escapes repository: {value}")
    return resolved


def scheduled_turns(path: Path) -> list[int]:
    turns: list[int] = []
    for row in read_jsonl(path):
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"invalid scheduled turn {turn!r}: {path}")
        if row.get("model_name") != EXPECTED_MODEL:
            fail(f"wrong model in {path}: {row.get('model_name')!r}")
        turns.append(turn)
    if turns != list(range(len(turns))):
        fail(f"scheduled turns are not a contiguous prefix: {path}")
    if not turns:
        fail(f"canonical transcript has no scheduled turns: {path}")
    return turns


def load_canonical(*, require_complete: bool) -> list[dict[str, Any]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_config = {
        "benchmark": "aiwf_medium_context",
        "model": EXPECTED_MODEL,
        "provider": "BaseTen",
        "endpoint": (
            "https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1"
        ),
        "filler": None,
        "target_eligible_runs": TARGET,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
        },
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            fail(f"configuration mismatch for {key}")
    arm = config.get("arm") or {}
    if arm.get("name") != "none" or arm.get("enable_thinking") is not False:
        fail("judge accepts only the frozen thinking-off arm")
    frozen = read_tsv(FROZEN_PATH)
    if len(frozen) != TARGET:
        fail("frozen-order.tsv must contain 30 slots")
    canonical = read_tsv(CANONICAL_PATH)
    if [int(row["slot"]) for row in canonical] != list(
        range(1, len(canonical) + 1)
    ):
        fail("canonical rows must be a contiguous frozen prefix")
    if require_complete and len(canonical) != TARGET:
        fail(f"judging requires 30 canonical runs; found {len(canonical)}")
    seen_dirs: set[Path] = set()
    result: list[dict[str, Any]] = []
    for row in canonical:
        slot = int(row["slot"])
        if row["mode"] != "none" or frozen[slot - 1] != {
            "slot": str(slot),
            "mode": "none",
        }:
            fail(f"canonical/frozen mismatch at slot {slot}")
        run_dir = resolve_run_dir(row["run_dir"])
        if run_dir in seen_dirs:
            fail(f"duplicate canonical run directory: {run_dir}")
        seen_dirs.add(run_dir)
        transcript = run_dir / "transcript.jsonl"
        turns = scheduled_turns(transcript)
        if int(row["turns"]) != len(turns):
            fail(f"canonical turn count mismatch at slot {slot}")
        result.append(
            {
                "slot": slot,
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": sha256(transcript),
                "turns": turns,
            }
        )
    return result


def actual_judge_identity() -> tuple[str, str]:
    judge_source = ROOT / "src/multi_turn_eval/judging/claude_judge.py"
    text = judge_source.read_text(encoding="utf-8")
    model_match = re.search(r'^JUDGE_MODEL\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version_match = re.search(
        r'^JUDGE_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE
    )
    if not model_match or not version_match:
        fail("could not pin judge identity from claude_judge.py")
    model = model_match.group(1)
    version = version_match.group(1)
    if model != EXPECTED_JUDGE_MODEL or version != EXPECTED_JUDGE_VERSION:
        fail(
            f"judge identity changed: model={model!r}, version={version!r}"
        )
    if "pro" in model.lower() or not model.startswith("claude-"):
        fail(f"disallowed judge model: {model}")
    return model, version


def source_hash_text() -> str:
    paths = (
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/judging/claude_judge.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
        Path(__file__).resolve(),
        CONFIG_PATH,
        FROZEN_PATH,
        CANONICAL_PATH,
    )
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in paths
    )


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
    if INPUTS_PATH.exists():
        existing = read_tsv(INPUTS_PATH)
        normalized = [{key: str(row[key]) for key in INPUT_FIELDS} for row in rows]
        if existing != normalized:
            fail("canonical inputs changed after judging was initialized")
    else:
        write_tsv(INPUTS_PATH, INPUT_FIELDS, rows)

    hashes = source_hash_text()
    if SOURCE_HASH_PATH.exists():
        if SOURCE_HASH_PATH.read_text(encoding="utf-8") != hashes:
            fail("judge or campaign source changed after judging was initialized")
    else:
        SOURCE_HASH_PATH.write_text(hashes, encoding="utf-8")


def validate_outputs(entry: dict[str, Any]) -> tuple[bool, str]:
    run_dir = entry["run_dir"]
    judged_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    analysis_path = run_dir / "claude_analysis.md"
    try:
        judged = read_jsonl(judged_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not analysis_path.is_file() or analysis_path.stat().st_size == 0:
            fail(f"missing claude_analysis.md: {run_dir}")
        judged_turns = [row.get("turn") for row in judged]
        if judged_turns != entry["turns"]:
            fail(
                f"judged turns {judged_turns} do not match transcript "
                f"{entry['turns']}"
            )
        for row in judged:
            scores = row.get("scores")
            if not isinstance(scores, dict):
                fail("judged row lacks scores")
            for key in (
                "tool_use_correct",
                "instruction_following",
                "kb_grounding",
            ):
                if not isinstance(scores.get(key), bool):
                    fail(f"judged row lacks boolean {key}")
        if summary.get("judge_model") != EXPECTED_JUDGE_MODEL:
            fail(f"unexpected judge model: {summary.get('judge_model')!r}")
        if summary.get("judge_version") != EXPECTED_JUDGE_VERSION:
            fail(f"unexpected judge version: {summary.get('judge_version')!r}")
        if summary.get("model_name") != EXPECTED_MODEL:
            fail(f"unexpected judged model: {summary.get('model_name')!r}")
        if summary.get("turns_scored") != len(entry["turns"]):
            fail("summary turns_scored mismatch")
        if sha256(entry["transcript"]) != entry["transcript_sha256"]:
            fail("judging mutated the transcript")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def attempt_count(slot: int) -> int:
    if not ATTEMPTS_PATH.exists():
        return 0
    return max(
        (
            int(row["attempt"])
            for row in read_tsv(ATTEMPTS_PATH)
            if int(row["slot"]) == slot
        ),
        default=0,
    )


def snapshot_invalid(entry: dict[str, Any], attempt: int) -> None:
    paths = [
        entry["run_dir"] / "claude_judged.jsonl",
        entry["run_dir"] / "claude_summary.json",
        entry["run_dir"] / "claude_analysis.md",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    destination = (
        INVALID_DIR
        / f"slot{entry['slot']:02d}-before-attempt{attempt:02d}-{int(time.time())}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, destination / path.name)
        path.unlink()


def run_judge(
    entry: dict[str, Any],
    *,
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[int, bool, str]:
    valid, error = validate_outputs(entry)
    if valid:
        return entry["slot"], True, "already valid"

    while True:
        attempt = attempt_count(entry["slot"]) + 1
        if attempt > max_attempts:
            return (
                entry["slot"],
                False,
                f"attempt cap exhausted ({max_attempts}); last validation: {error}",
            )
        snapshot_invalid(entry, attempt)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"slot{entry['slot']:02d}-attempt{attempt:02d}.log"
        started_at = utc_now()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(ROOT / ".venv/bin/multi-turn-eval"),
                    "judge",
                    entry["run_dir_text"],
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished_at = utc_now()
        valid, error = validate_outputs(entry)
        append_tsv(
            ATTEMPTS_PATH,
            ATTEMPT_FIELDS,
            {
                "slot": entry["slot"],
                "run_dir": entry["run_dir_text"],
                "attempt": attempt,
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_code": result.returncode,
                "valid": int(valid),
                "transcript_sha256": entry["transcript_sha256"],
                "judge_model": EXPECTED_JUDGE_MODEL,
                "judge_version": EXPECTED_JUDGE_VERSION,
                "log": str(log_path.relative_to(ROOT)),
                "error": error.replace("\t", " ").replace("\n", " "),
            },
        )
        if valid:
            return entry["slot"], True, f"valid after attempt {attempt}"
        if attempt >= max_attempts:
            return entry["slot"], False, error
        time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send Claude judge requests; default is read-only preflight",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="total durable cap per run, including earlier invocations",
    )
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        fail("--workers must be between 1 and 4")
    if args.max_attempts < 1:
        fail("--max-attempts must be positive")

    judge_model, judge_version = actual_judge_identity()
    entries = load_canonical(require_complete=args.execute)
    valid_count = sum(validate_outputs(entry)[0] for entry in entries)
    print(
        f"Preflight: canonical={len(entries)}/{TARGET}, valid_judgments="
        f"{valid_count}/{len(entries)}, judge={judge_model} ({judge_version})"
    )
    if not args.execute:
        print("Read-only preflight only. No judge request was made.")
        return 0
    campaign_log = CAMPAIGN_LOG.read_text(encoding="utf-8")
    if "RUN_COLLECTION_DONE total=30 none=30" not in campaign_log:
        fail("campaign.log lacks the completed N=30 collection marker")
    freeze_inputs(entries)

    incomplete = [entry for entry in entries if not validate_outputs(entry)[0]]
    results: list[tuple[int, bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_judge,
                entry,
                max_attempts=args.max_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            for entry in incomplete
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"slot {result[0]:02d}: {'OK' if result[1] else 'FAILED'} "
                f"{result[2]}"
            )

    failures = [result for result in results if not result[1]]
    final_valid = sum(validate_outputs(entry)[0] for entry in entries)
    if failures or final_valid != TARGET:
        print(
            f"Judging incomplete: valid={final_valid}/{TARGET}, "
            f"failed_slots={[row[0] for row in failures]}",
            file=sys.stderr,
        )
        return 2
    COMPLETE_PATH.write_text(
        json.dumps(
            {
                "campaign": json.loads(
                    CONFIG_PATH.read_text(encoding="utf-8")
                )["campaign_id"],
                "completed_at": utc_now(),
                "canonical_runs": TARGET,
                "judge_model": judge_model,
                "judge_version": judge_version,
                "canonical_inputs_sha256": sha256(INPUTS_PATH),
                "judge_source_sha256": sha256(SOURCE_HASH_PATH),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Judging complete: {TARGET}/{TARGET} valid outputs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
