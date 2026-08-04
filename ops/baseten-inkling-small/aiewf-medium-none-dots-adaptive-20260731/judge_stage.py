#!/usr/bin/env python3
"""Preflight or resumably judge one completed Inkling Small dots stage.

The default invocation is read-only. ``--execute`` judges only canonical dot
transcripts, with at most two workers. The child environment contains the
Anthropic API key but no BaseTen credential or setting.
"""

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

from dotenv import dotenv_values

import collect


HERE = Path(__file__).resolve().parent
ROOT = collect.ROOT
JUDGING_DIR = HERE / "judging"
INPUTS_PATH = JUDGING_DIR / "canonical-inputs.tsv"
SOURCE_HASH_PATH = JUDGING_DIR / "judge-source-sha256.txt"
ATTEMPTS_PATH = JUDGING_DIR / "judge-attempts.tsv"
LOG_DIR = JUDGING_DIR / "logs"
INVALID_DIR = JUDGING_DIR / "invalid-output-snapshots"
LOCK = HERE / ".judge.lock"

EXPECTED_MODEL = collect.MODEL
EXPECTED_JUDGE_MODEL = "claude-opus-4-5"
EXPECTED_JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"

INPUT_FIELDS = ("slot", "run_dir", "transcript_sha256", "scheduled_turns")
ATTEMPT_FIELDS = (
    "slot", "run_dir", "attempt", "started_at", "finished_at", "exit_code",
    "valid", "transcript_sha256", "judge_model", "judge_version", "log", "error",
)

_ledger_lock = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise RuntimeError(message)


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
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with _ledger_lock:
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
                extrasaction="ignore",
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON at {path}:{line_number}: {exc}")
        if not isinstance(row, dict):
            fail(f"non-object row at {path}:{line_number}")
        rows.append(row)
    return rows


def actual_judge_identity() -> tuple[str, str]:
    source = ROOT / "src/multi_turn_eval/judging/claude_judge.py"
    text = source.read_text(encoding="utf-8")
    model_match = re.search(r'^JUDGE_MODEL\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version_match = re.search(r'^JUDGE_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not model_match or not version_match:
        fail("could not identify the repository judge")
    identity = (model_match.group(1), version_match.group(1))
    if identity != (EXPECTED_JUDGE_MODEL, EXPECTED_JUDGE_VERSION):
        fail(f"judge identity changed: {identity}")
    if "pro" in identity[0].lower() or not identity[0].startswith("claude-"):
        fail(f"disallowed judge model: {identity[0]}")
    return identity


def source_hash_text() -> str:
    paths = (
        Path(__file__).resolve(),
        HERE / "configuration.json",
        HERE / "frozen-order.tsv",
        HERE / "collect.py",
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/judging/claude_judge.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
        ROOT / "benchmarks/aiwf_medium_context/prompts/system.py",
        ROOT / "benchmarks/_shared/turns.py",
    )
    return "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in paths)


def load_stage_entries(stage: int) -> list[dict[str, Any]]:
    collect.validate_configuration()
    schedule = collect.validate_schedule()
    collect.validate_source_hashes()
    canonical = collect.validate_manifests(schedule)
    if len(canonical) != stage:
        fail(f"stage {stage} must be fully and exactly collected; found {len(canonical)}")
    marker = f"DOT_STAGE_DONE stage={stage} canonical={stage} control=30"
    if marker not in collect.CAMPAIGN_LOG.read_text(encoding="utf-8"):
        fail(f"campaign.log lacks completed-stage marker: {marker}")
    if collect.validate_or_freeze_control(execute=False) != 30:
        fail("the 30-run primary none control is not frozen")

    result: list[dict[str, Any]] = []
    for row in canonical:
        run_dir = collect.resolve_repo_path(row["run_dir"])
        transcript = run_dir / "transcript.jsonl"
        turns = [
            record["turn"] for record in collect.read_transcript(run_dir)
            if record.get("recovery_turn") is not True
        ]
        if turns != list(range(len(turns))):
            fail(f"scheduled turns are not a contiguous prefix: {run_dir}")
        result.append(
            {
                "slot": row["slot"],
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": sha256(transcript),
                "turns": turns,
            }
        )
    return result


def freeze_inputs(entries: list[dict[str, Any]]) -> None:
    proposed = [
        {
            "slot": entry["slot"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": len(entry["turns"]),
        }
        for entry in entries
    ]
    existing = read_tsv(INPUTS_PATH)
    if existing:
        if len(existing) > len(proposed) or existing != [
            {key: str(row[key]) for key in INPUT_FIELDS}
            for row in proposed[: len(existing)]
        ]:
            fail("dot judge inputs changed after they were frozen")
    if existing != [
        {key: str(row[key]) for key in INPUT_FIELDS} for row in proposed
    ]:
        write_tsv(INPUTS_PATH, INPUT_FIELDS, proposed)

    hashes = source_hash_text()
    if SOURCE_HASH_PATH.exists():
        if SOURCE_HASH_PATH.read_text(encoding="utf-8") != hashes:
            fail("judge source changed after judging was initialized")
    else:
        SOURCE_HASH_PATH.write_text(hashes, encoding="utf-8")


def verify_frozen_entry(entry: dict[str, Any]) -> None:
    if sha256(entry["transcript"]) != entry["transcript_sha256"]:
        fail(f"frozen transcript changed for {entry['slot']}")
    if not SOURCE_HASH_PATH.is_file():
        fail("judge source hashes have not been frozen")
    if SOURCE_HASH_PATH.read_text(encoding="utf-8") != source_hash_text():
        fail("judge source changed after judging was initialized")


def validate_outputs(entry: dict[str, Any]) -> tuple[bool, str]:
    run_dir = entry["run_dir"]
    try:
        judged = read_jsonl(run_dir / "claude_judged.jsonl")
        summary = json.loads((run_dir / "claude_summary.json").read_text(encoding="utf-8"))
        analysis = run_dir / "claude_analysis.md"
        if not analysis.is_file() or not analysis.stat().st_size:
            fail(f"missing claude_analysis.md: {run_dir}")
        if [row.get("turn") for row in judged] != entry["turns"]:
            fail(f"judged turns do not match transcript for {entry['slot']}")
        for row in judged:
            scores = row.get("scores")
            if not isinstance(scores, dict):
                fail("judged row lacks scores")
            for key in (
                "turn_taking", "tool_use_correct", "instruction_following", "kb_grounding",
            ):
                if not isinstance(scores.get(key), bool):
                    fail(f"judged row lacks boolean {key}")
        if summary.get("judge_model") != EXPECTED_JUDGE_MODEL:
            fail("unexpected judge model")
        if summary.get("judge_version") != EXPECTED_JUDGE_VERSION:
            fail("unexpected judge version")
        if summary.get("model_name") != EXPECTED_MODEL:
            fail("unexpected judged model")
        if summary.get("turns_scored") != len(entry["turns"]):
            fail("summary turns_scored mismatch")
        if sha256(entry["transcript"]) != entry["transcript_sha256"]:
            fail("judging mutated the transcript")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def load_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        for dotenv_path in (ROOT / ".env", ROOT.parent / "gb-benchmarks/.env"):
            if dotenv_path.is_file():
                key = dotenv_values(dotenv_path).get("ANTHROPIC_API_KEY")
                if key:
                    break
    if not key:
        fail("ANTHROPIC_API_KEY is unavailable")
    return str(key)


def judge_environment(anthropic_key: str) -> dict[str, str]:
    allowed = (
        "PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env["ANTHROPIC_API_KEY"] = anthropic_key
    # Prevent cli.py's load_dotenv() from repopulating provider credentials.
    env["PYTHON_DOTENV_DISABLED"] = "1"
    return env


def attempt_count(entry: dict[str, Any]) -> int:
    with _ledger_lock:
        rows = read_tsv(ATTEMPTS_PATH)
    matching = [row for row in rows if row["slot"] == entry["slot"]]
    for row in matching:
        if row["run_dir"] != entry["run_dir_text"]:
            fail(f"judge retry run changed for {entry['slot']}")
        if row["transcript_sha256"] != entry["transcript_sha256"]:
            fail(f"judge retry transcript changed for {entry['slot']}")
    return max((int(row["attempt"]) for row in matching), default=0)


def snapshot_invalid(entry: dict[str, Any], attempt: int) -> None:
    paths = [
        entry["run_dir"] / "claude_judged.jsonl",
        entry["run_dir"] / "claude_summary.json",
        entry["run_dir"] / "claude_analysis.md",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    destination = INVALID_DIR / (
        f"{entry['slot']}-before-attempt{attempt:02d}-{int(time.time())}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, destination / path.name)
        path.unlink()


def run_judge(
    entry: dict[str, Any],
    *,
    anthropic_key: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[str, bool, str]:
    valid, error = validate_outputs(entry)
    if valid:
        return entry["slot"], True, "already valid"
    while True:
        verify_frozen_entry(entry)
        attempt = attempt_count(entry) + 1
        if attempt > max_attempts:
            return entry["slot"], False, f"attempt cap exhausted ({max_attempts}): {error}"
        snapshot_invalid(entry, attempt)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{entry['slot']}-attempt{attempt:02d}.log"
        started = now()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(ROOT / ".venv/bin/multi-turn-eval"),
                    "judge", entry["run_dir_text"],
                    "--judge-model", EXPECTED_JUDGE_MODEL,
                ],
                cwd=ROOT,
                env=judge_environment(anthropic_key),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished = now()
        verify_frozen_entry(entry)
        valid, error = validate_outputs(entry)
        append_tsv(
            ATTEMPTS_PATH,
            ATTEMPT_FIELDS,
            {
                "slot": entry["slot"],
                "run_dir": entry["run_dir_text"],
                "attempt": attempt,
                "started_at": started,
                "finished_at": finished,
                "exit_code": result.returncode,
                "valid": int(valid),
                "transcript_sha256": entry["transcript_sha256"],
                "judge_model": EXPECTED_JUDGE_MODEL,
                "judge_version": EXPECTED_JUDGE_VERSION,
                "log": str(log_path.relative_to(ROOT)),
                "error": (error or "-").replace("\t", " ").replace("\n", " "),
            },
        )
        if valid:
            return entry["slot"], True, f"valid after attempt {attempt}"
        if attempt >= max_attempts:
            return entry["slot"], False, error
        time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))


def complete_path(stage: int) -> Path:
    return JUDGING_DIR / f"COMPLETE-stage-{stage}.json"


def write_complete(stage: int, entries: list[dict[str, Any]]) -> None:
    payload = {
        "campaign": json.loads((HERE / "configuration.json").read_text())["campaign_id"],
        "stage": stage,
        "completed_at": now(),
        "canonical_dots": stage,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "judge_version": EXPECTED_JUDGE_VERSION,
        "canonical_inputs_sha256": sha256(INPUTS_PATH),
        "judge_source_sha256": sha256(SOURCE_HASH_PATH),
        "transcript_sha256": {entry["slot"]: entry["transcript_sha256"] for entry in entries},
    }
    path = complete_path(stage)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=collect.STAGES, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        fail("--workers must be 1 or 2")
    if args.max_attempts < 1 or args.retry_delay_seconds < 0:
        fail("attempt count must be positive and retry delay nonnegative")

    identity = actual_judge_identity()
    entries = load_stage_entries(args.stage)
    valid = sum(validate_outputs(entry)[0] for entry in entries)
    print(
        f"preflight: stage={args.stage} dots={len(entries)}/{args.stage} "
        f"valid_judgments={valid}/{args.stage} judge={identity[0]} ({identity[1]})"
    )
    if not args.execute:
        print("Read-only preflight only. No Claude or BaseTen request was made.")
        return 0

    with LOCK.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fail("another dots judge owns the lock")
        entries = load_stage_entries(args.stage)
        freeze_inputs(entries)
        key = load_anthropic_key()
        incomplete = [entry for entry in entries if not validate_outputs(entry)[0]]
        results: list[tuple[str, bool, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_judge,
                    entry,
                    anthropic_key=key,
                    max_attempts=args.max_attempts,
                    retry_delay_seconds=args.retry_delay_seconds,
                )
                for entry in incomplete
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"{result[0]}: {'OK' if result[1] else 'FAILED'} {result[2]}")
        failures = [result for result in results if not result[1]]
        final_valid = sum(validate_outputs(entry)[0] for entry in entries)
        if failures or final_valid != args.stage:
            print(
                f"judging incomplete: valid={final_valid}/{args.stage}, "
                f"failures={[row[0] for row in failures]}",
                file=sys.stderr,
            )
            return 2
        write_complete(args.stage, entries)
        print(f"judging complete: {args.stage}/{args.stage} valid dots outputs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
