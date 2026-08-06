#!/usr/bin/env python3
"""Preflight or resumably judge one completed paired Gemma dots stage.

The default invocation is read-only. ``--execute`` judges the exact canonical
prefix for ``initial`` (10 pairs) or ``full`` (30 pairs), with no more than two
Claude workers. It never calls BaseTen or launches collection.
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
SOURCE_HASH_PATH = JUDGING_DIR / "judge-source-sha256.txt"
ATTEMPTS_PATH = JUDGING_DIR / "judge-attempts.tsv"
LOG_DIR = JUDGING_DIR / "logs"
INVALID_DIR = JUDGING_DIR / "invalid-output-snapshots"
LOCK_PATH = HERE / ".judge.lock"

EXPECTED_MODEL = collect.MODEL
EXPECTED_JUDGE_MODEL = "claude-opus-4-5"
EXPECTED_JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
STAGE_SLOTS = {"initial": collect.INITIAL_SLOTS, "full": collect.FULL_SLOTS}

INPUT_FIELDS = (
    "slot",
    "pair",
    "arm",
    "run_dir",
    "transcript_sha256",
    "scheduled_turns",
)
ATTEMPT_FIELDS = (
    "slot",
    "pair",
    "arm",
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
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
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
    temporary.replace(path)


def append_tsv(
    path: Path,
    fields: tuple[str, ...],
    row: dict[str, Any],
) -> None:
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or not path.stat().st_size
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
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def input_path(stage: str) -> Path:
    return JUDGING_DIR / f"canonical-inputs-{stage}.tsv"


def complete_path(stage: str) -> Path:
    return JUDGING_DIR / f"COMPLETE-{stage}.json"


def validate_headers() -> None:
    expected = (
        (input_path("initial"), INPUT_FIELDS),
        (input_path("full"), INPUT_FIELDS),
        (ATTEMPTS_PATH, ATTEMPT_FIELDS),
    )
    for path, fields in expected:
        if not path.is_file() or not path.stat().st_size:
            fail(f"missing judge ledger: {path}")
        header = path.read_text(encoding="utf-8").splitlines()[0]
        if header != "\t".join(fields):
            fail(f"unexpected judge ledger header: {path}")


def actual_judge_identity() -> tuple[str, str]:
    source = ROOT / "src/multi_turn_eval/judging/claude_judge.py"
    text = source.read_text(encoding="utf-8")
    model_match = re.search(r'^JUDGE_MODEL\s*=\s*"([^"]+)"', text, re.MULTILINE)
    version_match = re.search(
        r'^JUDGE_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE
    )
    if not model_match or not version_match:
        fail("could not identify the repository Claude judge")
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
        HERE / "protocol.md",
        HERE / "collect.py",
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/judging/claude_judge.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
        ROOT / "benchmarks/aiwf_medium_context/prompts/system.py",
        ROOT / "benchmarks/_shared/turns.py",
    )
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in paths
    )


def scheduled_turns(transcript: Path) -> list[int]:
    turns: list[int] = []
    for row in read_jsonl(transcript):
        if row.get("model_name") != EXPECTED_MODEL:
            fail(f"wrong model in {transcript}: {row.get('model_name')!r}")
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < collect.N_TURNS:
            fail(f"invalid scheduled turn {turn!r}: {transcript}")
        turns.append(turn)
    if turns != list(range(len(turns))) or not turns:
        fail(f"scheduled turns are not a nonempty contiguous prefix: {transcript}")
    return turns


def load_stage_entries(stage: str) -> list[dict[str, Any]]:
    validate_headers()
    collect.validate_configuration()
    schedule = collect.validate_schedule()
    collect.validate_source_hashes()
    canonical = collect.validate_manifests(schedule)
    target = STAGE_SLOTS[stage]
    if len(canonical) != target:
        fail(
            f"stage {stage} must be fully and exactly collected; "
            f"found {len(canonical)}/{target} canonical rows"
        )
    marker = (
        "INITIAL_COLLECTION_DONE total=20 nofiller=10 dots96=10"
        if stage == "initial"
        else "FULL_COLLECTION_DONE total=60 nofiller=30 dots96=30"
    )
    if not collect.CAMPAIGN_LOG.is_file() or marker not in collect.CAMPAIGN_LOG.read_text(
        encoding="utf-8"
    ):
        fail(f"campaign.log lacks completed-stage marker: {marker}")

    result: list[dict[str, Any]] = []
    for row in canonical:
        run_dir = collect.resolve_run_dir(row["run_dir"])
        transcript = run_dir / "transcript.jsonl"
        turns = scheduled_turns(transcript)
        if len(turns) != int(row["scheduled_rows"]):
            fail(f"canonical scheduled-turn mismatch at {row['slot']}")
        transcript_hash = sha256(transcript)
        if transcript_hash != row["transcript_sha256"]:
            fail(f"canonical transcript hash mismatch at {row['slot']}")
        result.append(
            {
                "slot": row["slot"],
                "pair": int(row["pair"]),
                "arm": row["arm"],
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": transcript_hash,
                "turns": turns,
                "classification": row["classification"],
            }
        )
    return result


def freeze_inputs(stage: str, entries: list[dict[str, Any]]) -> None:
    proposed = [
        {
            "slot": entry["slot"],
            "pair": entry["pair"],
            "arm": entry["arm"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": len(entry["turns"]),
        }
        for entry in entries
    ]
    path = input_path(stage)
    existing = read_tsv(path)
    normalized = [
        {field: str(row[field]) for field in INPUT_FIELDS} for row in proposed
    ]
    if existing and existing != normalized:
        fail(f"{stage} judge inputs changed after they were frozen")
    if not existing:
        write_tsv(path, INPUT_FIELDS, proposed)

    hashes = source_hash_text()
    if SOURCE_HASH_PATH.exists():
        if SOURCE_HASH_PATH.read_text(encoding="utf-8") != hashes:
            fail("judge or campaign source changed after judging was initialized")
    else:
        SOURCE_HASH_PATH.write_text(hashes, encoding="utf-8")


def verify_frozen_entry(entry: dict[str, Any]) -> None:
    if sha256(entry["transcript"]) != entry["transcript_sha256"]:
        fail(f"frozen transcript changed for {entry['slot']}")
    if not SOURCE_HASH_PATH.is_file():
        fail("judge source hashes have not been frozen")
    if SOURCE_HASH_PATH.read_text(encoding="utf-8") != source_hash_text():
        fail("judge or campaign source changed after judging was initialized")


def validate_outputs(entry: dict[str, Any]) -> tuple[bool, str]:
    run_dir = entry["run_dir"]
    try:
        judged = read_jsonl(run_dir / "claude_judged.jsonl")
        summary = json.loads(
            (run_dir / "claude_summary.json").read_text(encoding="utf-8")
        )
        analysis = run_dir / "claude_analysis.md"
        if not analysis.is_file() or not analysis.stat().st_size:
            fail(f"missing or empty claude_analysis.md: {run_dir}")
        if [row.get("turn") for row in judged] != entry["turns"]:
            fail(f"judged turns do not match transcript for {entry['slot']}")
        for row in judged:
            scores = row.get("scores")
            if not isinstance(scores, dict):
                fail(f"judged row lacks scores for {entry['slot']}")
            for key in (
                "turn_taking",
                "tool_use_correct",
                "instruction_following",
                "kb_grounding",
            ):
                if not isinstance(scores.get(key), bool):
                    fail(f"judged row lacks boolean {key} for {entry['slot']}")
        if summary.get("judge_model") != EXPECTED_JUDGE_MODEL:
            fail(f"unexpected judge model for {entry['slot']}")
        if summary.get("judge_version") != EXPECTED_JUDGE_VERSION:
            fail(f"unexpected judge version for {entry['slot']}")
        if summary.get("model_name") != EXPECTED_MODEL:
            fail(f"unexpected judged model for {entry['slot']}")
        if summary.get("turns_scored") != len(entry["turns"]):
            fail(f"summary turns_scored mismatch for {entry['slot']}")
        if sha256(entry["transcript"]) != entry["transcript_sha256"]:
            fail(f"judging mutated the transcript for {entry['slot']}")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def load_anthropic_key(key_file: Path | None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    candidates = [key_file] if key_file is not None else []
    candidates.extend((ROOT / ".env", ROOT.parent / "gb-benchmarks/.env"))
    if not key:
        for path in candidates:
            if path is not None and path.is_file():
                key = dotenv_values(path).get("ANTHROPIC_API_KEY")
                if key:
                    break
    if not key:
        fail("ANTHROPIC_API_KEY is unavailable")
    return str(key)


def judge_environment(anthropic_key: str) -> dict[str, str]:
    allowed = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env["ANTHROPIC_API_KEY"] = anthropic_key
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
        if row["judge_model"] != EXPECTED_JUDGE_MODEL:
            fail(f"judge retry identity changed for {entry['slot']}")
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
            return (
                entry["slot"],
                False,
                f"attempt cap exhausted ({max_attempts}): {error}",
            )
        snapshot_invalid(entry, attempt)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{entry['slot']}-attempt{attempt:02d}.log"
        started_at = utc_now()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(ROOT / ".venv/bin/multi-turn-eval"),
                    "judge",
                    entry["run_dir_text"],
                    "--judge-model",
                    EXPECTED_JUDGE_MODEL,
                ],
                cwd=ROOT,
                env=judge_environment(anthropic_key),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished_at = utc_now()
        verify_frozen_entry(entry)
        valid, error = validate_outputs(entry)
        append_tsv(
            ATTEMPTS_PATH,
            ATTEMPT_FIELDS,
            {
                "slot": entry["slot"],
                "pair": entry["pair"],
                "arm": entry["arm"],
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
                "error": (error or "-").replace("\t", " ").replace("\n", " "),
            },
        )
        if valid:
            return entry["slot"], True, f"valid after attempt {attempt}"
        if attempt >= max_attempts:
            return entry["slot"], False, error
        time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))


def write_complete(stage: str, entries: list[dict[str, Any]]) -> None:
    inputs = input_path(stage)
    payload = {
        "campaign": collect.CAMPAIGN_ID,
        "stage": stage,
        "completed_at": utc_now(),
        "canonical_runs": len(entries),
        "canonical_pairs": len(entries) // 2,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "judge_version": EXPECTED_JUDGE_VERSION,
        "canonical_inputs_sha256": sha256(inputs),
        "judge_source_sha256": sha256(SOURCE_HASH_PATH),
        "transcript_sha256": {
            entry["slot"]: entry["transcript_sha256"] for entry in entries
        },
    }
    path = complete_path(stage)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_SLOTS), required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send Claude judge requests; default is read-only",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        fail("--workers must be 1 or 2")
    if args.max_attempts < 1 or args.retry_delay_seconds < 0:
        fail("attempt count must be positive and retry delay nonnegative")

    identity = actual_judge_identity()
    entries = load_stage_entries(args.stage)
    valid = sum(validate_outputs(entry)[0] for entry in entries)
    print(
        f"Preflight: stage={args.stage}, canonical={len(entries)}/"
        f"{STAGE_SLOTS[args.stage]}, valid_judgments={valid}/{len(entries)}, "
        f"judge={identity[0]} ({identity[1]}), max_workers=2"
    )
    if not args.execute:
        print("Read-only preflight only. No Claude or BaseTen request was made.")
        return 0

    with LOCK_PATH.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another paired-dots judge owns .judge.lock")

        entries = load_stage_entries(args.stage)
        freeze_inputs(args.stage, entries)
        anthropic_key = load_anthropic_key(args.key_file)
        incomplete = [
            entry for entry in entries if not validate_outputs(entry)[0]
        ]
        results: list[tuple[str, bool, str]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = [
                pool.submit(
                    run_judge,
                    entry,
                    anthropic_key=anthropic_key,
                    max_attempts=args.max_attempts,
                    retry_delay_seconds=args.retry_delay_seconds,
                )
                for entry in incomplete
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"{result[0]}: {'OK' if result[1] else 'FAILED'} "
                    f"{result[2]}"
                )

        failures = [result for result in results if not result[1]]
        final_valid = sum(validate_outputs(entry)[0] for entry in entries)
        if failures or final_valid != len(entries):
            print(
                f"Judging incomplete: valid={final_valid}/{len(entries)}, "
                f"failed_slots={[row[0] for row in failures]}",
                file=sys.stderr,
            )
            return 2
        write_complete(args.stage, entries)
        print(
            f"Judging complete: stage={args.stage}, "
            f"valid={len(entries)}/{len(entries)}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
