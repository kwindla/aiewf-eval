#!/usr/bin/env python3
"""Resumable immutable-transcript judging for a collected Qwen3.6 filler lane."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path("/home/khkramer/src/aiewf-eval")
DATA = ROOT / "docs/filler-study-data/qwen36-dots-2026-07-28"
LANES = ("qwen35-control", "qwen27-dots", "qwen35-dots")
LEDGER_FIELDS = [
    "slot",
    "run_dir",
    "attempt",
    "start_utc",
    "end_utc",
    "judge_rc",
    "valid",
    "transcript_sha256",
    "judge_model",
    "judge_version",
    "log",
    "error",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def observed_turns(run_dir: Path) -> set[int]:
    turns = set()
    for line in (run_dir / "transcript.jsonl").read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < 30
            and row.get("recovery_turn") is not True
        ):
            turns.add(turn)
    return turns


def validate_judgment(run_dir: Path) -> tuple[bool, str, str, str]:
    transcript = run_dir / "transcript.jsonl"
    judged = run_dir / "claude_judged.jsonl"
    summary = run_dir / "claude_summary.json"
    try:
        observed = observed_turns(run_dir)
        final = {}
        for line in judged.read_text().splitlines():
            row = json.loads(line)
            turn = row.get("turn")
            if isinstance(turn, int) and turn in observed:
                final[turn] = row
        if set(final) != observed:
            return False, "", "", f"coverage {sorted(final)} != {sorted(observed)}"
        for turn, row in final.items():
            scores = row.get("scores") or {}
            if not all(
                isinstance(scores.get(key), bool)
                for key in ("tool_use_correct", "instruction_following", "kb_grounding")
            ):
                return False, "", "", f"invalid score schema turn={turn}"
        metadata = json.loads(summary.read_text())
        if metadata.get("turns_scored") != len(observed):
            return False, "", "", "summary turns_scored mismatch"
        model = str(metadata.get("judge_model") or "")
        version = str(metadata.get("judge_version") or "")
        if not model or not version:
            return False, "", "", "missing judge provenance"
        return True, model, version, ""
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
        return False, "", "", f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", choices=LANES)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    state = DATA / "state" / args.lane
    canonical_path = state / "canonical.tsv"
    if not canonical_path.is_file():
        raise SystemExit(f"missing collected lane: {canonical_path}")
    canonical = read_tsv(canonical_path)
    if not canonical:
        raise SystemExit("canonical lane is empty")

    judging = state / "judging"
    logs = judging / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ledger = judging / "attempts.tsv"
    lock_stream = (judging / "driver.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another judge driver owns {args.lane}")

    ledger_lock = threading.Lock()

    def append_ledger(row: dict[str, object]) -> None:
        with ledger_lock:
            exists = ledger.is_file() and ledger.stat().st_size > 0
            with ledger.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS, delimiter="\t")
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())

    def judge_one(item: dict[str, str]) -> tuple[str, bool, str]:
        slot = item["slot"]
        run_dir = Path(item["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        transcript = run_dir / "transcript.jsonl"
        expected_hash = item["transcript_sha256"]
        if sha256(transcript) != expected_hash:
            return slot, False, "transcript hash mismatch before judging"
        valid, model, version, error = validate_judgment(run_dir)
        if valid:
            return slot, True, "existing valid judgment"
        prior = [
            row
            for row in read_tsv(ledger)
            if row.get("slot") == slot
        ]
        for attempt in range(len(prior) + 1, args.max_attempts + 1):
            start = utc()
            log_path = logs / f"{slot}-attempt{attempt:02d}.log"
            with log_path.open("w") as output:
                result = subprocess.run(
                    [str(ROOT / ".venv/bin/multi-turn-eval"), "judge", str(run_dir)],
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            end = utc()
            unchanged = sha256(transcript) == expected_hash
            valid, model, version, error = validate_judgment(run_dir)
            if not unchanged:
                valid = False
                error = "transcript hash changed during judging"
            append_ledger(
                {
                    "slot": slot,
                    "run_dir": str(run_dir),
                    "attempt": attempt,
                    "start_utc": start,
                    "end_utc": end,
                    "judge_rc": result.returncode,
                    "valid": int(valid),
                    "transcript_sha256": expected_hash,
                    "judge_model": model,
                    "judge_version": version,
                    "log": str(log_path.relative_to(ROOT)),
                    "error": error,
                }
            )
            if valid:
                return slot, True, f"valid attempt={attempt}"
            time.sleep(min(30, 2**attempt))
        return slot, False, error or "attempt ceiling exhausted"

    failures = []
    print(
        f"{utc()} JUDGE_START lane={args.lane} runs={len(canonical)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_one, item): item["slot"] for item in canonical}
        for future in as_completed(futures):
            slot, valid, detail = future.result()
            print(
                f"{utc()} JUDGE_RESULT slot={slot} valid={int(valid)} detail={detail}",
                flush=True,
            )
            if not valid:
                failures.append((slot, detail))

    if failures:
        print(f"{utc()} JUDGE_INCOMPLETE failures={failures}", flush=True)
        return 1
    complete = {
        "lane": args.lane,
        "runs": len(canonical),
        "completed_utc": utc(),
        "transcripts_immutable": True,
    }
    (judging / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(f"{utc()} JUDGE_COMPLETE lane={args.lane} runs={len(canonical)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
