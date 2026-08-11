#!/usr/bin/env python3
"""Judge completed extension runs from immutable canonical snapshots.

This is an incremental companion to ``../judge_extension.py``.  It only
selects runs already committed to an extension campaign's ``canonical.tsv``,
freezes their transcript hashes in a snapshot, and then uses the frozen judge
implementation.  The final 120-run validation remains the responsibility of
``judge_extension.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import importlib.util
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LOCAL_OPS = HERE.parent
ROOT = HERE.parents[2]
DRIVER = LOCAL_OPS / "judge_extension.py"
SNAPSHOTS = HERE / "judging-snapshots"
TARGET = 120
SNAPSHOT_FIELDS = ("slot", "run_dir", "transcript_sha256", "scheduled_turns")


def load_driver():
    spec = importlib.util.spec_from_file_location("gemma4_extension_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension judge driver: {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a concurrently appended TSV only after two identical observations."""
    for _ in range(10):
        first = path.read_bytes() if path.is_file() else b""
        time.sleep(0.05)
        second = path.read_bytes() if path.is_file() else b""
        if first != second or (second and not second.endswith(b"\n")):
            continue
        if not second:
            return []
        text = second.decode("utf-8")
        return list(csv.DictReader(text.splitlines(), delimiter="\t"))
    raise RuntimeError(f"could not obtain a stable read of {path}")


def canonical_entries(judge: Any) -> list[dict[str, Any]]:
    rows = stable_read_tsv(judge.CANONICAL)
    if len(rows) > TARGET:
        raise RuntimeError(f"canonical cohort exceeds target: {len(rows)}/{TARGET}")
    slots: list[int] = []
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for row in rows:
        try:
            slot = int(row["slot"])
            run_dir_text = row["run_dir"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("invalid or partially written canonical row") from error
        slots.append(slot)
        run_dir = (ROOT / run_dir_text).resolve()
        run_dir.relative_to(ROOT.resolve())
        if run_dir in seen:
            raise RuntimeError(f"duplicate canonical run directory: {run_dir}")
        seen.add(run_dir)
        transcript = run_dir / "transcript.jsonl"
        turns = judge.scheduled_turns(run_dir)
        entries.append(
            {
                "slot": slot,
                "run_dir": run_dir,
                "run_dir_text": str(run_dir.relative_to(ROOT)),
                "transcript": transcript,
                "transcript_sha256": judge.sha256(transcript),
                "turns": turns,
            }
        )
    if slots != list(range(1, len(rows) + 1)):
        raise RuntimeError(
            f"canonical slots are not a contiguous prefix: found {slots[:5]}..."
        )
    return entries


def snapshot_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot": entry["slot"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": len(entry["turns"]),
        }
        for entry in entries
    ]


def freeze_snapshot(judge: Any, arm: str, entries: list[dict[str, Any]]) -> Path:
    path = SNAPSHOTS / arm / f"canonical-through-{len(entries):03d}.tsv"
    rows = snapshot_rows(entries)
    expected = [{key: str(row[key]) for key in SNAPSHOT_FIELDS} for row in rows]
    if path.exists():
        if judge.read_tsv(path) != expected:
            raise RuntimeError(f"existing canonical snapshot changed: {path}")
    else:
        judge.write_tsv(path, SNAPSHOT_FIELDS, rows)
    return path


def judge_snapshot(
    judge: Any,
    arm: str,
    entries: list[dict[str, Any]],
    *,
    workers: int,
    max_attempts: int,
) -> tuple[int, int]:
    snapshot = freeze_snapshot(judge, arm, entries)
    pending = [entry for entry in entries if not judge.validate_judgment(entry)[0]]
    existing = len(entries) - len(pending)
    print(
        f"snapshot arm={arm} canonical={len(entries)}/{TARGET} "
        f"valid={existing}/{len(entries)} pending={len(pending)} "
        f"path={snapshot.relative_to(ROOT)}",
        flush=True,
    )
    if not pending:
        return len(entries), existing

    judge.JUDGING.mkdir(parents=True, exist_ok=True)
    judge.LOGS.mkdir(parents=True, exist_ok=True)
    with (judge.JUDGING / ".judge.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another {arm} extension judge is active") from error
        failures: list[tuple[int, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(judge.judge_one, entry, max_attempts)
                for entry in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                slot, valid, detail = future.result()
                print(
                    f"judged arm={arm} slot={slot:03d} valid={int(valid)} {detail}",
                    flush=True,
                )
                if not valid:
                    failures.append((slot, detail))
        if failures:
            raise RuntimeError(f"unexpected judging failures: {failures}")

    valid = sum(judge.validate_judgment(entry)[0] for entry in entries)
    if valid != len(entries):
        raise RuntimeError(
            f"post-snapshot validation found only {valid}/{len(entries)} valid"
        )
    print(f"status arm={arm} canonical={len(entries)}/{TARGET} valid={valid}", flush=True)
    return len(entries), valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-cache", choices=("fp8", "bf16"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-batch", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.min_batch < 1:
        parser.error("--min-batch must be positive")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")

    driver = load_driver()
    judge = driver.load_judge_module(driver.CAMPAIGNS[args.kv_cache])
    watcher_dir = SNAPSHOTS / args.kv_cache
    watcher_dir.mkdir(parents=True, exist_ok=True)
    with (watcher_dir / ".watch.lock").open("w") as watcher_lock:
        try:
            fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"another {args.kv_cache} snapshot watcher is active") from error

        last_status: tuple[int, int] | None = None
        while True:
            entries = canonical_entries(judge)
            valid = sum(judge.validate_judgment(entry)[0] for entry in entries)
            status = (len(entries), valid)
            if status != last_status:
                print(
                    f"preflight arm={args.kv_cache} canonical={len(entries)}/{TARGET} "
                    f"valid={valid}/{len(entries)}",
                    flush=True,
                )
                last_status = status
            pending = len(entries) - valid
            should_run = args.execute and pending and (
                not args.watch
                or pending >= args.min_batch
                or len(entries) == TARGET
            )
            if should_run:
                status = judge_snapshot(
                    judge,
                    args.kv_cache,
                    entries,
                    workers=args.workers,
                    max_attempts=args.max_attempts,
                )
                last_status = status
            if not args.watch or (len(entries) == TARGET and status[1] == TARGET):
                break
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
