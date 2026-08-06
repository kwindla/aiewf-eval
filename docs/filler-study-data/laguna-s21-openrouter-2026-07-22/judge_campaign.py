#!/usr/bin/env python3
"""Judge the final Laguna S 2.1 30/30 campaign, resumably and serially."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATE = HERE / "state"
MODEL = "laguna_s21"
ARMS = {"nofiller", "dots96"}
N_TURNS = 30
MAX_JUDGE_ATTEMPTS = 3
SCHEDULES = (
    (
        HERE / "schedule.tsv",
        "ece7b3e83708f018627c78343c74db97642683f1adc77a4d77526ce80970886e",
    ),
    (
        HERE / "schedule-dots-topup.tsv",
        "6521d0be0ab91bc3f64a631b4635e17de2e38dcfcec536cccb1a50aab0da6491",
    ),
    (
        HERE / "schedule-n30-topup.tsv",
        "7ea9b6e3dfc53d104aca9d91eafdb4487623a8862a62c2aca3ae78b836d259e7",
    ),
)
RUN_COMPLETION_SENTINELS = (
    STATE / "RUNS_COMPLETE",
    STATE / "dots-topup" / "RUNS_COMPLETE",
    STATE / "n30-topup" / "RUNS_COMPLETE",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        if new:
            writer.writeheader()
        writer.writerow(row)


def observed_turns(run_dir: Path) -> set[int]:
    turns: set[int] = set()
    transcript = run_dir / "transcript.jsonl"
    if not transcript.is_file() or not transcript.stat().st_size:
        return turns
    try:
        for line in transcript.read_text().splitlines():
            row = json.loads(line)
            turn = row.get("turn")
            if (
                isinstance(turn, int)
                and 0 <= turn < N_TURNS
                and row.get("recovery_turn") is not True
            ):
                if turn in turns:
                    return set()
                turns.add(turn)
    except (json.JSONDecodeError, OSError):
        return set()
    return turns


def valid_judgment(run_dir: Path) -> bool:
    judged = run_dir / "claude_judged.jsonl"
    summary = run_dir / "claude_summary.json"
    observed = observed_turns(run_dir)
    if not observed or not all(
        path.is_file() and path.stat().st_size for path in (judged, summary)
    ):
        return False
    try:
        rows = [json.loads(line) for line in judged.read_text().splitlines()]
        if len(rows) != len(observed):
            return False
        final: dict[int, dict] = {}
        for row in rows:
            turn = row.get("turn")
            if not isinstance(turn, int) or turn in final:
                return False
            final[turn] = row
        if set(final) != observed:
            return False
        for row in final.values():
            scores = row.get("scores") or {}
            if not all(
                isinstance(scores.get(key), bool)
                for key in (
                    "tool_use_correct",
                    "instruction_following",
                    "kb_grounding",
                )
            ):
                return False
        meta = json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError, TypeError):
        return False
    return (
        meta.get("turns_scored") == len(observed)
        and bool(meta.get("judge_model"))
        and bool(meta.get("judge_version"))
    )


def load_manifest() -> list[tuple[str, Path]]:
    schedule: list[dict[str, str]] = []
    for schedule_path, expected_hash in SCHEDULES:
        if sha256(schedule_path) != expected_hash:
            raise RuntimeError(f"frozen schedule changed: {schedule_path.name}")
        schedule.extend(read_tsv(schedule_path))
    missing_sentinels = [
        path.relative_to(HERE) for path in RUN_COMPLETION_SENTINELS if not path.is_file()
    ]
    if missing_sentinels:
        raise RuntimeError(
            "run completion sentinel(s) absent; refusing to judge a partial campaign: "
            f"{missing_sentinels}"
        )
    manifest = read_tsv(STATE / "manifest.tsv")
    expected = {row["slot"]: row for row in schedule}
    if len(expected) != 60 or len(schedule) != 60:
        raise RuntimeError("frozen schedule union must contain 60 unique slots")
    expected_arm_counts = {
        arm: sum(row["arm"] == arm for row in schedule) for arm in ARMS
    }
    if expected_arm_counts != {"nofiller": 30, "dots96": 30}:
        raise RuntimeError(f"schedule arm counts are invalid: {expected_arm_counts}")
    included: list[tuple[str, Path]] = []
    seen_slots: set[str] = set()
    seen_dirs: set[Path] = set()
    for row in manifest:
        slot = row.get("slot", "")
        if slot in seen_slots or slot not in expected:
            raise RuntimeError(f"duplicate or unexpected manifest slot: {slot!r}")
        assignment = expected[slot]
        if row.get("model") != MODEL or row.get("arm") not in ARMS:
            raise RuntimeError(f"manifest policy failure in {slot}")
        if (
            row["arm"] != assignment["arm"]
            or assignment["model"] != MODEL
            or assignment["requested_model"] != "poolside/laguna-s-2.1"
            or assignment["service"] != "openrouter"
        ):
            raise RuntimeError(f"manifest/schedule mismatch in {slot}")
        run_dir = Path(row["run_dir"])
        run_dir = (run_dir if run_dir.is_absolute() else ROOT / run_dir).resolve()
        try:
            run_dir.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"run outside repository: {run_dir}") from exc
        if run_dir in seen_dirs:
            raise RuntimeError(f"duplicate included run: {run_dir}")
        if not observed_turns(run_dir):
            raise RuntimeError(f"missing, malformed, or duplicate-turn transcript: {run_dir}")
        seen_slots.add(slot)
        seen_dirs.add(run_dir)
        included.append((slot, run_dir))
    if seen_slots != set(expected):
        raise RuntimeError(f"manifest is incomplete; missing={sorted(set(expected) - seen_slots)}")
    return included


def main() -> None:
    included = load_manifest()
    logs = STATE / "judge-logs"
    logs.mkdir(parents=True, exist_ok=True)
    attempts_path = STATE / "judge-attempts.tsv"
    # The stage-2 n=10 sentinel intentionally remains as historical evidence.
    # Use a distinct sentinel so final analysis cannot mistake it for proof
    # that the 40 newly added runs have also been judged.
    complete_path = STATE / "N30_JUDGING_COMPLETE"
    fields = [
        "slot",
        "run_dir",
        "attempt",
        "start_utc",
        "end_utc",
        "judge_rc",
        "valid",
        "transcript_sha256",
        "log",
    ]

    with (STATE / "judge.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another Laguna judge campaign is active") from exc

        for slot, run_dir in included:
            if valid_judgment(run_dir):
                print(f"valid existing judgment slot={slot} run={run_dir}", flush=True)
                continue
            prior = [row for row in read_tsv(attempts_path) if row["slot"] == slot]
            next_attempt = len(prior) + 1
            while next_attempt <= MAX_JUDGE_ATTEMPTS:
                transcript = run_dir / "transcript.jsonl"
                transcript_hash = sha256(transcript)
                started = now()
                proc = subprocess.run(
                    ["uv", "run", "multi-turn-eval", "judge", str(run_dir)],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if sha256(transcript) != transcript_hash:
                    raise RuntimeError(f"judge mutated immutable transcript: {run_dir}")
                log_path = logs / f"{slot}-attempt-{next_attempt}.log"
                log_path.write_text(proc.stdout)
                valid = proc.returncode == 0 and valid_judgment(run_dir)
                append_tsv(
                    attempts_path,
                    fields,
                    {
                        "slot": slot,
                        "run_dir": run_dir.relative_to(ROOT),
                        "attempt": next_attempt,
                        "start_utc": started,
                        "end_utc": now(),
                        "judge_rc": proc.returncode,
                        "valid": int(valid),
                        "transcript_sha256": transcript_hash,
                        "log": log_path.relative_to(ROOT),
                    },
                )
                print(
                    f"judge slot={slot} attempt={next_attempt} "
                    f"rc={proc.returncode} valid={valid}",
                    flush=True,
                )
                if valid:
                    break
                next_attempt += 1
            if not valid_judgment(run_dir):
                raise RuntimeError(f"judge failed after retries: {run_dir}")

        complete_path.touch()
        print("N30_JUDGING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
