#!/usr/bin/env python3
"""Strictly sequential, resumable 30+30 Inkling Small AIEWF collector."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
SCHEDULE = HERE / "frozen-order.tsv"
ATTEMPTS = HERE / "attempts.tsv"
CANONICAL = HERE / "canonical.tsv"
SOURCE_HASHES = HERE / "source-sha256.txt"
CAMPAIGN_LOG = HERE / "campaign.log"
LOG_DIR = HERE / "logs"
LOCK = HERE / ".collection.lock"

MODEL = "thinkingmachines/inkling-small"
ENDPOINT = "https://inference.baseten.co/v1"
ARMS = {"none", "low"}
TARGET_PER_ARM = 30
N_TURNS = 30

ATTEMPT_FIELDS = (
    "slot", "pair", "arm", "attempt", "started_at", "finished_at",
    "exit_code", "run_dir", "scheduled_rows", "response_turns",
    "end_session_turn", "classification", "log",
)
CANONICAL_FIELDS = (
    "slot", "pair", "arm", "attempt", "run_dir", "scheduled_rows",
    "response_turns", "end_session_turn", "classification",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with CAMPAIGN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_hashes() -> None:
    for line in SOURCE_HASHES.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source integrity failure for {relative}: {actual} != {expected}"
            )


def validate_configuration(*, execute: bool) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "benchmark": "aiwf_medium_context",
        "provider": "BaseTen Model API",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "service": "baseten",
        "pipeline": "text",
        "arms": ["none", "low"],
        "target_valid_conversations_per_arm": TARGET_PER_ARM,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "schedule_seed": 20260731,
        "filler": None,
        "max_attempts_per_slot": 4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"configuration mismatch for {key}: {config.get(key)!r}")
    if config.get("sampling") != {"temperature": 1.0, "max_tokens": 16384}:
        raise RuntimeError("sampling settings are not frozen")
    runtime = config.get("runtime") or {}
    if runtime != {
        "provider_endpoint_concurrency": 1,
        "enable_recovery": True,
        "dedupe_tool_calls": True,
        "tool_result_run_llm": False,
        "text_idle_timeout_seconds": 45,
    }:
        raise RuntimeError("runtime settings are not frozen")
    if execute and not all((config.get("smoke_gate") or {}).get(f"{arm}_passed") is True for arm in ARMS):
        raise RuntimeError("both excluded smoke gates must pass before --execute")
    return config


def validate_schedule() -> list[dict[str, str]]:
    rows = read_tsv(SCHEDULE)
    if len(rows) != 60:
        raise RuntimeError(f"schedule must contain 60 assignments, found {len(rows)}")
    if [row["slot"] for row in rows] != [f"IS-{i:02d}" for i in range(1, 61)]:
        raise RuntimeError("schedule slots must be IS-01 through IS-60")
    for pair in range(1, 31):
        pair_rows = [row for row in rows if int(row["pair"]) == pair]
        if len(pair_rows) != 2 or {row["arm"] for row in pair_rows} != ARMS:
            raise RuntimeError(f"pair {pair} does not contain exactly none and low")
        order = pair_rows[0]["pair_order"]
        if any(row["pair_order"] != order for row in pair_rows):
            raise RuntimeError(f"pair-order mismatch for pair {pair}")
        if [row["arm"] for row in pair_rows] != order.split("-"):
            raise RuntimeError(f"pair sequence mismatch for pair {pair}")
    for block in range(5):
        pair_rows = rows[block * 12:(block + 1) * 12:2]
        orders = [row["pair_order"] for row in pair_rows]
        if orders.count("none-low") != 3 or orders.count("low-none") != 3:
            raise RuntimeError(f"block {block + 1} is not order-balanced")
    return rows


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def read_transcript(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "transcript.jsonl"
    if not path.is_file() or not path.stat().st_size:
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid transcript JSON {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object transcript row {path}:{line_number}")
        rows.append(row)
    return rows


def validate_run(run_dir: Path, arm: str) -> dict[str, int | str]:
    rows = read_transcript(run_dir)
    scheduled = [row for row in rows if row.get("recovery_turn") is not True]
    turns = [row.get("turn") for row in scheduled]
    if turns != list(range(len(turns))) or len(turns) > N_TURNS:
        raise RuntimeError(f"scheduled turns are not a contiguous prefix: {run_dir}")
    response_turns = 0
    end_turn = -1
    for row in scheduled:
        if row.get("model_name") != MODEL:
            raise RuntimeError(f"model mismatch in {run_dir}: {row.get('model_name')!r}")
        calls = row.get("tool_calls") or []
        text = row.get("assistant_text")
        if (isinstance(text, str) and text.strip()) or calls:
            response_turns += 1
        if any(call.get("name") == "end_session" for call in calls):
            end_turn = max(end_turn, int(row["turn"]))
    recovery_end = any(
        any(call.get("name") == "end_session" for call in row.get("tool_calls") or [])
        for row in rows
        if row.get("recovery_turn") is True
    )
    if recovery_end and end_turn < 0:
        end_turn = 30

    run_log = run_dir / "run.log"
    if not run_log.is_file():
        raise RuntimeError(f"run log is missing: {run_dir}")
    log_text = run_log.read_text(encoding="utf-8")
    signature = (
        f"Using BaseTen with base_url={ENDPOINT}, model={MODEL}, "
        f"reasoning_effort={arm}, enable_thinking=(unset), "
        "max_tokens=16384, temperature=1.0"
    )
    required = (
        signature,
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    if any(item not in log_text for item in required):
        raise RuntimeError(f"runtime signature mismatch: {run_dir}")
    if "MTE_FILLER_" in log_text or "MTE_FILLER_DOTS active:" in log_text:
        raise RuntimeError(f"filler leaked into campaign run: {run_dir}")

    if response_turns == 0:
        classification = "zero_response"
    elif end_turn == 29 and len(scheduled) == 30:
        classification = "strict_complete"
    elif recovery_end:
        classification = "recovery_end_session"
    elif end_turn >= 0:
        classification = "model_abort"
    else:
        classification = "incomplete_no_end_session"
    return {
        "scheduled_rows": len(scheduled),
        "response_turns": response_turns,
        "end_session_turn": end_turn,
        "classification": classification,
    }


def objective_infrastructure_failure(output: str) -> bool:
    patterns = (
        r"Pipeline failed", r"Idle timeout detected", r"ReadTimeout",
        r"ConnectTimeout", r"Connection(?:Error|Reset|Refused)",
        r"rate.?limit", r"HTTP[/ ]+[45][0-9][0-9]", r"status.?429",
        r"APIError", r"InternalServerError", r"ServiceUnavailable",
        r"EngineCore", r"Upstream error", r"Traceback",
    )
    return any(re.search(pattern, output, flags=re.IGNORECASE) for pattern in patterns)


def load_key() -> str:
    key = os.environ.get("BASETEN_API_KEY")
    if not key:
        key = dotenv_values(ROOT.parent / "gb-benchmarks/.env").get("BASETEN_API_KEY")
    if not key:
        raise RuntimeError("BASETEN_API_KEY is unavailable")
    return str(key)


def run_attempt(arm: str, api_key: str) -> tuple[int, str, Path | None]:
    env = os.environ.copy()
    env.update({
        "BASETEN_API_KEY": api_key,
        "BASETEN_BASE_URL": ENDPOINT,
        "MTE_BASETEN_REASONING_EFFORT": arm,
        "MTE_BASETEN_MAX_TOKENS": "16384",
        "MTE_BASETEN_TEMPERATURE": "1.0",
        "MTE_ENABLE_RECOVERY": "1",
        "MTE_DEDUPE_TOOL_CALLS": "1",
        "MTE_TOOL_RESULT_RUN_LLM": "0",
        "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
    })
    for name in (
        "MTE_BASETEN_ENABLE_THINKING", "MTE_FILLER_DOTS", "MTE_FILLER_TOKEN",
        "MTE_FILLER_POSITION",
    ):
        env.pop(name, None)
    proc = subprocess.run(
        [
            str(ROOT / ".venv/bin/multi-turn-eval"), "run", "aiwf_medium_context",
            "--model", MODEL, "--service", "baseten", "--pipeline", "text",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    matches = re.findall(r"^Output directory: (.+)$", proc.stdout, flags=re.MULTILINE)
    run_dir = resolve_run_dir(matches[-1]) if matches else None
    return proc.returncode, proc.stdout, run_dir


def preflight(*, execute: bool) -> tuple[dict[str, Any], list[dict[str, str]]]:
    config = validate_configuration(execute=execute)
    schedule = validate_schedule()
    validate_source_hashes()
    if ATTEMPTS.read_text(encoding="utf-8").splitlines()[0] != "\t".join(ATTEMPT_FIELDS):
        raise RuntimeError("attempts.tsv header mismatch")
    if CANONICAL.read_text(encoding="utf-8").splitlines()[0] != "\t".join(CANONICAL_FIELDS):
        raise RuntimeError("canonical.tsv header mismatch")
    canonical = read_tsv(CANONICAL)
    if len({row["slot"] for row in canonical}) != len(canonical):
        raise RuntimeError("canonical slots are not unique")
    expected_prefix = [row["slot"] for row in schedule[:len(canonical)]]
    if [row["slot"] for row in canonical] != expected_prefix:
        raise RuntimeError("canonical assignments are not a schedule prefix")
    for row in canonical:
        current = validate_run(resolve_run_dir(row["run_dir"]), row["arm"])
        for field in ("scheduled_rows", "response_turns", "end_session_turn"):
            if int(row[field]) != int(current[field]):
                raise RuntimeError(f"canonical {field} mismatch for {row['slot']}")
        if row["classification"] != current["classification"]:
            raise RuntimeError(f"canonical classification mismatch for {row['slot']}")
    return config, schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, schedule = preflight(execute=args.execute)
    canonical = read_tsv(CANONICAL)
    counts = {arm: sum(row["arm"] == arm for row in canonical) for arm in ARMS}
    print(f"preflight ok: canonical={len(canonical)}/60 none={counts['none']} low={counts['low']}")
    if not args.execute:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another collector owns the campaign lock") from exc

        api_key = load_key()
        completed = {row["slot"] for row in read_tsv(CANONICAL)}
        for assignment in schedule:
            slot = assignment["slot"]
            if slot in completed:
                continue
            pair = assignment["pair"]
            arm = assignment["arm"]
            attempts = [row for row in read_tsv(ATTEMPTS) if row["slot"] == slot]
            while len(attempts) < int(config["max_attempts_per_slot"]):
                validate_source_hashes()
                attempt = len(attempts) + 1
                started = now()
                log(f"starting slot={slot} pair={pair} arm={arm} attempt={attempt}")
                rc, output, run_dir = run_attempt(arm, api_key)
                finished = now()
                console_log = LOG_DIR / f"{slot}-attempt-{attempt}.log"
                console_log.write_text(output, encoding="utf-8")
                if run_dir is None:
                    metrics = {
                        "scheduled_rows": 0, "response_turns": 0,
                        "end_session_turn": -1, "classification": "zero_response",
                    }
                else:
                    metrics = validate_run(run_dir, arm)
                classification = str(metrics["classification"])
                if classification == "zero_response" and objective_infrastructure_failure(output):
                    classification = "infra_zero_response_replaced"
                attempt_row = {
                    "slot": slot, "pair": pair, "arm": arm, "attempt": attempt,
                    "started_at": started, "finished_at": finished, "exit_code": rc,
                    "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else "",
                    "scheduled_rows": metrics["scheduled_rows"],
                    "response_turns": metrics["response_turns"],
                    "end_session_turn": metrics["end_session_turn"],
                    "classification": classification,
                    "log": str(console_log.relative_to(ROOT)),
                }
                append_tsv(ATTEMPTS, ATTEMPT_FIELDS, attempt_row)
                attempts.append({key: str(value) for key, value in attempt_row.items()})
                if classification == "infra_zero_response_replaced":
                    log(f"replacing objective zero-response infrastructure failure slot={slot}")
                    continue
                if classification == "zero_response":
                    raise RuntimeError(f"unclassified zero-response attempt requires review: {slot}")
                canonical_row = {field: attempt_row[field] for field in CANONICAL_FIELDS}
                append_tsv(CANONICAL, CANONICAL_FIELDS, canonical_row)
                log(
                    f"counted slot={slot} arm={arm} class={classification} "
                    f"rows={metrics['scheduled_rows']} responses={metrics['response_turns']}"
                )
                completed.add(slot)
                break
            if slot not in completed:
                raise RuntimeError(f"slot {slot} exhausted the attempt ceiling")

    canonical = read_tsv(CANONICAL)
    counts = {arm: sum(row["arm"] == arm for row in canonical) for arm in ARMS}
    log(f"COLLECTION_DONE total={len(canonical)} none={counts['none']} low={counts['low']}")


if __name__ == "__main__":
    main()
