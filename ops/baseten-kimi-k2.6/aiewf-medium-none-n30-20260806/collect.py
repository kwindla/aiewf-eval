#!/usr/bin/env python3
"""Concurrent, resumable, complete-only BaseTen Kimi K2.6 collector."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
SCHEDULE = HERE / "frozen-order.tsv"
ATTEMPTS = HERE / "attempts.tsv"
CANONICAL = HERE / "canonical.tsv"
SOURCE_HASHES = HERE / "source-sha256.txt"
CAMPAIGN_LOG = HERE / "campaign.log"
LOG_DIR = HERE / "logs"
PENDING_DIR = HERE / "pending"
LOCK = HERE / ".collection.lock"
RUN_ONE = ROOT / "ops/aiewf-campaign-template/run_one.py"

MODEL = "moonshotai/Kimi-K2.6"
ENDPOINT = "https://inference.baseten.co/v1"
ARM = "none"
TARGET = 30
N_TURNS = 30
CONCURRENCY = 1

ATTEMPT_FIELDS = (
    "slot",
    "attempt",
    "started_at",
    "finished_at",
    "exit_code",
    "run_dir",
    "scheduled_rows",
    "response_turns",
    "tool_calls",
    "token_rows",
    "end_session_turn",
    "classification",
    "log",
)
CANONICAL_FIELDS = (
    "slot",
    "attempt",
    "run_dir",
    "scheduled_rows",
    "response_turns",
    "tool_calls",
    "token_rows",
    "end_session_turn",
    "classification",
)

_log_lock = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with _log_lock:
        with CAMPAIGN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, fields: tuple[str, ...], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object JSON row {path}:{number}")
        rows.append(value)
    return rows


def validate_configuration(*, execute: bool) -> dict[str, Any]:
    config = read_json(CONFIG)
    exact = {
        "schema_version": 1,
        "campaign_id": "aiewf-medium-baseten-kimi-k2.6-none-n30-20260806",
        "benchmark": "aiwf_medium_context",
        "provider": "BaseTen Model API",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "service": "baseten",
        "pipeline": "text",
        "arm": ARM,
        "target_complete_conversations": TARGET,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "filler": None,
        "sampling": {
            "reasoning_effort": "none",
            "temperature": 0.6,
            "max_tokens": 8192,
        },
        "runtime": {
            "provider_endpoint_concurrency": CONCURRENCY,
            "enable_recovery": True,
            "dedupe_tool_calls": True,
            "tool_result_run_llm": False,
            "text_idle_timeout_seconds": 45,
            "conversation_timeout_seconds": 300,
            "inter_attempt_cooldown_seconds": 30,
            "max_attempts_per_slot": 5,
        },
        "paths": {"run_output_root": "runs/aiwf_medium_context"},
        "arms": {"none": {}},
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"frozen configuration mismatch for {key}: {config.get(key)!r}"
            )
    if config.get("smoke_gate", {}).get("passed") is not True and execute:
        raise RuntimeError("excluded smoke gate has not passed")
    seed = config.get("seed_runs")
    if not isinstance(seed, list) or len(seed) != 1:
        raise RuntimeError("configuration must identify exactly one seed run")
    if seed[0].get("slot") != "K26-01":
        raise RuntimeError("the frozen seed must be slot K26-01")
    return config


def validate_schedule() -> list[dict[str, str]]:
    if SCHEDULE.read_text(encoding="utf-8").splitlines()[0] != "slot\tarm":
        raise RuntimeError("unexpected schedule header")
    rows = read_tsv(SCHEDULE)
    expected = [f"K26-{index:02d}" for index in range(1, TARGET + 1)]
    if [row.get("slot") for row in rows] != expected:
        raise RuntimeError("schedule must contain K26-01 through K26-30 in order")
    if any(row.get("arm") != ARM for row in rows):
        raise RuntimeError("schedule contains a non-none arm")
    return rows


def source_hash_text() -> str:
    paths = (
        CONFIG,
        SCHEDULE,
        Path(__file__).resolve(),
        RUN_ONE,
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/pipelines/base.py",
        ROOT / "src/multi_turn_eval/pipelines/text.py",
        ROOT / "src/multi_turn_eval/recording/transcript_recorder.py",
        ROOT / "src/multi_turn_eval/services/baseten_logged.py",
        ROOT / "src/multi_turn_eval/services/lilac_logged.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
        ROOT / "benchmarks/aiwf_medium_context/prompts/system.py",
        ROOT / "benchmarks/aiwf_medium_context/data/knowledge_base.txt",
        ROOT / "benchmarks/_shared/__init__.py",
        ROOT / "benchmarks/_shared/tools.py",
        ROOT / "benchmarks/_shared/turns.py",
    )
    lines = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"source-integrity path is missing: {path}")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative(path)}")
    return "\n".join(lines) + "\n"


def verify_source_hashes() -> None:
    if SOURCE_HASHES.exists() and SOURCE_HASHES.read_text(encoding="utf-8") != source_hash_text():
        raise RuntimeError("campaign source hashes changed after collection began")


def validate_provenance(run_dir: Path) -> None:
    path = run_dir / "run.log"
    if not path.is_file():
        raise RuntimeError(f"run.log is missing: {run_dir}")
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        f"Using BaseTen with base_url={ENDPOINT}, model={MODEL}, "
        "reasoning_effort=none, enable_thinking=(unset), max_tokens=8192, "
        "temperature=0.6",
        "Text pipeline idle_timeout_secs=45.0",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise RuntimeError(f"runtime signature mismatch ({missing}): {run_dir}")
    if "MTE_FILLER_DOTS active:" in text:
        raise RuntimeError(f"filler is active in frozen no-filler run: {run_dir}")


def inspect_run(run_dir: Path) -> dict[str, int | str]:
    metrics: dict[str, int | str] = {
        "scheduled_rows": 0,
        "response_turns": 0,
        "tool_calls": 0,
        "token_rows": 0,
        "end_session_turn": -1,
        "classification": "incomplete_missing_artifacts",
    }
    transcript = run_dir / "transcript.jsonl"
    runtime_path = run_dir / "runtime.json"
    if not transcript.is_file() or not transcript.stat().st_size:
        return metrics
    try:
        rows = read_jsonl(transcript)
        runtime = read_json(runtime_path) if runtime_path.is_file() else {}
        validate_provenance(run_dir)
    except Exception:
        metrics["classification"] = "invalid_artifact_or_signature"
        return metrics

    scheduled = [row for row in rows if row.get("recovery_turn") is not True]
    turns = [row.get("turn") for row in scheduled]
    if turns != list(range(len(turns))) or len(turns) > N_TURNS:
        metrics["classification"] = "invalid_transcript_shape"
        return metrics
    if any(row.get("model_name") != MODEL for row in rows):
        metrics["classification"] = "invalid_model_identity"
        return metrics

    response_turns = 0
    tool_calls = 0
    token_rows = 0
    end_turn = -1
    for row in scheduled:
        calls = row.get("tool_calls") or []
        text = row.get("assistant_text")
        if (isinstance(text, str) and text.strip()) or calls:
            response_turns += 1
        if isinstance(calls, list):
            tool_calls += len(calls)
            if any(call.get("name") == "end_session" for call in calls):
                end_turn = max(end_turn, int(row["turn"]))
        if isinstance(row.get("tokens"), dict):
            token_rows += 1
    for row in rows:
        if row.get("recovery_turn") is not True:
            continue
        calls = row.get("tool_calls") or []
        if any(call.get("name") == "end_session" for call in calls):
            turn = row.get("turn")
            if isinstance(turn, int):
                end_turn = max(end_turn, turn)

    metrics.update(
        {
            "scheduled_rows": len(scheduled),
            "response_turns": response_turns,
            "tool_calls": tool_calls,
            "token_rows": token_rows,
            "end_session_turn": end_turn,
        }
    )
    if len(scheduled) != N_TURNS:
        metrics["classification"] = "incomplete_scheduled_turns"
    elif response_turns != N_TURNS:
        metrics["classification"] = "incomplete_response_turns"
    elif token_rows != N_TURNS:
        metrics["classification"] = "incomplete_token_accounting"
    elif runtime.get("model_name") != MODEL:
        metrics["classification"] = "invalid_runtime_model"
    elif runtime.get("status") != "completed" or runtime.get("valid") is not True:
        metrics["classification"] = "incomplete_runtime"
    else:
        metrics["classification"] = "strict_complete"
    return metrics


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    resolved.relative_to((ROOT / "runs/aiwf_medium_context").resolve())
    return resolved


def validate_manifests(schedule: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    expected_attempt_header = "\t".join(ATTEMPT_FIELDS)
    expected_canonical_header = "\t".join(CANONICAL_FIELDS)
    if ATTEMPTS.read_text(encoding="utf-8").splitlines()[0] != expected_attempt_header:
        raise RuntimeError("attempts.tsv header mismatch")
    if CANONICAL.read_text(encoding="utf-8").splitlines()[0] != expected_canonical_header:
        raise RuntimeError("canonical.tsv header mismatch")
    attempts = read_tsv(ATTEMPTS)
    canonical = read_tsv(CANONICAL)
    allowed_slots = {row["slot"] for row in schedule}
    if any(row.get("slot") not in allowed_slots for row in (*attempts, *canonical)):
        raise RuntimeError("manifest contains a slot outside the frozen schedule")
    keys: set[tuple[str, int]] = set()
    numbers: dict[str, list[int]] = {}
    for row in attempts:
        key = (row["slot"], int(row["attempt"]))
        if key in keys:
            raise RuntimeError(f"duplicate attempt: {key}")
        keys.add(key)
        numbers.setdefault(row["slot"], []).append(int(row["attempt"]))
    for slot, values in numbers.items():
        if values != list(range(1, len(values) + 1)):
            raise RuntimeError(f"attempt sequence is not contiguous for {slot}")
    if len({row["slot"] for row in canonical}) != len(canonical):
        raise RuntimeError("canonical.tsv contains duplicate slots")
    if len({row["run_dir"] for row in canonical}) != len(canonical):
        raise RuntimeError("canonical.tsv contains duplicate run directories")
    for row in canonical:
        key = (row["slot"], int(row["attempt"]))
        if key not in keys:
            raise RuntimeError(f"canonical row references missing attempt: {key}")
        current = inspect_run(resolve_run_dir(row["run_dir"]))
        if current["classification"] != "strict_complete":
            raise RuntimeError(f"canonical run is no longer complete: {row['slot']}")
        for field in (
            "scheduled_rows",
            "response_turns",
            "tool_calls",
            "token_rows",
            "end_session_turn",
        ):
            if int(row[field]) != int(current[field]):
                raise RuntimeError(f"canonical {field} mismatch for {row['slot']}")
        if row["classification"] != current["classification"]:
            raise RuntimeError(f"canonical classification mismatch for {row['slot']}")
    return attempts, canonical


def load_key(config: dict[str, Any]) -> str:
    credential = config["credential"]
    key = os.environ.get(credential["source_environment"])
    if key:
        return key
    dotenv_path = (ROOT / credential["fallback_dotenv"]).resolve()
    prefix = credential["fallback_key"] + "="
    if dotenv_path.is_file():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                value = line[len(prefix) :].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                if value:
                    return value
    raise RuntimeError("BASETEN_API_KEY is unavailable")


def child_environment(key: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("MTE_FILLER_") or name.startswith("MTE_BASETEN_"):
            env.pop(name, None)
    env.update(
        {
            "BASETEN_API_KEY": key,
            "BASETEN_BASE_URL": ENDPOINT,
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "0.6",
            "MTE_ENABLE_RECOVERY": "1",
            "MTE_DEDUPE_TOOL_CALLS": "1",
            "MTE_TOOL_RESULT_RUN_LLM": "0",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
        }
    )
    return env


def next_attempt(attempts: list[dict[str, str]], slot: str) -> int:
    return 1 + max(
        (int(row["attempt"]) for row in attempts if row["slot"] == slot),
        default=0,
    )


def make_run_dir(slot: str, attempt: int) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = (
        f"{timestamp}_moonshotai_Kimi-K2.6_none_{slot}_attempt{attempt:02d}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    return (ROOT / "runs/aiwf_medium_context" / name).resolve()


def pending_path(slot: str) -> Path:
    return PENDING_DIR / f"{slot}.json"


def run_attempt(
    *, slot: str, attempt: int, key: str, timeout_seconds: int, lock_fd: int
) -> dict[str, Any]:
    run_dir = make_run_dir(slot, attempt)
    log_path = LOG_DIR / f"{slot}-attempt-{attempt:02d}.log"
    pending = {
        "slot": slot,
        "attempt": attempt,
        "started_at": now(),
        "finished_at": "",
        "exit_code": "",
        "run_dir": relative(run_dir),
        "log": relative(log_path),
        "pid": None,
    }
    atomic_write(pending_path(slot), json.dumps(pending, indent=2, sort_keys=True) + "\n")
    command = (
        sys.executable,
        str(RUN_ONE),
        "--config",
        str(CONFIG),
        "--arm",
        ARM,
        "--run-dir",
        str(run_dir),
    )
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment(key),
            stdout=output,
            stderr=subprocess.STDOUT,
            pass_fds=(lock_fd,),
        )
        pending["pid"] = process.pid
        atomic_write(pending_path(slot), json.dumps(pending, indent=2, sort_keys=True) + "\n")
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 124
            output.write(f"\nCOLLECTOR_TIMEOUT seconds={timeout_seconds} at={now()}\n")
            output.flush()
            os.fsync(output.fileno())
    pending["finished_at"] = now()
    pending["exit_code"] = exit_code
    atomic_write(pending_path(slot), json.dumps(pending, indent=2, sort_keys=True) + "\n")
    return pending


def finalize_pending(pending: dict[str, Any]) -> bool:
    slot = str(pending["slot"])
    attempt = int(pending["attempt"])
    run_dir = resolve_run_dir(str(pending["run_dir"]))
    metrics = inspect_run(run_dir)
    pending_exit_code = pending.get("exit_code")
    row = {
        "slot": slot,
        "attempt": attempt,
        "started_at": pending.get("started_at", ""),
        "finished_at": pending.get("finished_at", "") or now(),
        "exit_code": (
            "recovered"
            if pending_exit_code is None or pending_exit_code == ""
            else pending_exit_code
        ),
        "run_dir": relative(run_dir),
        **metrics,
        "log": str(pending.get("log", "")),
    }
    append_tsv(ATTEMPTS, ATTEMPT_FIELDS, row)
    complete = metrics["classification"] == "strict_complete"
    if complete:
        append_tsv(
            CANONICAL,
            CANONICAL_FIELDS,
            {field: row[field] for field in CANONICAL_FIELDS},
        )
    pending_path(slot).unlink(missing_ok=True)
    log(
        f"finished slot={slot} attempt={attempt} rc={row['exit_code']} "
        f"classification={metrics['classification']} canonical={str(complete).lower()} "
        f"run_dir={row['run_dir']}"
    )
    return complete


def recover_pending(schedule: list[dict[str, str]]) -> None:
    allowed = {row["slot"] for row in schedule}
    for path in sorted(PENDING_DIR.glob("K26-*.json")):
        pending = read_json(path)
        slot = pending.get("slot")
        if slot not in allowed or path != pending_path(str(slot)):
            raise RuntimeError(f"invalid pending record: {path}")
        pid = pending.get("pid")
        if isinstance(pid, int) and Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"pending child process is unexpectedly still live: {pid}")
        attempts = read_tsv(ATTEMPTS)
        if int(pending["attempt"]) != next_attempt(attempts, str(slot)):
            raise RuntimeError(f"pending attempt is not next for {slot}")
        log(f"recovering pending slot={slot} attempt={pending['attempt']}")
        finalize_pending(pending)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = validate_configuration(execute=args.execute)
    schedule = validate_schedule()
    verify_source_hashes()
    attempts, canonical = validate_manifests(schedule)
    print(
        f"preflight ok: campaign={config['campaign_id']} canonical={len(canonical)}/{TARGET} "
        f"concurrency={CONCURRENCY}"
    )
    if not args.execute:
        print("read-only preflight; pass --execute to start or resume collection")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another collector or child holds the campaign lock") from exc
        if not SOURCE_HASHES.exists():
            atomic_write(SOURCE_HASHES, source_hash_text())
        recover_pending(schedule)
        attempts, canonical = validate_manifests(schedule)
        key = load_key(config)
        max_attempts = int(config["runtime"]["max_attempts_per_slot"])
        timeout_seconds = int(config["runtime"]["conversation_timeout_seconds"])
        cooldown_seconds = int(config["runtime"]["inter_attempt_cooldown_seconds"])
        log(f"collector start canonical={len(canonical)}/{TARGET} concurrency={CONCURRENCY}")

        while len(canonical) < TARGET:
            verify_source_hashes()
            attempts, canonical = validate_manifests(schedule)
            complete_slots = {row["slot"] for row in canonical}
            missing = [row["slot"] for row in schedule if row["slot"] not in complete_slots]
            assignments: list[tuple[str, int]] = []
            for slot in missing:
                attempt = next_attempt(attempts, slot)
                if attempt > max_attempts:
                    raise RuntimeError(
                        f"attempt cap exhausted for {slot}; review before increasing it"
                    )
                assignments.append((slot, attempt))
                if len(assignments) == CONCURRENCY:
                    break
            for slot, attempt in assignments:
                log(
                    f"provider cooldown seconds={cooldown_seconds} before "
                    f"slot={slot} attempt={attempt}"
                )
                time.sleep(cooldown_seconds)
                log(f"starting slot={slot} attempt={attempt}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {
                    pool.submit(
                        run_attempt,
                        slot=slot,
                        attempt=attempt,
                        key=key,
                        timeout_seconds=timeout_seconds,
                        lock_fd=lock_handle.fileno(),
                    ): (slot, attempt)
                    for slot, attempt in assignments
                }
                for future in concurrent.futures.as_completed(futures):
                    pending = future.result()
                    finalize_pending(pending)

        _, canonical = validate_manifests(schedule)
        log(f"campaign collection complete canonical={len(canonical)}/{TARGET}")
        print(f"collection complete: {len(canonical)} strict-complete conversations")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
