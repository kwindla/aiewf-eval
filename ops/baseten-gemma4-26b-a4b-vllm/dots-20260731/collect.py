#!/usr/bin/env python3
"""Collect the frozen paired Gemma 4 26B A4B no-filler/96-dot campaign.

The default invocation is a read-only local preflight.  ``--execute`` scales
the exact dedicated BaseTen deployment to one replica, runs the requested
frozen stage strictly sequentially, and scales it back to zero in ``finally``.
No judging or score-dependent decision is performed by this driver.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG_PATH = HERE / "configuration.json"
SCHEDULE_PATH = HERE / "frozen-order.tsv"
ATTEMPTS_PATH = HERE / "attempts.tsv"
CANONICAL_PATH = HERE / "canonical.tsv"
RESOURCE_EVENTS_PATH = HERE / "resource-events.tsv"
SOURCE_HASH_PATH = HERE / "source-sha256.txt"
PROMOTION_RECORD_PATH = HERE / "promotion-decision.json"
CAMPAIGN_LOG = HERE / "campaign.log"
LOG_DIR = HERE / "logs"
LOCK_PATH = HERE / ".collection.lock"

CAMPAIGN_ID = "aiewf-medium-gemma4-26b-a4b-dots-paired-20260731"
MODEL = "google/gemma-4-26B-A4B-it"
ENDPOINT = "https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1"
CONTROL_ENDPOINT = (
    "https://api.baseten.co/v1/models/qel1y223/deployments/qz4zpye"
)
ARMS = ("nofiller", "dots96")
N_TURNS = 30
INITIAL_SLOTS = 20
FULL_SLOTS = 60

ACTIVE_AUTOSCALING = {
    "min_replica": 1,
    "max_replica": 1,
    "autoscaling_window": 60,
    "scale_down_delay": 120,
    "concurrency_target": 1,
    "target_utilization_percentage": 70,
}
TEARDOWN_AUTOSCALING = {**ACTIVE_AUTOSCALING, "min_replica": 0}

ATTEMPT_FIELDS = (
    "slot",
    "pair",
    "stage",
    "arm",
    "attempt",
    "started_at",
    "finished_at",
    "exit_code",
    "run_dir",
    "scheduled_rows",
    "response_turns",
    "tool_calls",
    "end_session_turn",
    "classification",
    "transcript_sha256",
    "log",
)
CANONICAL_FIELDS = (
    "slot",
    "pair",
    "stage",
    "arm",
    "attempt",
    "run_dir",
    "scheduled_rows",
    "response_turns",
    "tool_calls",
    "end_session_turn",
    "classification",
    "transcript_sha256",
)
RESOURCE_FIELDS = (
    "timestamp",
    "event",
    "requested_min",
    "requested_max",
    "status",
    "active_replicas",
    "detail",
)
PROMOTION_TRIGGERS = {
    "ci_excludes_zero",
    "absolute_effect_ge_3_and_aligned_same_turn_recurs_ge_3",
    "completion_differs",
}
INFRASTRUCTURE_ZERO_RESPONSE = re.compile(
    r"DeadlineExceeded|ResourceExhausted|ReadTimeout|ConnectTimeout|"
    r"Connection(?:Error|Reset|Refused)|APIConnectionError|rate.?limit|"
    r"HTTP[/ ]+5\d\d|(?:^|\D)429(?:\D|$)|InternalServerError|"
    r"ServiceUnavailable|Upstream error",
    re.IGNORECASE | re.MULTILINE,
)


class CampaignInterrupted(RuntimeError):
    """Controlled signal interruption that still permits teardown."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise ValueError(message)


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


def append_tsv(
    path: Path,
    fields: tuple[str, ...],
    row: dict[str, Any],
) -> None:
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


def append_log(message: str) -> None:
    rendered = f"[{utc_now()}] {message}"
    print(rendered, flush=True)
    with CAMPAIGN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_resource_event(
    event: str,
    *,
    requested_min: int | str = "",
    requested_max: int | str = "",
    status: str = "",
    active_replicas: int | str = "",
    detail: str = "",
) -> None:
    append_tsv(
        RESOURCE_EVENTS_PATH,
        RESOURCE_FIELDS,
        {
            "timestamp": utc_now(),
            "event": event,
            "requested_min": requested_min,
            "requested_max": requested_max,
            "status": status,
            "active_replicas": active_replicas,
            "detail": detail.replace("\t", " ").replace("\n", " "),
        },
    )


def validate_headers() -> None:
    expected = (
        (ATTEMPTS_PATH, ATTEMPT_FIELDS),
        (CANONICAL_PATH, CANONICAL_FIELDS),
        (RESOURCE_EVENTS_PATH, RESOURCE_FIELDS),
    )
    for path, fields in expected:
        header = path.read_text(encoding="utf-8").splitlines()[0]
        if header != "\t".join(fields):
            fail(f"unexpected TSV header: {path}")


def validate_configuration() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "benchmark": "aiwf_medium_context",
        "provider": "BaseTen",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "control_endpoint": CONTROL_ENDPOINT,
        "deployment_id": "qz4zpye",
        "model_id": "qel1y223",
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
        },
        "schedule_seed": 42620260731,
        "initial_target_per_arm": 10,
        "promoted_target_per_arm": 30,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "max_attempts_per_slot": 4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(
                f"configuration mismatch for {key}: expected {value!r}, "
                f"found {config.get(key)!r}"
            )
    serving = config.get("serving") or {}
    if serving != {
        "vllm_version": "0.26.1rc1.dev77+g6f91edf96",
        "automatic_prefix_caching": True,
        "streaming": True,
        "automatic_tool_choice": True,
        "tool_call_parser": "gemma4",
        "reasoning_parser": "gemma4",
        "enable_thinking": False,
        "mtp": {
            "assistant_model": "google/gemma-4-26B-A4B-it-assistant",
            "num_speculative_tokens": 1,
        },
    }:
        fail("serving configuration does not match the frozen deployment")
    if config.get("autoscaling") != {
        "collection": ACTIVE_AUTOSCALING,
        "teardown": TEARDOWN_AUTOSCALING,
    }:
        fail("autoscaling configuration is not frozen")
    arms = config.get("arms") or {}
    if arms != {
        "nofiller": {"filler": None},
        "dots96": {
            "count": 96,
            "token": ".",
            "position": "suffix",
            "request_only": True,
        },
    }:
        fail("arm configuration is not the exact no-filler/96-dot contrast")
    return config


def validate_schedule() -> list[dict[str, str]]:
    rows = read_tsv(SCHEDULE_PATH)
    if len(rows) != FULL_SLOTS:
        fail(f"frozen schedule must contain {FULL_SLOTS} assignments")
    if [row["slot"] for row in rows] != [
        f"G4D-{index:02d}" for index in range(1, FULL_SLOTS + 1)
    ]:
        fail("frozen slots must be G4D-01 through G4D-60")
    for pair in range(1, 31):
        pair_rows = [row for row in rows if int(row["pair"]) == pair]
        if len(pair_rows) != 2 or {row["arm"] for row in pair_rows} != set(ARMS):
            fail(f"pair {pair} does not contain one nofiller and one dots96")
        expected_stage = "initial_n10" if pair <= 10 else "promoted_n30"
        if any(row["stage"] != expected_stage for row in pair_rows):
            fail(f"stage mismatch in pair {pair}")
        order = pair_rows[0]["pair_order"]
        if order not in {"nofiller-dots96", "dots96-nofiller"}:
            fail(f"invalid pair order in pair {pair}: {order}")
        if any(row["pair_order"] != order for row in pair_rows):
            fail(f"pair-order mismatch in pair {pair}")
        if [row["arm"] for row in pair_rows] != order.split("-"):
            fail(f"slot order does not match pair order in pair {pair}")
    for block in range(3):
        first_rows = rows[block * 20 : (block + 1) * 20 : 2]
        first_arms = [row["arm"] for row in first_rows]
        if first_arms.count("nofiller") != 5 or first_arms.count("dots96") != 5:
            fail(f"ten-pair block {block + 1} is not order-balanced")
    return rows


def validate_source_hashes() -> None:
    if not SOURCE_HASH_PATH.is_file():
        fail(f"source hash manifest is missing: {SOURCE_HASH_PATH}")
    for line in SOURCE_HASH_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.is_file():
            fail(f"hashed source is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            fail(
                f"source integrity failure for {relative}: "
                f"expected {expected}, found {actual}"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                fail(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run directory escapes repository root: {value}")
    return resolved


def tool_name(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def inspect_transcript(run_dir: Path) -> dict[str, int | str]:
    transcript = run_dir / "transcript.jsonl"
    if not transcript.is_file() or not transcript.stat().st_size:
        fail(f"missing or empty transcript: {transcript}")
    all_rows = read_jsonl(transcript)
    scheduled: dict[int, dict[str, Any]] = {}
    response_turns = 0
    tool_calls = 0
    scheduled_ends: list[int] = []
    recovery_end = False
    for row in all_rows:
        if row.get("model_name") != MODEL:
            fail(f"model mismatch in {transcript}: {row.get('model_name')!r}")
        calls = row.get("tool_calls") or []
        if row.get("recovery_turn") is True:
            recovery_end = recovery_end or any(
                tool_name(call) == "end_session" for call in calls
            )
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"invalid scheduled turn {turn!r}: {transcript}")
        if turn in scheduled:
            fail(f"duplicate scheduled turn {turn}: {transcript}")
        scheduled[turn] = row
        text = row.get("assistant_text")
        if (isinstance(text, str) and text.strip()) or calls:
            response_turns += 1
        tool_calls += len(calls) if isinstance(calls, list) else 0
        if any(tool_name(call) == "end_session" for call in calls):
            scheduled_ends.append(turn)
    if sorted(scheduled) != list(range(len(scheduled))):
        fail(f"scheduled turns are not a contiguous prefix: {transcript}")
    if response_turns < 1:
        fail(f"attempt has no valid model response: {transcript}")
    end_turn = max(scheduled_ends, default=-1)
    if recovery_end and end_turn < 0:
        end_turn = 30
    if len(scheduled) == N_TURNS and scheduled_ends == [29]:
        classification = "strict_complete"
    elif recovery_end:
        classification = "recovery_end_session"
    elif scheduled_ends:
        classification = "model_abort"
    else:
        classification = "incomplete_no_end_session"
    return {
        "scheduled_rows": len(scheduled),
        "response_turns": response_turns,
        "tool_calls": tool_calls,
        "end_session_turn": end_turn,
        "classification": classification,
        "transcript_sha256": sha256(transcript),
    }


def validate_run_provenance(run_dir: Path, arm: str) -> None:
    path = run_dir / "run.log"
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty run.log: {run_dir}")
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        f"Using vllm-openai with base_url={ENDPOINT}, model={MODEL}, "
        "thinking=False, thinking_budget=None, T=1.0, top_p=0.95, "
        "top_k=64, max_tokens=8192",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"runtime provenance missing {missing}: {run_dir}")
    filler_active = (
        "MTE_FILLER_DOTS active: 96 x '.' filler tokens, "
        "position=suffix (history left filler-free)"
    ) in text
    if arm == "dots96" and not filler_active:
        fail(f"96-dot activation evidence is missing: {run_dir}")
    if arm == "nofiller" and filler_active:
        fail(f"filler leaked into nofiller control: {run_dir}")


def validate_manifests(
    schedule: list[dict[str, str]],
) -> list[dict[str, str]]:
    validate_headers()
    canonical = read_tsv(CANONICAL_PATH)
    if [row["slot"] for row in canonical] != [
        row["slot"] for row in schedule[: len(canonical)]
    ]:
        fail("canonical manifest is not a contiguous frozen-schedule prefix")
    if len(canonical) > FULL_SLOTS:
        fail("canonical manifest exceeds the frozen schedule")
    seen_dirs: set[Path] = set()
    for row, assignment in zip(canonical, schedule):
        for field in ("slot", "pair", "stage", "arm"):
            if row[field] != assignment[field]:
                fail(f"canonical/frozen {field} mismatch at {assignment['slot']}")
        run_dir = resolve_run_dir(row["run_dir"])
        if run_dir in seen_dirs:
            fail(f"duplicate canonical run directory: {run_dir}")
        seen_dirs.add(run_dir)
        validation = inspect_transcript(run_dir)
        validate_run_provenance(run_dir, row["arm"])
        for field in (
            "scheduled_rows",
            "response_turns",
            "tool_calls",
            "end_session_turn",
        ):
            if int(row[field]) != int(validation[field]):
                fail(f"canonical {field} mismatch at {row['slot']}")
        for field in ("classification", "transcript_sha256"):
            if row[field] != validation[field]:
                fail(f"canonical {field} mismatch at {row['slot']}")
    return canonical


def selected_schedule(
    schedule: list[dict[str, str]], stage: str
) -> list[dict[str, str]]:
    return schedule[: INITIAL_SLOTS if stage == "initial" else FULL_SLOTS]


def validate_promotion_decision(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("campaign_id") != CAMPAIGN_ID:
        fail("promotion decision campaign_id mismatch")
    if payload.get("decision_after_n_per_arm") != 10:
        fail("promotion decision must use the frozen n=10/arm look")
    if payload.get("promote_to_n30") is not True:
        fail("full stage requires promote_to_n30=true")
    triggers = payload.get("triggered_rules")
    if (
        not isinstance(triggers, list)
        or not triggers
        or any(trigger not in PROMOTION_TRIGGERS for trigger in triggers)
    ):
        fail("promotion decision does not name a valid triggered rule")
    for field in ("aggregates_sha256", "included_runs_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            fail(f"promotion decision {field} must be 64 lowercase hex digits")
    decided_at = payload.get("decided_at")
    if not isinstance(decided_at, str) or "T" not in decided_at:
        fail("promotion decision lacks an ISO-8601 decided_at timestamp")
    return payload


def freeze_promotion_decision(source: Path) -> dict[str, Any]:
    payload = validate_promotion_decision(source)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if PROMOTION_RECORD_PATH.exists():
        existing = json.loads(PROMOTION_RECORD_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            fail("promotion decision changed after it was recorded")
    else:
        temporary = PROMOTION_RECORD_PATH.with_suffix(".json.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(PROMOTION_RECORD_PATH)
    return payload


def next_attempt(slot: str) -> int:
    attempts = [
        int(row["attempt"])
        for row in read_tsv(ATTEMPTS_PATH)
        if row["slot"] == slot
    ]
    return max(attempts, default=0) + 1


def load_key(key_file: Path) -> str:
    key = os.environ.get("BASETEN_API_KEY")
    if not key and key_file.is_file():
        key = dotenv_values(key_file).get("BASETEN_API_KEY")
    if not key:
        fail(
            "BASETEN_API_KEY is unavailable; set it or provide a dotenv "
            "file containing only the expected named key"
        )
    return str(key)


def api_json(
    url: str,
    key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Api-Key {key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        body = response.read()
    parsed = json.loads(body) if body else {}
    if not isinstance(parsed, dict):
        fail(f"BaseTen control API returned a non-object from {url}")
    return parsed


def set_autoscaling(key: str, settings: dict[str, Any], *, event: str) -> None:
    append_resource_event(
        f"{event}_REQUEST",
        requested_min=settings["min_replica"],
        requested_max=settings["max_replica"],
    )
    api_json(
        CONTROL_ENDPOINT + "/autoscaling_settings",
        key,
        "PATCH",
        settings,
    )
    append_resource_event(
        f"{event}_ACCEPTED",
        requested_min=settings["min_replica"],
        requested_max=settings["max_replica"],
    )


def deployment_state(key: str) -> tuple[str, int]:
    payload = api_json(CONTROL_ENDPOINT, key)
    status = str(payload.get("status", "UNKNOWN"))
    replicas = int(payload.get("active_replica_count", 0))
    return status, replicas


def wait_for_state(
    key: str,
    *,
    active: bool,
    attempts: int = 180,
    interval_seconds: float = 5.0,
) -> None:
    target = "ACTIVE" if active else "SCALED_TO_ZERO"
    last_status = "UNKNOWN"
    last_replicas = -1
    for attempt in range(1, attempts + 1):
        try:
            last_status, last_replicas = deployment_state(key)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_status = f"ERROR:{type(exc).__name__}"
            last_replicas = -1
        ready = (
            last_status == "ACTIVE" and last_replicas >= 1
            if active
            else last_status == "SCALED_TO_ZERO" and last_replicas == 0
        )
        if ready:
            append_resource_event(
                f"DEPLOYMENT_{target}",
                status=last_status,
                active_replicas=last_replicas,
                detail=f"poll_attempt={attempt}",
            )
            append_log(
                f"DEPLOYMENT_{target} status={last_status} "
                f"replicas={last_replicas} poll_attempt={attempt}"
            )
            return
        if attempt % 6 == 0:
            append_log(
                f"DEPLOYMENT_WAIT target={target} status={last_status} "
                f"replicas={last_replicas} poll_attempt={attempt}"
            )
        time.sleep(interval_seconds)
    fail(
        f"deployment did not reach {target}: "
        f"status={last_status}, replicas={last_replicas}"
    )


def child_environment(api_key: str, arm: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if (
            name.startswith("MTE_FILLER_")
            or name.startswith("MTE_VLLM_")
            or name in {"VLLM_BASE_URL", "VLLM_API_KEY", "BASETEN_API_KEY"}
        ):
            env.pop(name, None)
    env.update(
        {
            "VLLM_BASE_URL": ENDPOINT,
            "VLLM_API_KEY": api_key,
            "MTE_VLLM_THINKING": "0",
            "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_TOP_P": "0.95",
            "MTE_VLLM_TOP_K": "64",
            "MTE_VLLM_MAX_TOKENS": "8192",
            "MTE_ENABLE_RECOVERY": "1",
            "MTE_DEDUPE_TOOL_CALLS": "1",
            "MTE_TOOL_RESULT_RUN_LLM": "0",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
        }
    )
    if arm == "dots96":
        env.update(
            {
                "MTE_FILLER_DOTS": "96",
                "MTE_FILLER_TOKEN": ".",
                "MTE_FILLER_POSITION": "suffix",
            }
        )
    elif arm != "nofiller":
        fail(f"unknown arm: {arm}")
    return env


def candidate_run_dir(output: str) -> Path | None:
    matches = re.findall(r"^Output directory: (.+)$", output, flags=re.MULTILINE)
    return resolve_run_dir(matches[-1]) if matches else None


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_model_attempt(
    assignment: dict[str, str],
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, str, Path | None, bool]:
    command = [
        str(ROOT / ".venv/bin/multi-turn-eval"),
        "run",
        "aiwf_medium_context",
        "--model",
        MODEL,
        "--service",
        "vllm-openai",
        "--pipeline",
        "text",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_environment(api_key, assignment["arm"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    interrupted = False
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, output, candidate_run_dir(output), interrupted
    except subprocess.TimeoutExpired:
        terminate_process(process)
        output, _ = process.communicate()
        output += f"\nCOLLECTOR_TIMEOUT seconds={timeout_seconds}\n"
        return 124, output, candidate_run_dir(output), interrupted
    except CampaignInterrupted:
        interrupted = True
        terminate_process(process)
        output, _ = process.communicate()
        output += "\nCOLLECTOR_OPERATOR_INTERRUPTED\n"
        return 130, output, candidate_run_dir(output), interrupted


def install_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: FrameType | None) -> None:
        raise CampaignInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


def set_teardown_signal_guard(previous: dict[int, Any]) -> None:
    for signum in previous:
        signal.signal(signum, signal.SIG_IGN)


def restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def execute_collection(
    *,
    config: dict[str, Any],
    schedule: list[dict[str, str]],
    stage: str,
    api_key: str,
) -> int:
    target = selected_schedule(schedule, stage)
    canonical = validate_manifests(schedule)
    completed = {row["slot"] for row in canonical}
    if all(assignment["slot"] in completed for assignment in target):
        print(f"Stage {stage} is already complete; no BaseTen request was made.")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous_handlers = install_signal_handlers()
    lifecycle_started = False
    primary_error: BaseException | None = None
    teardown_error: BaseException | None = None
    try:
        lifecycle_started = True
        set_autoscaling(api_key, ACTIVE_AUTOSCALING, event="SCALE_UP")
        wait_for_state(api_key, active=True)
        for assignment in target:
            if assignment["slot"] in completed:
                continue
            slot = assignment["slot"]
            while slot not in completed:
                validate_source_hashes()
                attempt = next_attempt(slot)
                if attempt > int(config["max_attempts_per_slot"]):
                    fail(f"slot {slot} exhausted the four-attempt ceiling")
                started_at = utc_now()
                append_log(
                    f"RUN_START slot={slot} pair={assignment['pair']} "
                    f"stage={assignment['stage']} arm={assignment['arm']} "
                    f"attempt={attempt}"
                )
                exit_code, output, run_dir, interrupted = run_model_attempt(
                    assignment,
                    api_key,
                    int(config["runtime"]["conversation_timeout_seconds"]),
                )
                finished_at = utc_now()
                log_path = LOG_DIR / f"{slot}-attempt{attempt:02d}.log"
                log_path.write_text(output, encoding="utf-8")
                validation: dict[str, int | str] | None = None
                validation_error = ""
                if run_dir is not None:
                    try:
                        validation = inspect_transcript(run_dir)
                        validate_run_provenance(run_dir, assignment["arm"])
                    except Exception as exc:
                        validation_error = str(exc)
                if interrupted:
                    classification = "operator_interrupted"
                elif validation is not None:
                    classification = str(validation["classification"])
                elif INFRASTRUCTURE_ZERO_RESPONSE.search(output):
                    classification = "infra_zero_response_replaced"
                else:
                    classification = "zero_response_unclassified"
                values: dict[str, int | str] = {
                    "scheduled_rows": 0,
                    "response_turns": 0,
                    "tool_calls": 0,
                    "end_session_turn": -1,
                    "transcript_sha256": "",
                }
                if validation is not None:
                    values.update(validation)
                attempt_row = {
                    "slot": slot,
                    "pair": assignment["pair"],
                    "stage": assignment["stage"],
                    "arm": assignment["arm"],
                    "attempt": attempt,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "exit_code": exit_code,
                    "run_dir": (
                        str(run_dir.relative_to(ROOT)) if run_dir else ""
                    ),
                    **values,
                    "classification": classification,
                    "log": str(log_path.relative_to(ROOT)),
                }
                append_tsv(ATTEMPTS_PATH, ATTEMPT_FIELDS, attempt_row)
                append_log(
                    f"ATTEMPT_CLASSIFIED slot={slot} attempt={attempt} "
                    f"classification={classification} "
                    f"responses={values['response_turns']}"
                )
                if interrupted:
                    raise CampaignInterrupted(
                        f"operator interrupted slot {slot}; attempt recorded"
                    )
                if classification == "infra_zero_response_replaced":
                    continue
                if classification == "zero_response_unclassified":
                    fail(
                        f"slot {slot} has an unclassified zero-response "
                        f"attempt: {validation_error or 'no transcript'}"
                    )
                append_tsv(
                    CANONICAL_PATH,
                    CANONICAL_FIELDS,
                    {field: attempt_row[field] for field in CANONICAL_FIELDS},
                )
                completed.add(slot)
                append_log(
                    f"RUN_CANONICAL slot={slot} arm={assignment['arm']} "
                    f"classification={classification}"
                )
        counts = {
            arm: sum(row["arm"] == arm for row in read_tsv(CANONICAL_PATH))
            for arm in ARMS
        }
        marker = (
            "INITIAL_COLLECTION_DONE total=20 nofiller=10 dots96=10"
            if stage == "initial"
            else "FULL_COLLECTION_DONE total=60 nofiller=30 dots96=30"
        )
        append_log(marker)
        append_log(
            f"STAGE_DONE stage={stage} nofiller={counts['nofiller']} "
            f"dots96={counts['dots96']}"
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        set_teardown_signal_guard(previous_handlers)
        if lifecycle_started:
            try:
                set_autoscaling(
                    api_key,
                    TEARDOWN_AUTOSCALING,
                    event="TEARDOWN",
                )
                wait_for_state(api_key, active=False)
            except BaseException as exc:
                teardown_error = exc
                append_resource_event(
                    "TEARDOWN_FAILED",
                    requested_min=0,
                    requested_max=1,
                    detail=str(exc),
                )
                append_log(f"TEARDOWN_FAILED error={exc}")
        restore_signal_handlers(previous_handlers)

    if primary_error is not None:
        if teardown_error is not None:
            fail(
                f"collection failed ({primary_error}); teardown also failed "
                f"({teardown_error})"
            )
        raise primary_error
    if teardown_error is not None:
        raise teardown_error
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform BaseTen control and model requests; default is read-only",
    )
    parser.add_argument(
        "--stage",
        choices=("initial", "full"),
        default="initial",
        help="initial collects 10/arm; full requires a reviewed promotion decision",
    )
    parser.add_argument(
        "--decision-file",
        type=Path,
        help="reviewed, hash-linked promotion decision required for --stage full",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=ROOT.parent / "gb-benchmarks/.env",
        help="fallback dotenv file from which only BASETEN_API_KEY is read",
    )
    args = parser.parse_args()

    config = validate_configuration()
    schedule = validate_schedule()
    validate_source_hashes()
    canonical = validate_manifests(schedule)
    counts = Counter(row["arm"] for row in canonical)
    target_size = INITIAL_SLOTS if args.stage == "initial" else FULL_SLOTS
    print(
        f"Preflight OK: stage={args.stage}, canonical={len(canonical)}/60, "
        f"nofiller={counts.get('nofiller', 0)}, "
        f"dots96={counts.get('dots96', 0)}, target_slots={target_size}, "
        "provider_concurrency=1"
    )
    if not args.execute:
        print("Read-only preflight only. No BaseTen request was made.")
        return 0

    with LOCK_PATH.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another collection process holds .collection.lock")

        # Reload mutable state after the lock; the read-only preflight above may
        # have raced another process that was just finishing a slot.
        validate_source_hashes()
        canonical = validate_manifests(schedule)
        if args.stage == "full":
            if len(canonical) < INITIAL_SLOTS:
                fail("full stage requires the complete initial 10/arm stage")
            if args.decision_file is None:
                fail("--stage full requires --decision-file")
            decision = freeze_promotion_decision(args.decision_file)
            append_log(
                "PROMOTION_RECORDED triggers="
                + ",".join(decision["triggered_rules"])
            )
        elif args.decision_file is not None:
            fail("--decision-file is accepted only with --stage full")

        api_key = load_key(args.key_file)
        return execute_collection(
            config=config,
            schedule=schedule,
            stage=args.stage,
            api_key=api_key,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
