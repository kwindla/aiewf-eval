#!/usr/bin/env python3
"""Resumable, sequential BaseTen runner for the frozen Qwen3.6 filler lanes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path("/home/khkramer/src/aiewf-eval")
DATA = ROOT / "docs/filler-study-data/qwen36-dots-2026-07-28"
ENV_FILE = Path("/home/khkramer/src/gb-benchmarks/.env")
SOURCE_MANIFEST = DATA / "source-manifest.sha256"

MODELS = {
    "qwen36_27b": {
        "model": "Qwen/Qwen3.6-27B",
        "endpoint": "https://model-w67n482q.api.baseten.co/deployment/wxpnlg5/sync/v1",
        "control": "https://api.baseten.co/v1/models/w67n482q/deployments/wxpnlg5",
    },
    "qwen36_35b": {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "endpoint": "https://model-qzkm8mpq.api.baseten.co/deployment/qe20zvr/sync/v1",
        "control": "https://api.baseten.co/v1/models/qzkm8mpq/deployments/qe20zvr",
    },
}

LANES = {
    "qwen35-control": ("schedule-qwen35-control.tsv", "qwen36_35b"),
    "qwen27-dots": ("schedule-qwen27-dots.tsv", "qwen36_27b"),
    "qwen35-dots": ("schedule-qwen35-dots.tsv", "qwen36_35b"),
}

INFRA_RE = re.compile(
    r"DeadlineExceeded|ResourceExhausted|ReadTimeout|ConnectTimeout|"
    r"Connection(?:Error|Reset|Refused)|rate.?limit|HTTP[/ ]+5\\d\\d|"
    r"(?:^|\\D)429(?:\\D|$)|InternalServerError|ServiceUnavailable|"
    r"Upstream error",
    re.IGNORECASE | re.MULTILINE,
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_key() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("BASETEN_API_KEY="):
            value = line.split("=", 1)[1]
            if value:
                return value
    raise RuntimeError("BASETEN_API_KEY is missing")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    for raw in SOURCE_MANIFEST.read_text().splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = ROOT / relative.strip()
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source integrity failure: {relative}: expected {expected}, got {actual}"
            )


def api_json(url: str, key: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Api-Key {key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else {}


def set_autoscaling(control: str, key: str, minimum: int) -> None:
    api_json(
        control + "/autoscaling_settings",
        key,
        "PATCH",
        {
            "min_replica": minimum,
            "max_replica": 1,
            "autoscaling_window": 60,
            "scale_down_delay": 120,
            "concurrency_target": 1,
            "target_utilization_percentage": 70,
        },
    )


def deployment_state(control: str, key: str) -> tuple[str, int]:
    payload = api_json(control, key)
    return str(payload.get("status", "UNKNOWN")), int(payload.get("active_replica_count", 0))


def wait_active(control: str, key: str, log) -> None:
    for attempt in range(1, 181):
        try:
            status, replicas = deployment_state(control, key)
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            status, replicas = f"ERROR:{type(error).__name__}", 0
        if status == "ACTIVE" and replicas >= 1:
            log(f"DEPLOYMENT_ACTIVE replicas={replicas} attempt={attempt}")
            return
        if attempt % 6 == 0:
            log(f"DEPLOYMENT_WAIT status={status} replicas={replicas} attempt={attempt}")
        time.sleep(5)
    raise RuntimeError("deployment did not become active")


def wait_zero(control: str, key: str, log) -> bool:
    for attempt in range(1, 181):
        try:
            status, replicas = deployment_state(control, key)
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            status, replicas = f"ERROR:{type(error).__name__}", -1
        if status == "SCALED_TO_ZERO" and replicas == 0:
            log(f"DEPLOYMENT_SCALED_TO_ZERO replicas=0 attempt={attempt}")
            return True
        if attempt % 6 == 0:
            log(f"SCALE_ZERO_WAIT status={status} replicas={replicas} attempt={attempt}")
        time.sleep(5)
    log("ERROR deployment did not scale to zero")
    return False


def append_tsv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def inspect_transcript(path: Path) -> dict[str, object]:
    rows = []
    for line in path.read_text().splitlines():
        rows.append(json.loads(line))
    scripted = [
        row
        for row in rows
        if isinstance(row.get("turn"), int)
        and 0 <= row["turn"] < 30
        and row.get("recovery_turn") is not True
    ]
    turns = [row["turn"] for row in scripted]
    if len(turns) != len(set(turns)):
        raise RuntimeError(f"duplicate scripted turn in {path}")
    responses = [
        row
        for row in rows
        if (row.get("assistant_text") or "")
        or len(row.get("tool_calls") or []) > 0
    ]
    scripted_responses = [
        row
        for row in scripted
        if (row.get("assistant_text") or "")
        or len(row.get("tool_calls") or []) > 0
    ]
    thought_turns = sum(bool(row.get("assistant_thought")) for row in scripted)
    tool_calls = sum(len(row.get("tool_calls") or []) for row in rows)
    end_turns = [
        int(row["turn"])
        for row in rows
        if isinstance(row.get("turn"), int)
        and any(call.get("name") == "end_session" for call in row.get("tool_calls") or [])
    ]
    return {
        "records": len(rows),
        "turns": len(scripted),
        "response_turns": len(scripted_responses),
        "captured_responses": len(responses),
        "thought_turns": thought_turns,
        "tool_calls": tool_calls,
        "end_session_turn": max(end_turns, default=-1),
    }


def run_request(
    assignment: dict[str, str],
    spec: dict[str, str],
    key: str,
    attempt: int,
    run_log: Path,
    log,
) -> tuple[int, str]:
    env = os.environ.copy()
    for name in (
        "MTE_FILLER_DOTS",
        "MTE_FILLER_TOKEN",
        "MTE_FILLER_POSITION",
        "MTE_VLLM_THINKING_BUDGET",
        "MTE_VLLM_NATIVE_BUDGET",
        "MTE_VLLM_GRACE",
        "MTE_VLLM_TOP_K",
    ):
        env.pop(name, None)
    env.update(
        {
            "VLLM_BASE_URL": spec["endpoint"],
            "VLLM_API_KEY": key,
            "MTE_VLLM_THINKING": "0",
            "MTE_VLLM_TEMPERATURE": "0.6",
            "MTE_VLLM_TOP_P": "0.95",
            "MTE_VLLM_MAX_TOKENS": "8192",
            "MTE_ENABLE_RECOVERY": "1",
            "MTE_DEDUPE_TOOL_CALLS": "1",
            "MTE_TOOL_RESULT_RUN_LLM": "0",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
        }
    )
    if assignment["arm"] == "dots96":
        env.update(
            {
                "MTE_FILLER_DOTS": "96",
                "MTE_FILLER_TOKEN": ".",
                "MTE_FILLER_POSITION": "suffix",
            }
        )
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "900s",
        str(ROOT / ".venv/bin/multi-turn-eval"),
        "run",
        "aiwf_medium_context",
        "--model",
        spec["model"],
        "--service",
        "vllm-openai",
        "--pipeline",
        "text",
    ]
    log(
        f"RUN_START slot={assignment['slot']} model={assignment['model']} "
        f"arm={assignment['arm']} stage={assignment['stage']} attempt={attempt}"
    )
    run_dir = ""
    with run_log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("Output directory: "):
                run_dir = line.removeprefix("Output directory: ").strip()
        rc = process.wait()
    log(
        f"RUN_EXIT slot={assignment['slot']} arm={assignment['arm']} "
        f"attempt={attempt} rc={rc} run_dir={run_dir or 'NA'}"
    )
    return rc, run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", choices=sorted(LANES))
    parser.add_argument(
        "--through-stage",
        required=True,
        choices=("control", "1", "2", "3"),
        help="Execute only assignments reached by the frozen adaptive protocol.",
    )
    args = parser.parse_args()

    schedule_name, expected_model = LANES[args.lane]
    schedule = DATA / schedule_name
    state = DATA / "state" / args.lane
    state.mkdir(parents=True, exist_ok=True)
    attempts_path = state / "attempts.tsv"
    canonical_path = state / "canonical.tsv"
    driver_log = state / "driver.log"
    lock_stream = (state / "driver.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another driver owns lane {args.lane}", file=sys.stderr)
        return 2

    def log(message: str) -> None:
        rendered = f"{utc()} {message}"
        print(rendered, flush=True)
        with driver_log.open("a") as stream:
            stream.write(rendered + "\n")

    verify_sources()
    assignments = read_tsv(schedule)
    for row in assignments:
        if row["model"] != expected_model:
            raise RuntimeError(f"schedule model mismatch: {row}")
    if args.through_stage == "control":
        selected = [row for row in assignments if row["stage"] == "control"]
    else:
        limit = int(args.through_stage)
        selected = [
            row for row in assignments if row["stage"].isdigit() and int(row["stage"]) <= limit
        ]
    if not selected:
        raise RuntimeError("selected stage has no assignments")

    key = load_key()
    spec = MODELS[expected_model]
    attempt_fields = [
        "slot",
        "model",
        "arm",
        "stage",
        "attempt",
        "start_utc",
        "end_utc",
        "run_rc",
        "run_dir",
        "records",
        "turns",
        "response_turns",
        "thought_turns",
        "tool_calls",
        "end_session_turn",
        "classification",
        "transcript_sha256",
        "log",
    ]
    canonical_fields = [
        "slot",
        "model",
        "arm",
        "stage",
        "attempt",
        "run_dir",
        "turns",
        "response_turns",
        "tool_calls",
        "end_session_turn",
        "classification",
        "transcript_sha256",
    ]

    log(
        f"LANE_START lane={args.lane} through_stage={args.through_stage} "
        f"assignments={len(selected)}"
    )
    set_autoscaling(spec["control"], key, 1)
    wait_active(spec["control"], key, log)
    failure = False
    try:
        for assignment in selected:
            canonical = {row["slot"]: row for row in read_tsv(canonical_path)}
            if assignment["slot"] in canonical:
                log(f"SLOT_SKIP slot={assignment['slot']} reason=already_canonical")
                continue
            while True:
                verify_sources()
                prior = [
                    row
                    for row in read_tsv(attempts_path)
                    if row["slot"] == assignment["slot"]
                ]
                attempt = len(prior) + 1
                if attempt > 4:
                    log(f"LANE_STOP slot={assignment['slot']} reason=attempt_ceiling")
                    failure = True
                    break
                run_log = state / f"{assignment['slot']}-attempt{attempt:02d}.log"
                start = utc()
                rc, run_dir_raw = run_request(
                    assignment, spec, key, attempt, run_log, log
                )
                end = utc()
                run_dir = ROOT / run_dir_raw if run_dir_raw else None
                details = {
                    "records": 0,
                    "turns": 0,
                    "response_turns": 0,
                    "captured_responses": 0,
                    "thought_turns": 0,
                    "tool_calls": 0,
                    "end_session_turn": -1,
                }
                transcript_hash = ""
                transcript = run_dir / "transcript.jsonl" if run_dir else None
                if transcript and transcript.is_file() and transcript.stat().st_size:
                    details = inspect_transcript(transcript)
                    transcript_hash = sha256(transcript)
                if int(details["thought_turns"]) != 0:
                    raise RuntimeError(
                        f"thinking-off lane captured thoughts: slot={assignment['slot']}"
                    )
                valid_response = int(details["captured_responses"]) > 0
                run_text = run_log.read_text(errors="replace")
                config_signature = (
                    f"Using vllm-openai with base_url={spec['endpoint']}, "
                    f"model={spec['model']}, thinking=False, thinking_budget=None, "
                    "T=0.6, top_p=0.95, top_k=None, max_tokens=8192"
                )
                if valid_response and config_signature not in run_text:
                    raise RuntimeError(
                        f"missing exact run configuration signature: slot={assignment['slot']}"
                    )
                filler_logged = "MTE_FILLER_DOTS active: 96 x '.' filler tokens" in run_text
                if valid_response and assignment["arm"] == "dots96" and not filler_logged:
                    raise RuntimeError(
                        f"missing filler activation evidence: slot={assignment['slot']}"
                    )
                if assignment["arm"] == "nofiller" and filler_logged:
                    raise RuntimeError(
                        f"filler leaked into control: slot={assignment['slot']}"
                    )
                if valid_response:
                    if int(details["end_session_turn"]) == 29:
                        classification = "strict_complete"
                    else:
                        classification = "fixed_denominator_outcome"
                elif INFRA_RE.search(run_text):
                    classification = "infra_zero_response_replaced"
                else:
                    classification = "zero_response_unclassified"
                row = {
                    "slot": assignment["slot"],
                    "model": assignment["model"],
                    "arm": assignment["arm"],
                    "stage": assignment["stage"],
                    "attempt": attempt,
                    "start_utc": start,
                    "end_utc": end,
                    "run_rc": rc,
                    "run_dir": run_dir_raw or "NA",
                    "records": details["records"],
                    "turns": details["turns"],
                    "response_turns": details["response_turns"],
                    "thought_turns": details["thought_turns"],
                    "tool_calls": details["tool_calls"],
                    "end_session_turn": details["end_session_turn"],
                    "classification": classification,
                    "transcript_sha256": transcript_hash,
                    "log": str(run_log.relative_to(ROOT)),
                }
                append_tsv(attempts_path, attempt_fields, row)
                log(
                    f"ATTEMPT_CLASSIFIED slot={assignment['slot']} "
                    f"classification={classification} turns={details['turns']} "
                    f"responses={details['response_turns']}"
                )
                if classification == "infra_zero_response_replaced":
                    continue
                if classification == "zero_response_unclassified":
                    failure = True
                    break
                append_tsv(
                    canonical_path,
                    canonical_fields,
                    {field: row[field] for field in canonical_fields},
                )
                break
            if failure:
                break
    finally:
        log("TEARDOWN_START min_replica=0 max_replica=1 scale_down_delay=120")
        try:
            set_autoscaling(spec["control"], key, 0)
            wait_zero(spec["control"], key, log)
        except Exception as error:
            log(f"TEARDOWN_ERROR type={type(error).__name__} message={error}")
            failure = True

    canonical_count = len(read_tsv(canonical_path))
    log(
        f"LANE_DONE lane={args.lane} canonical_total={canonical_count} "
        f"selected_target={len(selected)} failure={int(failure)}"
    )
    if not failure and all(
        row["slot"] in {item["slot"] for item in read_tsv(canonical_path)}
        for row in selected
    ):
        (state / f"COMPLETE-stage-{args.through_stage}").write_text(utc() + "\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
