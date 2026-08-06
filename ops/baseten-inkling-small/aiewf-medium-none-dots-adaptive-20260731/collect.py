#!/usr/bin/env python3
"""Strictly sequential, resumable Inkling Small +96-dots collector.

The default invocation is read-only. ``--execute`` collects only the requested
stage cap (6, 10, or 30) and has no code path that runs a control conversation.
"""

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
PRIMARY = ROOT / "ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731"
CONFIG = HERE / "configuration.json"
SCHEDULE = HERE / "frozen-order.tsv"
ATTEMPTS = HERE / "attempts.tsv"
CANONICAL = HERE / "canonical.tsv"
CONTROL_INPUTS = HERE / "control-inputs.tsv"
STAGE_DECISIONS = HERE / "stage-decisions.tsv"
SOURCE_HASHES = HERE / "source-sha256.txt"
CAMPAIGN_LOG = HERE / "campaign.log"
LOG_DIR = HERE / "logs"
LOCK = HERE / ".collection.lock"

MODEL = "thinkingmachines/inkling-small"
ENDPOINT = "https://inference.baseten.co/v1"
ARM = "dots96"
STAGES = (6, 10, 30)
N_TURNS = 30

ATTEMPT_FIELDS = (
    "slot", "stage_cap", "arm", "attempt", "started_at", "finished_at",
    "exit_code", "run_dir", "scheduled_rows", "response_turns",
    "end_session_turn", "classification", "log",
)
CANONICAL_FIELDS = (
    "slot", "stage_cap", "arm", "attempt", "run_dir", "scheduled_rows",
    "response_turns", "end_session_turn", "classification",
)
CONTROL_FIELDS = ("slot", "run_dir", "transcript_sha256", "scheduled_turns")
DECISION_FIELDS = (
    "completed_stage", "requested_stage", "decision", "decided_at",
    "analysis_artifact", "analysis_sha256", "rationale",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {value}") from exc
    return resolved


def validate_source_hashes() -> None:
    if not SOURCE_HASHES.is_file():
        raise RuntimeError(f"missing source hash manifest: {SOURCE_HASHES}")
    for line in SOURCE_HASHES.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source integrity failure for {relative}: {actual} != {expected}"
            )


def validate_configuration() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "campaign_id": "aiewf-medium-inkling-small-baseten-none-dots-adaptive-20260731",
        "benchmark": "aiwf_medium_context",
        "provider": "BaseTen Model API",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "service": "baseten",
        "pipeline": "text",
        "thinking_effort": "none",
        "test_arm": ARM,
        "stage_caps": list(STAGES),
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "sampling": {"temperature": 1.0, "max_tokens": 16384},
        "runtime": {
            "provider_endpoint_concurrency": 1,
            "enable_recovery": True,
            "dedupe_tool_calls": True,
            "tool_result_run_llm": False,
            "text_idle_timeout_seconds": 45,
        },
        "filler": {
            "count": 96,
            "token": ".",
            "position": "suffix",
            "outgoing_request_only": True,
        },
        "max_attempts_per_slot": 4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"configuration mismatch for {key}: {config.get(key)!r}")
    control = config.get("control") or {}
    if control != {
        "campaign": "ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731",
        "arm": "none",
        "target_conversations": 30,
        "policy": "freeze and reuse the completed primary cohort; never rerun or top up control",
    }:
        raise RuntimeError("control configuration is not frozen")
    return config


def validate_schedule() -> list[dict[str, str]]:
    rows = read_tsv(SCHEDULE)
    if len(rows) != 30:
        raise RuntimeError(f"dot schedule must contain 30 assignments, found {len(rows)}")
    if [row["slot"] for row in rows] != [f"DOT-{n:02d}" for n in range(1, 31)]:
        raise RuntimeError("dot slots must be DOT-01 through DOT-30")
    expected_caps = [6] * 6 + [10] * 4 + [30] * 20
    if [int(row["stage_cap"]) for row in rows] != expected_caps:
        raise RuntimeError("stage-cap boundaries are not frozen at 6, 10, and 30")
    if any(row["arm"] != ARM for row in rows):
        raise RuntimeError("schedule contains a non-dots assignment")
    return rows


def read_transcript(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "transcript.jsonl"
    if not path.is_file() or not path.stat().st_size:
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid transcript JSON {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object transcript row {path}:{line_number}")
        rows.append(row)
    return rows


def scheduled_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = read_transcript(run_dir)
    scheduled = [row for row in rows if row.get("recovery_turn") is not True]
    turns = [row.get("turn") for row in scheduled]
    if turns != list(range(len(turns))) or len(turns) > N_TURNS:
        raise RuntimeError(f"scheduled turns are not a contiguous prefix: {run_dir}")
    for row in rows:
        if row.get("model_name") != MODEL:
            raise RuntimeError(f"model mismatch in {run_dir}: {row.get('model_name')!r}")
    return scheduled


def inspect_primary_control(*, require_complete: bool) -> list[dict[str, Any]]:
    primary_canonical = PRIMARY / "canonical.tsv"
    primary_log = PRIMARY / "campaign.log"
    if not primary_canonical.is_file() or not primary_log.is_file():
        if require_complete:
            raise RuntimeError("primary Inkling Small campaign artifacts are missing")
        return []
    rows = read_tsv(primary_canonical)
    if require_complete:
        marker = "COLLECTION_DONE total=60 none=30 low=30"
        if marker not in primary_log.read_text(encoding="utf-8"):
            raise RuntimeError(f"primary campaign lacks completed marker: {marker}")
        if len(rows) != 60:
            raise RuntimeError(f"primary canonical must contain 60 rows, found {len(rows)}")
    controls = [row for row in rows if row.get("arm") == "none"]
    if require_complete and len(controls) != 30:
        raise RuntimeError(f"primary none control must contain 30 rows, found {len(controls)}")
    result: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for row in controls:
        run_dir = resolve_repo_path(row["run_dir"])
        if run_dir in seen_dirs:
            raise RuntimeError(f"duplicate primary control run: {run_dir}")
        seen_dirs.add(run_dir)
        scheduled = scheduled_rows(run_dir)
        if int(row["scheduled_rows"]) != len(scheduled):
            raise RuntimeError(f"primary control row count mismatch at {row['slot']}")
        result.append(
            {
                "slot": row["slot"],
                "run_dir": str(run_dir.relative_to(ROOT)),
                "transcript_sha256": sha256(run_dir / "transcript.jsonl"),
                "scheduled_turns": len(scheduled),
            }
        )
    return result


def validate_or_freeze_control(*, execute: bool) -> int:
    frozen = read_tsv(CONTROL_INPUTS)
    controls = inspect_primary_control(require_complete=execute or bool(frozen))
    normalized = [{field: str(row[field]) for field in CONTROL_FIELDS} for row in controls]
    if frozen:
        if len(frozen) != 30 or frozen != normalized:
            raise RuntimeError("primary none control changed after it was frozen")
        return len(frozen)
    if execute:
        if len(normalized) != 30:
            raise RuntimeError("cannot freeze control before all 30 primary none runs exist")
        write_tsv(CONTROL_INPUTS, CONTROL_FIELDS, normalized)
        return len(normalized)
    return len(controls)


def validate_stage_gate(stage: int) -> None:
    if stage == 6:
        return
    prior = 6 if stage == 10 else 10
    matches = [
        row for row in read_tsv(STAGE_DECISIONS)
        if int(row["completed_stage"]) == prior
        and int(row["requested_stage"]) == stage
        and row["decision"] == "extend"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"stage {stage} requires exactly one frozen extend decision from stage {prior}"
        )
    row = matches[0]
    artifact = resolve_repo_path(row["analysis_artifact"])
    if not artifact.is_file() or not artifact.stat().st_size:
        raise RuntimeError(f"stage decision analysis artifact is missing: {artifact}")
    if sha256(artifact) != row["analysis_sha256"]:
        raise RuntimeError(f"stage decision analysis artifact changed: {artifact}")
    if not row["rationale"].strip():
        raise RuntimeError("stage decision rationale is empty")


def validate_run(run_dir: Path) -> dict[str, int | str]:
    rows = read_transcript(run_dir)
    scheduled = scheduled_rows(run_dir)
    response_turns = 0
    end_turn = -1
    for row in scheduled:
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
    required = (
        f"Using BaseTen with base_url={ENDPOINT}, model={MODEL}, "
        "reasoning_effort=none, enable_thinking=(unset), "
        "max_tokens=16384, temperature=1.0",
        "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix "
        "(history left filler-free)",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    if any(item not in log_text for item in required):
        raise RuntimeError(f"runtime or filler signature mismatch: {run_dir}")

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


def build_attempt_environment(api_key: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BASETEN_API_KEY": api_key,
            "BASETEN_BASE_URL": ENDPOINT,
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "16384",
            "MTE_BASETEN_TEMPERATURE": "1.0",
            "MTE_FILLER_DOTS": "96",
            "MTE_FILLER_TOKEN": ".",
            "MTE_FILLER_POSITION": "suffix",
            "MTE_ENABLE_RECOVERY": "1",
            "MTE_DEDUPE_TOOL_CALLS": "1",
            "MTE_TOOL_RESULT_RUN_LLM": "0",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
        }
    )
    env.pop("MTE_BASETEN_ENABLE_THINKING", None)
    return env


def run_attempt(api_key: str) -> tuple[int, str, Path | None]:
    proc = subprocess.run(
        [
            str(ROOT / ".venv/bin/multi-turn-eval"), "run", "aiwf_medium_context",
            "--model", MODEL, "--service", "baseten", "--pipeline", "text",
        ],
        cwd=ROOT,
        env=build_attempt_environment(api_key),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    matches = re.findall(r"^Output directory: (.+)$", proc.stdout, flags=re.MULTILINE)
    run_dir = resolve_repo_path(matches[-1]) if matches else None
    return proc.returncode, proc.stdout, run_dir


def validate_manifests(schedule: list[dict[str, str]]) -> list[dict[str, str]]:
    if ATTEMPTS.read_text(encoding="utf-8").splitlines()[0] != "\t".join(ATTEMPT_FIELDS):
        raise RuntimeError("attempts.tsv header mismatch")
    if CANONICAL.read_text(encoding="utf-8").splitlines()[0] != "\t".join(CANONICAL_FIELDS):
        raise RuntimeError("canonical.tsv header mismatch")
    if CONTROL_INPUTS.read_text(encoding="utf-8").splitlines()[0] != "\t".join(CONTROL_FIELDS):
        raise RuntimeError("control-inputs.tsv header mismatch")
    if STAGE_DECISIONS.read_text(encoding="utf-8").splitlines()[0] != "\t".join(DECISION_FIELDS):
        raise RuntimeError("stage-decisions.tsv header mismatch")
    canonical = read_tsv(CANONICAL)
    if len({row["slot"] for row in canonical}) != len(canonical):
        raise RuntimeError("canonical dot slots are not unique")
    expected_prefix = [row["slot"] for row in schedule[: len(canonical)]]
    if [row["slot"] for row in canonical] != expected_prefix:
        raise RuntimeError("canonical dot assignments are not a frozen schedule prefix")
    for row, assignment in zip(canonical, schedule):
        for field in ("slot", "stage_cap", "arm"):
            if row[field] != assignment[field]:
                raise RuntimeError(f"canonical {field} mismatch for {row['slot']}")
        current = validate_run(resolve_repo_path(row["run_dir"]))
        for field in ("scheduled_rows", "response_turns", "end_session_turn"):
            if int(row[field]) != int(current[field]):
                raise RuntimeError(f"canonical {field} mismatch for {row['slot']}")
        if row["classification"] != current["classification"]:
            raise RuntimeError(f"canonical classification mismatch for {row['slot']}")
    return canonical


def preflight(*, execute: bool, stage: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if stage not in STAGES:
        raise RuntimeError(f"stage must be one of {STAGES}")
    config = validate_configuration()
    schedule = validate_schedule()
    validate_source_hashes()
    canonical = validate_manifests(schedule)
    if len(canonical) > stage:
        raise RuntimeError(
            f"requested stage {stage} is below already-collected n={len(canonical)}"
        )
    if stage > 6 and len(canonical) < (6 if stage == 10 else 10):
        raise RuntimeError(f"cannot request stage {stage} before completing the prior stage")
    validate_stage_gate(stage)
    control_count = validate_or_freeze_control(execute=execute)
    if execute and control_count != 30:
        raise RuntimeError("--execute requires all 30 frozen primary controls")
    print(
        f"preflight ok: requested_stage={stage} dots={len(canonical)}/{stage} "
        f"primary_none_control={control_count}/30 execute={execute}"
    )
    return config, schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=STAGES, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, schedule = preflight(execute=False, stage=args.stage)
    if not args.execute:
        print("Read-only preflight only. No BaseTen request was made.")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another collector or stage-gate process owns the lock") from exc

        # Re-run every execution gate while holding the campaign lock. This is
        # the first point at which the control freeze may write locally.
        config, schedule = preflight(execute=True, stage=args.stage)
        api_key = load_key()
        completed = {row["slot"] for row in read_tsv(CANONICAL)}
        for assignment in schedule[: args.stage]:
            slot = assignment["slot"]
            if slot in completed:
                continue
            attempts = [row for row in read_tsv(ATTEMPTS) if row["slot"] == slot]
            while len(attempts) < int(config["max_attempts_per_slot"]):
                validate_source_hashes()
                validate_or_freeze_control(execute=True)
                validate_stage_gate(args.stage)
                attempt = len(attempts) + 1
                started = now()
                log(
                    f"starting slot={slot} stage_cap={assignment['stage_cap']} "
                    f"arm={ARM} attempt={attempt}"
                )
                rc, output, run_dir = run_attempt(api_key)
                finished = now()
                console_log = LOG_DIR / f"{slot}-attempt-{attempt}.log"
                console_log.write_text(output, encoding="utf-8")
                if run_dir is None:
                    metrics: dict[str, int | str] = {
                        "scheduled_rows": 0,
                        "response_turns": 0,
                        "end_session_turn": -1,
                        "classification": "zero_response",
                    }
                else:
                    metrics = validate_run(run_dir)
                classification = str(metrics["classification"])
                if classification == "zero_response" and objective_infrastructure_failure(output):
                    classification = "infra_zero_response_replaced"
                attempt_row = {
                    "slot": slot,
                    "stage_cap": assignment["stage_cap"],
                    "arm": ARM,
                    "attempt": attempt,
                    "started_at": started,
                    "finished_at": finished,
                    "exit_code": rc,
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
                    raise RuntimeError(
                        f"unclassified zero-response attempt requires review: {slot}"
                    )
                canonical_row = {field: attempt_row[field] for field in CANONICAL_FIELDS}
                append_tsv(CANONICAL, CANONICAL_FIELDS, canonical_row)
                log(
                    f"counted slot={slot} arm={ARM} class={classification} "
                    f"rows={metrics['scheduled_rows']} responses={metrics['response_turns']}"
                )
                completed.add(slot)
                break
            if slot not in completed:
                raise RuntimeError(f"slot {slot} exhausted the attempt ceiling")

        canonical = read_tsv(CANONICAL)
        if len(canonical) != args.stage:
            raise RuntimeError(
                f"stage {args.stage} ended at unexpected canonical count {len(canonical)}"
            )
        log(f"DOT_STAGE_DONE stage={args.stage} canonical={len(canonical)} control=30")


if __name__ == "__main__":
    main()
