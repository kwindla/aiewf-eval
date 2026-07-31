#!/usr/bin/env python3
"""Portable, strictly sequential, resumable AIEWF campaign collector.

The default invocation is a read-only preflight. Live requests require
``--execute`` and a configuration whose serving gate is explicitly verified.
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
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ONE = HERE / "run_one.py"

ATTEMPT_FIELDS = (
    "slot",
    "arm",
    "attempt",
    "started_at",
    "finished_at",
    "exit_code",
    "run_dir",
    "response_turns",
    "classification",
    "log",
)
CANONICAL_FIELDS = (
    "slot",
    "arm",
    "attempt",
    "run_dir",
    "turns",
    "response_turns",
    "tool_calls",
    "classification",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise ValueError(message)


def resolve_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON configuration: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON configuration {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"configuration must be a JSON object: {path}")
    return value


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
                fail(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


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


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def scalar_environment(value: Any, *, name: str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)):
        return str(value)
    fail(f"environment value for {name} must be a string, number, or boolean")


def validate_environment_mapping(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        fail(f"{field} must be an object")
    result: dict[str, str] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            fail(f"invalid environment name in {field}: {name!r}")
        result[name] = scalar_environment(raw, name=name)
    return result


def validate_configuration(
    config_path: Path, *, require_serving_verified: bool
) -> dict[str, Any]:
    config = read_json(config_path)
    required_strings = (
        "campaign_id",
        "benchmark",
        "model",
        "service",
        "pipeline",
        "schedule_path",
    )
    for field in required_strings:
        if not isinstance(config.get(field), str) or not config[field].strip():
            fail(f"configuration field {field!r} must be a non-empty string")
    if config.get("schema_version") != 1:
        fail("schema_version must be 1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config["campaign_id"]):
        fail("campaign_id may contain only letters, numbers, dot, underscore, and dash")

    target = config.get("target_eligible_runs")
    turns = config.get("fixed_scheduled_turns_per_conversation")
    if not isinstance(target, int) or target < 1:
        fail("target_eligible_runs must be a positive integer")
    if not isinstance(turns, int) or turns < 1:
        fail("fixed_scheduled_turns_per_conversation must be a positive integer")

    accepted = config.get("accepted_response_model_ids")
    if not isinstance(accepted, list) or not accepted or not all(
        isinstance(item, str) and item for item in accepted
    ):
        fail("accepted_response_model_ids must be a non-empty string list")
    if config["model"] not in accepted:
        fail("accepted_response_model_ids must include the requested model")

    endpoint = config.get("endpoint")
    if not isinstance(endpoint, dict):
        fail("endpoint must be an object")
    if not isinstance(endpoint.get("url"), str) or not endpoint["url"].strip():
        fail("endpoint.url must be a non-empty string")
    if not isinstance(endpoint.get("request_env"), str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", endpoint["request_env"]
    ):
        fail("endpoint.request_env must be an environment-variable name")

    credential = config.get("credential")
    if not isinstance(credential, dict):
        fail("credential must be an object")
    for field in ("request_env", "source_env", "source_file_key"):
        if not isinstance(credential.get(field), str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", credential[field]
        ):
            fail(f"credential.{field} must be an environment-variable name")
    if not isinstance(credential.get("source_file"), str):
        fail("credential.source_file must be a string (empty disables file fallback)")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        fail("paths must be an object")
    for field in ("campaign_artifact_dir", "run_output_root"):
        if not isinstance(paths.get(field), str) or not paths[field].strip():
            fail(f"paths.{field} must be a non-empty string")
    artifact_dir = resolve_path(paths["campaign_artifact_dir"])
    run_output_root = resolve_path(paths["run_output_root"])
    if artifact_dir == run_output_root:
        fail("campaign_artifact_dir and run_output_root must be different")
    if artifact_dir in run_output_root.parents or run_output_root in artifact_dir.parents:
        fail("campaign_artifact_dir and run_output_root must not contain one another")

    integrity_paths = config.get("source_integrity_paths", [])
    if not isinstance(integrity_paths, list) or not all(
        isinstance(value, str) and value for value in integrity_paths
    ):
        fail("source_integrity_paths must be a non-empty-string list")
    resolved_integrity_paths = [resolve_path(value) for value in integrity_paths]

    collection = config.get("collection")
    if not isinstance(collection, dict):
        fail("collection must be an object")
    if collection.get("provider_endpoint_concurrency") != 1:
        fail("provider_endpoint_concurrency must be exactly 1")
    for field in ("max_attempts_per_slot_default", "timeout_seconds_default"):
        if not isinstance(collection.get(field), int) or collection[field] < 1:
            fail(f"collection.{field} must be a positive integer")

    eligibility = config.get("eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("policy") != "first_valid_response":
        fail("eligibility.policy must be 'first_valid_response'")
    if eligibility.get("missing_future_turns") != "retain_as_fixed_denominator_failures":
        fail(
            "eligibility.missing_future_turns must be "
            "'retain_as_fixed_denominator_failures'"
        )

    common_env = validate_environment_mapping(
        config.get("common_environment", {}), field="common_environment"
    )
    arms = config.get("arms")
    if not isinstance(arms, dict) or not arms:
        fail("arms must be a non-empty object")
    normalized_arms: dict[str, dict[str, Any]] = {}
    for arm_name, arm in arms.items():
        if not isinstance(arm_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", arm_name
        ):
            fail(f"invalid arm name: {arm_name!r}")
        if not isinstance(arm, dict):
            fail(f"arm {arm_name!r} must be an object")
        needles = arm.get("provenance_log_needles", [])
        if not isinstance(needles, list) or not all(
            isinstance(needle, str) and needle for needle in needles
        ):
            fail(f"arm {arm_name!r} provenance_log_needles must be a string list")
        normalized_arms[arm_name] = {
            **arm,
            "environment": validate_environment_mapping(
                arm.get("environment", {}), field=f"arms.{arm_name}.environment"
            ),
            "provenance_log_needles": needles,
        }

    unset = config.get("unset_environment", [])
    prefixes = config.get("unset_environment_prefixes", [])
    if not isinstance(unset, list) or not all(
        isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        for name in unset
    ):
        fail("unset_environment must be a list of environment-variable names")
    if not isinstance(prefixes, list) or not all(
        isinstance(prefix, str) and prefix for prefix in prefixes
    ):
        fail("unset_environment_prefixes must be a non-empty-string list")

    serving = config.get("serving")
    if not isinstance(serving, dict):
        fail("serving must be an object")
    if require_serving_verified and serving.get("verified") is not True:
        fail(
            "serving smoke gate is not complete: set serving.verified=true only "
            "after endpoint, streaming, tools, continuation, and serving checks pass"
        )

    config["_config_path"] = config_path.resolve()
    config["_schedule_path"] = resolve_path(config["schedule_path"])
    config["_artifact_dir"] = artifact_dir
    config["_run_output_root"] = run_output_root
    config["_common_environment"] = common_env
    config["_arms"] = normalized_arms
    config["_source_integrity_paths"] = resolved_integrity_paths
    return config


def validate_schedule(config: dict[str, Any]) -> list[dict[str, str]]:
    schedule_path = config["_schedule_path"]
    if not schedule_path.is_file():
        fail(f"missing frozen schedule: {schedule_path}")
    lines = schedule_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "slot\tarm":
        fail("frozen schedule header must be exactly 'slot\\tarm'")
    rows = read_tsv(schedule_path)
    target = config["target_eligible_runs"]
    if len(rows) != target:
        fail(f"frozen schedule must contain exactly {target} slots")
    expected_slots = list(range(1, target + 1))
    try:
        slots = [int(row["slot"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        fail("frozen schedule contains a non-integer slot")
    if slots != expected_slots:
        fail(f"frozen schedule slots must be exactly 1..{target}")
    unknown = sorted({row.get("arm", "") for row in rows} - set(config["_arms"]))
    if unknown:
        fail(f"frozen schedule references unknown arms: {unknown}")
    return rows


def artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    artifact_dir = config["_artifact_dir"]
    return {
        "attempts": artifact_dir / "attempts.tsv",
        "canonical": artifact_dir / "canonical.tsv",
        "campaign_log": artifact_dir / "campaign.log",
        "lock": artifact_dir / ".collection.lock",
        "logs": artifact_dir / "logs",
        "source_hash": artifact_dir / "source-sha256.txt",
        "pending": artifact_dir / "pending-attempt.json",
    }


def initialize_artifacts(config: dict[str, Any]) -> None:
    paths = artifact_paths(config)
    config["_artifact_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    if not paths["attempts"].exists():
        atomic_write(paths["attempts"], "\t".join(ATTEMPT_FIELDS) + "\n")
    if not paths["canonical"].exists():
        atomic_write(paths["canonical"], "\t".join(CANONICAL_FIELDS) + "\n")


def append_campaign_log(config: dict[str, Any], message: str) -> None:
    path = artifact_paths(config)["campaign_log"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_manifest_rows(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = artifact_paths(config)
    attempts_exists = paths["attempts"].exists()
    canonical_exists = paths["canonical"].exists()
    if attempts_exists != canonical_exists:
        fail("attempts.tsv and canonical.tsv must either both exist or both be absent")
    if not attempts_exists:
        return [], []
    attempt_lines = paths["attempts"].read_text(encoding="utf-8").splitlines()
    canonical_lines = paths["canonical"].read_text(encoding="utf-8").splitlines()
    if not attempt_lines or attempt_lines[0] != "\t".join(ATTEMPT_FIELDS):
        fail("unexpected attempts.tsv header")
    if not canonical_lines or canonical_lines[0] != "\t".join(CANONICAL_FIELDS):
        fail("unexpected canonical.tsv header")
    return read_tsv(paths["attempts"]), read_tsv(paths["canonical"])


def path_from_manifest(value: str) -> Path:
    if not value:
        fail("manifest path is empty")
    return resolve_path(value)


def validate_transcript(config: dict[str, Any], run_dir: Path) -> dict[str, int | str]:
    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.is_file() or transcript_path.stat().st_size == 0:
        fail(f"missing transcript: {transcript_path}")
    rows = read_jsonl(transcript_path)
    scheduled: dict[int, dict[str, Any]] = {}
    response_turns = 0
    tool_calls = 0
    n_turns = config["fixed_scheduled_turns_per_conversation"]
    accepted_models = set(config["accepted_response_model_ids"])
    for row in rows:
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < n_turns:
            fail(f"invalid scheduled turn in {transcript_path}: {turn!r}")
        if turn in scheduled:
            fail(f"duplicate scheduled turn {turn} in {transcript_path}")
        if row.get("model_name") not in accepted_models:
            fail(
                f"model mismatch at turn {turn}: {row.get('model_name')!r} not in "
                f"{sorted(accepted_models)!r}"
            )
        scheduled[turn] = row
        assistant_text = row.get("assistant_text")
        calls = row.get("tool_calls") or []
        if (isinstance(assistant_text, str) and assistant_text.strip()) or calls:
            response_turns += 1
        if isinstance(calls, list):
            tool_calls += len(calls)
    if sorted(scheduled) != list(range(len(scheduled))):
        fail("scheduled transcript turns must form a contiguous prefix from zero")
    if response_turns < 1:
        fail("attempt has no valid model response")
    return {
        "turns": len(scheduled),
        "response_turns": response_turns,
        "tool_calls": tool_calls,
        "classification": (
            f"complete_{n_turns}"
            if len(scheduled) == n_turns
            else "fixed_denominator_short"
        ),
    }


def validate_run_provenance(config: dict[str, Any], arm: str, run_dir: Path) -> None:
    run_log = run_dir / "run.log"
    if not run_log.is_file():
        fail(f"missing standard run.log: {run_log}")
    text = run_log.read_text(encoding="utf-8", errors="replace")
    required = [
        f"base_url={config['endpoint']['url']}",
        f"model={config['model']}",
        *config["_arms"][arm]["provenance_log_needles"],
    ]
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"run provenance is missing {missing}: {run_log}")
    forbidden = config["_arms"][arm].get("forbidden_log_needles", [])
    if not isinstance(forbidden, list) or not all(
        isinstance(needle, str) and needle for needle in forbidden
    ):
        fail(f"arm {arm!r} forbidden_log_needles must be a string list")
    present = [needle for needle in forbidden if needle in text]
    if present:
        fail(f"run provenance contains forbidden markers {present}: {run_log}")


def validate_manifests(
    config: dict[str, Any], schedule: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    attempts, canonical = load_manifest_rows(config)
    target = config["target_eligible_runs"]
    if len(canonical) > target:
        fail(f"canonical.tsv has more than {target} rows")
    try:
        canonical_slots = [int(row["slot"]) for row in canonical]
    except (KeyError, TypeError, ValueError):
        fail("canonical.tsv contains a non-integer slot")
    if canonical_slots != list(range(1, len(canonical) + 1)):
        fail("canonical.tsv must be a unique contiguous prefix of frozen slots")
    if len({row["run_dir"] for row in canonical}) != len(canonical):
        fail("canonical.tsv contains duplicate run directories")

    attempts_by_slot: dict[int, list[int]] = {}
    attempt_keys: set[tuple[int, int]] = set()
    for row in attempts:
        try:
            slot = int(row["slot"])
            attempt = int(row["attempt"])
        except (KeyError, TypeError, ValueError):
            fail("attempts.tsv contains a non-integer slot or attempt")
        if not 1 <= slot <= target or row.get("arm") != schedule[slot - 1]["arm"]:
            fail(f"attempt/schedule mismatch at slot {slot}")
        key = (slot, attempt)
        if key in attempt_keys:
            fail(f"duplicate attempt row for slot {slot}, attempt {attempt}")
        attempt_keys.add(key)
        attempts_by_slot.setdefault(slot, []).append(attempt)
    for slot, numbers in attempts_by_slot.items():
        if numbers != list(range(1, len(numbers) + 1)):
            fail(f"attempt numbers for slot {slot} must be contiguous from one")
    allowed_attempt_slots = set(range(1, len(canonical) + 2))
    if set(attempts_by_slot) - allowed_attempt_slots:
        fail("attempts.tsv contains work beyond the next non-canonical slot")

    for row in canonical:
        slot = int(row["slot"])
        attempt = int(row["attempt"])
        arm = schedule[slot - 1]["arm"]
        if row.get("arm") != arm:
            fail(f"canonical/schedule arm mismatch at slot {slot}")
        if (slot, attempt) not in attempt_keys:
            fail(f"canonical slot {slot} references a missing attempt")
        run_dir = path_from_manifest(row["run_dir"])
        validation = validate_transcript(config, run_dir)
        validate_run_provenance(config, arm, run_dir)
        for field in ("turns", "response_turns", "tool_calls"):
            if int(row[field]) != int(validation[field]):
                fail(f"canonical {field} mismatch at slot {slot}")
        if row["classification"] != validation["classification"]:
            fail(f"canonical classification mismatch at slot {slot}")
    return attempts, canonical


def source_hashes(config: dict[str, Any]) -> str:
    base_paths = (
        config["_config_path"],
        config["_schedule_path"],
        Path(__file__).resolve(),
        RUN_ONE.resolve(),
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/pipelines/base.py",
        ROOT / "src/multi_turn_eval/pipelines/text.py",
        ROOT / "src/multi_turn_eval/recording/transcript_recorder.py",
        ROOT / f"benchmarks/{config['benchmark']}/config.py",
    )
    paths = list(dict.fromkeys((*base_paths, *config["_source_integrity_paths"])))
    lines: list[str] = []
    for path in paths:
        if not path.is_file():
            fail(f"source-integrity file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {display_path(path)}")
    return "\n".join(lines) + "\n"


def verify_source_hashes(config: dict[str, Any]) -> None:
    path = artifact_paths(config)["source_hash"]
    if path.exists() and path.read_text(encoding="utf-8") != source_hashes(config):
        fail("campaign source hashes changed after collection began")


def extract_credential(config: dict[str, Any]) -> str:
    credential = config["credential"]
    value = os.environ.get(credential["source_env"])
    if value:
        return value
    source_file = credential["source_file"]
    if source_file:
        path = resolve_path(source_file)
        if path.is_file():
            prefix = credential["source_file_key"] + "="
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith(prefix):
                    candidate = line[len(prefix) :].strip()
                    if (
                        len(candidate) >= 2
                        and candidate[0] == candidate[-1]
                        and candidate[0] in {"'", '"'}
                    ):
                        candidate = candidate[1:-1]
                    if candidate:
                        return candidate
    fail(
        f"credential unavailable: set {credential['source_env']} or provide "
        f"{credential['source_file_key']} in {source_file or '(configured source file)'}"
    )


def child_environment(config: dict[str, Any], arm: str, credential: str) -> dict[str, str]:
    env = os.environ.copy()
    for prefix in config.get("unset_environment_prefixes", []):
        for name in tuple(env):
            if name.startswith(prefix):
                env.pop(name, None)
    for name in config.get("unset_environment", []):
        env.pop(name, None)
    env.update(config["_common_environment"])
    env.update(config["_arms"][arm]["environment"])
    env[config["endpoint"]["request_env"]] = config["endpoint"]["url"]
    env[config["credential"]["request_env"]] = credential
    return env


def next_attempt_number(attempts: Iterable[dict[str, str]], slot: int) -> int:
    numbers = [int(row["attempt"]) for row in attempts if int(row["slot"]) == slot]
    return max(numbers, default=0) + 1


def make_run_dir(config: dict[str, Any], *, slot: int, arm: str, attempt: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", config["model"]).strip("_")
    name = (
        f"{timestamp}_{safe_model}_{arm}_slot{slot:03d}_attempt{attempt:02d}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    return (config["_run_output_root"] / name).resolve()


def write_pending(config: dict[str, Any], pending: dict[str, Any]) -> None:
    atomic_write(
        artifact_paths(config)["pending"],
        json.dumps(pending, indent=2, sort_keys=True) + "\n",
    )


def run_attempt(
    config: dict[str, Any],
    *,
    slot: int,
    arm: str,
    attempt: int,
    credential: str,
    timeout_seconds: int,
    lock_fd: int,
) -> dict[str, Any]:
    paths = artifact_paths(config)
    run_dir = make_run_dir(config, slot=slot, arm=arm, attempt=attempt)
    log_path = paths["logs"] / f"slot{slot:03d}-{arm}-attempt{attempt:02d}.log"
    pending: dict[str, Any] = {
        "slot": slot,
        "arm": arm,
        "attempt": attempt,
        "started_at": utc_now(),
        "finished_at": "",
        "exit_code": "",
        "run_dir": display_path(run_dir),
        "log": display_path(log_path),
    }
    write_pending(config, pending)
    append_campaign_log(
        config,
        f"RUN_START slot={slot} arm={arm} attempt={attempt} "
        f"run_dir={display_path(run_dir)} log={display_path(log_path)}",
    )
    command = [
        sys.executable,
        str(RUN_ONE),
        "--config",
        str(config["_config_path"]),
        "--arm",
        arm,
        "--run-dir",
        str(run_dir),
    ]
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=child_environment(config, arm, credential),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                pass_fds=(lock_fd,),
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            log_handle.write(
                f"\nCOLLECTOR_TIMEOUT seconds={timeout_seconds} at={utc_now()}\n"
            )
            log_handle.flush()
            os.fsync(log_handle.fileno())
    pending["finished_at"] = utc_now()
    pending["exit_code"] = exit_code
    write_pending(config, pending)
    append_campaign_log(
        config,
        f"RUN_EXIT slot={slot} arm={arm} attempt={attempt} rc={exit_code} "
        f"timeout={str(timed_out).lower()} run_dir={display_path(run_dir)}",
    )
    return pending


def finalize_pending(
    config: dict[str, Any],
    schedule: list[dict[str, str]],
    pending: dict[str, Any],
) -> bool:
    paths = artifact_paths(config)
    attempts, canonical = load_manifest_rows(config)
    slot = int(pending["slot"])
    arm = str(pending["arm"])
    attempt = int(pending["attempt"])
    expected_slot = len(canonical) + 1
    if slot != expected_slot or not 1 <= slot <= len(schedule):
        fail(f"pending attempt slot {slot} is not next canonical slot {expected_slot}")
    if arm != schedule[slot - 1]["arm"]:
        fail(f"pending attempt arm {arm!r} does not match frozen schedule")
    if attempt != next_attempt_number(attempts, slot):
        fail(f"pending attempt number {attempt} is not the next durable attempt")
    run_dir = path_from_manifest(str(pending["run_dir"]))
    log_path = path_from_manifest(str(pending["log"]))
    validation: dict[str, int | str] | None = None
    validation_error = ""
    try:
        validation = validate_transcript(config, run_dir)
        validate_run_provenance(config, arm, run_dir)
    except Exception as exc:
        validation_error = str(exc)
    classification = (
        str(validation["classification"])
        if validation is not None
        else "ineligible_no_valid_response_or_provenance"
    )
    response_turns = int(validation["response_turns"]) if validation else 0
    append_tsv(
        paths["attempts"],
        ATTEMPT_FIELDS,
        {
            "slot": slot,
            "arm": arm,
            "attempt": attempt,
            "started_at": pending.get("started_at", ""),
            "finished_at": pending.get("finished_at", "") or utc_now(),
            "exit_code": pending.get("exit_code", "unknown") or "unknown",
            "run_dir": display_path(run_dir),
            "response_turns": response_turns,
            "classification": classification,
            "log": display_path(log_path),
        },
    )
    if validation is not None:
        append_tsv(
            paths["canonical"],
            CANONICAL_FIELDS,
            {
                "slot": slot,
                "arm": arm,
                "attempt": attempt,
                "run_dir": display_path(run_dir),
                **validation,
            },
        )
        append_campaign_log(
            config,
            f"RUN_CANONICAL slot={slot} arm={arm} attempt={attempt} "
            f"classification={classification} run_dir={display_path(run_dir)}",
        )
    else:
        append_campaign_log(
            config,
            f"RUN_INELIGIBLE slot={slot} arm={arm} attempt={attempt} "
            f"reason={json.dumps(validation_error or classification)}",
        )
    paths["pending"].unlink()
    return validation is not None


def validate_pending_shape(config: dict[str, Any], pending: Any) -> dict[str, Any]:
    if not isinstance(pending, dict):
        fail("pending-attempt.json must contain an object")
    for field in ("slot", "attempt"):
        if not isinstance(pending.get(field), int) or pending[field] < 1:
            fail(f"pending-attempt.json field {field!r} must be a positive integer")
    for field in ("arm", "run_dir", "log", "started_at"):
        if not isinstance(pending.get(field), str) or not pending[field]:
            fail(f"pending-attempt.json field {field!r} must be a non-empty string")
    run_dir = path_from_manifest(pending["run_dir"])
    try:
        run_dir.relative_to(config["_run_output_root"])
    except ValueError:
        fail("pending run directory is outside configured run_output_root")
    log_path = path_from_manifest(pending["log"])
    try:
        log_path.relative_to(config["_artifact_dir"])
    except ValueError:
        fail("pending log is outside configured campaign_artifact_dir")
    return pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "configuration.example.json",
        help="JSON campaign configuration (default: configuration.example.json)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform live model requests; default is read-only preflight",
    )
    parser.add_argument(
        "--max-attempts-per-slot",
        type=int,
        help="durable attempt cap per slot; defaults to configuration value",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="wall-clock timeout per conversation; defaults to configuration value",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = validate_configuration(
        config_path, require_serving_verified=args.execute
    )
    schedule = validate_schedule(config)
    attempts, canonical = validate_manifests(config, schedule)
    verify_source_hashes(config)
    max_attempts = (
        args.max_attempts_per_slot
        if args.max_attempts_per_slot is not None
        else config["collection"]["max_attempts_per_slot_default"]
    )
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else config["collection"]["timeout_seconds_default"]
    )
    if max_attempts < 1 or timeout_seconds < 1:
        fail("attempt cap and timeout must be positive")
    print(
        f"Preflight OK: campaign={config['campaign_id']}, model={config['model']}, "
        f"endpoint={config['endpoint']['url']}, canonical={len(canonical)}/"
        f"{config['target_eligible_runs']}, concurrency=1"
    )
    print(f"  artifacts: {config['_artifact_dir']}")
    print(f"  run output root: {config['_run_output_root']}")
    if not args.execute:
        print(
            "Read-only preflight only. No directory, manifest, lock, credential, "
            "or endpoint was touched. Pass --execute after the smoke gate is recorded."
        )
        return 0

    initialize_artifacts(config)
    paths = artifact_paths(config)
    credential = extract_credential(config)
    with paths["lock"].open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another collection process holds the configured campaign lock")

        if not paths["source_hash"].exists():
            atomic_write(paths["source_hash"], source_hashes(config))

        append_campaign_log(
            config,
            f"COLLECTOR_START max_attempts_per_slot={max_attempts} "
            f"timeout_seconds={timeout_seconds} canonical={len(canonical)}/"
            f"{config['target_eligible_runs']}",
        )

        if paths["pending"].exists():
            pending = validate_pending_shape(config, read_json(paths["pending"]))
            print(
                f"recovering pending slot {pending['slot']:03d} "
                f"attempt {pending['attempt']}"
            )
            finalize_pending(config, schedule, pending)

        while True:
            attempts, canonical = validate_manifests(config, schedule)
            if len(canonical) == config["target_eligible_runs"]:
                break
            slot = len(canonical) + 1
            arm = schedule[slot - 1]["arm"]
            attempt = next_attempt_number(attempts, slot)
            if attempt > max_attempts:
                append_campaign_log(
                    config,
                    f"RUN_COLLECTION_STOP slot={slot} reason=attempt_cap cap={max_attempts}",
                )
                print(
                    f"slot {slot:03d}: attempt cap exhausted; rerun with a "
                    "deliberately higher --max-attempts-per-slot",
                    file=sys.stderr,
                )
                return 2
            print(f"slot {slot:03d}: starting arm={arm} attempt={attempt}")
            pending = run_attempt(
                config,
                slot=slot,
                arm=arm,
                attempt=attempt,
                credential=credential,
                timeout_seconds=timeout_seconds,
                lock_fd=lock_handle.fileno(),
            )
            eligible = finalize_pending(config, schedule, pending)
            if eligible:
                print(f"slot {slot:03d}: canonical")
            else:
                print(f"slot {slot:03d}: ineligible; replacement required")

        final = read_tsv(paths["canonical"])
        counts: dict[str, int] = {}
        for row in final:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        count_text = " ".join(f"{arm}={count}" for arm, count in sorted(counts.items()))
        append_campaign_log(
            config,
            f"RUN_COLLECTION_DONE total={len(final)} {count_text}".rstrip(),
        )
        append_campaign_log(config, "CAMPAIGN_COLLECTION_DONE")
        print(f"Collection complete: {len(final)} canonical conversations ({count_text}).")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
