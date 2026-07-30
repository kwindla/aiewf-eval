#!/usr/bin/env python3
"""Strictly sequential, resumable N=30 AIEWF collector for Gemma 4 26B A4B.

The default invocation is a read-only preflight. Pass ``--execute`` only after
the deployment smoke gate has been recorded in ``configuration.json``.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "configuration.json"
FROZEN_PATH = HERE / "frozen-order.tsv"
ATTEMPTS_PATH = HERE / "attempts.tsv"
CANONICAL_PATH = HERE / "canonical.tsv"
CAMPAIGN_LOG = HERE / "campaign.log"
LOCK_PATH = HERE / ".collection.lock"
RUN_LOG_DIR = HERE / "logs"
SOURCE_HASH_PATH = HERE / "source-sha256.txt"

EXPECTED_ENDPOINT = (
    "https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1"
)
EXPECTED_MODEL = "google/gemma-4-26B-A4B-it"
EXPECTED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_tokens": 8192,
}
TARGET = 30
N_TURNS = 30

ATTEMPT_FIELDS = (
    "slot",
    "mode",
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
    "mode",
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


def append_campaign_log(message: str) -> None:
    with CAMPAIGN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def validate_configuration(*, require_serving_verified: bool) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "benchmark": "aiwf_medium_context",
        "model": EXPECTED_MODEL,
        "endpoint": EXPECTED_ENDPOINT,
        "filler": None,
        "target_eligible_runs": TARGET,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(
                f"configuration mismatch for {key}: "
                f"expected {value!r}, found {config.get(key)!r}"
            )
    if config.get("sampling") != EXPECTED_SAMPLING:
        fail(f"sampling configuration is not frozen: {config.get('sampling')!r}")
    arm = config.get("arm") or {}
    if arm.get("name") != "none" or arm.get("enable_thinking") is not False:
        fail("campaign must contain only the explicitly thinking-off arm")
    collection = config.get("collection") or {}
    if collection.get("provider_endpoint_concurrency") != 1:
        fail("provider endpoint concurrency must be exactly one")
    serving = config.get("serving") or {}
    if require_serving_verified:
        if serving.get("verified") is not True:
            fail(
                "deployment smoke gate is not complete: set serving.verified=true "
                "only after streaming/tool/continuation/prefix/MTP validation"
            )
        if str(serving.get("vllm_version", "")).startswith("PENDING"):
            fail("record the exact live vLLM version before collection")
        mtp = serving.get("mtp") or {}
        if str(mtp.get("status", "")).startswith("PENDING"):
            fail("record the MTP smoke-gate disposition before collection")
    return config


def validate_manifests() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if FROZEN_PATH.read_text(encoding="utf-8").splitlines()[0] != "slot\tmode":
        fail("unexpected frozen-order.tsv header")
    if (
        CANONICAL_PATH.read_text(encoding="utf-8").splitlines()[0]
        != "\t".join(CANONICAL_FIELDS)
    ):
        fail("unexpected canonical.tsv header")
    if (
        ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines()[0]
        != "\t".join(ATTEMPT_FIELDS)
    ):
        fail("unexpected attempts.tsv header")
    frozen = read_tsv(FROZEN_PATH)
    if len(frozen) != TARGET:
        fail(f"frozen-order.tsv must contain {TARGET} slots")
    expected_slots = list(range(1, TARGET + 1))
    slots = [int(row["slot"]) for row in frozen]
    if slots != expected_slots or any(row["mode"] != "none" for row in frozen):
        fail("frozen order must be exactly slots 1..30 in mode none")

    canonical = read_tsv(CANONICAL_PATH)
    if len(canonical) > TARGET:
        fail(f"canonical.tsv has more than {TARGET} rows")
    canonical_slots = [int(row["slot"]) for row in canonical]
    if canonical_slots != list(range(1, len(canonical) + 1)):
        fail("canonical.tsv must be a unique contiguous prefix of frozen slots")
    if len(set(row["run_dir"] for row in canonical)) != len(canonical):
        fail("canonical.tsv contains duplicate run directories")
    for row in canonical:
        slot = int(row["slot"])
        if row["mode"] != "none" or frozen[slot - 1]["mode"] != "none":
            fail(f"canonical/frozen mismatch at slot {slot}")
        run_dir = resolve_run_dir(row["run_dir"])
        validation = validate_transcript(run_dir)
        validate_run_provenance(run_dir)
        for field in ("turns", "response_turns", "tool_calls"):
            if int(row[field]) != int(validation[field]):
                fail(f"canonical {field} mismatch at slot {slot}")
        if row["classification"] != validation["classification"]:
            fail(f"canonical classification mismatch at slot {slot}")
    return frozen, canonical


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run directory escapes repository root: {value}")
    return resolved


def validate_transcript(run_dir: Path) -> dict[str, int | str]:
    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.is_file() or transcript_path.stat().st_size == 0:
        fail(f"missing transcript: {transcript_path}")
    rows = read_jsonl(transcript_path)
    scheduled: dict[int, dict[str, Any]] = {}
    response_turns = 0
    tool_calls = 0
    for row in rows:
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"invalid scheduled turn in {transcript_path}: {turn!r}")
        if turn in scheduled:
            fail(f"duplicate scheduled turn {turn} in {transcript_path}")
        if row.get("model_name") != EXPECTED_MODEL:
            fail(
                f"model mismatch at turn {turn}: "
                f"{row.get('model_name')!r} != {EXPECTED_MODEL!r}"
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
            "complete_30" if len(scheduled) == N_TURNS else "fixed_denominator_short"
        ),
    }


def validate_run_provenance(run_dir: Path) -> None:
    run_log = run_dir / "run.log"
    if not run_log.is_file():
        fail(f"missing standard run.log: {run_log}")
    text = run_log.read_text(encoding="utf-8", errors="replace")
    required = (
        f"base_url={EXPECTED_ENDPOINT}",
        f"model={EXPECTED_MODEL}",
        "thinking=False",
        "T=1.0",
        "top_p=0.95",
        "top_k=64",
        "max_tokens=8192",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"run provenance is missing {missing}: {run_log}")


def candidate_run_dir(log_path: Path) -> Path | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"(?:Transcript:|Output directory:)\s+"
        r"(runs/aiwf_medium_context/[^\s/]+)(?:/transcript\.jsonl)?",
        text,
    )
    if not matches:
        return None
    return resolve_run_dir(matches[-1])


def extract_key(explicit_env_name: str, fallback_path: Path) -> str:
    for name in (explicit_env_name, "BASETEN_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    if fallback_path.is_file():
        prefix = "BASETEN_API_KEY="
        for line in fallback_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                value = line[len(prefix) :]
                if value:
                    return value
    fail(
        "BaseTen key unavailable; set VLLM_API_KEY or BASETEN_API_KEY, "
        "or provide --key-file"
    )


def child_environment(api_key: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("MTE_FILLER_"):
            env.pop(name, None)
    env.update(
        {
            "VLLM_BASE_URL": EXPECTED_ENDPOINT,
            "VLLM_API_KEY": api_key,
            "MTE_VLLM_THINKING": "0",
            "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_TOP_P": "0.95",
            "MTE_VLLM_TOP_K": "64",
            "MTE_VLLM_MAX_TOKENS": "8192",
        }
    )
    env.pop("MTE_VLLM_THINKING_BUDGET", None)
    env.pop("MTE_VLLM_NATIVE_BUDGET", None)
    return env


def source_hashes() -> str:
    paths = (
        CONFIG_PATH,
        FROZEN_PATH,
        Path(__file__).resolve(),
        ROOT / "src/multi_turn_eval/cli.py",
        ROOT / "src/multi_turn_eval/pipelines/base.py",
        ROOT / "src/multi_turn_eval/pipelines/text.py",
        ROOT / "src/multi_turn_eval/services/vllm_openai.py",
        ROOT / "benchmarks/aiwf_medium_context/config.py",
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    return "\n".join(lines) + "\n"


def next_attempt_number(slot: int) -> int:
    attempts = [
        int(row["attempt"])
        for row in read_tsv(ATTEMPTS_PATH)
        if int(row["slot"]) == slot
    ]
    return max(attempts, default=0) + 1


def run_attempt(
    *,
    slot: int,
    attempt: int,
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, Path, Path | None, str | None]:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"slot{slot:02d}-none-attempt{attempt:02d}.log"
    command = [
        str(ROOT / ".venv/bin/multi-turn-eval"),
        "run",
        "aiwf_medium_context",
        "--model",
        EXPECTED_MODEL,
        "--service",
        "vllm-openai",
    ]
    started_at = utc_now()
    append_campaign_log(
        f"RUN_START slot={slot} mode=none attempt={attempt} log={log_path.relative_to(ROOT)}"
    )
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=child_environment(api_key),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            log_handle.write(
                f"\nCOLLECTOR_TIMEOUT seconds={timeout_seconds} at={utc_now()}\n"
            )
    run_dir = candidate_run_dir(log_path)
    finished_at = utc_now()
    append_campaign_log(
        f"RUN_EXIT slot={slot} mode=none attempt={attempt} rc={exit_code} "
        f"timeout={str(timed_out).lower()} run_dir="
        f"{run_dir.relative_to(ROOT) if run_dir else '(none)'}"
    )
    return exit_code, log_path, run_dir, f"{started_at}\t{finished_at}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform live model requests; default is read-only preflight",
    )
    parser.add_argument(
        "--max-attempts-per-slot",
        type=int,
        default=3,
        help="total durable attempt cap per slot, including earlier invocations",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="wall-clock timeout for one 30-turn conversation",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("/home/khkramer/src/gb-benchmarks/.env"),
        help="fallback file from which only BASETEN_API_KEY is read",
    )
    args = parser.parse_args()
    if args.max_attempts_per_slot < 1:
        fail("--max-attempts-per-slot must be positive")

    validate_configuration(require_serving_verified=args.execute)
    frozen, canonical = validate_manifests()
    if SOURCE_HASH_PATH.exists():
        if SOURCE_HASH_PATH.read_text(encoding="utf-8") != source_hashes():
            fail("campaign source hashes changed after collection began")
    print(
        f"Preflight OK: model={EXPECTED_MODEL}, endpoint={EXPECTED_ENDPOINT}, "
        f"canonical={len(canonical)}/{TARGET}, concurrency=1"
    )
    if not args.execute:
        print(
            "Read-only preflight only. No endpoint request was made. "
            "Pass --execute after the smoke gate is recorded."
        )
        return 0

    api_key = extract_key("VLLM_API_KEY", args.key_file)
    with LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another collection process holds .collection.lock")

        if not SOURCE_HASH_PATH.exists():
            SOURCE_HASH_PATH.write_text(source_hashes(), encoding="utf-8")

        canonical_slots = {int(row["slot"]) for row in canonical}
        for frozen_row in frozen:
            slot = int(frozen_row["slot"])
            if slot in canonical_slots:
                print(f"slot {slot:02d}: already canonical; skipping")
                continue
            while True:
                attempt = next_attempt_number(slot)
                if attempt > args.max_attempts_per_slot:
                    append_campaign_log(
                        f"RUN_COLLECTION_STOP slot={slot} reason=attempt_cap "
                        f"cap={args.max_attempts_per_slot}"
                    )
                    print(
                        f"slot {slot:02d}: attempt cap exhausted; "
                        "rerun later with a deliberately higher cap",
                        file=sys.stderr,
                    )
                    return 2
                print(f"slot {slot:02d}: starting attempt {attempt}")
                exit_code, log_path, run_dir, times = run_attempt(
                    slot=slot,
                    attempt=attempt,
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                )
                started_at, finished_at = (times or "\t").split("\t", 1)
                classification = "ineligible_no_valid_response"
                response_turns = 0
                validation: dict[str, int | str] | None = None
                validation_error = ""
                if run_dir is not None:
                    try:
                        validation = validate_transcript(run_dir)
                        validate_run_provenance(run_dir)
                        classification = str(validation["classification"])
                        response_turns = int(validation["response_turns"])
                    except Exception as exc:
                        validation_error = str(exc)
                append_tsv(
                    ATTEMPTS_PATH,
                    ATTEMPT_FIELDS,
                    {
                        "slot": slot,
                        "mode": "none",
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "exit_code": exit_code,
                        "run_dir": (
                            str(run_dir.relative_to(ROOT)) if run_dir else ""
                        ),
                        "response_turns": response_turns,
                        "classification": classification,
                        "log": str(log_path.relative_to(ROOT)),
                    },
                )
                if validation is None:
                    append_campaign_log(
                        f"RUN_INELIGIBLE slot={slot} attempt={attempt} "
                        f"reason={json.dumps(validation_error or classification)}"
                    )
                    print(
                        f"slot {slot:02d}: ineligible attempt {attempt}: "
                        f"{validation_error or classification}",
                        file=sys.stderr,
                    )
                    continue
                assert run_dir is not None
                append_tsv(
                    CANONICAL_PATH,
                    CANONICAL_FIELDS,
                    {
                        "slot": slot,
                        "mode": "none",
                        "attempt": attempt,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        **validation,
                    },
                )
                canonical_slots.add(slot)
                append_campaign_log(
                    f"RUN_CANONICAL slot={slot} attempt={attempt} "
                    f"classification={classification} "
                    f"run_dir={run_dir.relative_to(ROOT)}"
                )
                print(
                    f"slot {slot:02d}: canonical {classification}, "
                    f"responses={response_turns}"
                )
                break

        final = read_tsv(CANONICAL_PATH)
        if len(final) != TARGET:
            fail(f"collection ended with {len(final)}/{TARGET} canonical runs")
        append_campaign_log("RUN_COLLECTION_DONE total=30 none=30")
        append_campaign_log("CAMPAIGN_COLLECTION_DONE")
        print("Collection complete: 30 canonical thinking-off conversations.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
