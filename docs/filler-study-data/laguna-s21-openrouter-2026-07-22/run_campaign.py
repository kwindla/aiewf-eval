#!/usr/bin/env python3
"""Resumable run-only driver for the Laguna S 2.1 OpenRouter screen."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "poolside/laguna-s-2.1"
MODEL_KEY = "laguna_s21"
MAX_ATTEMPTS = 4
INFRA_RE = re.compile(
    r"DeadlineExceeded|ResourceExhausted|ReadTimeout|ConnectTimeout|"
    r"Connection(?:Error|Reset|Refused)|RateLimitError|Too Many Requests|"
    r"rate limit(?: exceeded| error)|HTTP[/ ]+5\d\d|"
    r"HTTP[/ ]+429|status(?: code)?[=: ]+429|InternalServerError|"
    r"ServiceUnavailable|Upstream error",
    re.IGNORECASE | re.MULTILINE,
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def end_session_turn(run_dir: Path) -> int:
    best = -1
    transcript = run_dir / "transcript.jsonl"
    if not transcript.is_file():
        return best
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            best = max(best, int(row.get("turn", -1)))
    return best


def validate_run(run_dir: Path, arm: str) -> None:
    transcript = run_dir / "transcript.jsonl"
    run_log = run_dir / "run.log"
    if not transcript.is_file() or not transcript.stat().st_size or not run_log.is_file():
        raise RuntimeError(f"incomplete run artifact: {run_dir}")
    log = run_log.read_text()
    required = (
        "Using OpenRouter with base_url=https://openrouter.ai/api/v1, reasoning_off=True, max_tokens=8192",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    if any(signature not in log for signature in required):
        raise RuntimeError(f"runtime signature mismatch: {run_dir}")
    filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    if arm == "dots96" and log.count(filler) != 1:
        raise RuntimeError(f"missing or repeated filler signature: {run_dir}")
    if arm == "nofiller" and "MTE_FILLER_DOTS active:" in log:
        raise RuntimeError(f"filler leaked into control: {run_dir}")
    thinking_tokens = 0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != MODEL:
            raise RuntimeError(f"model mismatch: {run_dir}")
        thinking_tokens += int((row.get("tokens") or {}).get("thinking_tokens") or 0)
    if thinking_tokens:
        raise RuntimeError(f"reasoning-off run reported {thinking_tokens} thinking tokens: {run_dir}")


def run_attempt(arm: str, log_path: Path) -> tuple[int, str, Path | None]:
    env = os.environ.copy()
    for key in (
        "MTE_FILLER_DOTS",
        "MTE_FILLER_TOKEN",
        "MTE_FILLER_POSITION",
        "MTE_OPENROUTER_TEMPERATURE",
    ):
        env.pop(key, None)
    env.update(
        {
            "MTE_OPENROUTER_REASONING_OFF": "1",
            "MTE_OPENROUTER_MAX_TOKENS": "8192",
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
    proc = subprocess.run(
        [
            "uv",
            "run",
            "multi-turn-eval",
            "run",
            "aiwf_medium_context",
            "--model",
            MODEL,
            "--service",
            "openrouter",
            "--pipeline",
            "text",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout)
    matches = re.findall(r"^Output directory: (.+)$", proc.stdout, flags=re.MULTILINE)
    run_dir = Path(matches[-1]) if matches else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return proc.returncode, proc.stdout, run_dir


def main() -> None:
    schedule = HERE / "schedule.tsv"
    state = HERE / "state"
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    attempts_path = state / "attempts.tsv"
    counted_path = state / "counted.tsv"
    manifest_path = state / "manifest.tsv"
    driver_log = state / "driver.log"
    with (state / "driver.lock").open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another Laguna campaign driver is active") from exc

        def log(message: str) -> None:
            line = f"[{now()}] {message}"
            print(line, flush=True)
            with driver_log.open("a") as handle:
                handle.write(line + "\n")

        counted = {row["slot"] for row in read_tsv(counted_path)}
        for assignment in read_tsv(schedule):
            slot = assignment["slot"]
            arm = assignment["arm"]
            if (
                assignment["model"] != MODEL_KEY
                or assignment["requested_model"] != MODEL
                or assignment["service"] != "openrouter"
                or arm not in {"nofiller", "dots96"}
            ):
                raise RuntimeError(f"schedule policy failure in {slot}")
            if slot in counted:
                continue

            prior = [
                row
                for row in read_tsv(attempts_path)
                if row["slot"] == slot and not row["classification"].startswith("infra_")
            ]
            candidate: Path | None = None
            classification = ""
            attempt = len([row for row in read_tsv(attempts_path) if row["slot"] == slot])
            if prior:
                candidate = Path(prior[-1]["run_dir"])
                classification = prior[-1]["classification"]
                log(f"adopting slot={slot} run={candidate}")

            while candidate is None:
                attempt += 1
                if attempt > MAX_ATTEMPTS:
                    raise RuntimeError(f"replacement limit reached: {slot}")
                run_output = logs / f"{slot}-attempt-{attempt}.log"
                started = now()
                log(f"run slot={slot} attempt={attempt} arm={arm}")
                rc, output, run_dir = run_attempt(arm, run_output)
                rows = 0
                es_turn = -1
                if run_dir is not None and (run_dir / "transcript.jsonl").is_file():
                    rows = len((run_dir / "transcript.jsonl").read_text().splitlines())
                    es_turn = end_session_turn(run_dir)
                # Scan the concise CLI output, not run.log: the latter includes
                # the complete prompt and ordinary decimal timing values, both
                # of which can contain strings resembling HTTP error markers.
                infra = es_turn < 0 and bool(INFRA_RE.search(output))
                if infra:
                    classification = "infra_zero_response_replaced" if rows == 0 else "infra_partial_response_replaced"
                elif rows == 0:
                    classification = "zero_response_unclassified"
                elif es_turn == 29:
                    classification = "strict_complete"
                elif es_turn >= 0:
                    classification = "model_abort"
                else:
                    classification = "incomplete_no_end_session"
                append_tsv(
                    attempts_path,
                    [
                        "slot",
                        "model",
                        "arm",
                        "attempt",
                        "start_utc",
                        "end_utc",
                        "run_rc",
                        "run_dir",
                        "transcript_rows",
                        "end_session_turn",
                        "classification",
                        "log",
                    ],
                    {
                        "slot": slot,
                        "model": MODEL_KEY,
                        "arm": arm,
                        "attempt": attempt,
                        "start_utc": started,
                        "end_utc": now(),
                        "run_rc": rc,
                        "run_dir": run_dir or "NA",
                        "transcript_rows": rows,
                        "end_session_turn": es_turn,
                        "classification": classification,
                        "log": run_output,
                    },
                )
                log(f"attempt slot={slot} rc={rc} rows={rows} end_session={es_turn} class={classification}")
                if classification.startswith("infra_"):
                    continue
                if classification == "zero_response_unclassified" or run_dir is None:
                    raise RuntimeError(f"unclassified zero-response: {slot}")
                candidate = run_dir

            validate_run(candidate, arm)
            append_tsv(
                counted_path,
                ["slot", "model", "arm", "attempt", "run_dir", "classification"],
                {
                    "slot": slot,
                    "model": MODEL_KEY,
                    "arm": arm,
                    "attempt": attempt,
                    "run_dir": candidate,
                    "classification": classification,
                },
            )
            append_tsv(
                manifest_path,
                ["slot", "model", "arm", "run_dir", "classification"],
                {
                    "slot": slot,
                    "model": MODEL_KEY,
                    "arm": arm,
                    "run_dir": candidate.relative_to(ROOT),
                    "classification": classification,
                },
            )
            counted.add(slot)
            log(f"counted slot={slot} class={classification} run={candidate}")

        (state / "RUNS_COMPLETE").touch()
        log("RUNS_COMPLETE")


if __name__ == "__main__":
    main()
