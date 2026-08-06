#!/usr/bin/env python3
"""Resumable lane runner for the Gemini 2.5 Flash thinking-off campaign."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "gemini-2.5-flash"
MODEL_KEY = "gemini25flash"
LANES = {"control", "control-topup", "dots", "dots-topup", "focused"}
MAX_ATTEMPTS = 4
SOURCE_HASHES_STAGE1 = {
    ROOT / "benchmarks/aiwf_medium_context/config.py": "ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164",
    ROOT / "benchmarks/aiwf_medium_context/prompts/system.py": "6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6",
    ROOT / "benchmarks/_shared/turns.py": "c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b",
    ROOT / "src/multi_turn_eval/pipelines/base.py": "eaa2b36ce5efd591d0657b37e904f64c339cd8feb7102754e670c01e0bd53d35",
    ROOT / "src/multi_turn_eval/services/google_logged.py": "97294f5a086d9516ff501c638aa14d525e67cceb11e8df692f50c8f0d1c227c3",
    ROOT / "src/multi_turn_eval/judging/claude_judge.py": "3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18",
    ROOT / "tests/test_google_filler.py": "f7eb1859cb10c8264b7a1f9897666d878e79b73e80a16a42135997f46015ba44",
}
SOURCE_HASHES_CONTROL_TOPUP = {
    **SOURCE_HASHES_STAGE1,
    ROOT / "src/multi_turn_eval/pipelines/base.py": "70b77c51da6dd6232d4aa44aa2b1c95922e21200cabbb65ee5abf76cbbb06a98",
    ROOT / "tests/test_google_filler.py": "0cda16dbefc4c48da5beb6bc5e0b14f1140e596399bd757b55e1d814bddc21fb",
}
SCHEDULE_HASHES = {
    "control": "de13ddb7039f196eac4a144618dd936f28e688b65d188d0cec35990a916306f0",
    "control-topup": "5911a44345638fa322fd19d1d01d63e8fdf5378aea0b6389d2022bbf3c7763cc",
    "dots": "1f5e855e3f39e80b22bd5ed7f8ac80f5fb5caaf29e9aa360b139bc8437e1ff14",
}
INFRA_RE = re.compile(
    r"DeadlineExceeded|ResourceExhausted|ReadTimeout|ConnectTimeout|"
    r"Connection(?:Error|Reset|Refused)|rate.?limit|HTTP[/ ]+5\d\d|"
    r"(?:^|\D)429(?:\D|$)|InternalServerError|ServiceUnavailable|Upstream error",
    re.IGNORECASE | re.MULTILINE,
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_integrity(lane: str) -> None:
    source_hashes = (
        SOURCE_HASHES_CONTROL_TOPUP if lane == "control-topup" else SOURCE_HASHES_STAGE1
    )
    for path, expected in source_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen source changed: {path}")
    schedule = HERE / f"schedule-{lane}.tsv"
    if lane in SCHEDULE_HASHES and sha256(schedule) != SCHEDULE_HASHES[lane]:
        raise RuntimeError(f"frozen schedule changed: {schedule}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_tsv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def valid_judgment(run_dir: Path) -> bool:
    transcript = run_dir / "transcript.jsonl"
    judged = run_dir / "claude_judged.jsonl"
    summary = run_dir / "claude_summary.json"
    if not all(path.is_file() and path.stat().st_size for path in (transcript, judged, summary)):
        return False
    observed = set()
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < 30 and row.get("recovery_turn") is not True:
            observed.add(turn)
    final = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and turn in observed:
            final[turn] = row
    if set(final) != observed:
        return False
    for row in final.values():
        scores = row.get("scores") or {}
        if not all(isinstance(scores.get(key), bool) for key in (
            "tool_use_correct", "instruction_following", "kb_grounding"
        )):
            return False
    meta = json.loads(summary.read_text())
    return meta.get("turns_scored") == len(observed) and bool(meta.get("judge_model"))


def end_session_turn(run_dir: Path) -> int:
    best = -1
    for line in (run_dir / "transcript.jsonl").read_text().splitlines():
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
        "Configured gemini-2.5-flash with thinking_budget=0 (disabled)",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    if any(signature not in log for signature in required):
        raise RuntimeError(f"runtime signature mismatch: {run_dir}")
    filler = "MTE_FILLER_DOTS active: 96 x '.' filler tokens, position=suffix"
    if (arm == "dots96" and log.count(filler) != 1) or (arm == "nofiller" and "MTE_FILLER_DOTS active:" in log):
        raise RuntimeError(f"filler signature mismatch: {run_dir}")
    thought_tokens = 0
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if row.get("model_name") != MODEL:
            raise RuntimeError(f"model mismatch: {run_dir}")
        thought_tokens += int((row.get("tokens") or {}).get("thinking_tokens") or 0)
    if thought_tokens != 0:
        raise RuntimeError(f"thinking-off run reported {thought_tokens} thought tokens: {run_dir}")


def judge(slot: str, run_dir: Path, judge_log: Path, judge_lock: Path) -> None:
    if valid_judgment(run_dir):
        return
    judge_lock.parent.mkdir(parents=True, exist_ok=True)
    with judge_lock.open("a") as lock_handle:
        for attempt in range(1, 4):
            started = now()
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                proc = subprocess.run(
                    ["uv", "run", "multi-turn-eval", "judge", str(run_dir)],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                output = proc.stdout
                rc = proc.returncode
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
            log_path = run_dir / f"judge-gemini25-attempt-{attempt}.log"
            log_path.write_text(output)
            append_tsv(judge_log, ["slot", "run_dir", "attempt", "start_utc", "end_utc", "judge_rc", "log"], {
                "slot": slot, "run_dir": run_dir, "attempt": attempt,
                "start_utc": started, "end_utc": now(), "judge_rc": rc, "log": log_path,
            })
            if rc == 0 and valid_judgment(run_dir):
                return
    raise RuntimeError(f"judge failed after retries: {run_dir}")


def run_attempt(arm: str, log_path: Path) -> tuple[int, str, Path | None]:
    env = os.environ.copy()
    for key in ("MTE_FILLER_DOTS", "MTE_FILLER_TOKEN", "MTE_FILLER_POSITION"):
        env.pop(key, None)
    env.update({
        "MTE_GOOGLE_THINKING_MODE": "disabled",
        "MTE_ENABLE_RECOVERY": "1",
        "MTE_DEDUPE_TOOL_CALLS": "1",
        "MTE_TOOL_RESULT_RUN_LLM": "0",
        "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
    })
    if arm == "dots96":
        env.update({"MTE_FILLER_DOTS": "96", "MTE_FILLER_TOKEN": ".", "MTE_FILLER_POSITION": "suffix"})
    proc = subprocess.run(
        ["uv", "run", "multi-turn-eval", "run", "aiwf_medium_context", "--model", MODEL,
         "--service", "google", "--pipeline", "text"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log_path.write_text(proc.stdout)
    matches = re.findall(r"^Output directory: (.+)$", proc.stdout, flags=re.MULTILINE)
    run_dir = Path(matches[-1]) if matches else None
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return proc.returncode, proc.stdout, run_dir


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in LANES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} {{{'|'.join(sorted(LANES))}}}")
    lane = sys.argv[1]
    schedule = HERE / f"schedule-{lane}.tsv"
    if not schedule.is_file():
        raise SystemExit(f"schedule does not exist: {schedule}")
    validate_integrity(lane)
    state = HERE / "state" / lane
    logs = state / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = state / "driver.lock"
    attempts_path = state / "attempts.tsv"
    counted_path = state / "counted.tsv"
    manifest_path = state / "manifest.tsv"
    judge_log = state / "judge-attempts.tsv"
    judge_lock = HERE / "state" / "judge.lock"
    driver_log = state / "driver.log"
    with lock_path.open("a") as lane_lock:
        try:
            fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another driver owns lane {lane}") from exc

        def log(message: str) -> None:
            line = f"[{now()}] {message}"
            print(line, flush=True)
            with driver_log.open("a") as handle:
                handle.write(line + "\n")

        counted = {row["slot"] for row in read_tsv(counted_path)}
        for assignment in read_tsv(schedule):
            slot = assignment["slot"]
            model = assignment["model"]
            arm = assignment["arm"]
            requested = assignment["requested_model"]
            if model != MODEL_KEY or requested != MODEL or arm not in {"nofiller", "dots96"}:
                raise RuntimeError(f"model policy failure in slot {slot}")
            if slot in counted:
                continue
            prior = [row for row in read_tsv(attempts_path) if row["slot"] == slot and not row["classification"].startswith("infra_")]
            run_dir: Path | None = None
            classification = ""
            attempt = len([row for row in read_tsv(attempts_path) if row["slot"] == slot])
            if prior:
                run_dir = Path(prior[-1]["run_dir"])
                classification = prior[-1]["classification"]
                log(f"adopting uncommitted slot={slot} run={run_dir}")
            while run_dir is None:
                validate_integrity(lane)
                attempt += 1
                if attempt > MAX_ATTEMPTS:
                    raise RuntimeError(f"replacement limit reached: {slot}")
                run_output = logs / f"{slot}-attempt-{attempt}.log"
                started = now()
                log(f"run slot={slot} attempt={attempt} arm={arm}")
                rc, output, candidate = run_attempt(arm, run_output)
                rows = 0
                es_turn = -1
                combined = output
                if candidate is not None and (candidate / "transcript.jsonl").is_file():
                    rows = len((candidate / "transcript.jsonl").read_text().splitlines())
                    es_turn = end_session_turn(candidate)
                    if (candidate / "run.log").is_file():
                        combined += "\n" + (candidate / "run.log").read_text()
                infra = es_turn < 0 and bool(INFRA_RE.search(combined))
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
                append_tsv(attempts_path, [
                    "slot", "model", "arm", "attempt", "start_utc", "end_utc", "run_rc", "run_dir",
                    "transcript_rows", "end_session_turn", "classification", "log",
                ], {
                    "slot": slot, "model": model, "arm": arm, "attempt": attempt,
                    "start_utc": started, "end_utc": now(), "run_rc": rc,
                    "run_dir": candidate or "NA", "transcript_rows": rows,
                    "end_session_turn": es_turn, "classification": classification, "log": run_output,
                })
                log(f"attempt slot={slot} rc={rc} rows={rows} end_session={es_turn} class={classification}")
                if classification.startswith("infra_"):
                    continue
                if classification == "zero_response_unclassified" or candidate is None:
                    raise RuntimeError(f"unclassified zero-response: {slot}")
                run_dir = candidate
            validate_run(run_dir, arm)
            judge(slot, run_dir, judge_log, judge_lock)
            append_tsv(counted_path, ["slot", "model", "arm", "attempt", "run_dir", "classification", "judge_rc"], {
                "slot": slot, "model": model, "arm": arm, "attempt": attempt,
                "run_dir": run_dir, "classification": classification, "judge_rc": 0,
            })
            append_tsv(manifest_path, ["model", "arm", "run_dir"], {
                "model": model, "arm": arm, "run_dir": run_dir.relative_to(ROOT),
            })
            counted.add(slot)
            log(f"counted slot={slot} class={classification} run={run_dir}")
        (state / "COMPLETE").touch()
        log(f"COMPLETE lane={lane}")


if __name__ == "__main__":
    main()
