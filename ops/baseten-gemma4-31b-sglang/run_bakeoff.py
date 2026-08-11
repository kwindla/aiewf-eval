#!/usr/bin/env python3
"""Run the matched Gemma 4 31B SGLang/vLLM serving bakeoff."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "bakeoff-20260806-normalized"
LOGS = ARTIFACTS / "logs"
RESULTS = ARTIFACTS / "attempts.json"
MODEL = "google/gemma-4-31B-it"
RUN_COMMAND = [
    str(ROOT / ".venv" / "bin" / "multi-turn-eval"),
    "run",
    "aiwf_medium_context",
    "--model",
    MODEL,
    "--service",
    "vllm-openai",
    "--pipeline",
    "text",
]

DEPLOYMENTS = {
    "sglang_no_mtp": {
        "endpoint": "https://model-qel8v803.api.baseten.co/deployment/324j8o2/sync/v1",
        "control": "https://api.baseten.co/v1/models/qel8v803/deployments/324j8o2",
    },
    "sglang_mtp": {
        "endpoint": "https://model-q951m16w.api.baseten.co/deployment/q862ez8/sync/v1",
        "control": "https://api.baseten.co/v1/models/q951m16w/deployments/q862ez8",
    },
    "vllm_no_mtp": {
        "endpoint": "https://model-qzk215kq.api.baseten.co/deployment/wgvde5j/sync/v1",
        "control": "https://api.baseten.co/v1/models/qzk215kq/deployments/wgvde5j",
    },
}
EXISTING_VLLM_RUN = (
    ROOT / "runs/aiwf_medium_context/20260806T094200_google_gemma-4-31B-it_8b8db844"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_key() -> str:
    for path in (ROOT / ".env", ROOT.parent / "gb-benchmarks" / ".env"):
        if path.exists():
            values = dotenv_values(path)
            if values.get("BASETEN_API_KEY"):
                return str(values["BASETEN_API_KEY"])
    raise RuntimeError("BASETEN_API_KEY not found")


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}


def set_minimum(key: str, label: str, minimum: int) -> None:
    response = requests.patch(
        DEPLOYMENTS[label]["control"] + "/autoscaling_settings",
        headers=headers(key),
        json={
            "min_replica": minimum,
            "max_replica": 1,
            "autoscaling_window": 60,
            "scale_down_delay": 120,
            "concurrency_target": 1,
            "target_utilization_percentage": 70,
        },
        timeout=30,
    )
    response.raise_for_status()


def wait_state(key: str, label: str, *, active: bool) -> dict[str, Any]:
    deadline = time.monotonic() + (30 * 60 if active else 15 * 60)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = requests.get(
            DEPLOYMENTS[label]["control"], headers=headers(key), timeout=30
        )
        response.raise_for_status()
        last = response.json()
        status = last.get("status")
        replicas = int(last.get("active_replica_count") or 0)
        if active and status == "ACTIVE" and replicas >= 1:
            return last
        if not active and status == "SCALED_TO_ZERO" and replicas == 0:
            return last
        time.sleep(10)
    raise TimeoutError(
        f"{label} did not reach {'ACTIVE' if active else 'SCALED_TO_ZERO'}: {last}"
    )


def clean_environment(
    key: str, endpoint: str, *, normalize_tool_call_indices: bool
) -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("MTE_") or name in {
            "VLLM_BASE_URL",
            "VLLM_API_KEY",
            "BASETEN_BASE_URL",
            "LILAC_BASE_URL",
        }:
            env.pop(name, None)
    env.update(
        {
            "VLLM_BASE_URL": endpoint,
            "VLLM_API_KEY": key,
            "MTE_ENABLE_RECOVERY": "1",
            "MTE_DEDUPE_TOOL_CALLS": "1",
            "MTE_TOOL_RESULT_RUN_LLM": "0",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
            "MTE_VLLM_THINKING": "0",
            "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_TOP_P": "0.95",
            "MTE_VLLM_TOP_K": "64",
            "MTE_VLLM_MAX_TOKENS": "8192",
        }
    )
    if normalize_tool_call_indices:
        env["MTE_VLLM_NORMALIZE_TOOL_CALL_INDICES"] = "1"
    return env


def parse_run_dir(log_path: Path) -> Path | None:
    match = re.search(r"^Output directory: (.+)$", log_path.read_text(), re.MULTILINE)
    if not match:
        return None
    path = Path(match.group(1).strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def summarize_run(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"run_dir": None, "complete": False}
    runtime_path = run_dir / "runtime.json"
    transcript_path = run_dir / "transcript.jsonl"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    rows = (
        [json.loads(line) for line in transcript_path.read_text().splitlines() if line]
        if transcript_path.exists()
        else []
    )
    scripted = [
        row
        for row in rows
        if isinstance(row.get("turn"), int) and 0 <= int(row["turn"]) < 30
    ]
    ttfb = [float(row["ttfb_ms"]) for row in scripted if row.get("ttfb_ms") is not None]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def token_total(name: str) -> int:
        return sum(int((row.get("tokens") or {}).get(name) or 0) for row in rows)

    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "runtime_status": runtime.get("status"),
        "runtime_valid": runtime.get("valid"),
        "rows": len(rows),
        "scripted_turns": len(scripted),
        "complete": runtime.get("status") == "completed"
        and runtime.get("valid") is True
        and len(scripted) == 30,
        "ttfat_p50_ms": percentile(ttfb, 0.5),
        "ttfat_p95_ms": percentile(ttfb, 0.95),
        "ttfat_max_ms": max(ttfb) if ttfb else None,
        "prompt_tokens": token_total("prompt_tokens"),
        "completion_tokens": token_total("completion_tokens"),
        "cache_read_input_tokens": token_total("cache_read_input_tokens"),
        "thinking_tokens": token_total("thinking_tokens"),
    }


def run_one(key: str, label: str, index: int) -> dict[str, Any]:
    log_path = LOGS / f"{label}-run{index:02d}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            RUN_COMMAND,
            cwd=ROOT,
            env=clean_environment(
                key,
                DEPLOYMENTS[label]["endpoint"],
                normalize_tool_call_indices=label.startswith("sglang_"),
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    result = summarize_run(parse_run_dir(log_path))
    result.update(
        {
            "arm": label,
            "index": index,
            "exit_code": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(ROOT)),
            "finished_at": utc_now(),
        }
    )
    return result


def write_results(payload: dict[str, Any]) -> None:
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    key = load_key()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "model": MODEL,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "runs": [],
        "resource_states": [],
    }
    if EXISTING_VLLM_RUN.exists():
        existing = summarize_run(EXISTING_VLLM_RUN)
        existing.update({"arm": "vllm_no_mtp", "index": 1, "preexisting": True})
        payload["runs"].append(existing)
    write_results(payload)

    stopping = False

    def stop_handler(signum, _frame):
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGHUP, stop_handler)

    sglang_labels = ("sglang_no_mtp", "sglang_mtp")
    try:
        for label in sglang_labels:
            set_minimum(key, label, 1)
        for label in sglang_labels:
            payload["resource_states"].append(
                {"at": utc_now(), "arm": label, "state": "ACTIVE", "detail": wait_state(key, label, active=True)}
            )
        write_results(payload)
        for index in range(1, 4):
            print(f"[{utc_now()}] matched SGLang pair {index}/3", flush=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    label: executor.submit(run_one, key, label, index)
                    for label in sglang_labels
                }
                pair = [futures[label].result() for label in sglang_labels]
            payload["runs"].extend(pair)
            write_results(payload)
            print(json.dumps(pair, indent=2, sort_keys=True), flush=True)
            if stopping or any(
                item["exit_code"] != 0 or not item["complete"] for item in pair
            ):
                raise RuntimeError("unexpected or incomplete SGLang bakeoff run")
    finally:
        for label in sglang_labels:
            try:
                set_minimum(key, label, 0)
            except Exception as error:
                payload["resource_states"].append(
                    {"at": utc_now(), "arm": label, "state": "TEARDOWN_ERROR", "error": repr(error)}
                )
        for label in sglang_labels:
            try:
                state = wait_state(key, label, active=False)
                payload["resource_states"].append(
                    {"at": utc_now(), "arm": label, "state": "SCALED_TO_ZERO", "detail": state}
                )
            except Exception as error:
                payload["resource_states"].append(
                    {"at": utc_now(), "arm": label, "state": "TEARDOWN_ERROR", "error": repr(error)}
                )
            write_results(payload)

    try:
        set_minimum(key, "vllm_no_mtp", 1)
        payload["resource_states"].append(
            {"at": utc_now(), "arm": "vllm_no_mtp", "state": "ACTIVE", "detail": wait_state(key, "vllm_no_mtp", active=True)}
        )
        write_results(payload)
        for index in (2, 3):
            print(f"[{utc_now()}] vLLM control {index}/3", flush=True)
            item = run_one(key, "vllm_no_mtp", index)
            payload["runs"].append(item)
            write_results(payload)
            print(json.dumps(item, indent=2, sort_keys=True), flush=True)
            if item["exit_code"] != 0 or not item["complete"]:
                raise RuntimeError("unexpected or incomplete vLLM bakeoff run")
    finally:
        try:
            set_minimum(key, "vllm_no_mtp", 0)
            state = wait_state(key, "vllm_no_mtp", active=False)
            payload["resource_states"].append(
                {"at": utc_now(), "arm": "vllm_no_mtp", "state": "SCALED_TO_ZERO", "detail": state}
            )
        except Exception as error:
            payload["resource_states"].append(
                {"at": utc_now(), "arm": "vllm_no_mtp", "state": "TEARDOWN_ERROR", "error": repr(error)}
            )

    payload["finished_at"] = utc_now()
    write_results(payload)
    print(f"[{utc_now()}] bakeoff complete: {RESULTS.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
