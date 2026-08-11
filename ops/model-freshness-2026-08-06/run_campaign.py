#!/usr/bin/env python3
"""Run one complete AIEWF medium-context conversation per retained README row.

The campaign is split into independent lanes so hosted providers and dedicated
inference deployments can run concurrently.  Results are written per lane and
can be resumed safely.  Secrets are read from the repository .env files and are
never written to the result artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from dotenv import dotenv_values
import requests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ROOT = ROOT / "runs" / "aiwf_medium_context"
LOG_ROOT = HERE / "logs"


@dataclass(frozen=True)
class Row:
    key: str
    label: str
    lane: str
    model: str
    service: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    baseten_control: str | None = None
    baseten_concurrency: int = 1


COMMON_ENV = {
    "MTE_ENABLE_RECOVERY": "1",
    "MTE_DEDUPE_TOOL_CALLS": "1",
    "MTE_TOOL_RESULT_RUN_LLM": "0",
    "MTE_TEXT_IDLE_TIMEOUT_SECS": "45",
}


ROWS = (
    # OpenAI. OpenAI Pro models are intentionally absent and rejected by code.
    Row("gpt41", "gpt-4.1", "openai", "gpt-4.1", "openai"),
    Row("gpt51", "gpt-5.1", "openai", "gpt-5.1", "openai"),
    Row(
        "terra_medium", "gpt-5.6-terra (medium)", "openai",
        "gpt-5.6-terra", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "medium"},
    ),
    Row(
        "gpt54_low", "gpt-5.4 (low)", "openai", "gpt-5.4", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "low"},
    ),
    Row(
        "gpt55_none", "gpt-5.5 (none)", "openai", "gpt-5.5", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row(
        "sol_none", "gpt-5.6-sol (none)", "openai", "gpt-5.6-sol", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row(
        "gpt54_none_dots", "gpt-5.4 (none, +96 dots)", "openai",
        "gpt-5.4", "openai",
        env={
            "MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none",
            "MTE_FILLER_DOTS": "96",
            "MTE_FILLER_TOKEN": ".",
            "MTE_FILLER_POSITION": "suffix",
        },
    ),
    Row("gpt4o", "gpt-4o", "openai", "gpt-4o", "openai"),
    Row(
        "terra_none", "gpt-5.6-terra (none)", "openai",
        "gpt-5.6-terra", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row(
        "gpt54mini_medium", "gpt-5.4-mini (medium)", "openai",
        "gpt-5.4-mini", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "medium"},
    ),
    Row(
        "gpt54_none", "gpt-5.4 (none)", "openai", "gpt-5.4", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row("gpt52", "gpt-5.2", "openai", "gpt-5.2", "openai"),
    Row(
        "luna_none", "gpt-5.6-luna (none)", "openai", "gpt-5.6-luna", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row("gpt41mini", "gpt-4.1-mini", "openai", "gpt-4.1-mini", "openai"),
    Row("gpt5mini", "gpt-5-mini", "openai", "gpt-5-mini", "openai"),
    Row(
        "gpt54mini_none", "gpt-5.4-mini (none)", "openai",
        "gpt-5.4-mini", "openai",
        env={"MTE_OPENAI_RESPONSES_REASONING_EFFORT": "none"},
    ),
    Row("gpt4omini", "gpt-4o-mini", "openai", "gpt-4o-mini", "openai"),

    # Other hosted APIs.
    Row("sonnet46", "claude-sonnet-4-6", "hosted", "claude-sonnet-4-6", "anthropic"),
    Row(
        "fable5_low", "claude-fable-5 (low)", "hosted", "claude-fable-5", "anthropic",
        env={
            "MTE_ANTHROPIC_EFFORT": "low",
            "MTE_ANTHROPIC_THINKING_DISPLAY": "summarized",
        },
    ),
    Row(
        "fable5_default", "claude-fable-5 (default)", "hosted",
        "claude-fable-5", "anthropic",
        env={"MTE_ANTHROPIC_THINKING_DISPLAY": "summarized"},
    ),
    Row("haiku45", "claude-haiku-4-5", "hosted", "claude-haiku-4-5", "anthropic"),
    Row(
        "sonnet5", "claude-sonnet-5", "hosted", "claude-sonnet-5", "anthropic",
        env={"MTE_ANTHROPIC_THINKING": "disabled"},
    ),
    Row(
        "glm52_none", "glm-5.2 (none)", "hosted", "zai-org/GLM-5.2", "baseten",
        env={
            "BASETEN_BASE_URL": "https://inference.baseten.co/v1",
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "1.0",
        },
    ),
    Row(
        "kimi26_none", "kimi-k2.6 (thinking off)", "hosted",
        "moonshotai/Kimi-K2.6", "baseten",
        env={
            "BASETEN_BASE_URL": "https://inference.baseten.co/v1",
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "0.6",
        },
    ),
    Row(
        "inkling_none", "inkling (none)", "hosted", "thinkingmachines/inkling", "baseten",
        env={
            "BASETEN_BASE_URL": "https://inference.baseten.co/v1",
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "1.0",
        },
    ),
    Row(
        "inkling_small_none", "inkling-small (none)", "hosted",
        "thinkingmachines/inkling-small", "baseten",
        env={
            "BASETEN_BASE_URL": "https://inference.baseten.co/v1",
            "MTE_BASETEN_REASONING_EFFORT": "none",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "1.0",
        },
    ),
    Row(
        "inkling_small_low", "inkling-small (low)", "hosted",
        "thinkingmachines/inkling-small", "baseten",
        env={
            "BASETEN_BASE_URL": "https://inference.baseten.co/v1",
            "MTE_BASETEN_REASONING_EFFORT": "low",
            "MTE_BASETEN_MAX_TOKENS": "8192",
            "MTE_BASETEN_TEMPERATURE": "1.0",
        },
    ),
    Row(
        "gemini36", "gemini-3.6-flash (minimal)", "hosted",
        "gemini-3.6-flash", "google", args=("--thinking", "minimal"),
    ),
    Row(
        "gemini35", "gemini-3.5-flash (minimal)", "hosted",
        "gemini-3.5-flash", "google", args=("--thinking", "minimal"),
    ),
    Row(
        "gemini25", "gemini-2.5-flash (thinking off)", "hosted",
        "gemini-2.5-flash", "google", args=("--thinking", "disabled"),
    ),
    Row(
        "gemini35lite", "gemini-3.5-flash-lite (minimal)", "hosted",
        "gemini-3.5-flash-lite", "google", args=("--thinking", "minimal"),
    ),
    Row(
        "gemma431", "lilac/gemma-4-31b-it (thinking off)", "hosted",
        "lilac/gemma-4-31b-it", "lilac",
        env={"MTE_LILAC_THINKING": "0", "MTE_TEXT_IDLE_TIMEOUT_SECS": "120"},
    ),
    Row("gptoss_groq", "gpt-oss-120b (groq)", "hosted", "openai/gpt-oss-120b", "groq"),
    Row(
        "laguna", "poolside/laguna-s-2.1 (thinking off)", "hosted",
        "poolside/laguna-s-2.1", "openrouter",
        env={
            "MTE_OPENROUTER_REASONING_OFF": "1",
            "MTE_OPENROUTER_MAX_TOKENS": "8192",
        },
    ),

    # Current self-hosted deployments. Each BaseTen deployment is brought up,
    # used for one run, and returned to scale-to-zero before the next row.
    Row(
        "glm5_thinking", "glm-5 (thinking)", "infra", "zai-org/GLM-5-FP8", "modal",
        env={
            "MODAL_BASE_URL": "https://api.us-west-2.modal.direct/v1",
            "MTE_MODAL_THINKING": "1",
            "MTE_TEXT_IDLE_TIMEOUT_SECS": "180",
        },
    ),
    Row(
        "qwen36_27", "qwen3.6-27b (thinking off)", "infra",
        "Qwen/Qwen3.6-27B", "vllm-openai",
        env={
            "VLLM_BASE_URL": "https://model-w67n482q.api.baseten.co/deployment/wxpnlg5/sync/v1",
            "MTE_VLLM_THINKING": "0", "MTE_VLLM_TEMPERATURE": "0.6",
            "MTE_VLLM_TOP_P": "0.95", "MTE_VLLM_MAX_TOKENS": "8192",
        },
        baseten_control="https://api.baseten.co/v1/models/w67n482q/deployments/wxpnlg5",
    ),
    Row(
        "qwen36_35", "qwen3.6-35b-a3b (thinking off, FP8)", "infra",
        "Qwen/Qwen3.6-35B-A3B-FP8", "vllm-openai",
        env={
            "VLLM_BASE_URL": "https://model-qzkm8mpq.api.baseten.co/deployment/qe20zvr/sync/v1",
            "MTE_VLLM_THINKING": "0", "MTE_VLLM_TEMPERATURE": "0.6",
            "MTE_VLLM_TOP_P": "0.95", "MTE_VLLM_MAX_TOKENS": "8192",
        },
        baseten_control="https://api.baseten.co/v1/models/qzkm8mpq/deployments/qe20zvr",
    ),
    Row(
        "qwen3_8b", "qwen3-8b (thinking off, BaseTen)", "infra",
        "qwen/qwen3-8b", "vllm-openai",
        env={
            "VLLM_BASE_URL": "https://model-wnp6rky3.api.baseten.co/deployment/wgvnndv/sync/v1",
            "MTE_VLLM_THINKING": "0", "MTE_VLLM_TEMPERATURE": "0.7",
            "MTE_VLLM_TOP_P": "0.8", "MTE_VLLM_TOP_K": "20",
            "MTE_VLLM_MAX_TOKENS": "8192",
        },
        baseten_control="https://api.baseten.co/v1/models/wnp6rky3/deployments/wgvnndv",
        baseten_concurrency=16,
    ),
    Row(
        "gemma426", "gemma-4-26b-a4b-it (thinking off)", "infra",
        "google/gemma-4-26B-A4B-it", "vllm-openai",
        env={
            "VLLM_BASE_URL": "https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1",
            "MTE_VLLM_THINKING": "0", "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_TOP_P": "0.95", "MTE_VLLM_TOP_K": "64",
            "MTE_VLLM_MAX_TOKENS": "8192",
        },
        baseten_control="https://api.baseten.co/v1/models/qel1y223/deployments/qz4zpye",
    ),
    Row(
        "gemma431_baseten", "gemma-4-31b-it (thinking off)", "infra",
        "google/gemma-4-31B-it", "vllm-openai",
        env={
            "VLLM_BASE_URL": "https://model-qzk215kq.api.baseten.co/deployment/wgvde5j/sync/v1",
            "MTE_VLLM_THINKING": "0", "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_TOP_P": "0.95", "MTE_VLLM_TOP_K": "64",
            "MTE_VLLM_MAX_TOKENS": "8192",
        },
        baseten_control="https://api.baseten.co/v1/models/qzk215kq/deployments/wgvde5j",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_secrets() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (ROOT / ".env", ROOT.parent / "gb-benchmarks" / ".env"):
        if path.exists():
            values.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return values


def clean_environment(secrets: dict[str, str], row: Row) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in secrets.items():
        env.setdefault(key, value)
    for key in tuple(env):
        if key.startswith("MTE_") or key in {
            "VLLM_BASE_URL", "VLLM_API_KEY", "BASETEN_BASE_URL", "LILAC_BASE_URL",
        }:
            env.pop(key, None)
    env.update(COMMON_ENV)
    env.update(row.env)
    if row.service == "vllm-openai":
        env["VLLM_API_KEY"] = secrets["BASETEN_API_KEY"]
    return env


def control_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}


def set_baseten_minimum(row: Row, key: str, minimum: int) -> None:
    assert row.baseten_control
    response = requests.patch(
        row.baseten_control + "/autoscaling_settings",
        headers=control_headers(key),
        json={
            "min_replica": minimum,
            "max_replica": 1,
            "autoscaling_window": 60,
            "scale_down_delay": 120,
            "concurrency_target": row.baseten_concurrency,
            "target_utilization_percentage": 70,
        },
        timeout=30,
    )
    response.raise_for_status()


def wait_for_baseten(row: Row, key: str, active: bool) -> None:
    assert row.baseten_control
    deadline = time.monotonic() + (30 * 60 if active else 15 * 60)
    last = "UNKNOWN"
    while time.monotonic() < deadline:
        try:
            response = requests.get(row.baseten_control, headers=control_headers(key), timeout=30)
            response.raise_for_status()
            payload = response.json()
            last = str(payload.get("status", "UNKNOWN"))
            replicas = int(payload.get("active_replica_count", 0))
            if active and last == "ACTIVE" and replicas >= 1:
                return
            if not active and last == "SCALED_TO_ZERO" and replicas == 0:
                return
        except requests.RequestException as error:
            last = type(error).__name__
        time.sleep(10)
    raise TimeoutError(f"BaseTen state timeout for {row.key}: last={last}")


def read_results(lane: str) -> list[dict[str, Any]]:
    path = HERE / f"results-{lane}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def write_results(lane: str, results: list[dict[str, Any]]) -> None:
    path = HERE / f"results-{lane}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def new_run_dir(before: set[Path], expected_model: str) -> Path | None:
    after = {path.parent for path in RUN_ROOT.glob("*/runtime.json")}
    created: list[Path] = []
    for path in after - before:
        try:
            runtime = json.loads((path / "runtime.json").read_text())
        except (OSError, ValueError):
            continue
        if runtime.get("model_name") == expected_model:
            created.append(path)
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def summarize_run(run_dir: Path | None, elapsed_seconds: float) -> dict[str, Any]:
    summary: dict[str, Any] = {"run_dir": None, "elapsed_seconds": round(elapsed_seconds, 3)}
    if run_dir is None:
        return summary
    summary["run_dir"] = str(run_dir.relative_to(ROOT))
    runtime_path = run_dir / "runtime.json"
    transcript_path = run_dir / "transcript.jsonl"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    rows = [json.loads(line) for line in transcript_path.read_text().splitlines() if line.strip()] if transcript_path.exists() else []
    scripted = {int(item["turn"]) for item in rows if isinstance(item.get("turn"), int) and 0 <= int(item["turn"]) < 30}
    token_rows = [item for item in rows if isinstance(item.get("tokens"), dict)]

    def token_total(name: str) -> int:
        return sum(int((item.get("tokens") or {}).get(name) or 0) for item in rows)

    summary.update(
        {
            "runtime_status": runtime.get("status"),
            "runtime_valid": runtime.get("valid"),
            "transcript_rows": len(rows),
            "scripted_turns": len(scripted),
            "token_rows": len(token_rows),
            "all_rows_have_tokens": len(token_rows) == len(rows),
            "prompt_tokens": token_total("prompt_tokens"),
            "completion_tokens": token_total("completion_tokens"),
            "cache_read_input_tokens": token_total("cache_read_input_tokens"),
            "cache_creation_input_tokens": token_total("cache_creation_input_tokens"),
            "thinking_tokens": token_total("thinking_tokens"),
            "user_words": sum(len(str(item.get("user_text", "")).split()) for item in rows),
            "assistant_words": sum(len(str(item.get("assistant_text", "")).split()) for item in rows),
        }
    )
    summary["complete"] = (
        runtime.get("status") == "completed"
        and runtime.get("valid") is True
        and len(scripted) == 30
    )
    return summary


def run_row(
    row: Row,
    secrets: dict[str, str],
    max_attempts: int,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = list(prior_attempts or [])
    baseten_key = secrets.get("BASETEN_API_KEY", "")
    try:
        if row.baseten_control:
            print(f"[{utc_now()}] {row.key}: scaling BaseTen deployment up", flush=True)
            set_baseten_minimum(row, baseten_key, 1)
            wait_for_baseten(row, baseten_key, True)
        for _ in range(max_attempts):
            attempt = len(attempts) + 1
            before = {path.parent for path in RUN_ROOT.glob("*/runtime.json")}
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = LOG_ROOT / f"{row.lane}-{row.key}-attempt{attempt}.log"
            cmd = [
                str(ROOT / ".venv" / "bin" / "multi-turn-eval"),
                "run", "aiwf_medium_context", "--model", row.model,
                "--service", row.service, "--pipeline", "text", *row.args,
            ]
            print(f"[{utc_now()}] {row.key}: attempt {attempt} starting", flush=True)
            start = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=ROOT, env=clean_environment(secrets, row), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - start
            item = summarize_run(new_run_dir(before, row.model), elapsed)
            item.update({"attempt": attempt, "exit_code": proc.returncode, "log": str(log_path.relative_to(ROOT))})
            attempts.append(item)
            print(
                f"[{utc_now()}] {row.key}: attempt {attempt} rc={proc.returncode} "
                f"complete={item.get('complete')} turns={item.get('scripted_turns')} "
                f"tokens={item.get('all_rows_have_tokens')}",
                flush=True,
            )
            if item.get("complete"):
                break
    finally:
        if row.baseten_control:
            print(f"[{utc_now()}] {row.key}: returning BaseTen deployment to scale zero", flush=True)
            set_baseten_minimum(row, baseten_key, 0)
            wait_for_baseten(row, baseten_key, False)

    return {
        "key": row.key,
        "label": row.label,
        "lane": row.lane,
        "model": row.model,
        "service": row.service,
        "finished_at": utc_now(),
        "complete": bool(attempts and attempts[-1].get("complete")),
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("openai", "hosted", "infra"), required=True)
    parser.add_argument("--row", action="append", default=[], help="Run only these row keys")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    secrets = load_secrets()
    selected = [row for row in ROWS if row.lane == args.lane and (not args.row or row.key in args.row)]
    if not selected:
        parser.error("no matching rows")

    results = read_results(args.lane)
    successful = {item["key"] for item in results if item.get("complete")}
    for row in selected:
        if row.key in successful:
            print(f"[{utc_now()}] {row.key}: already complete; skipping", flush=True)
            continue
        prior = next((item for item in results if item.get("key") == row.key), None)
        result = run_row(
            row,
            secrets,
            args.max_attempts,
            prior_attempts=(prior or {}).get("attempts", []),
        )
        results = [item for item in results if item.get("key") != row.key]
        results.append(result)
        write_results(args.lane, results)
    return 0 if all(item.get("complete") for item in results if item.get("key") in {row.key for row in selected}) else 1


if __name__ == "__main__":
    sys.exit(main())
