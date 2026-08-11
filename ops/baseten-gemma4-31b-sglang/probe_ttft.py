#!/usr/bin/env python3
"""Measure tiny-prompt and cached-long-prefix TTFT across BaseTen stacks."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "ttft-probe-20260806.json"
MODEL = "google/gemma-4-31B-it"
N = 30
DEPLOYMENTS = {
    "sglang_mtp": {
        "endpoint": "https://model-q951m16w.api.baseten.co/deployment/q862ez8/sync/v1",
        "control": "https://api.baseten.co/v1/models/q951m16w/deployments/q862ez8",
    },
    "vllm_no_mtp": {
        "endpoint": "https://model-qzk215kq.api.baseten.co/deployment/wgvde5j/sync/v1",
        "control": "https://api.baseten.co/v1/models/qzk215kq/deployments/wgvde5j",
    },
}


def load_key() -> str:
    for path in (ROOT / ".env", ROOT.parent / "gb-benchmarks/.env"):
        if path.is_file():
            key = dotenv_values(path).get("BASETEN_API_KEY")
            if key:
                return str(key)
    raise RuntimeError("BASETEN_API_KEY not found")


def control_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}


def set_minimum(key: str, label: str, minimum: int) -> None:
    response = requests.patch(
        DEPLOYMENTS[label]["control"] + "/autoscaling_settings",
        headers=control_headers(key),
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
            DEPLOYMENTS[label]["control"], headers=control_headers(key), timeout=30
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
    target = "ACTIVE" if active else "SCALED_TO_ZERO"
    raise TimeoutError(f"{label} did not reach {target}: {last}")


def request_once(
    key: str, label: str, messages: list[dict[str, str]], *, sequence: int
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = requests.post(
        DEPLOYMENTS[label]["endpoint"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    first_event_ms = None
    first_token_ms = None
    usage = None
    content = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        elapsed_ms = (time.perf_counter() - started) * 1000
        if first_event_ms is None:
            first_event_ms = elapsed_ms
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            token = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
            )
            if token and first_token_ms is None:
                first_token_ms = elapsed_ms
            if delta.get("content"):
                content.append(delta["content"])
    elapsed_ms = (time.perf_counter() - started) * 1000
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "sequence": sequence,
        "first_sse_ms": first_event_ms,
        "ttft_ms": first_token_ms,
        "elapsed_ms": elapsed_ms,
        "content": "".join(content),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def selected_metrics(key: str, label: str) -> list[str]:
    url = DEPLOYMENTS[label]["endpoint"].rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {key}"}, timeout=30
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return [f"metrics_error {error}"]
    needles = (
        "time_to_first_token",
        "request_prefill_time",
        "request_decode_time",
        "cache_hit_rate",
        "cached_tokens",
        "spec_accept",
    )
    return [
        line
        for line in response.text.splitlines()
        if not line.startswith("#") and any(needle in line for needle in needles)
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    first_sse = [
        float(row["first_sse_ms"])
        for row in rows
        if row.get("first_sse_ms") is not None
    ]
    cached = [int(row["cached_tokens"] or 0) for row in rows]
    return {
        "n": len(rows),
        "ttft_p50_ms": percentile(ttft, 0.5),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_min_ms": min(ttft),
        "ttft_max_ms": max(ttft),
        "first_sse_p50_ms": percentile(first_sse, 0.5),
        "cached_tokens_p50": percentile([float(value) for value in cached], 0.5),
    }


def run_stack(key: str, label: str) -> dict[str, Any]:
    print(f"waking {label}", flush=True)
    set_minimum(key, label, 1)
    wait_state(key, label, active=True)
    try:
        # One excluded request ensures the route, engine, and CUDA graphs are warm.
        warmup = request_once(
            key,
            label,
            [
                {"role": "system", "content": "Reply with exactly OK."},
                {"role": "user", "content": "Reply now."},
            ],
            sequence=-1,
        )
        metrics_before = selected_metrics(key, label)
        tiny = []
        for index in range(N):
            tiny.append(
                request_once(
                    key,
                    label,
                    [
                        {"role": "system", "content": "Reply with exactly OK."},
                        {
                            "role": "user",
                            "content": f"Reply now. Unique request {index:02d}.",
                        },
                    ],
                    sequence=index,
                )
            )
        knowledge = (ROOT / "benchmarks/aiwf_medium_context/data/knowledge_base.txt").read_text()
        long_prefix = (
            "Use this reference silently, then reply with exactly OK.\n\n" + knowledge
        )
        cold_long = request_once(
            key,
            label,
            [
                {"role": "system", "content": long_prefix},
                {"role": "user", "content": "Reply now. Cold prime."},
            ],
            sequence=-1,
        )
        warm_long = []
        for index in range(N):
            warm_long.append(
                request_once(
                    key,
                    label,
                    [
                        {"role": "system", "content": long_prefix},
                        {
                            "role": "user",
                            "content": f"Reply now. Unique request {index:02d}.",
                        },
                    ],
                    sequence=index,
                )
            )
        return {
            "warmup": warmup,
            "tiny_unique": tiny,
            "tiny_unique_summary": summarize(tiny),
            "long_prefix_cold": cold_long,
            "long_prefix_warm": warm_long,
            "long_prefix_warm_summary": summarize(warm_long),
            "metrics_before": metrics_before,
            "metrics_after": selected_metrics(key, label),
        }
    finally:
        print(f"scaling zero {label}", flush=True)
        set_minimum(key, label, 0)
        wait_state(key, label, active=False)


def main() -> int:
    key = load_key()
    output: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "requests_per_phase": N,
        "stacks": {},
    }
    for label in DEPLOYMENTS:
        output["stacks"][label] = run_stack(key, label)
        OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output["stacks"][label], indent=2)[:4000], flush=True)
    output["completed_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"complete: {OUTPUT.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
