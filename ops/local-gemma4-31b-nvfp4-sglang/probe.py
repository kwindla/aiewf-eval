#!/usr/bin/env python3
"""Protocol, tool-call, prefix-cache, and latency smoke for the local server."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "probe-results.json"
BASE_URL = "http://127.0.0.1:30000/v1"
MODEL = "google/gemma-4-31B-it"


def stream_request(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            **payload,
        },
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    first_sse_ms = None
    ttft_ms = None
    content: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        value = raw_line.removeprefix("data:").strip()
        if value == "[DONE]":
            break
        elapsed_ms = (time.perf_counter() - started) * 1000
        if first_sse_ms is None:
            first_sse_ms = elapsed_ms
        event = json.loads(value)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            visible = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
            )
            if visible and ttft_ms is None:
                ttft_ms = elapsed_ms
            if delta.get("content"):
                content.append(delta["content"])
            tool_calls.extend(delta.get("tool_calls") or [])
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "first_sse_ms": first_sse_ms,
        "ttft_ms": ttft_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "content": "".join(content),
        "tool_calls": tool_calls,
        "usage": usage,
        "cached_tokens": prompt_details.get("cached_tokens"),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    health = requests.get(BASE_URL.removesuffix("/v1") + "/health", timeout=10)
    health.raise_for_status()
    models = requests.get(BASE_URL + "/models", timeout=10)
    models.raise_for_status()

    # Excluded engine/CUDA-graph warmup.
    warmup = stream_request(
        {
            "messages": [
                {"role": "system", "content": "Reply with exactly OK."},
                {"role": "user", "content": "Reply now."},
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
    )

    tiny = []
    for index in range(10):
        tiny.append(
            stream_request(
                {
                    "messages": [
                        {"role": "system", "content": "Reply with exactly OK."},
                        {"role": "user", "content": f"Reply now. Probe {index:02d}."},
                    ],
                    "temperature": 0,
                    "max_tokens": 8,
                }
            )
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_session_suggestion",
                "description": "Submit a suggested session.",
                "parameters": {
                    "type": "object",
                    "properties": {"session_name": {"type": "string"}},
                    "required": ["session_name"],
                },
            },
        }
    ]
    forced_tool = stream_request(
        {
            "messages": [
                {"role": "system", "content": "Use the available tool as requested."},
                {"role": "user", "content": "Submit the session named Voice Agents."},
            ],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "submit_session_suggestion"}},
            "temperature": 0,
            "max_tokens": 128,
        }
    )

    knowledge = (ROOT / "benchmarks/aiwf_medium_context/data/knowledge_base.txt").read_text()
    long_system = "Use this reference silently. Reply with exactly OK.\n\n" + knowledge
    cold_long = stream_request(
        {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Reply now. Cold prime."},
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
    )
    warm_long = stream_request(
        {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Reply now. Warm continuation."},
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
    )

    ttft = [float(row["ttft_ms"]) for row in tiny if row["ttft_ms"] is not None]
    result = {
        "model": MODEL,
        "checkpoint": "RedHatAI/gemma-4-31B-it-NVFP4",
        "checkpoint_revision": "edafdf3dcaef23ff76f75b91edd6a4a975a399cf",
        "warmup": warmup,
        "tiny": tiny,
        "tiny_summary": {
            "n": len(ttft),
            "ttft_p50_ms": statistics.median(ttft),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "ttft_min_ms": min(ttft),
            "ttft_max_ms": max(ttft),
        },
        "forced_tool": forced_tool,
        "cold_long": cold_long,
        "warm_long": warm_long,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["tiny_summary"], indent=2))
    print(
        json.dumps(
            {
                "forced_tool_calls": forced_tool["tool_calls"],
                "cold_long_ttft_ms": cold_long["ttft_ms"],
                "warm_long_ttft_ms": warm_long["ttft_ms"],
                "warm_long_cached_tokens": warm_long["cached_tokens"],
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
