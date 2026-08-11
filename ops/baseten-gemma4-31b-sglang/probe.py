#!/usr/bin/env python3
"""Protocol, cache, and MTP probe for the Gemma 4 SGLang bakeoff."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values
from openai import OpenAI


MODEL = "google/gemma-4-31B-it"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path)
    return parser.parse_args()


def load_api_key(dotenv_path: Path | None) -> str:
    key = os.environ.get("BASETEN_API_KEY") or os.environ.get("VLLM_API_KEY")
    if not key and dotenv_path:
        values = dotenv_values(dotenv_path)
        key = values.get("BASETEN_API_KEY") or values.get("VLLM_API_KEY")
    if not key:
        raise RuntimeError("BASETEN_API_KEY or VLLM_API_KEY is required")
    return str(key)


def as_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def stream_chat(client: OpenAI, **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    first_chunk_seconds: float | None = None
    first_visible_seconds: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    finish_reasons: list[str] = []
    usage: Any = None
    response_model: str | None = None
    chunks = 0

    stream = client.chat.completions.create(
        model=MODEL,
        stream=True,
        stream_options={"include_usage": True},
        temperature=kwargs.pop("temperature", 1.0),
        top_p=kwargs.pop("top_p", 0.95),
        max_tokens=kwargs.pop("max_tokens", 256),
        extra_body={
            "top_k": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "return_cached_tokens_details": True,
            **kwargs.pop("extra_body", {}),
        },
        **kwargs,
    )
    for chunk in stream:
        chunks += 1
        elapsed = time.perf_counter() - started
        if first_chunk_seconds is None:
            first_chunk_seconds = elapsed
        response_model = getattr(chunk, "model", None) or response_model
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", ()) or ():
            if choice.finish_reason:
                finish_reasons.append(str(choice.finish_reason))
            delta = choice.delta
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if reasoning:
                reasoning_parts.append(str(reasoning))
            if delta.content:
                if first_visible_seconds is None:
                    first_visible_seconds = elapsed
                content_parts.append(delta.content)
            for tool_call in delta.tool_calls or ():
                if first_visible_seconds is None:
                    first_visible_seconds = elapsed
                part = tool_parts.setdefault(
                    tool_call.index, {"id": "", "name": "", "arguments": ""}
                )
                if tool_call.id:
                    part["id"] += tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        part["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        part["arguments"] += tool_call.function.arguments

    elapsed = time.perf_counter() - started
    tools = [tool_parts[index] for index in sorted(tool_parts)]
    return {
        "elapsed_seconds": elapsed,
        "first_chunk_seconds": first_chunk_seconds,
        "first_visible_seconds": first_visible_seconds,
        "chunks": chunks,
        "response_model": response_model,
        "finish_reasons": finish_reasons,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tools,
        "usage": as_dict(usage),
    }


def metrics_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/metrics"


def selected_metrics(text: str) -> list[str]:
    needles = (
        "sglang:cache_hit_rate",
        "sglang:cached_tokens_total",
        "sglang:spec_accept_length",
        "sglang:spec_accept_rate",
    )
    return [
        line
        for line in text.splitlines()
        if not line.startswith("#") and any(needle in line for needle in needles)
    ]


def main() -> int:
    args = parse_args()
    key = load_api_key(args.dotenv)
    base_url = args.base_url.rstrip("/")
    client = OpenAI(base_url=base_url, api_key=key, timeout=180)
    results: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "base_url": base_url,
        "model": MODEL,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    results["models"] = as_dict(client.models.list())
    results["plain"] = stream_chat(
        client,
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Reply with exactly: protocol pass"},
        ],
        max_tokens=32,
    )
    results["forced_tool"] = stream_chat(
        client,
        messages=[
            {"role": "system", "content": "Use the supplied tools when asked."},
            {
                "role": "user",
                "content": "Get the weather for Paris in celsius using the tool.",
            },
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=128,
    )
    forced_calls = results["forced_tool"]["tool_calls"]
    if not forced_calls:
        raise RuntimeError("forced tool probe returned no tool call")
    first_call = forced_calls[0]
    json.loads(first_call["arguments"])
    results["tool_continuation"] = stream_chat(
        client,
        messages=[
            {"role": "system", "content": "Use the supplied tools when asked."},
            {
                "role": "user",
                "content": "Get the weather for Paris in celsius using the tool.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": first_call["id"],
                        "type": "function",
                        "function": {
                            "name": first_call["name"],
                            "arguments": first_call["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_call["id"],
                "content": json.dumps(
                    {"city": "Paris", "temperature": 18, "unit": "celsius"}
                ),
            },
        ],
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    knowledge = Path(
        "benchmarks/aiwf_medium_context/data/knowledge_base.txt"
    ).read_text(encoding="utf-8")
    cache_messages = [
        {
            "role": "system",
            "content": (
                "Use the reference below. Answer with only one word.\n\n" + knowledge
            ),
        },
        {"role": "user", "content": "Is this reference available?"},
    ]
    results["cache_cold"] = stream_chat(
        client, messages=cache_messages, max_tokens=16, temperature=0
    )
    results["cache_warm"] = stream_chat(
        client, messages=cache_messages, max_tokens=16, temperature=0
    )

    # SGLang publishes decode/cache gauges on a periodic scheduler interval,
    # not synchronously with the completed request. Give that interval time to
    # expose the MTP acceptance values generated by this probe.
    time.sleep(12)
    response = requests.get(
        metrics_url(base_url),
        headers={"Authorization": f"Api-Key {key}"},
        timeout=30,
    )
    results["metrics"] = {
        "url": metrics_url(base_url),
        "status_code": response.status_code,
        "selected": selected_metrics(response.text) if response.ok else [],
        "error_preview": None if response.ok else response.text[:500],
    }
    results["finished_at"] = datetime.now(timezone.utc).isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
