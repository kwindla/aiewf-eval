#!/usr/bin/env python3
"""Capture raw automatic-tool SSE for a Gemma 4 OpenAI-compatible server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

from benchmarks._shared.turns import turns
from benchmarks.aiwf_medium_context.config import BenchmarkConfig


MODEL = "google/gemma-4-31B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path)
    return parser.parse_args()


def load_key(path: Path | None) -> str:
    key = os.environ.get("BASETEN_API_KEY") or os.environ.get("VLLM_API_KEY")
    if not key and path:
        values = dotenv_values(path)
        key = values.get("BASETEN_API_KEY") or values.get("VLLM_API_KEY")
    if not key:
        raise RuntimeError("BASETEN_API_KEY or VLLM_API_KEY is required")
    return str(key)


def openai_tools() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": schema.to_default_dict()}
        for schema in BenchmarkConfig.tools_schema.standard_tools
    ]


def conversation_messages(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows_by_turn = {
        int(row["turn"]): row
        for row in rows
        if isinstance(row.get("turn"), int) and 0 <= int(row["turn"]) <= 10
    }
    if sorted(rows_by_turn) != list(range(11)):
        raise ValueError("transcript must contain exactly the scripted prefix 0..10")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": BenchmarkConfig.system_instruction}
    ]
    for index in range(11):
        row = rows_by_turn[index]
        messages.extend(
            [
                {"role": "user", "content": row["user_text"]},
                {"role": "assistant", "content": row["assistant_text"]},
            ]
        )
    messages.append({"role": "user", "content": turns[11]["input"]})
    return messages


def stream_request(
    url: str, key: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        stream=True,
        timeout=180,
    )
    status = response.status_code
    response.raise_for_status()
    raw_events: list[dict[str, Any]] = []
    first_event: float | None = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        if first_event is None:
            first_event = time.perf_counter() - started
        raw_events.append(json.loads(data))

    content: list[str] = []
    reasoning: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    finishes: list[str] = []
    usage = None
    for event in raw_events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                reasoning.append(thought)
            for call in delta.get("tool_calls") or []:
                part = tool_parts.setdefault(
                    int(call.get("index") or 0),
                    {"id": "", "name": "", "arguments": ""},
                )
                part["id"] += call.get("id") or ""
                function = call.get("function") or {}
                part["name"] += function.get("name") or ""
                part["arguments"] += function.get("arguments") or ""
    return {
        "http_status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "first_event_seconds": first_event,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [tool_parts[index] for index in sorted(tool_parts)],
        "finish_reasons": finishes,
        "usage": usage,
        "raw_events": raw_events,
    }


def main() -> int:
    args = parse_args()
    key = load_key(args.dotenv)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    tools = openai_tools()
    full_messages = conversation_messages(args.transcript)
    simple_messages = [
        {
            "role": "system",
            "content": "You must call the supplied tool whenever the user asks you to submit a suggestion.",
        },
        {
            "role": "user",
            "content": "Submit a session suggestion for Jennifer Smith about OpenTelemetry tracing.",
        },
    ]
    output = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "endpoint": endpoint,
        "model": MODEL,
        "tools": tools,
        "system_sha256": hashlib.sha256(
            BenchmarkConfig.system_instruction.encode()
        ).hexdigest(),
        "full_message_count": len(full_messages),
        "simple_auto": stream_request(endpoint, key, simple_messages, tools),
        "turn11_auto": stream_request(endpoint, key, full_messages, tools),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
