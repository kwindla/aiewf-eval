"""Anthropic service wrapper with optional exact payload logging."""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger
from pipecat.services.anthropic.llm import AnthropicLLMService


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
    return ""


class LoggedAnthropicLLMService(AnthropicLLMService):
    """Anthropic service with optional logging of exact request payloads."""

    async def _create_message_stream(self, api_call, params):  # type: ignore[override]
        if _env_bool("MTE_LOG_ANTHROPIC_PAYLOADS", False):
            safe_params = _json_safe(params)
            messages = safe_params.get("messages", [])
            last_user = (
                _extract_last_user_text(messages) if isinstance(messages, list) else ""
            )
            logger.debug(
                f"{self}: Anthropic exact request payload "
                f"(message_count={len(messages) if isinstance(messages, list) else 'n/a'}, "
                f"last_user_text={last_user!r}) | "
                f"{json.dumps(safe_params, ensure_ascii=False)}"
            )
        return await super()._create_message_stream(api_call, params)

