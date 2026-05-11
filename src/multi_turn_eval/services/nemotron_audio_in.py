"""Nemotron audio-input service using OpenAI-compatible chat completions."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class ConversationCacheMissError(RuntimeError):
    """Raised when the remote conversation cache cannot attach to a suffix."""


class NemotronAudioInLLMService(LLMService):
    """Audio-input/text-output LLM service for Nemotron Omni on vLLM."""

    def __init__(
        self,
        *,
        model: str = "nemotron_3_nano_omni",
        api_key: str | None = None,
        base_url: str = "http://192.168.7.228:8000/v1",
        temperature: float | None = 0.2,
        max_tokens: int | None = 1024,
        top_p: float | None = None,
        top_k: int | None = 1,
        chat_template_kwargs: dict[str, Any] | None = None,
        request_timeout_secs: float = 180.0,
        conversation_cache_enabled: bool = False,
        suffix_only_conversation: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_completions_url = f"{self._base_url}/chat/completions"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._chat_template_kwargs = (
            copy.deepcopy(chat_template_kwargs)
            if chat_template_kwargs is not None
            else {"enable_thinking": False}
        )
        self._request_timeout_secs = float(request_timeout_secs)

        self._conversation_cache_enabled = bool(conversation_cache_enabled)
        self._conversation_id: str | None = None
        self._suffix_only_conversation = bool(suffix_only_conversation)
        self._conversation_cache_committed = False

        self._session: aiohttp.ClientSession | None = None
        self._generation_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Return whether the service emits Pipecat metrics frames."""
        return True

    async def run_inference(self, context: LLMContext) -> str | None:
        """Out-of-band inference is not supported by this pipeline service."""
        raise NotImplementedError(f"run_inference() not supported by {self.__class__.__name__}")

    async def start(self, frame: StartFrame):
        """Create the HTTP client when the pipeline starts."""
        await super().start(frame)
        await self._ensure_session()

    async def stop(self, frame: EndFrame):
        """Cancel any in-flight generation and close HTTP resources."""
        await super().stop(frame)
        await self._cancel_generation_task()
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel any in-flight generation and close HTTP resources."""
        await super().cancel(frame)
        await self._cancel_generation_task()
        await self._close_session()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle context-triggered inference and pass through unrelated frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context_frame(frame.context)
        elif isinstance(frame, LLMRunFrame):
            logger.debug(f"{self}: ignoring {frame.name}; LLMContextFrame triggers inference")
        elif isinstance(frame, InterruptionFrame):
            await self._cancel_generation_task()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_context_frame(self, context: LLMContext):
        context_messages = copy.deepcopy(
            context.get_messages(llm_specific_filter=self.__class__.__name__)
        )
        converted_messages: list[dict[str, Any]] = []
        for message in context_messages:
            converted = self._convert_context_message(message)
            if converted is not None:
                converted_messages.append(converted)

        if not converted_messages:
            logger.debug(f"{self}: ignoring empty LLM context")
            return

        await self._cancel_generation_task()
        payload = self._build_payload_from_messages(converted_messages)
        self._generation_task = self.create_task(
            self._run_completion_payload(payload),
            name="nemotron_audio_in_completion",
        )

    async def _cancel_generation_task(self):
        if self._generation_task:
            await self.cancel_task(self._generation_task)
            self._generation_task = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout_secs)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _build_payload_from_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": copy.deepcopy(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._top_p is not None:
            payload["top_p"] = self._top_p
        if self._top_k is not None:
            payload["top_k"] = self._top_k
        if self._chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = copy.deepcopy(self._chat_template_kwargs)

        return payload

    def _convert_context_message(self, message: Any) -> dict[str, Any] | None:
        if isinstance(message, LLMSpecificMessage):
            logger.debug(f"{self}: skipping LLM-specific context message for {message.llm}")
            return None
        if not isinstance(message, dict):
            logger.debug(f"{self}: skipping unsupported context message: {message!r}")
            return None

        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            logger.debug(f"{self}: skipping context message with unsupported role {role!r}")
            return None

        converted: dict[str, Any] = {"role": role}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                converted[key] = copy.deepcopy(message[key])

        content = message.get("content")
        if isinstance(content, str) or content is None:
            converted["content"] = content
            return converted

        if not isinstance(content, list):
            logger.debug(f"{self}: skipping unsupported content in context message: {content!r}")
            return None

        converted_content: list[dict[str, Any]] = []
        for item in content:
            converted_item = self._convert_context_content_part(item)
            if converted_item:
                converted_content.append(converted_item)

        if not converted_content:
            return None

        converted["content"] = converted_content
        return converted

    def _convert_context_content_part(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")
        if item_type == "text":
            return {"type": "text", "text": item.get("text", "")}
        if item_type == "audio_url":
            return copy.deepcopy(item)
        if item_type == "input_audio":
            input_audio = item.get("input_audio") or {}
            data = input_audio.get("data") or item.get("audio")
            if not data:
                return None
            audio_format = input_audio.get("format") or item.get("format") or "wav"
            url = (
                str(data)
                if str(data).startswith("data:")
                else f"data:audio/{audio_format};base64,{data}"
            )
            return {"type": "audio_url", "audio_url": {"url": url}}
        if item_type == "image_url":
            return copy.deepcopy(item)
        if "text" in item:
            return {"type": "text", "text": item["text"]}

        logger.debug(f"{self}: skipping unsupported context content part: {item!r}")
        return None

    async def _run_completion_payload(self, payload: dict[str, Any]):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            headers = {"Accept": "text/event-stream"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            session = await self._ensure_session()
            await self._stream_completion(session, payload, headers)
        except asyncio.CancelledError:
            logger.debug(f"{self}: completion cancelled")
            raise
        except Exception as exc:
            logger.error(f"{self}: completion failed: {exc}")
            await self.push_frame(ErrorFrame(error=str(exc)))
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            if self._generation_task is asyncio.current_task():
                self._generation_task = None

    async def _stream_completion(
        self,
        session: aiohttp.ClientSession,
        payload: dict[str, Any],
        headers: dict[str, str],
    ):
        ttfb_stopped = False
        async with session.post(
            self._chat_completions_url,
            json=payload,
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"vLLM request failed with {response.status}: {error_text}")

            async for event in self._iter_sse_events(response):
                if event == "[DONE]":
                    break

                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    logger.debug(f"{self}: skipping malformed SSE event: {event!r}")
                    continue

                if not isinstance(chunk, dict):
                    logger.debug(f"{self}: skipping non-object SSE event: {event!r}")
                    continue

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    await self.start_llm_usage_metrics(
                        LLMTokenUsage(
                            prompt_tokens=usage.get("prompt_tokens") or 0,
                            completion_tokens=usage.get("completion_tokens") or 0,
                            total_tokens=usage.get("total_tokens") or 0,
                        )
                    )

                choices = chunk.get("choices")
                if not isinstance(choices, list):
                    continue

                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue

                    tool_call_deltas = delta.get("tool_calls") or []
                    if tool_call_deltas and not ttfb_stopped:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True

                    text = delta.get("content") or ""
                    if not text:
                        continue
                    if not ttfb_stopped:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                    await self._push_llm_text(text)

    async def _iter_sse_events(self, response: aiohttp.ClientResponse):
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                yield line[len("data:") :].strip()

        line = buffer.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()
