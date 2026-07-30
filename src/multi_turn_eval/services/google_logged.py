"""Google LLM service with content-aware TTFB.

The upstream ``GoogleLLMService`` calls ``stop_ttfb_metrics()`` on the very
first stream chunk, regardless of whether that chunk carries a thought, content,
or function-call part. For models that emit reasoning before content (e.g.
Gemma 4 31B on Google AI Studio, which has thinking permanently on), that
understates true user-visible TTFT — the metric reflects time to first
*thought*, not time to first content.

This subclass overrides ``_process_context`` and gates ``stop_ttfb_metrics`` on
the first part that's actually user-visible (non-thought text, function call,
or inline_data). Token accounting, grounding metadata, thought-signature
bookmarks, and frame emission are otherwise byte-for-byte the same as upstream.
Mirrors the same fix already used by ``LoggedCerebrasLLMService``.
"""

import io
import os
import time
import uuid
from typing import Any, AsyncIterator

from google.genai.types import Content, Part
from loguru import logger
from PIL import Image
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    AssistantImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    MetricsFrame,
)
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.utils.tracing.service_decorators import traced_llm

# DeadlineExceeded is imported the same way upstream does.
from google.api_core.exceptions import DeadlineExceeded

from multi_turn_eval.metrics import RawTTFBMetricsData
from multi_turn_eval.services.filler import filler_suffix


_filler_logged = False


def _filler_position() -> str:
    position = os.getenv("MTE_FILLER_POSITION", "suffix").strip().lower()
    return position if position in {"suffix", "prefix", "system"} else "suffix"


def _weave_filler(text: str, filler: str, position: str) -> str:
    if position == "prefix":
        return filler + " " + text.lstrip()
    return text.rstrip() + " " + filler


def _log_filler(filler: str, position: str) -> None:
    global _filler_logged
    if _filler_logged:
        return
    _filler_logged = True
    parts = filler.split()
    logger.info(
        f"MTE_FILLER_DOTS active: {len(parts)} x {parts[0]!r} filler tokens, "
        f"position={position} (history left filler-free)"
    )


def _part_text(part: Any) -> str | None:
    if isinstance(part, Part):
        return part.text if isinstance(part.text, str) else None
    if isinstance(part, dict):
        text = part.get("text")
        return text if isinstance(text, str) else None
    return None


def _content_role(content: Any) -> str | None:
    if isinstance(content, Content):
        return content.role
    if isinstance(content, dict):
        role = content.get("role")
        return role if isinstance(role, str) else None
    return None


def _content_parts(content: Any) -> list[Any] | None:
    if isinstance(content, Content):
        return content.parts
    if isinstance(content, dict):
        parts = content.get("parts")
        return parts if isinstance(parts, list) else None
    return None


def _copy_part_with_text(part: Any, text: str) -> Any:
    if isinstance(part, Part):
        return part.model_copy(update={"text": text})
    copied = dict(part)
    copied["text"] = text
    return copied


def _copy_content_with_parts(content: Any, parts: list[Any]) -> Any:
    if isinstance(content, Content):
        return content.model_copy(update={"parts": parts})
    copied = dict(content)
    copied["parts"] = parts
    return copied


def _apply_filler_google(
    messages: list[Any], system_instruction: str | None
) -> tuple[list[Any], str | None]:
    """Copy and fill only the current outgoing Gemini request.

    Google represents conversation turns as ``Content`` objects containing
    ``Part`` objects. Tool responses also use ``role="user"``, so this walks
    backward to the last *text-bearing* user turn rather than adding text to a
    tool-response-only turn. The caller's list, Content, and Part instances are
    never mutated. When the filler knob is off (or no target exists), the
    original objects are returned unchanged.
    """
    filler = filler_suffix()
    if not filler:
        return messages, system_instruction

    position = _filler_position()
    if position == "system":
        if not isinstance(system_instruction, str):
            return messages, system_instruction
        _log_filler(filler, position)
        return messages, _weave_filler(system_instruction, filler, position)

    for content_idx in range(len(messages) - 1, -1, -1):
        content = messages[content_idx]
        if _content_role(content) != "user":
            continue
        parts = _content_parts(content)
        if not parts:
            continue
        for part_idx in range(len(parts) - 1, -1, -1):
            text = _part_text(parts[part_idx])
            if text is None:
                continue
            copied_parts = list(parts)
            copied_parts[part_idx] = _copy_part_with_text(
                parts[part_idx], _weave_filler(text, filler, position)
            )
            copied_messages = list(messages)
            copied_messages[content_idx] = _copy_content_with_parts(content, copied_parts)
            _log_filler(filler, position)
            return copied_messages, system_instruction

    return messages, system_instruction


class FillerGoogleLLMAdapter(GeminiLLMAdapter):
    """Gemini adapter that fills the ephemeral, API-ready request only."""

    def get_llm_invocation_params(
        self, context: LLMContext, *, system_instruction: str | None = None
    ):
        params = super().get_llm_invocation_params(
            context, system_instruction=system_instruction
        )
        messages, effective_system = _apply_filler_google(
            params["messages"], params["system_instruction"]
        )
        if (
            messages is params["messages"]
            and effective_system is params["system_instruction"]
        ):
            return params
        copied = dict(params)
        copied["messages"] = messages
        copied["system_instruction"] = effective_system
        return copied


class LoggedGoogleLLMService(GoogleLLMService):
    """Google service that measures TTFB to first content/tool/inline-data part,
    not the first thought."""

    adapter_class = FillerGoogleLLMAdapter

    @traced_llm
    async def _process_context(self, context: LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        grounding_metadata = None
        accumulated_text = ""
        ttfb_stopped = False
        raw_ttfb_emitted = False
        # Reference for raw TTFB (first chunk of any kind). The pipecat TTFB
        # metric we stop later is content-aware; this one is the raw floor.
        raw_t0 = time.monotonic()

        try:
            response = await self._stream_content(context)

            function_calls = []
            async for chunk in response:
                # Token accounting — assign (not accumulate); final chunk is authoritative.
                if chunk.usage_metadata:
                    prompt_tokens = chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens = chunk.usage_metadata.candidates_token_count or 0
                    total_tokens = chunk.usage_metadata.total_token_count or 0
                    cache_read_input_tokens = chunk.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens = chunk.usage_metadata.thoughts_token_count or 0

                if not chunk.candidates:
                    continue

                # Raw TTFB: first chunk that actually carries candidates,
                # regardless of whether parts are thoughts or content.
                if not raw_ttfb_emitted:
                    raw_ms = time.monotonic() - raw_t0
                    await self.push_frame(
                        MetricsFrame(
                            data=[RawTTFBMetricsData(processor=self.name, value=raw_ms)]
                        )
                    )
                    raw_ttfb_emitted = True

                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            function_call_id = None
                            if part.text:
                                if part.thought:
                                    # Thought-only — does NOT count toward TTFB.
                                    await self.push_frame(LLMThoughtStartFrame())
                                    await self.push_frame(LLMThoughtTextFrame(part.text))
                                    await self.push_frame(LLMThoughtEndFrame())
                                else:
                                    if not ttfb_stopped:
                                        await self.stop_ttfb_metrics()
                                        ttfb_stopped = True
                                    accumulated_text += part.text
                                    await self._push_llm_text(part.text)
                            elif part.function_call:
                                if not ttfb_stopped:
                                    await self.stop_ttfb_metrics()
                                    ttfb_stopped = True
                                function_call = part.function_call
                                function_call_id = function_call.id or str(uuid.uuid4())
                                logger.debug(
                                    f"Function call: {function_call.name}:{function_call_id}"
                                )
                                function_calls.append(
                                    FunctionCallFromLLM(
                                        context=context,
                                        tool_call_id=function_call_id,
                                        function_name=function_call.name,
                                        arguments=function_call.args or {},
                                    )
                                )
                            elif part.inline_data and part.inline_data.data:
                                if not ttfb_stopped:
                                    await self.stop_ttfb_metrics()
                                    ttfb_stopped = True
                                image = Image.open(io.BytesIO(part.inline_data.data))
                                await self.push_frame(
                                    AssistantImageRawFrame(
                                        image=image.tobytes(),
                                        size=image.size,
                                        format="RGB",
                                        original_data=part.inline_data.data,
                                        original_mime_type=part.inline_data.mime_type,
                                    )
                                )

                            # Thought-signature bookmarks (same as upstream).
                            if part.thought_signature:
                                bookmark = {}
                                if part.function_call:
                                    bookmark["function_call"] = function_call_id
                                elif part.inline_data and part.inline_data.data:
                                    bookmark["inline_data"] = part.inline_data
                                elif part.text is not None:
                                    bookmark["text"] = accumulated_text
                                else:
                                    logger.warning("Thought signature found on unhandled Part type")
                                if bookmark:
                                    await self.push_frame(
                                        LLMMessagesAppendFrame(
                                            [
                                                self.get_llm_adapter().create_llm_specific_message(
                                                    {
                                                        "type": "thought_signature",
                                                        "signature": part.thought_signature,
                                                        "bookmark": bookmark,
                                                    }
                                                )
                                            ]
                                        )
                                    )

                    if (
                        candidate.grounding_metadata
                        and candidate.grounding_metadata.grounding_chunks
                    ):
                        m = candidate.grounding_metadata
                        rendered_content = (
                            m.search_entry_point.rendered_content if m.search_entry_point else None
                        )
                        origins = [
                            {
                                "site_uri": grounding_chunk.web.uri
                                if grounding_chunk.web
                                else None,
                                "site_title": grounding_chunk.web.title
                                if grounding_chunk.web
                                else None,
                                "results": [
                                    {
                                        "text": grounding_support.segment.text
                                        if grounding_support.segment
                                        else "",
                                        "confidence": grounding_support.confidence_scores,
                                    }
                                    for grounding_support in (
                                        m.grounding_supports if m.grounding_supports else []
                                    )
                                    if grounding_support.grounding_chunk_indices
                                    and index in grounding_support.grounding_chunk_indices
                                ],
                            }
                            for index, grounding_chunk in enumerate(
                                m.grounding_chunks if m.grounding_chunks else []
                            )
                        ]
                        grounding_metadata = {
                            "rendered_content": rendered_content,
                            "origins": origins,
                        }

            await self.run_function_calls(function_calls)
        except DeadlineExceeded:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            if grounding_metadata and isinstance(grounding_metadata, dict):
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=accumulated_text,
                    origins=grounding_metadata["origins"],
                    rendered_content=grounding_metadata["rendered_content"],
                )
                await self.push_frame(llm_search_frame)

            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())
