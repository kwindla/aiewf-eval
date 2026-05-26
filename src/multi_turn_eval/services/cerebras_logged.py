"""Cerebras LLM service with content-aware TTFB.

The base OpenAI service (which ``CerebrasLLMService`` inherits unchanged) stops
the TTFB metric on the *first non-empty ``choices`` chunk* — which, for a
reasoning model like Kimi K2.6 in Thinking mode, is the first ``reasoning``
token, not the first user-visible ``content`` token. That understates TTFT by
roughly 3-4x (e.g. ~300ms to first reasoning token vs ~1000ms to first content).

This subclass overrides ``_process_context`` to stop TTFB only on the first
user-visible delta (``content`` / audio transcript) or first ``tool_calls``
delta — reasoning-only chunks do not count. The rest of the method mirrors the
upstream base implementation (token accounting, tool-call coalescing, stream
cleanup), so behavior is otherwise identical. Mirrors the same fix already used
by ``NemotronOmniAudioLLMService``.
"""

from contextlib import asynccontextmanager

from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.utils.tracing.service_decorators import traced_llm

import json


class LoggedCerebrasLLMService(CerebrasLLMService):
    """Cerebras service that measures TTFB to first content/tool, not reasoning."""

    @traced_llm
    async def _process_context(self, context: LLMContext):
        """Stream a completion, stopping TTFB on first content/tool delta.

        Reasoning-only chunks (``delta.reasoning``) are ignored for TTFB so the
        metric reflects true user-visible latency in Thinking mode.
        """
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""
        ttfb_stopped = False

        await self.start_ttfb_metrics()

        chunk_stream = await self.get_chat_completions(context)

        # Mirror upstream's defensive stream/iterator cleanup (see base_llm.py).
        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        async with _closing(chunk_stream) as chunk_iter:
            async for chunk in chunk_iter:
                if chunk.usage:
                    cached_tokens = (
                        chunk.usage.prompt_tokens_details.cached_tokens
                        if chunk.usage.prompt_tokens_details
                        else None
                    )
                    reasoning_tokens = (
                        chunk.usage.completion_tokens_details.reasoning_tokens
                        if chunk.usage.completion_tokens_details
                        else None
                    )
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cache_read_input_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.model and self.get_full_model_name() != chunk.model:
                    self.set_full_model_name(chunk.model)

                if chunk.choices is None or len(chunk.choices) == 0:
                    continue

                if not chunk.choices[0].delta:
                    continue

                delta = chunk.choices[0].delta

                content = getattr(delta, "content", None)
                audio = getattr(delta, "audio", None)
                transcript = (
                    audio.get("transcript")
                    if audio is not None and hasattr(audio, "get")
                    else None
                )
                has_content = bool(content)
                has_transcript = bool(transcript)
                has_tool_call = bool(getattr(delta, "tool_calls", None))

                # Content-aware TTFB: ignore reasoning-only chunks.
                if not ttfb_stopped and (has_content or has_transcript or has_tool_call):
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                if has_tool_call:
                    tool_call = delta.tool_calls[0]
                    if tool_call.index != func_idx:
                        functions_list.append(function_name)
                        arguments_list.append(arguments or "{}")
                        tool_id_list.append(tool_call_id)
                        function_name = ""
                        arguments = ""
                        tool_call_id = ""
                        func_idx += 1
                    if tool_call.function and tool_call.function.name:
                        function_name += tool_call.function.name
                        tool_call_id = tool_call.id
                    if tool_call.function and tool_call.function.arguments:
                        arguments += tool_call.function.arguments
                elif has_content:
                    await self._push_llm_text(content)
                elif has_transcript:
                    await self.push_frame(LLMTextFrame(transcript))

        if function_name:
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []
            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Failed to parse function call arguments: {arguments}")
                    continue
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)
