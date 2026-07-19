"""Lilac (OpenAI-compatible) LLM service with content-aware TTFB.

Lilac serves open-weights models at api.getlilac.com over the OpenAI Chat
Completions API. The stock ``OpenAILLMService`` stops its TTFB metric on the
*first non-empty ``choices`` chunk* — which, when a model runs with thinking
ON, is the first ``reasoning`` delta, not the first user-visible ``content``
token. For gemma-4-31b that understates TTFT by ~10x (e.g. ~300ms to first
reasoning token vs ~3800ms to first content).

This subclass mirrors ``LoggedCerebrasLLMService``: it stops the (content-aware)
TTFB metric on the first ``content`` / transcript / ``tool_calls`` delta —
reasoning-only chunks do not count — and separately emits ``RawTTFBMetricsData``
on the first chunk of any kind. So every turn records both:

- ``ttfb_ms``     — time to first user-visible (non-thinking) token
- ``raw_ttfb_ms`` — time to first stream chunk of any kind

With thinking OFF (no separate ``reasoning`` stream) the two coincide; with
thinking ON they diverge, exposing the reasoning latency the stock service
would otherwise have hidden.
"""

import json
import time
from contextlib import asynccontextmanager

from loguru import logger
from pipecat.frames.frames import LLMTextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm

from multi_turn_eval.metrics import RawTTFBMetricsData
from multi_turn_eval.services.filler import apply_filler_to_last_user


class LoggedLilacLLMService(OpenAILLMService):
    """Lilac service that measures TTFB to first content/tool, not reasoning,
    and also records raw (first-chunk) TTFB.

    Also injects the MTE_FILLER_DOTS filler suffix onto the final user turn of
    each request (latent-scratchpad probe, arxiv 2607.03502; services/filler.py),
    via the build_chat_completion_params seam — applied to the outgoing request
    only, so the persisted context / transcript stays filler-free. Inherited by
    the BaseTen service.
    """

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        if isinstance(params, dict) and isinstance(params.get("messages"), list):
            params["messages"] = apply_filler_to_last_user(params["messages"])
        return params

    @traced_llm
    async def _process_context(self, context: LLMContext):
        """Stream a completion, stopping content-aware TTFB on first
        content/tool delta and emitting raw TTFB on the first chunk."""
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""
        ttfb_stopped = False
        raw_ttfb_emitted = False

        await self.start_ttfb_metrics()
        # Reference for raw TTFB (first chunk of any kind, incl. reasoning). The
        # pipecat TTFB metric we stop later is content-aware; this is the raw
        # network+prefill+1-token floor.
        raw_t0 = time.monotonic()

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
                    # getattr-guarded: Lilac's usage object may omit the
                    # *_details sub-objects that the OpenAI SDK type defines.
                    ptd = getattr(chunk.usage, "prompt_tokens_details", None)
                    ctd = getattr(chunk.usage, "completion_tokens_details", None)
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cache_read_input_tokens=getattr(ptd, "cached_tokens", None),
                        reasoning_tokens=getattr(ctd, "reasoning_tokens", None),
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.model and self.get_full_model_name() != chunk.model:
                    self.set_full_model_name(chunk.model)

                if chunk.choices is None or len(chunk.choices) == 0:
                    continue

                # Raw TTFB: fires on the first chunk-with-choices, before we
                # peek at delta type. Mirrors what the stock service records.
                if not raw_ttfb_emitted:
                    raw_ms = time.monotonic() - raw_t0
                    await self.push_frame(
                        MetricsFrame(
                            data=[RawTTFBMetricsData(processor=self.name, value=raw_ms)]
                        )
                    )
                    raw_ttfb_emitted = True

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
