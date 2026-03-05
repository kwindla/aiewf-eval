"""Nemotron service wrapper built on OpenAI-compatible chat completions."""

import asyncio
import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, Optional

from loguru import logger
from openai import NOT_GIVEN, APITimeoutError
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {value!r}; ignoring")
        return None
    if parsed < 0:
        logger.warning(f"Invalid negative integer for {name}: {value!r}; ignoring")
        return None
    return parsed


class _AsyncListIterator:
    """Minimal async iterator wrapper for pseudo-stream chunks."""

    def __init__(self, items: list[Any]):
        self._items = items
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


class NemotronLLMService(OpenAILLMService):
    """OpenAI-compatible Nemotron service with sensible default sampling params.

    Defaults follow NVIDIA recommendations unless explicitly overridden.
    """

    DEFAULT_TEMPERATURE = 0.6
    DEFAULT_TOP_P = 0.95
    NON_STREAM_ENV = "MTE_NEMOTRON_NON_STREAMING"
    THINKING_OFF_ENV = "MTE_NEMOTRON_THINKING_OFF"
    THINKING_BUDGET_ENV = "MTE_NEMOTRON_THINKING_BUDGET"

    def __init__(
        self,
        *,
        model: str,
        params: Optional[OpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        if params is None:
            params = OpenAILLMService.InputParams(
                temperature=self.DEFAULT_TEMPERATURE,
                top_p=self.DEFAULT_TOP_P,
            )
        else:
            if params.temperature is NOT_GIVEN:
                params.temperature = self.DEFAULT_TEMPERATURE
            if params.top_p is NOT_GIVEN:
                params.top_p = self.DEFAULT_TOP_P

        super().__init__(model=model, params=params, **kwargs)
        self._non_streaming = _env_bool(self.NON_STREAM_ENV, False)
        self._thinking_off = _env_bool(self.THINKING_OFF_ENV, False)
        self._thinking_budget = _env_int(self.THINKING_BUDGET_ENV)
        enable_thinking = not self._thinking_off

        extra = self._settings.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        extra_body = extra.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}

        if self._thinking_off:
            # vLLM-style switch used by Nemotron endpoints to suppress thinking output.
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            extra_body["chat_template_kwargs"] = {
                **chat_template_kwargs,
                **({"enable_thinking": False} if not enable_thinking else {}),
            }

        if self._thinking_budget is not None:
            if self._thinking_off:
                logger.warning(
                    f"Both {self.THINKING_OFF_ENV} and {self.THINKING_BUDGET_ENV} are set; "
                    "ignoring thinking budget because thinking is disabled"
                )
            else:
                # vLLM extension supported by budget endpoints.
                vllm_xargs = extra_body.get("vllm_xargs")
                if not isinstance(vllm_xargs, dict):
                    vllm_xargs = {}
                vllm_xargs["thinking_budget"] = self._thinking_budget
                extra_body["vllm_xargs"] = vllm_xargs

        if extra_body:
            extra["extra_body"] = extra_body
            self._settings["extra"] = extra

        logger.info(
            f"Configured {model} with temperature={params.temperature}, "
            f"top_p={params.top_p}, non_streaming={self._non_streaming}, "
            f"thinking_off={self._thinking_off}, "
            f"thinking_budget={self._thinking_budget}"
        )

    async def get_chat_completions(self, params_from_context: OpenAILLMInvocationParams):
        """Get completions; optionally force non-streaming mode for compatibility.

        Some OpenAI-compatible endpoints emit mostly reasoning deltas in streaming mode,
        which can starve text-only turn handling. When NON_STREAM_ENV is enabled, use
        non-streaming completions and adapt them into pseudo stream chunks.
        """
        if not self._non_streaming:
            return await super().get_chat_completions(params_from_context)

        params = self.build_chat_completion_params(params_from_context)
        params["stream"] = False
        params.pop("stream_options", None)

        if self._retry_on_timeout:
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**params),
                    timeout=self._retry_timeout_secs,
                )
            except (APITimeoutError, asyncio.TimeoutError):
                logger.debug(f"{self}: Retrying non-stream completion due to timeout")
                response = await self._client.chat.completions.create(**params)
        else:
            response = await self._client.chat.completions.create(**params)

        return _AsyncListIterator(self._completion_to_pseudo_chunks(response))

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """Process context with content/tool TTFT semantics for Nemotron.

        Unlike the base OpenAI service (which stops TTFB on first non-empty
        `choices` chunk), this variant stops on the first user-visible output
        delta (`content`/transcript) or first tool-call delta. Reasoning-only
        chunks do not count toward TTFT.
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

        chunk_stream = await (
            self._stream_chat_completions_specific_context(context)
            if isinstance(context, OpenAILLMContext)
            else self._stream_chat_completions_universal_context(context)
        )

        async for chunk in chunk_stream:
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

            if not ttfb_stopped and (has_content or has_transcript or has_tool_call):
                await self.stop_ttfb_metrics()
                ttfb_stopped = True

            if has_tool_call:
                tool_call = delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
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
                await self.push_frame(LLMTextFrame(content))
            elif has_transcript:
                await self.push_frame(LLMTextFrame(transcript))

        if function_name and arguments:
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                arguments = json.loads(arguments)
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            await self.run_function_calls(function_calls)

    def _completion_to_pseudo_chunks(self, response: Any) -> list[Any]:
        """Convert non-stream completion response to stream-like chunk objects."""
        model_name = getattr(response, "model", self.model_name)
        chunks: list[Any] = []

        usage = getattr(response, "usage", None)
        if usage is not None:
            chunks.append(
                SimpleNamespace(
                    model=model_name,
                    usage=usage,
                    choices=[],
                )
            )

        choices = getattr(response, "choices", None) or []
        if not choices:
            return chunks

        message = getattr(choices[0], "message", None)
        if message is None:
            return chunks

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for idx, tool_call in enumerate(tool_calls):
                function = getattr(tool_call, "function", None)
                name = getattr(function, "name", "") if function else ""
                arguments = getattr(function, "arguments", "{}") if function else "{}"
                tool_call_id = getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex[:24]}"

                delta = SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            index=idx,
                            id=tool_call_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                    content=None,
                )
                chunks.append(
                    SimpleNamespace(
                        model=model_name,
                        usage=None,
                        choices=[SimpleNamespace(delta=delta)],
                    )
                )
            return chunks

        content = getattr(message, "content", None) or ""
        delta = SimpleNamespace(tool_calls=None, content=content)
        chunks.append(
            SimpleNamespace(
                model=model_name,
                usage=None,
                choices=[SimpleNamespace(delta=delta)],
            )
        )
        return chunks
