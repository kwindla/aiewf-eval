"""OpenAI Responses API service adapter for text-mode benchmark runs.

This service preserves Pipecat frame semantics while using /v1/responses
under the hood, so existing text pipelines and aggregators continue to work.
"""

import json
from copy import deepcopy
import time
from typing import Any, Dict, List

from loguru import logger
from openai import NOT_GIVEN
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm

from multi_turn_eval.metrics import RawTTFBMetricsData
from multi_turn_eval.services.filler import apply_filler_to_last_user, filler_suffix


class OpenAIResponsesLLMService(OpenAILLMService):
    """OpenAI service implementation backed by the Responses API."""

    def __init__(self, *, stateful_responses_enabled: bool = False, **kwargs: Any):
        self._stateful_responses_enabled = bool(stateful_responses_enabled)
        self._previous_response_id: str | None = None
        self._external_context: list[dict[str, Any]] = []
        self._native_history: list[dict[str, Any]] = []
        self._pending_history: tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool] | None = None
        self._previous_had_filler = False
        self._pending_context_messages: list[Any] | None = None
        super().__init__(**kwargs)

    def _setting(self, key: str, default: Any = NOT_GIVEN) -> Any:
        settings = getattr(self, "_settings", {})
        if isinstance(settings, dict):
            return settings.get(key, default)
        return getattr(settings, key, default)

    def _context_to_openai_params(
        self, context: LLMContext
    ) -> OpenAILLMInvocationParams:
        if isinstance(context, LLMContext):
            adapter = self.get_llm_adapter()
            # Mirror upstream OpenAI base_llm: both kwargs are required on the
            # pipecat 1.1.0 adapter.
            return adapter.get_llm_invocation_params(
                context,
                system_instruction=self._settings.system_instruction,
                convert_developer_to_user=not self.supports_developer_role,
            )

        return OpenAILLMInvocationParams(
            messages=context.messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
        )

    @staticmethod
    def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _is_not_given(value: Any) -> bool:
        return value is NOT_GIVEN

    @staticmethod
    def _to_json_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _message_to_responses_items(self, message: Any) -> List[Dict[str, Any]]:
        # Accept an explicitly seeded native Responses history without flattening.
        if self._get_attr(message, "type") in {
            "message", "reasoning", "function_call", "function_call_output"
        }:
            return [self._native_item(message)]
        role = self._get_attr(message, "role")
        content = self._get_attr(message, "content")
        tool_calls = self._get_attr(message, "tool_calls")
        items: List[Dict[str, Any]] = []

        if role == "tool":
            tool_call_id = self._get_attr(message, "tool_call_id")
            if tool_call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": self._to_json_text(content),
                    }
                )
            return items

        if role == "assistant" and tool_calls:
            for tool_call in tool_calls:
                function = self._get_attr(tool_call, "function") or {}
                function_name = self._get_attr(function, "name")
                arguments = self._get_attr(function, "arguments", "{}")
                call_id = self._get_attr(tool_call, "id")
                if function_name and call_id:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": function_name,
                            "arguments": arguments if isinstance(arguments, str) else "{}",
                        }
                    )

        # Preserve text content for standard conversation state.
        if content is None:
            return items

        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        text_parts.append(part)
                    continue
                part_type = self._get_attr(part, "type")
                if part_type in {"text", "output_text", "input_text"}:
                    text = self._get_attr(part, "text")
                    if text:
                        text_parts.append(text)
            content_text = "\n".join(text_parts).strip()
        else:
            content_text = str(content)

        if role in {"system", "developer", "user", "assistant"} and content_text:
            content_type = "output_text" if role == "assistant" else "input_text"
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": content_text}],
                }
            )

        return items

    def _messages_to_responses_input(self, messages: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for message in messages:
            items.extend(self._message_to_responses_items(message))
        return items

    def _tools_to_responses_tools(self, tools: Any) -> Any:
        if self._is_not_given(tools):
            return NOT_GIVEN

        # Use the existing adapter conversion first (Chat Completions shape),
        # then normalize to Responses API shape.
        adapter = self.get_llm_adapter()
        chat_tools = adapter.from_standard_tools(tools)
        if self._is_not_given(chat_tools):
            return NOT_GIVEN

        response_tools: List[Dict[str, Any]] = []
        for tool in chat_tools:
            if self._get_attr(tool, "type") != "function":
                continue
            fn = self._get_attr(tool, "function") or {}
            name = self._get_attr(fn, "name")
            if not name:
                continue
            resp_tool: Dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": self._get_attr(fn, "parameters"),
                "strict": self._get_attr(fn, "strict"),
            }
            description = self._get_attr(fn, "description")
            if description:
                resp_tool["description"] = description
            response_tools.append(resp_tool)

        return response_tools if response_tools else NOT_GIVEN

    def _tool_choice_to_responses_tool_choice(self, tool_choice: Any) -> Any:
        if self._is_not_given(tool_choice):
            return NOT_GIVEN

        if isinstance(tool_choice, str):
            return tool_choice

        # Chat Completions forced function syntax:
        # {"type":"function","function":{"name":"x"}}
        if isinstance(tool_choice, dict):
            if (
                tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
                and tool_choice["function"].get("name")
            ):
                return {"type": "function", "name": tool_choice["function"]["name"]}
            return tool_choice

        return tool_choice

    def _responses_request_params(self, context: LLMContext) -> Dict[str, Any]:
        params_from_context = self._context_to_openai_params(context)

        messages = params_from_context.get("messages") or []
        if self._pending_history is not None and messages != self._pending_context_messages:
            raise RuntimeError(
                "Cannot continue a changed context after an unfinished Responses request; "
                "retry the unchanged request or start a new conversation"
            )
        tools = params_from_context.get("tools", NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", NOT_GIVEN)

        request: Dict[str, Any] = {
            "model": getattr(self, "model_name", None) or self._setting("model"),
            "input": [],
            "tools": self._tools_to_responses_tools(tools),
            "tool_choice": self._tool_choice_to_responses_tool_choice(tool_choice),
            "temperature": self._setting("temperature"),
            "top_p": self._setting("top_p"),
            "service_tier": self._setting("service_tier"),
        }

        # Responses uses max_output_tokens.
        max_completion_tokens = self._setting("max_completion_tokens")
        max_tokens = self._setting("max_tokens")
        if not self._is_not_given(max_completion_tokens):
            request["max_output_tokens"] = max_completion_tokens
        elif not self._is_not_given(max_tokens):
            request["max_output_tokens"] = max_tokens

        # Pass model-specific settings (e.g., reasoning).
        extra = self._setting("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        request.update(extra)
        if request.get("store") is False:
            # Unstored reasoning cannot be recovered by response ID. Request
            # its encrypted replay payload without mutating caller settings.
            include = request.get("include")
            include = [] if include is None or self._is_not_given(include) else list(include)
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            request["include"] = include

        # Pipecat aggregates assistant text and tool calls into Chat-style
        # messages. Those are display/context projections, not replayable native
        # Responses output: flattening them loses phase, reasoning and boundaries.
        # Track only external inputs, whose prefix survives removal of empty
        # assistant stubs by the pipeline sanitizer.
        external = self._messages_to_responses_input(
            [message for message in messages if self._get_attr(message, "role") != "assistant"]
        )
        previous_external = self._external_context
        if external[:len(previous_external)] != previous_external:
            if any(self._get_attr(message, "role") in {"assistant", "tool"} for message in messages):
                raise ValueError("Responses context external history was rewritten; start a new service for a new conversation")
            self._native_history = []
            self._external_context = []
            self._previous_response_id = None
            previous_external = []
        if self._native_history:
            fresh_input = external[len(previous_external):]
        else:
            # Also allow a caller-supplied initial conversation. From this point
            # onward, provider output is retained verbatim, never reconstructed.
            fresh_input = self._messages_to_responses_input(messages)
        clean_history = deepcopy(self._native_history + fresh_input)
        has_filler = bool(filler_suffix())
        can_chain = (
            self._stateful_responses_enabled
            and self._previous_response_id
            and request.get("store") is not False
            and not has_filler
            and not self._previous_had_filler
        )
        if self._stateful_responses_enabled:
            request.setdefault("store", True)
        if can_chain:
            request["previous_response_id"] = self._previous_response_id
            request["input"] = deepcopy(fresh_input)
        else:
            # Filler studies intentionally keep earlier user messages filler-free.
            # Chaining would retain old filler on the server and change that study.
            request["input"] = apply_filler_to_last_user(deepcopy(clean_history))
        self._pending_history = (deepcopy(external), clean_history, has_filler, request.get("store") is not False)
        self._pending_context_messages = deepcopy(messages)

        # Drop NOT_GIVEN and empty markers.
        cleaned: Dict[str, Any] = {}
        for k, v in request.items():
            if self._is_not_given(v):
                continue
            cleaned[k] = v

        return cleaned

    @classmethod
    def _native_item(cls, item: Any) -> Any:
        """Copy SDK output items without dropping phase or encrypted reasoning."""
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", exclude_none=True)
        if isinstance(item, dict):
            return {key: cls._native_item(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [cls._native_item(value) for value in item]
        if hasattr(item, "__dict__"):
            return cls._native_item(vars(item))
        return deepcopy(item)

    def _remember_completed_response(self, response: Any) -> None:
        status = self._get_attr(response, "status")
        if status not in {None, "completed"}:
            raise RuntimeError(f"Cannot commit an unfinished Responses result: {status}")
        pending = self._pending_history
        if pending is None:
            raise RuntimeError("Responses completion has no matching request boundary")
        external, history, had_filler, stored = pending
        output = self._get_attr(response, "output") or []
        self._native_history = history + [self._native_item(item) for item in output]
        self._external_context = external
        self._previous_had_filler = had_filler
        self._pending_history = None
        self._pending_context_messages = None
        response_id = self._get_attr(response, "id")
        self._previous_response_id = str(response_id) if response_id and stored else None

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_items = getattr(response, "output", None) or []
        parts: List[str] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type != "message":
                continue
            content_parts = getattr(item, "content", None) or []
            for part in content_parts:
                part_type = getattr(part, "type", None)
                if part_type in {"output_text", "text", "input_text"}:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
        return "".join(parts)

    async def run_inference(
        self, context: LLMContext, max_tokens: int | None = None
    ) -> str | None:
        params = self._responses_request_params(context)
        params["stream"] = False
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        response = await self._client.responses.create(**params)
        self._remember_completed_response(response)
        return self._extract_response_text(response)

    @traced_llm
    async def _process_context(self, context: LLMContext):
        await self.start_ttfb_metrics()
        ttfb_stopped = False
        raw_ttfb_emitted = False
        raw_t0 = time.monotonic()
        function_call_map: Dict[str, Dict[str, str]] = {}
        processed_call_ids: set[str] = set()
        pending_function_calls: List[FunctionCallFromLLM] = []
        response_completed = False
        terminal_error = False
        completed_output_items: dict[int, Any] = {}

        async def run_pending_function_calls() -> None:
            """Run tool callbacks only after response usage has been emitted.

            Responses streams report a function call before ``response.completed``,
            while the latter is the event that carries billable token usage. The
            benchmark's tool callback writes and advances the transcript turn. If
            it runs immediately, the turn is persisted with ``tokens: null`` and
            the later usage frame is attributed to transient next-turn state.
            """

            if not pending_function_calls:
                return
            calls = list(pending_function_calls)
            pending_function_calls.clear()
            await self.run_function_calls(calls)

        params = self._responses_request_params(context)

        async with self._client.responses.stream(**params) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                # Metadata-only lifecycle events are not token arrival. The
                # first output/reasoning/function event is our raw TTFT; visible
                # text or a tool call separately stops the TTFAT timer below.
                is_raw_output_event = bool(
                    event_type
                    and (
                        event_type.startswith("response.output_")
                        or "reasoning" in event_type
                        or "function_call" in event_type
                    )
                )
                if is_raw_output_event and not raw_ttfb_emitted:
                    await self.push_frame(
                        MetricsFrame(
                            data=[
                                RawTTFBMetricsData(
                                    processor=self.name,
                                    value=time.monotonic() - raw_t0,
                                )
                            ]
                        )
                    )
                    raw_ttfb_emitted = True

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                        await self._push_llm_text(delta)
                    continue

                if event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", None) == "function_call":
                        item_id = getattr(item, "id", None)
                        call_id = getattr(item, "call_id", None)
                        name = getattr(item, "name", None)
                        if item_id:
                            function_call_map[item_id] = {
                                "call_id": call_id or item_id,
                                "name": name or "",
                            }
                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                    continue

                if event_type == "response.function_call_arguments.done":
                    item_id = getattr(event, "item_id", None)
                    name = getattr(event, "name", "")
                    args_text = getattr(event, "arguments", "{}")

                    mapping = function_call_map.get(item_id, {}) if item_id else {}
                    call_id = mapping.get("call_id") or item_id
                    function_name = name or mapping.get("name") or ""

                    if not call_id or not function_name:
                        logger.warning(
                            f"{self}: Skipping function call due to missing call_id/name "
                            f"(item_id={item_id}, name={function_name})"
                        )
                        continue

                    if call_id in processed_call_ids:
                        continue

                    try:
                        arguments = json.loads(args_text) if isinstance(args_text, str) else {}
                    except Exception:
                        logger.warning(
                            f"{self}: Failed to parse function arguments as JSON for {function_name}; "
                            "using raw payload"
                        )
                        arguments = {"_raw_arguments": args_text}

                    # Retain the complete native call even on compatible streams
                    # that omit output_item.done and response.completed.
                    completed_output_items.setdefault(
                        getattr(event, "output_index", len(completed_output_items)),
                        {"type": "function_call", "id": item_id, "call_id": call_id,
                         "name": function_name, "arguments": args_text},
                    )
                    if not ttfb_stopped:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True

                    processed_call_ids.add(call_id)
                    pending_function_calls.append(
                        FunctionCallFromLLM(
                            context=context,
                            tool_call_id=call_id,
                            function_name=function_name,
                            arguments=arguments,
                        )
                    )
                    continue

                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item is not None:
                        completed_output_items[getattr(event, "output_index", len(completed_output_items))] = item
                    if item is not None and getattr(item, "type", None) == "function_call":
                        item_id = getattr(item, "id", None)
                        call_id = getattr(item, "call_id", None) or item_id
                        function_name = getattr(item, "name", "")
                        args_text = getattr(item, "arguments", "{}")

                        if not call_id or not function_name or call_id in processed_call_ids:
                            continue

                        try:
                            arguments = json.loads(args_text) if isinstance(args_text, str) else {}
                        except Exception:
                            arguments = {"_raw_arguments": args_text}

                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True

                        processed_call_ids.add(call_id)
                        pending_function_calls.append(
                            FunctionCallFromLLM(
                                context=context,
                                tool_call_id=call_id,
                                function_name=function_name,
                                arguments=arguments,
                            )
                        )
                    continue

                if event_type == "response.completed":
                    response = getattr(event, "response", None)
                    if response is not None:
                        self._remember_completed_response(response)
                        effective_service_tier = getattr(response, "service_tier", None)
                        if effective_service_tier:
                            logger.info(
                                f"{self}: Responses API effective "
                                f"service_tier={effective_service_tier}"
                            )
                        model_name = getattr(response, "model", None)
                        if model_name and self.get_full_model_name() != model_name:
                            self.set_full_model_name(model_name)

                        usage = getattr(response, "usage", None)
                        if usage is not None:
                            input_details = getattr(usage, "input_tokens_details", None)
                            output_details = getattr(usage, "output_tokens_details", None)
                            tokens = LLMTokenUsage(
                                prompt_tokens=getattr(usage, "input_tokens", None),
                                completion_tokens=getattr(usage, "output_tokens", None),
                                total_tokens=getattr(usage, "total_tokens", None),
                                cache_read_input_tokens=(
                                    getattr(input_details, "cached_tokens", None)
                                    if input_details
                                    else None
                                ),
                                reasoning_tokens=(
                                    getattr(output_details, "reasoning_tokens", None)
                                    if output_details
                                    else None
                                ),
                            )
                            await self.start_llm_usage_metrics(tokens)
                    response_completed = True
                    await run_pending_function_calls()
                    continue

                if event_type in {"response.failed", "response.error", "response.incomplete", "error"}:
                    terminal_error = True
                    await self.push_error(error_msg=f"Responses API error event: {event}")

        # Defensive compatibility fallback for a provider stream that closes after
        # a complete function-call item but omits response.completed. Do not execute
        # a tool from an explicitly failed response.
        if pending_function_calls and not terminal_error:
            if not response_completed:
                # A compatible provider may omit the terminal response event.
                # Retain completed native items, but do not chain an unknown ID.
                self._remember_completed_response({
                    "output": [completed_output_items[index] for index in sorted(completed_output_items)]
                })
                logger.warning(
                    f"{self}: Responses stream ended without response.completed; "
                    "running completed function calls without usage metrics"
                )
            await run_pending_function_calls()

        if not ttfb_stopped:
            await self.stop_ttfb_metrics()
