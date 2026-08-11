"""vLLM OpenAI-compatible LLM service that times TTFT to the first NON-THOUGHT token.

Stock pipecat (`BaseOpenAILLMService._process_context`, base_llm.py:467) stops the
TTFB clock on the first streamed chunk that carries any `choices` — i.e. the first
role / reasoning delta. For a reasoning model served with a reasoning parser, the
answer (`content`) tokens do not begin until the model finishes thinking, so the
stock metric badly understates TTFT when thinking is enabled (observed: ~270 ms
reported vs. ~2.2 s to the first real answer token on Nemotron-3-Super).

This subclass defers the TTFB stop until a delta actually carries user-visible
output (text `content` or a `tool_call`). It does so WITHOUT duplicating the large
`_process_context` method:

  * `get_chat_completions` wraps the chunk stream and "arms" a flag on the first
    content/tool delta (resetting it per invocation);
  * `stop_ttfb_metrics` is gated on that flag.

Pipecat already calls `stop_ttfb_metrics()` on every chunk, so once armed the
existing call records TTFB at the correct moment; before that it is a no-op.
`reasoning_content`-only, role-only, and empty deltas never arm it. (Nemotron over
vLLM emits only text or tool-call tokens — there is no audio-transcript path.)
"""

from typing import Any

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService

from multi_turn_eval.services.filler import apply_filler_to_last_user


def is_qwen_reasoning_history_model(model: str | None) -> bool:
    """Return whether *model* needs Qwen thinking preserved across tool calls.

    The deployed 27B model has appeared under both ``Qwen3.5`` and ``Qwen3.6``
    labels in run infrastructure. Keep the compatibility gate narrow to those
    generations instead of changing every model routed through vllm-openai.
    """

    compact = "".join(ch for ch in (model or "").lower() if ch.isalnum())
    return "qwen35" in compact or "qwen36" in compact


class QwenReasoningAssistantAggregator(LLMAssistantAggregator):
    """Universal assistant aggregator that retains Qwen reasoning history.

    vLLM streams Qwen's thought text separately as ``reasoning_content``. The
    service converts those deltas to Pipecat thought frames; this aggregator
    retains the completed thought and adds it to the assistant message created
    for the following tool call. That produces the history shape expected by
    Qwen's ``preserve_thinking`` chat-template path::

        {
            "role": "assistant",
            "reasoning_content": "...",
            "tool_calls": [...],
        }

    Pipecat's normal thought event still fires, so callers can record the
    thought without exposing it as assistant answer text.
    """

    def __init__(
        self,
        context: LLMContext,
        *,
        params: LLMAssistantAggregatorParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(context, params=params, **kwargs)
        self._qwen_reasoning_parts: list[str] = []
        self._qwen_completed_reasoning: str | None = None

    async def _handle_thought_start(self, frame: LLMThoughtStartFrame) -> None:
        self._qwen_reasoning_parts = []
        self._qwen_completed_reasoning = None
        await super()._handle_thought_start(frame)

    async def _handle_thought_text(self, frame: LLMThoughtTextFrame) -> None:
        self._qwen_reasoning_parts.append(frame.text)
        await super()._handle_thought_text(frame)

    async def _handle_thought_end(self, frame: LLMThoughtEndFrame) -> None:
        reasoning = "".join(self._qwen_reasoning_parts)
        self._qwen_completed_reasoning = reasoning if reasoning else None
        self._qwen_reasoning_parts = []
        await super()._handle_thought_end(frame)

    def _attach_reasoning_to_tool_call(self, tool_call_id: str) -> None:
        reasoning = self._qwen_completed_reasoning
        if not reasoning:
            return

        # The stock universal aggregator appends a tool-result placeholder
        # immediately after the assistant tool-call message, so locate the
        # matching assistant message rather than assuming it is last.
        for message in reversed(self._context.get_messages()):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            if any(
                isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id
                for tool_call in tool_calls
            ):
                # A single assistant response can issue multiple tool calls.
                # Each assistant tool-call message must carry the reasoning
                # that preceded that response, so retain it until the next
                # thought starts rather than consuming it after the first call.
                message["reasoning_content"] = reasoning
                return

    async def _handle_function_call_in_progress(
        self, frame: FunctionCallInProgressFrame
    ) -> None:
        await super()._handle_function_call_in_progress(frame)
        self._attach_reasoning_to_tool_call(frame.tool_call_id)


class QwenReasoningContextAggregatorPair(LLMContextAggregatorPair):
    """Universal context-aggregator pair with a Qwen-aware assistant side."""

    def __init__(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
        add_tool_change_messages: bool | None = None,
    ) -> None:
        user_params = user_params or LLMUserAggregatorParams()
        assistant_params = assistant_params or LLMAssistantAggregatorParams()
        if add_tool_change_messages is not None:
            user_params.add_tool_change_messages = add_tool_change_messages
            assistant_params.add_tool_change_messages = add_tool_change_messages
        self._user = LLMUserAggregator(context, params=user_params)
        self._assistant = QwenReasoningAssistantAggregator(
            context, params=assistant_params
        )


class VLLMOpenAILLMService(OpenAILLMService):
    """OpenAI-compatible vLLM service whose TTFB metric is the first answer token.

    Thinking-budget cap is handled by `aiewf-eval/pipelines/base.py`'s `vllm-openai`
    service-construction branch: when `MTE_VLLM_THINKING=1 MTE_VLLM_THINKING_BUDGET=N`
    are set, it ships `extra_body.vllm_xargs.thinking_budget=N` per request → vLLM
    populates `SamplingParams.extra_args` → the server-side
    `ThinkingBudgetLogitsProcessor` plugin reads it per request. Nothing in this
    service file needs to know about it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Some OpenAI-compatible servers expose a parser bug where the streamed
        # ``tool_calls[].index`` is the called tool's position in the request's
        # tool schema rather than the call's zero-based position in the model
        # response. Pipecat (correctly) uses the latter to coalesce streaming
        # fragments. Keep this compatibility behavior opt-in so conforming
        # vLLM/OpenAI streams remain untouched.
        self._normalize_tool_call_indices = bool(
            kwargs.pop("normalize_tool_call_indices", False)
        )
        model = kwargs.get("model")
        settings = kwargs.get("settings")
        if settings is not None and getattr(settings, "model", None):
            # Pipecat's canonical ``settings`` object wins over the deprecated
            # direct model argument when callers provide both.
            model = settings.model
        self._qwen_reasoning_history_enabled = is_qwen_reasoning_history_model(model)
        super().__init__(*args, **kwargs)
        self._ttft_armed = False

    def create_reasoning_context_aggregator_pair(
        self, context: LLMContext
    ) -> QwenReasoningContextAggregatorPair | None:
        """Provide a reasoning-aware aggregator only for the gated Qwen models."""

        if not self._qwen_reasoning_history_enabled:
            return None
        return QwenReasoningContextAggregatorPair(context)

    def build_chat_completion_params(self, params_from_context):
        # MTE_FILLER_DOTS latent-scratchpad filler on the final user turn of the
        # outgoing request only (persisted context untouched) — same seam as
        # LoggedLilacLLMService; see services/filler.py.
        params = super().build_chat_completion_params(params_from_context)
        if isinstance(params, dict) and isinstance(params.get("messages"), list):
            params["messages"] = apply_filler_to_last_user(params["messages"])
        return params

    async def get_chat_completions(self, params_from_context: Any) -> Any:
        # _process_context calls this once per turn, right after start_ttfb_metrics()
        # and before iterating the stream — reset the per-turn arming flag here.
        self._ttft_armed = False
        stream = await super().get_chat_completions(params_from_context)

        async def _armed_stream() -> Any:
            thought_active = False
            tool_index_maps: dict[int, dict[int, int]] = {}
            try:
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None)
                    if self._normalize_tool_call_indices and choices:
                        for choice_position, choice in enumerate(choices):
                            choice_delta = getattr(choice, "delta", None)
                            tool_calls = (
                                getattr(choice_delta, "tool_calls", None)
                                if choice_delta
                                else None
                            )
                            if not tool_calls:
                                continue
                            choice_index = getattr(choice, "index", choice_position)
                            index_map = tool_index_maps.setdefault(choice_index, {})
                            for tool_call in tool_calls:
                                raw_index = getattr(tool_call, "index", None)
                                if raw_index is None:
                                    continue
                                if raw_index not in index_map:
                                    index_map[raw_index] = len(index_map)
                                tool_call.index = index_map[raw_index]
                    delta = getattr(choices[0], "delta", None) if choices else None
                    if delta is not None:
                        # vLLM/OpenAI-compatible routes expose the same Qwen
                        # thought stream under either provider extension. The
                        # dedicated BaseTen route currently emits ``reasoning``;
                        # other deployments and captured port-to-port artifacts
                        # use ``reasoning_content``.
                        reasoning = (
                            getattr(delta, "reasoning", None)
                            or getattr(delta, "reasoning_content", None)
                        )
                        if self._qwen_reasoning_history_enabled and reasoning:
                            if not thought_active:
                                await self.push_frame(LLMThoughtStartFrame())
                                thought_active = True
                            await self.push_frame(LLMThoughtTextFrame(reasoning))

                        # First non-thought token = first text content or tool call.
                        has_visible_output = bool(
                            getattr(delta, "content", None)
                            or getattr(delta, "tool_calls", None)
                        )
                        if thought_active and has_visible_output:
                            thought_active = False
                            await self.push_frame(LLMThoughtEndFrame())
                        if not self._ttft_armed and has_visible_output:
                            self._ttft_armed = True
                    yield chunk
            finally:
                if thought_active:
                    await self.push_frame(LLMThoughtEndFrame())
                # Preserve pipecat's explicit stream cleanup (uvloop asyncgen safety
                # on Python 3.12+); the parent only closes this wrapper, not the
                # underlying OpenAI stream.
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        return _armed_stream()

    async def stop_ttfb_metrics(self, *, end_time: float | None = None) -> None:
        # Defer the parent's per-chunk stop until a user-visible (non-thought)
        # token has actually streamed. Keep the keyword-only end_time from the
        # FrameProcessor signature so timestamped call sites stay compatible.
        if self._ttft_armed:
            await super().stop_ttfb_metrics(end_time=end_time)
