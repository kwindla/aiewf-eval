"""Deliver realtime tool results independently of local speech-detector timing.

Pipecat 1.10's universal assistant aggregator suppresses a tool-result context
push while the user is speaking. In realtime mode, a server can call a tool
before local VAD stops; no subsequent inference push releases that result.
The provider already owns inference and turn detection, so return its requested
tool result without waiting for local user/bot speech flags.
"""

from pipecat.frames.frames import FunctionCallResultFrame
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection


class RealtimeAssistantAggregator(LLMAssistantAggregator):
    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        # Retain Pipecat's validation, grouping, run_llm=False, and callbacks;
        # only remove the cascade-specific local speech gate.
        user_speaking = self._user_speaking
        self._user_speaking = False
        try:
            await super()._handle_function_call_result(frame)
        finally:
            self._user_speaking = user_speaking

    async def _maybe_push_context_after_function_result(self):
        if not self.has_queued_frame(FunctionCallResultFrame):
            await self.push_context_frame(FrameDirection.UPSTREAM)


class RealtimeContextAggregatorPair(LLMContextAggregatorPair):
    """Universal pair with a realtime-only tool-result delivery fix."""

    def __init__(self, context, *, user_params=None):
        self._user = LLMUserAggregator(
            context,
            params=user_params or LLMUserAggregatorParams(),
            _realtime_service_mode=True,
        )
        self._assistant = RealtimeAssistantAggregator(
            context,
            params=LLMAssistantAggregatorParams(),
            _realtime_service_mode=True,
            _paired_user_aggregator=self._user,
        )
