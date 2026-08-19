"""Z.AI (OpenAI-compatible) GLM service with TTFAT and preserved thinking.

GLM-5.3 always reasons — the API rejects any attempt to disable thinking with
code 1210 ("This model always engages in thinking and cannot be disabled;
please use low, high, or max") — and streams its chain-of-thought as separate
``reasoning_content`` deltas. Two consequences for this benchmark:

1. **Latency.** The stock ``OpenAILLMService`` stops TTFB on the first
   ``choices`` chunk, which for this model is always a *reasoning* token. Every
   GLM-5.3 row would then report a thinking-free latency that no user ever
   experiences. :class:`VLLMOpenAILLMService` arms TTFT on the first answer
   (``content`` / ``tool_calls``) delta, which gives the TTFAT half. It does
   **not** emit ``RawTTFBMetricsData`` — only :class:`LoggedLilacLLMService` and
   its subclasses do — so this class adds that emission itself, and each turn
   records both TTFAT (``ttfb_ms``) and raw TTFB (``raw_ttfb_ms``), whose
   difference is the per-turn thinking delay.

   The two halves cannot simply be inherited from one place: the Lilac base
   records both metrics but pushes no Pipecat thought frames, and the reasoning
   aggregator in (2) is driven entirely by those frames. Keeping the vLLM base
   preserves the correctness contract and re-adds the missing metric here.

2. **Correctness across tool calls.** Z.AI's chat-completions reference: "To
   enable Preserved Thinking, you must forward the full, unmodified, and
   correctly ordered historical ``reasoning_content`` in ``messages``. Missing,
   truncated, rewritten, or reordered blocks may degrade performance." The
   Qwen-reasoning aggregator implements exactly that contract, so it is enabled
   unconditionally here rather than behind the Qwen model gate.

Nothing in the vLLM base class is vLLM-specific for this route: the
tool-call-index normalization is opt-in and stays off, and the thinking-budget
knob is only reachable from the ``vllm-openai`` construction branch.
"""

from __future__ import annotations

import time
from typing import Any

from pipecat.frames.frames import MetricsFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from multi_turn_eval.metrics import RawTTFBMetricsData
from multi_turn_eval.services.vllm_openai import (
    QwenReasoningContextAggregatorPair,
    VLLMOpenAILLMService,
)


class LoggedZaiLLMService(VLLMOpenAILLMService):
    """Z.AI GLM service recording TTFAT + raw TTFB and replaying reasoning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The base class only pushes thought frames when its Qwen model gate is
        # on, and "glm-5.3" does not match that gate. Without this the reasoning
        # aggregator below is installed but never receives any thought text, so
        # `reasoning_content` is silently dropped from tool-call history — the
        # exact failure Z.AI's preserved-thinking contract warns about.
        self._qwen_reasoning_history_enabled = True

    def create_reasoning_context_aggregator_pair(
        self, context: LLMContext
    ) -> QwenReasoningContextAggregatorPair:
        # Unconditional: every GLM-5.3 turn reasons, and Z.AI requires the
        # history to be replayed on assistant tool-call messages.
        return QwenReasoningContextAggregatorPair(context)

    async def get_chat_completions(self, params_from_context: Any) -> Any:
        # The parent wraps the stream to arm TTFT on the first *answer* delta.
        # Wrap that wrapper to time the first chunk of any kind — for this model
        # that is the first reasoning token, so raw_ttfb is the network + prefill
        # + first-token floor and ttfb - raw_ttfb is the thinking delay.
        # Start the clock BEFORE the await: that await is where the request is
        # issued and the response stream opens, so timing after it measures ~0.
        raw_t0 = time.monotonic()
        stream = await super().get_chat_completions(params_from_context)

        async def _raw_timed_stream() -> Any:
            raw_ttfb_emitted = False
            async for chunk in stream:
                if not raw_ttfb_emitted and getattr(chunk, "choices", None):
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
                yield chunk

        return _raw_timed_stream()
