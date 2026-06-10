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

from pipecat.services.openai.llm import OpenAILLMService


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
        super().__init__(*args, **kwargs)
        self._ttft_armed = False

    async def get_chat_completions(self, params_from_context: Any) -> Any:
        # _process_context calls this once per turn, right after start_ttfb_metrics()
        # and before iterating the stream — reset the per-turn arming flag here.
        self._ttft_armed = False
        stream = await super().get_chat_completions(params_from_context)

        async def _armed_stream() -> Any:
            try:
                async for chunk in stream:
                    if not self._ttft_armed:
                        choices = getattr(chunk, "choices", None)
                        delta = getattr(choices[0], "delta", None) if choices else None
                        # First non-thought token = first text content or tool call.
                        if delta is not None and (
                            getattr(delta, "content", None) or getattr(delta, "tool_calls", None)
                        ):
                            self._ttft_armed = True
                    yield chunk
            finally:
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
