"""Focused coverage for the Z.AI GLM-5.3 route.

GLM-5.3 always reasons: the API rejects `none`/`minimal`/`medium`/`xhigh` with
HTTP 400 code 1210, so the only valid efforts are low, high, and max.
"""

import pytest

from multi_turn_eval.cli import SERVICE_ALIASES
from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.vllm_openai import QwenReasoningContextAggregatorPair
from multi_turn_eval.services.zai_logged import LoggedZaiLLMService


class ConcretePipeline(BasePipeline):
    def _setup_context(self):
        pass

    def _setup_llm(self):
        pass

    def _build_task(self):
        pass

    async def _queue_first_turn(self):
        pass

    async def _queue_next_turn(self):
        pass


class CapturingService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def zai_pipeline_stub():
    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "zai"
    return pipeline


def test_service_alias_registered():
    assert SERVICE_ALIASES["zai"] == (
        "multi_turn_eval.services.zai_logged.LoggedZaiLLMService"
    )


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_thinking_enabled_and_effort_ride_in_extra_body(monkeypatch, effort):
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setenv("MTE_ZAI_REASONING_EFFORT", effort)
    monkeypatch.delenv("MTE_ZAI_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MTE_ZAI_TEMPERATURE", raising=False)

    service = zai_pipeline_stub()._create_llm(CapturingService, "glm-5.3")

    assert service.kwargs["base_url"] == "https://api.z.ai/api/paas/v4"
    assert service.kwargs["api_key"] == "test-zai-key"
    params = service.kwargs["params"]
    assert params.max_tokens == 16384
    # Neither field is a named argument of the OpenAI SDK's completions.create();
    # sending them top-level raises "unexpected keyword argument 'thinking'".
    assert params.extra == {
        "extra_body": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": effort,
        }
    }


@pytest.mark.parametrize("effort", ["none", "minimal", "medium", "xhigh"])
def test_efforts_the_api_rejects_fail_fast(monkeypatch, effort):
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setenv("MTE_ZAI_REASONING_EFFORT", effort)

    with pytest.raises(ValueError, match="cannot disable thinking"):
        zai_pipeline_stub()._create_llm(CapturingService, "glm-5.3")


def test_missing_key_is_an_explicit_error(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setenv("MTE_ZAI_REASONING_EFFORT", "low")

    with pytest.raises(EnvironmentError, match="ZAI_API_KEY"):
        zai_pipeline_stub()._create_llm(CapturingService, "glm-5.3")


def test_reasoning_history_is_preserved_unconditionally():
    # Z.AI requires the full historical reasoning_content on assistant tool-call
    # messages ("Preserved Thinking"); the Qwen model gate must not apply here.
    service = object.__new__(LoggedZaiLLMService)
    pair = service.create_reasoning_context_aggregator_pair(context=object())
    assert isinstance(pair, QwenReasoningContextAggregatorPair)


def test_raw_ttfb_is_emitted_on_the_first_chunk():
    """The vLLM base arms TTFT on the first *answer* delta but emits no raw
    TTFB; without the wrapper added here, raw_ttfb_ms is never recorded and the
    thinking delay cannot be separated from network + prefill."""

    import asyncio
    from types import SimpleNamespace

    from multi_turn_eval.metrics import RawTTFBMetricsData

    service = object.__new__(LoggedZaiLLMService)
    service._name = "zai-test"
    pushed = []

    async def fake_push_frame(frame):
        pushed.append(frame)

    service.push_frame = fake_push_frame

    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="thinking"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))]),
    ]

    async def parent_stream():
        for chunk in chunks:
            yield chunk

    async def fake_super(_params):
        # Stand in for the real request round-trip, which is what raw TTFB must
        # capture; the clock has to be running across this await.
        await asyncio.sleep(0.06)
        return parent_stream()

    # Bypass the parent's own wrapper; we are testing only the raw-TTFB layer.
    import multi_turn_eval.services.zai_logged as mod

    async def drive():
        stream = await LoggedZaiLLMService.get_chat_completions(service, object())
        return [c async for c in stream]

    original = mod.VLLMOpenAILLMService.get_chat_completions
    mod.VLLMOpenAILLMService.get_chat_completions = staticmethod(fake_super)
    try:
        seen = asyncio.run(drive())
    finally:
        mod.VLLMOpenAILLMService.get_chat_completions = original

    assert len(seen) == 2, "every chunk must pass through untouched"
    raw = [
        d
        for f in pushed
        for d in getattr(f, "data", [])
        if isinstance(d, RawTTFBMetricsData)
    ]
    assert len(raw) == 1, "raw TTFB must fire exactly once, on the first chunk"
    # The clock must start before the parent's await (which issues the request);
    # starting it after makes every raw TTFB read ~0.
    assert raw[0].value >= 0.05, "raw TTFB must include the request round-trip"


def test_raw_ttfb_clock_starts_before_the_request_is_issued():
    import asyncio
    import inspect

    from multi_turn_eval.services.zai_logged import LoggedZaiLLMService as svc

    src = inspect.getsource(svc.get_chat_completions)
    t0_line = next(i for i, l in enumerate(src.splitlines()) if "raw_t0 = time.monotonic()" in l)
    await_line = next(i for i, l in enumerate(src.splitlines()) if "await super().get_chat_completions" in l)
    assert t0_line < await_line, "raw_t0 must be captured before the awaited request"



def test_thought_frames_are_enabled_for_glm(monkeypatch):
    """The base class gates thought-frame emission on a Qwen model-name match.
    "glm-5.3" does not match it, so without forcing the gate the reasoning
    aggregator receives nothing and reasoning_content is dropped from history."""

    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.setenv("MTE_ZAI_REASONING_EFFORT", "low")

    from multi_turn_eval.services.vllm_openai import is_qwen_reasoning_history_model

    # Precondition: the inherited gate really is off for this model name.
    assert not is_qwen_reasoning_history_model("glm-5.3")

    service = LoggedZaiLLMService(api_key="test-zai-key", model="glm-5.3")
    assert service._qwen_reasoning_history_enabled is True
