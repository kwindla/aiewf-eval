"""Focused coverage for BaseTen reasoning, timing, and request shaping."""

import asyncio
from types import SimpleNamespace

import pytest
from openai import NOT_GIVEN

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.baseten_logged import LoggedBaseTenLLMService


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


def baseten_pipeline_stub():
    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "baseten"
    return pipeline


def test_baseten_kimi_thinking_request_uses_model_api_argument(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")
    monkeypatch.setenv("MTE_BASETEN_REASONING_EFFORT", "omit")
    monkeypatch.setenv("MTE_BASETEN_ENABLE_THINKING", "true")
    monkeypatch.setenv("MTE_BASETEN_MAX_TOKENS", "8192")
    monkeypatch.setenv("MTE_BASETEN_TEMPERATURE", "1.0")
    monkeypatch.setenv("MTE_BASETEN_TOP_P", "0.95")

    service = baseten_pipeline_stub()._create_llm(
        CapturingService, "moonshotai/Kimi-K2.6"
    )

    params = service.kwargs["params"]
    assert params.max_tokens == 8192
    assert params.temperature == 1.0
    assert params.top_p == 0.95
    assert params.extra == {
        "extra_body": {"chat_template_args": {"enable_thinking": True}}
    }


def test_baseten_top_p_is_optional(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")
    monkeypatch.setenv("MTE_BASETEN_REASONING_EFFORT", "none")
    monkeypatch.delenv("MTE_BASETEN_ENABLE_THINKING", raising=False)
    monkeypatch.delenv("MTE_BASETEN_TOP_P", raising=False)

    service = baseten_pipeline_stub()._create_llm(
        CapturingService, "moonshotai/Kimi-K2.6"
    )

    params = service.kwargs["params"]
    assert params.top_p is NOT_GIVEN
    assert params.extra == {"extra_body": {"reasoning_effort": "none"}}


def test_baseten_max_tokens_can_be_omitted(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")
    monkeypatch.setenv("MTE_BASETEN_REASONING_EFFORT", "omit")
    monkeypatch.setenv("MTE_BASETEN_MAX_TOKENS", "")

    service = baseten_pipeline_stub()._create_llm(
        CapturingService, "muse-glimmer-30b"
    )

    params = service.kwargs["params"]
    assert params.max_tokens is NOT_GIVEN


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1"])
def test_baseten_rejects_invalid_top_p(monkeypatch, value):
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")
    monkeypatch.setenv("MTE_BASETEN_TOP_P", value)

    with pytest.raises(ValueError, match="expected 0 < top_p <= 1"):
        baseten_pipeline_stub()._create_llm(
            CapturingService, "moonshotai/Kimi-K2.6"
        )


def test_baseten_ttfat_ignores_reasoning_content_deltas():
    chunks = [
        SimpleNamespace(
            usage=None,
            model=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning_content="I should calculate this.",
                content=None,
                audio=None,
                tool_calls=None,
            ))],
        ),
        SimpleNamespace(
            usage=None,
            model=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning_content=None,
                content="323",
                audio=None,
                tool_calls=None,
            ))],
        ),
    ]

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not chunks:
                raise StopAsyncIteration
            return chunks.pop(0)

        async def aclose(self):
            pass

    service = object.__new__(LoggedBaseTenLLMService)
    service._name = "LoggedBaseTenLLMService#test"
    stops = []
    visible_text = []

    async def noop(*args, **kwargs):
        pass

    async def stop_ttfb_metrics(*args, **kwargs):
        stops.append(len(visible_text))

    async def push_text(text):
        visible_text.append(text)

    async def get_chat_completions(context):
        return Stream()

    service.start_ttfb_metrics = noop
    service.stop_ttfb_metrics = stop_ttfb_metrics
    service.push_frame = noop
    service._push_llm_text = push_text
    service.get_chat_completions = get_chat_completions
    service.get_full_model_name = lambda: "moonshotai/Kimi-K2.6"
    service.set_full_model_name = lambda model: None
    service.run_function_calls = noop

    asyncio.run(service._process_context(SimpleNamespace()))

    assert stops == [0]
    assert visible_text == ["323"]
