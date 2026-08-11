"""Targeted coverage for OpenAI Responses routing in ``BasePipeline``."""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.services.openai.llm import OpenAILLMService

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService


class ConcretePipeline(BasePipeline):
    """Minimal concrete shell used to exercise the base service factory."""

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


@pytest.mark.parametrize("effort", ["none", "medium"])
def test_gpt55_routes_to_responses_with_requested_effort(monkeypatch, effort):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MTE_OPENAI_RESPONSES_REASONING_EFFORT", effort)

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openai"
    service = pipeline._create_llm(OpenAILLMService, "gpt-5.5")

    assert isinstance(service, OpenAIResponsesLLMService)
    assert service._settings.model == "gpt-5.5"
    assert service._settings.extra == {"reasoning": {"effort": effort}}


def test_gpt55_routing_only_applies_to_openai_service(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openrouter"
    service = pipeline._create_llm(OpenAILLMService, "gpt-5.5")

    assert type(service) is OpenAILLMService


@pytest.mark.parametrize("model", ["gpt-5.5-pro", "o3-pro", "future-model-pro"])
def test_factory_rejects_openai_pro_before_service_creation(monkeypatch, model):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openai"

    with pytest.raises(ValueError, match="Pro models are excluded"):
        pipeline._create_llm(OpenAILLMService, model)


def test_factory_rejects_namespaced_openai_pro_on_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openrouter"

    with pytest.raises(ValueError, match="Pro models are excluded"):
        pipeline._create_llm(OpenAILLMService, "openai/o3-pro")


class _ResponseStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def iterate():
            for event in self._events:
                yield event

        return iterate()


def test_tool_callback_runs_after_completed_usage_metrics():
    """Tool turns must persist usage before their callback advances the turn."""

    item = SimpleNamespace(
        type="function_call",
        id="item-1",
        call_id="call-1",
        name="end_session",
        arguments="{}",
    )
    usage = SimpleNamespace(
        input_tokens=123,
        output_tokens=17,
        total_tokens=140,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    events = [
        SimpleNamespace(type="response.output_item.added", item=item),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="item-1",
            name="end_session",
            arguments="{}",
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(model="gpt-5.5", usage=usage),
        ),
    ]

    service = object.__new__(OpenAIResponsesLLMService)
    service._name = "OpenAIResponsesLLMService#test"
    service._client = SimpleNamespace(
        responses=SimpleNamespace(stream=lambda **kwargs: _ResponseStream(events))
    )
    service._responses_request_params = lambda context: {}
    service.get_full_model_name = lambda: "gpt-5.5"
    service.set_full_model_name = lambda model: None

    order = []

    async def start_ttfb_metrics():
        return None

    async def stop_ttfb_metrics():
        return None

    async def start_usage_metrics(tokens):
        order.append(("usage", tokens.prompt_tokens, tokens.completion_tokens))

    async def run_function_calls(calls):
        order.append(("function", calls[0].function_name, calls[0].tool_call_id))

    async def push_error(*args, **kwargs):
        raise AssertionError("unexpected response error")

    service.start_ttfb_metrics = start_ttfb_metrics
    service.stop_ttfb_metrics = stop_ttfb_metrics
    service.start_llm_usage_metrics = start_usage_metrics
    service.run_function_calls = run_function_calls
    service.push_error = push_error

    asyncio.run(service._process_context(SimpleNamespace()))

    assert order == [
        ("usage", 123, 17),
        ("function", "end_session", "call-1"),
    ]
