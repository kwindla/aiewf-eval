"""Targeted coverage for OpenAI Responses routing in ``BasePipeline``."""

import asyncio
from types import SimpleNamespace

import pytest
from openai import NOT_GIVEN
from pipecat.services.openai.llm import OpenAILLMService

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService
from multi_turn_eval.metrics import RawTTFBMetricsData


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
    assert service._stateful_responses_enabled is True
    assert service._settings.model == "gpt-5.5"
    assert service._settings.extra == {
        "reasoning": {"effort": effort},
        "service_tier": "priority",
    }


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-4.1"])
def test_openai_fast_mode_is_sent_for_ttfat_models(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MTE_OPENAI_SERVICE_TIER", "fast")
    monkeypatch.setenv("MTE_OPENAI_RESPONSES_REASONING_EFFORT", "none")

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openai"
    service = pipeline._create_llm(OpenAILLMService, model)

    assert isinstance(service, OpenAIResponsesLLMService)
    assert service._settings.extra["service_tier"] == "fast"
    if model.startswith("gpt-5.6"):
        assert service._settings.extra["reasoning"] == {"effort": "none"}

    monkeypatch.setattr(
        service,
        "_context_to_openai_params",
        lambda _context: {
            "messages": [],
            "tools": NOT_GIVEN,
            "tool_choice": NOT_GIVEN,
        },
    )
    request = service._responses_request_params(SimpleNamespace())
    assert request["service_tier"] == "fast"


def test_gpt55_routing_only_applies_to_openai_service(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openrouter"
    service = pipeline._create_llm(OpenAILLMService, "gpt-5.5")

    assert type(service) is OpenAILLMService


def test_grok46_routes_to_stateful_xai_responses_low(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("MTE_XAI_REASONING_EFFORT", "low")

    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "xai"
    service = pipeline._create_llm(OpenAIResponsesLLMService, "grok-4.6")

    assert isinstance(service, OpenAIResponsesLLMService)
    assert service._settings.model == "grok-4.6"
    assert str(service._client.base_url) == "https://api.x.ai/v1/"
    assert service._stateful_responses_enabled is True
    assert service._settings.extra["reasoning"] == {"effort": "low"}
    assert service._settings.extra["store"] is True
    assert service._settings.extra["prompt_cache_key"].startswith("aiewf-grok46-")


def test_grok46_rejects_unsupported_effort(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("MTE_XAI_REASONING_EFFORT", "none")
    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "xai"

    with pytest.raises(ValueError, match="expected one of"):
        pipeline._create_llm(OpenAIResponsesLLMService, "grok-4.6")


@pytest.mark.parametrize("stateful", [False, True])
def test_responses_initial_request_keeps_full_context(stateful):
    service = OpenAIResponsesLLMService(
        model="grok-4.6", api_key="test-key", stateful_responses_enabled=stateful
    )
    messages = [
        {"role": "system", "content": "Answer from the event knowledge base."},
        {"role": "user", "content": "When does registration open?"},
    ]
    service._context_to_openai_params = lambda context: {"messages": messages}

    request = service._responses_request_params(SimpleNamespace())

    assert "previous_response_id" not in request
    assert [item["role"] for item in request["input"]] == ["system", "user"]


@pytest.mark.parametrize("tool_calls", [0, 1, 2])
def test_stateful_continuation_sends_only_new_user_or_tool_outputs(tool_calls):
    service = OpenAIResponsesLLMService(
        model="grok-4.6", api_key="test-key", stateful_responses_enabled=True
    )
    assistant = {"role": "assistant", "content": "Registration opens at 8 AM."}
    if tool_calls:
        assistant["tool_calls"] = [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            for index in range(tool_calls)
        ]
        followup = [
            {"role": "tool", "tool_call_id": f"call-{index}", "content": "found"}
            for index in range(tool_calls)
        ]
    else:
        followup = [{"role": "user", "content": "Where do I register?"}]
    messages = [
        {"role": "system", "content": "Answer event questions."},
        {"role": "user", "content": "When does registration open?"},
    ]
    service._context_to_openai_params = lambda context: {"messages": messages}
    service._responses_request_params(SimpleNamespace())
    service._remember_completed_response(SimpleNamespace(id="resp-previous", output=[
        {"type": "message", "role": "assistant", "phase": "final_answer", "content": []}
    ]))
    messages.extend([assistant, *followup])

    request = service._responses_request_params(SimpleNamespace())

    assert request["previous_response_id"] == "resp-previous"
    if tool_calls:
        assert request["input"] == [
            {"type": "function_call_output", "call_id": f"call-{index}", "output": "found"}
            for index in range(tool_calls)
        ]
    else:
        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Where do I register?"}],
            }
        ]


def test_stateless_responses_keep_history_after_a_completed_response():
    service = OpenAIResponsesLLMService(model="gpt-5.5", api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]
    service._context_to_openai_params = lambda context: {"messages": messages}
    service._responses_request_params(SimpleNamespace())
    native = {"id": "msg-1", "type": "message", "role": "assistant", "phase": "final_answer",
              "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}
    service._remember_completed_response(SimpleNamespace(id="resp-stateless", output=[native]))
    messages.extend([{"role": "assistant", "content": "Hi"},
                     {"role": "user", "content": "When does registration open?"}])
    request = service._responses_request_params(SimpleNamespace())
    assert "previous_response_id" not in request
    assert [item["role"] for item in request["input"]] == ["user", "assistant", "user"]
    assert request["input"][1] == native


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
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thinking"),
        SimpleNamespace(type="response.output_item.added", item=item),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="item-1",
            name="end_session",
            arguments="{}",
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp-1", model="gpt-5.5", usage=usage),
        ),
    ]

    service = object.__new__(OpenAIResponsesLLMService)
    service._name = "OpenAIResponsesLLMService#test"
    service._client = SimpleNamespace(
        responses=SimpleNamespace(stream=lambda **kwargs: _ResponseStream(events))
    )
    service._stateful_responses_enabled = True
    service._previous_response_id = None
    service._pending_history = ([], [], False, True)
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

    async def push_frame(frame):
        metric = frame.data[0]
        assert isinstance(metric, RawTTFBMetricsData)
        order.append(("raw",))

    service.push_frame = push_frame

    asyncio.run(service._process_context(SimpleNamespace()))

    assert order == [
        ("raw",),
        ("usage", 123, 17),
        ("function", "end_session", "call-1"),
    ]
    assert service._previous_response_id == "resp-1"
