"""Focused coverage for generic OpenRouter routing and request shaping."""

from openai._types import NOT_GIVEN

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.services.openrouter_logged import LoggedOpenRouterLLMService


LAGUNA_MODEL = "poolside/laguna-s-2.1"


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


class CapturingService:
    """Service double that exposes the kwargs produced by ``_create_llm``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def openrouter_pipeline_stub():
    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "openrouter"
    return pipeline


def test_laguna_slug_and_explicit_reasoning_off_request_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("MTE_OPENROUTER_REASONING_OFF", "1")
    monkeypatch.setenv("MTE_OPENROUTER_MAX_TOKENS", "8192")
    monkeypatch.delenv("MTE_OPENROUTER_TEMPERATURE", raising=False)

    service = openrouter_pipeline_stub()._create_llm(CapturingService, LAGUNA_MODEL)

    assert service.kwargs["model"] == LAGUNA_MODEL
    assert service.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    params = service.kwargs["params"]
    assert params.max_tokens == 8192
    assert params.temperature is NOT_GIVEN
    assert params.extra == {
        "extra_body": {"reasoning": {"enabled": False}},
    }


def test_laguna_default_omits_explicit_reasoning_control(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("MTE_OPENROUTER_REASONING_OFF", raising=False)
    monkeypatch.delenv("MTE_OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MTE_OPENROUTER_TEMPERATURE", raising=False)

    service = openrouter_pipeline_stub()._create_llm(CapturingService, LAGUNA_MODEL)

    assert service.kwargs["model"] == LAGUNA_MODEL
    assert "params" not in service.kwargs


def test_openrouter_filler_modifies_only_outgoing_request_copy(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "3")
    monkeypatch.setenv("MTE_FILLER_TOKEN", ".")
    monkeypatch.setenv("MTE_FILLER_POSITION", "suffix")
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Book the appointment."},
    ]
    service = LoggedOpenRouterLLMService(
        api_key="test-openrouter-key",
        base_url="https://openrouter.ai/api/v1",
        settings=LoggedOpenRouterLLMService.Settings(model=LAGUNA_MODEL),
    )

    request = service.build_chat_completion_params(
        {"messages": messages, "tools": [], "tool_choice": "auto"}
    )

    assert request["messages"] is not messages
    assert request["messages"][-1] is not messages[-1]
    assert request["messages"][-1]["content"] == "Book the appointment. . . ."
    assert messages[-1]["content"] == "Book the appointment."
