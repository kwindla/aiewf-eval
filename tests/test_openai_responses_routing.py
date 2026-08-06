"""Targeted coverage for OpenAI Responses routing in ``BasePipeline``."""

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
