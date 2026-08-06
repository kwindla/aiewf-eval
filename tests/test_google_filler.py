"""Focused coverage for request-local filler injection on Gemini requests."""

from google.genai.types import Content, FunctionResponse, Part
from pipecat.processors.aggregators.llm_context import LLMContext

from multi_turn_eval.services.google_logged import (
    FillerGoogleLLMAdapter,
    LoggedGoogleLLMService,
    _apply_filler_google,
)
from multi_turn_eval.pipelines.base import BasePipeline


class FakeGoogleService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubGooglePipeline(BasePipeline):
    def _setup_context(self):
        raise NotImplementedError

    def _setup_llm(self):
        raise NotImplementedError

    def _build_task(self):
        raise NotImplementedError

    async def _queue_first_turn(self):
        raise NotImplementedError

    async def _queue_next_turn(self):
        raise NotImplementedError


def google_pipeline_stub():
    pipeline = object.__new__(StubGooglePipeline)
    pipeline.service_name = "google"
    return pipeline


def test_suffix_dots_copy_google_content_and_part(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "3")
    monkeypatch.setenv("MTE_FILLER_TOKEN", ".")
    monkeypatch.setenv("MTE_FILLER_POSITION", "suffix")

    question_part = Part(text="What is next?  ")
    question = Content(role="user", parts=[question_part])
    tool_result = Content(
        role="user",
        parts=[Part(function_response=FunctionResponse(name="lookup", response={"ok": True}))],
    )
    messages = [question, tool_result]

    filled, system = _apply_filler_google(messages, "System prompt")

    assert system == "System prompt"
    assert filled is not messages
    assert filled[0] is not question
    assert filled[0].parts is not question.parts
    assert filled[0].parts[0] is not question_part
    assert filled[0].parts[0].text == "What is next? . . ."
    assert filled[1] is tool_result

    # Persisted request inputs are filler-free.
    assert messages == [question, tool_result]
    assert question.parts[0].text == "What is next?  "


def test_unset_filler_is_identity_noop(monkeypatch):
    monkeypatch.delenv("MTE_FILLER_DOTS", raising=False)
    messages = [Content(role="user", parts=[Part(text="Hello")])]

    filled, system = _apply_filler_google(messages, "System prompt")

    assert filled is messages
    assert system == "System prompt"


def test_active_filler_without_text_target_is_identity_noop(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "3")
    messages = [
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="lookup", response={"ok": True}
                    )
                )
            ],
        )
    ]

    filled, system = _apply_filler_google(messages, None)

    assert filled is messages
    assert system is None


def test_adapter_fills_outgoing_copy_not_context(monkeypatch):
    monkeypatch.setenv("MTE_FILLER_DOTS", "2")
    monkeypatch.setenv("MTE_FILLER_TOKEN", "-")
    monkeypatch.setenv("MTE_FILLER_POSITION", "prefix")
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    params = FillerGoogleLLMAdapter().get_llm_invocation_params(context)

    assert params["messages"][-1].parts[-1].text == "- - Hello"
    assert context.messages == [{"role": "user", "content": "Hello"}]
    assert LoggedGoogleLLMService.adapter_class is FillerGoogleLLMAdapter


def test_gemini25_disabled_pins_zero_thinking_budget(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("MTE_GOOGLE_THINKING_MODE", "disabled")

    service = google_pipeline_stub()._create_llm(FakeGoogleService, "gemini-2.5-flash")

    assert service.kwargs["params"].thinking.thinking_budget == 0


def test_gemini25_pipecat_default_omits_explicit_thinking_config(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("MTE_GOOGLE_THINKING_MODE", "default")

    service = google_pipeline_stub()._create_llm(FakeGoogleService, "gemini-2.5-flash")

    assert "params" not in service.kwargs


def test_gemini25_rejects_gemini3_thinking_levels(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("MTE_GOOGLE_THINKING_MODE", "minimal")

    try:
        google_pipeline_stub()._create_llm(FakeGoogleService, "gemini-2.5-flash")
    except ValueError as exc:
        assert "expected disabled or default" in str(exc)
    else:
        raise AssertionError("Gemini 2.5 must reject Gemini 3 thinking levels")
