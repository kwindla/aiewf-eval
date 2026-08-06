"""Offline coverage for vLLM Qwen reasoning/tool-history compatibility."""

import asyncio
from types import SimpleNamespace

import pytest

from openai import NOT_GIVEN
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.openai.llm import OpenAILLMService

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.pipelines.text import TextPipeline
from multi_turn_eval.services.vllm_openai import (
    QwenReasoningAssistantAggregator,
    QwenReasoningContextAggregatorPair,
    VLLMOpenAILLMService,
)


class ConcretePipeline(BasePipeline):
    """Minimal concrete shell used to exercise the service factory."""

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


def vllm_pipeline_stub():
    pipeline = object.__new__(ConcretePipeline)
    pipeline.service_name = "vllm-openai"
    return pipeline


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.6-27B"])
def test_qwen_request_sends_enable_and_preserve_thinking_together(
    monkeypatch, model
):
    monkeypatch.setenv("MTE_VLLM_THINKING", "1")
    monkeypatch.delenv("MTE_VLLM_THINKING_BUDGET", raising=False)

    service = vllm_pipeline_stub()._create_llm(CapturingService, model)

    chat_template_kwargs = service.kwargs["params"].extra["extra_body"][
        "chat_template_kwargs"
    ]
    assert chat_template_kwargs == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }


def test_non_qwen_vllm_request_shape_is_unchanged(monkeypatch):
    monkeypatch.setenv("MTE_VLLM_THINKING", "1")
    monkeypatch.delenv("MTE_VLLM_THINKING_BUDGET", raising=False)

    service = vllm_pipeline_stub()._create_llm(
        CapturingService, "meta-llama/Llama-3.3-70B-Instruct"
    )

    chat_template_kwargs = service.kwargs["params"].extra["extra_body"][
        "chat_template_kwargs"
    ]
    assert chat_template_kwargs == {"enable_thinking": True}


class FakeStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def _chunk(*, reasoning=None, reasoning_content=None, content=None, tool_calls=None):
    delta = SimpleNamespace(
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_qwen_stream_captures_reasoning_and_arms_ttfat_on_tool_call(monkeypatch):
    tool_delta = [SimpleNamespace(index=0)]
    stream = FakeStream(
        [
            _chunk(reasoning_content="Need "),
            _chunk(reasoning_content="the calendar."),
            _chunk(tool_calls=tool_delta),
        ]
    )

    async def fake_parent_get_chat_completions(self, params_from_context):
        return stream

    monkeypatch.setattr(
        OpenAILLMService,
        "get_chat_completions",
        fake_parent_get_chat_completions,
    )
    service = VLLMOpenAILLMService(
        model="Qwen/Qwen3.5-27B",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    frames = []

    async def capture_frame(frame, *args, **kwargs):
        frames.append(frame)

    service.push_frame = capture_frame

    async def consume():
        armed_after_chunk = []
        wrapped = await service.get_chat_completions({})
        async for _ in wrapped:
            armed_after_chunk.append(service._ttft_armed)
        return armed_after_chunk

    armed_after_chunk = asyncio.run(consume())

    assert armed_after_chunk == [False, False, True]
    assert [type(frame) for frame in frames] == [
        LLMThoughtStartFrame,
        LLMThoughtTextFrame,
        LLMThoughtTextFrame,
        LLMThoughtEndFrame,
    ]
    assert [frame.text for frame in frames if isinstance(frame, LLMThoughtTextFrame)] == [
        "Need ",
        "the calendar.",
    ]
    assert stream.closed


def test_qwen_stream_accepts_baseten_reasoning_alias(monkeypatch):
    stream = FakeStream(
        [
            _chunk(reasoning="Private thought."),
            _chunk(content="Visible answer."),
        ]
    )

    async def fake_parent_get_chat_completions(self, params_from_context):
        return stream

    monkeypatch.setattr(
        OpenAILLMService,
        "get_chat_completions",
        fake_parent_get_chat_completions,
    )
    service = VLLMOpenAILLMService(
        model="Qwen/Qwen3.6-27B",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    frames = []

    async def capture_frame(frame, *args, **kwargs):
        frames.append(frame)

    service.push_frame = capture_frame

    async def consume():
        wrapped = await service.get_chat_completions({})
        async for _ in wrapped:
            pass

    asyncio.run(consume())

    thought_text = [
        frame.text for frame in frames if isinstance(frame, LLMThoughtTextFrame)
    ]
    assert thought_text == ["Private thought."]
    assert [type(frame) for frame in frames] == [
        LLMThoughtStartFrame,
        LLMThoughtTextFrame,
        LLMThoughtEndFrame,
    ]


def test_reasoning_is_attached_to_assistant_tool_call_history():
    context = LLMContext([{"role": "user", "content": "Check my calendar."}])
    aggregator = QwenReasoningAssistantAggregator(context)

    async def aggregate_tool_call():
        await aggregator._handle_thought_start(LLMThoughtStartFrame())
        await aggregator._handle_thought_text(LLMThoughtTextFrame("Need calendar access."))
        await aggregator._handle_thought_end(LLMThoughtEndFrame())
        await aggregator._handle_function_call_in_progress(
            FunctionCallInProgressFrame(
                function_name="get_calendar",
                tool_call_id="call_123",
                arguments={"date": "tomorrow"},
            )
        )

    asyncio.run(aggregate_tool_call())

    assistant_messages = [
        message
        for message in context.get_messages()
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    assert assistant_messages == [
        {
            "role": "assistant",
            "reasoning_content": "Need calendar access.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "get_calendar",
                        "arguments": '{"date": "tomorrow"}',
                    },
                    "type": "function",
                }
            ],
        }
    ]

    service = VLLMOpenAILLMService(
        model="Qwen/Qwen3.5-27B",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    invocation = service.get_llm_adapter().get_llm_invocation_params(
        context,
        convert_developer_to_user=False,
    )
    outgoing_messages = service.build_chat_completion_params(invocation)["messages"]
    outgoing_assistant = [
        message for message in outgoing_messages if message.get("role") == "assistant"
    ]
    assert outgoing_assistant[0]["reasoning_content"] == "Need calendar access."


def test_same_reasoning_is_attached_to_each_tool_call_in_one_response():
    context = LLMContext([{"role": "user", "content": "Check both systems."}])
    aggregator = QwenReasoningAssistantAggregator(context)

    async def aggregate_tool_calls():
        await aggregator._handle_thought_start(LLMThoughtStartFrame())
        await aggregator._handle_thought_text(
            LLMThoughtTextFrame("Need both lookups.")
        )
        await aggregator._handle_thought_end(LLMThoughtEndFrame())
        for tool_call_id, function_name in (
            ("call_1", "lookup_calendar"),
            ("call_2", "lookup_contacts"),
        ):
            await aggregator._handle_function_call_in_progress(
                FunctionCallInProgressFrame(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments={},
                )
            )

    asyncio.run(aggregate_tool_calls())

    assistant_messages = [
        message
        for message in context.get_messages()
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 2
    assert all(
        message.get("reasoning_content") == "Need both lookups."
        for message in assistant_messages
    )


class BenchmarkStub:
    system_instruction = "Use tools."
    turns = [{"input": "Hello"}]
    tools_schema = NOT_GIVEN


def test_text_pipeline_uses_qwen_service_aggregator_and_non_qwen_falls_back():
    qwen_pipeline = TextPipeline(BenchmarkStub())
    qwen_pipeline.llm = VLLMOpenAILLMService(
        model="Qwen/Qwen3.5-27B",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    recorded_thoughts = []
    qwen_pipeline.recorder = SimpleNamespace(
        record_assistant_thought=recorded_thoughts.append
    )
    qwen_pipeline._setup_context()

    assert isinstance(
        qwen_pipeline.context_aggregator, QwenReasoningContextAggregatorPair
    )
    qwen_assistant = qwen_pipeline.context_aggregator.assistant()

    async def emit_thought():
        await qwen_assistant._handle_thought_start(LLMThoughtStartFrame())
        await qwen_assistant._handle_thought_text(LLMThoughtTextFrame("Captured thought."))
        await qwen_assistant._handle_thought_end(LLMThoughtEndFrame())

    asyncio.run(emit_thought())
    assert recorded_thoughts == ["Captured thought."]

    non_qwen_pipeline = TextPipeline(BenchmarkStub())
    non_qwen_pipeline.llm = VLLMOpenAILLMService(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    non_qwen_pipeline._setup_context()

    assert type(non_qwen_pipeline.context_aggregator) is LLMContextAggregatorPair


def test_text_pipeline_does_not_call_pipecat_legacy_aggregator_hook():
    class NonQwenService:
        def create_context_aggregator(self, context):
            raise AssertionError("legacy Pipecat hook must not be called")

    pipeline = TextPipeline(BenchmarkStub())
    pipeline.llm = NonQwenService()
    pipeline._setup_context()

    assert type(pipeline.context_aggregator) is LLMContextAggregatorPair
