"""A provider tool call must not deadlock behind slower local VAD."""

import asyncio
from functools import wraps
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from multi_turn_eval.processors.realtime_context import RealtimeContextAggregatorPair


def run_async(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@pytest.mark.parametrize(
    "user_speaking,bot_speaking",
    [(True, False), (False, True), (True, True), (False, False)],
)
@run_async
async def test_tool_result_is_delivered_despite_local_speech_state(
    user_speaking, bot_speaking
):
    context = LLMContext([])
    assistant = RealtimeContextAggregatorPair(context).assistant()
    assistant.push_context_frame = AsyncMock()
    assistant.push_frame = AsyncMock()
    assistant._user_speaking = user_speaking
    assistant._bot_speaking = bot_speaking
    await assistant._handle_function_call_in_progress(
        FunctionCallInProgressFrame("vote", "call-1", {"id": "123"})
    )
    await assistant._handle_function_call_result(
        FunctionCallResultFrame("vote", "call-1", {"id": "123"}, {"status": "success"})
    )
    assistant.push_context_frame.assert_awaited_once_with(FrameDirection.UPSTREAM)
    assert assistant._user_speaking == user_speaking
    assert assistant._bot_speaking == bot_speaking
    assert any(
        m.get("role") == "tool" and "success" in m["content"]
        for m in context.get_messages()
    )


@run_async
async def test_explicit_no_inference_is_preserved():
    assistant = RealtimeContextAggregatorPair(LLMContext([])).assistant()
    assistant.push_context_frame = AsyncMock()
    assistant.push_frame = AsyncMock()
    await assistant._handle_function_call_in_progress(
        FunctionCallInProgressFrame("end_session", "call-2", {})
    )
    await assistant._handle_function_call_result(
        FunctionCallResultFrame(
            "end_session", "call-2", {}, {"status": "success"}, run_llm=False
        )
    )
    assistant.push_context_frame.assert_not_awaited()
