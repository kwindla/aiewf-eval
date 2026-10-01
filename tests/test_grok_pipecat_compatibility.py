"""xAI session setup and whole-response audio boundaries on Pipecat 1.10."""

import asyncio
import json
from unittest.mock import AsyncMock
from pipecat.frames.frames import TTSStoppedFrame, LLMFullResponseEndFrame
from multi_turn_eval.pipelines.grok_realtime import XAIRealtimeLLMService


def test_xai_current_setup_and_audio_completion():
    async def scenario():
        service = object.__new__(XAIRealtimeLLMService)

        async def events():
            for e in [
                {"type": "conversation.created"},
                {
                    "type": "response.output_audio.done",
                    "event_id": "e",
                    "response_id": "r",
                    "item_id": "i",
                    "output_index": 0,
                    "content_index": 0,
                },
                {"type": "response.done", "response": {"output": []}},
            ]:
                yield json.dumps(e)

        service._websocket = events()
        service._current_audio_response = object()
        service._send_session_update = AsyncMock()
        service._handle_xai_response_done = AsyncMock()
        service.push_frame = AsyncMock()
        await service._receive_task_handler()
        service._send_session_update.assert_awaited_once()
        service._handle_xai_response_done.assert_awaited_once()
        frames = [c.args[0] for c in service.push_frame.await_args_list]
        assert [type(f) for f in frames] == [TTSStoppedFrame, LLMFullResponseEndFrame]

    asyncio.run(scenario())


def test_grok_pipeline_recognizes_current_voice_model_without_realtime_in_name():
    from multi_turn_eval.pipelines.grok_realtime import GrokRealtimePipeline
    from benchmarks.aiwf_medium_context.config import BenchmarkConfig

    pipeline = GrokRealtimePipeline(BenchmarkConfig())
    pipeline.model_name = "grok-voice-think-fast-1.0"
    assert pipeline._is_grok_realtime()
