"""Focused coverage for the current Gemini Live model and turn gating."""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types as genai_types
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from multi_turn_eval.pipelines.realtime import (
    GeminiLiveLLMServiceWithReconnection,
    RealtimePipeline,
    TurnGate,
)
from multi_turn_eval.processors.tts_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)


MODEL = "gemini-3.1-flash-live-preview"


def _pipeline() -> RealtimePipeline:
    benchmark = SimpleNamespace(
        turns=[],
        system_instruction="Use the supplied tools.",
        tools_schema=[],
    )
    pipeline = RealtimePipeline(benchmark)
    pipeline.model_name = MODEL
    return pipeline


def test_gemini31_live_uses_canonical_settings_and_minimal_thinking(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("MTE_GOOGLE_THINKING_MODE", raising=False)

    service = _pipeline()._create_llm(GeminiLiveLLMServiceWithReconnection, MODEL)

    assert service._settings.model == MODEL
    assert service._settings.system_instruction == "Use the supplied tools."
    assert service._settings.thinking.thinking_level == genai_types.ThinkingLevel.MINIMAL
    assert service._settings.thinking.include_thoughts is False


@pytest.mark.parametrize(
    ("mode", "level"),
    [
        ("low", genai_types.ThinkingLevel.LOW),
        ("medium", genai_types.ThinkingLevel.MEDIUM),
        ("high", genai_types.ThinkingLevel.HIGH),
    ],
)
def test_gemini31_live_accepts_supported_thinking_levels(monkeypatch, mode, level):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    pipeline = _pipeline()
    pipeline.thinking = mode

    service = pipeline._create_llm(GeminiLiveLLMServiceWithReconnection, MODEL)

    assert service._settings.thinking.thinking_level == level


def test_gemini31_live_rejects_thinking_disabled(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    pipeline = _pipeline()
    pipeline.thinking = "disabled"

    with pytest.raises(ValueError, match="does not support thinking_budget=0"):
        pipeline._create_llm(GeminiLiveLLMServiceWithReconnection, MODEL)


def test_turn_gate_completes_when_transcript_arrives_after_bot_stop():
    completed = []

    async def scenario():
        async def on_turn_ready(text):
            completed.append(text)

        gate = TurnGate(on_turn_ready=on_turn_ready, audio_drain_delay=0)
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        gate.set_pending_transcript("late transcript")
        await gate._turn_end_task

        assert gate._pending_transcript is None

    asyncio.run(scenario())
    assert completed == ["late transcript"]


def test_turn_gate_restores_transcript_for_follow_up_speech_segment():
    completed = []

    async def scenario():
        async def on_turn_ready(text):
            completed.append(text)

        gate = TurnGate(on_turn_ready=on_turn_ready, audio_drain_delay=60)
        gate.set_pending_transcript("two-part response")
        await asyncio.sleep(0)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._pending_transcript == "two-part response"

        gate._audio_drain_delay = 0
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await gate._turn_end_task

    asyncio.run(scenario())
    assert completed == ["two-part response"]


def test_turn_gate_merges_consecutive_transcript_segments():
    completed = []

    async def scenario():
        async def on_turn_ready(text):
            completed.append(text)

        gate = TurnGate(on_turn_ready=on_turn_ready, audio_drain_delay=60)
        gate.set_pending_transcript("Workshop day")
        await asyncio.sleep(0)
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        gate.set_pending_transcript("is on Tuesday.")
        assert gate._pending_transcript == "Workshop day is on Tuesday."

        gate._audio_drain_delay = 0
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await gate._turn_end_task

    asyncio.run(scenario())
    assert completed == ["Workshop day is on Tuesday."]


def test_assistant_transcript_flushes_when_playback_stops_without_turn_complete():
    updates = []

    async def scenario():
        processor = TTSStoppedAssistantTranscriptProcessor()

        @processor.event_handler("on_transcript_update")
        async def on_update(_processor, frame):
            updates.extend(message.content for message in frame.messages)

        await processor.process_frame(
            TTSTextFrame("fallback transcript", aggregated_by="test"),
            FrameDirection.DOWNSTREAM,
        )
        await processor.process_frame(
            BotStoppedSpeakingFrame(),
            FrameDirection.DOWNSTREAM,
        )

    asyncio.run(scenario())
    assert updates == ["fallback transcript"]
