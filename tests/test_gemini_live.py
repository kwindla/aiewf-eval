"""Focused coverage for the current Gemini Live model and turn gating."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from google.genai import types as genai_types
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

from multi_turn_eval.pipelines.realtime import (
    GeminiLiveLLMServiceWithReconnection,
    RealtimePipeline,
    TurnGate,
)
from multi_turn_eval.processors.tts_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder


MODEL = "gemini-3.1-flash-live-preview"
OPAQUE_LIVE_MODEL = "opaque-live-model"


def _pipeline(model: str = MODEL, **behavior) -> RealtimePipeline:
    benchmark = SimpleNamespace(
        turns=[],
        system_instruction="Use the supplied tools.",
        tools_schema=[],
    )
    pipeline = RealtimePipeline(benchmark, **behavior)
    pipeline.model_name = model
    return pipeline


def _async_live_pipeline() -> RealtimePipeline:
    pipeline = _pipeline(
        OPAQUE_LIVE_MODEL,
        gemini_3_protocol=True,
        require_interaction_status=True,
        explicit_audio_activity=True,
        allow_turn_replay=False,
    )
    pipeline.service_name = "gemini-live"
    return pipeline


def _enable_interaction_status_sdk(monkeypatch):
    """Make public-SDK unit tests advertise the private EAP field."""
    monkeypatch.setitem(
        genai_types.LiveServerContent.model_fields,
        "interaction_status",
        SimpleNamespace(),
    )


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


def test_explicit_behavior_configures_opaque_async_gemini_live_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("MTE_GOOGLE_THINKING_MODE", raising=False)
    _enable_interaction_status_sdk(monkeypatch)
    pipeline = _async_live_pipeline()

    service = pipeline._create_llm(
        GeminiLiveLLMServiceWithReconnection,
        OPAQUE_LIVE_MODEL,
    )

    assert pipeline._is_gemini_live() is True
    assert service._is_gemini_3 is True
    assert service._supports_non_blocking_tools is False
    assert service._settings.thinking.thinking_level == genai_types.ThinkingLevel.MINIMAL
    assert service._settings.vad.disabled is True
    assert pipeline._uses_explicit_audio_activity() is True


def test_required_interaction_status_rejects_incompatible_sdk(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delitem(
        genai_types.LiveServerContent.model_fields,
        "interaction_status",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="requires a google-genai SDK build"):
        _async_live_pipeline()._create_llm(
            GeminiLiveLLMServiceWithReconnection,
            OPAQUE_LIVE_MODEL,
        )


def test_async_turn_completion_waits_for_requires_action(monkeypatch):
    parent_messages = []

    async def record_parent_completion(_service, message):
        parent_messages.append(message)

    async def scenario():
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        _enable_interaction_status_sdk(monkeypatch)
        monkeypatch.setattr(
            GeminiLiveLLMService,
            "_handle_msg_turn_complete",
            record_parent_completion,
        )
        service = _async_live_pipeline()._create_llm(
            GeminiLiveLLMServiceWithReconnection,
            OPAQUE_LIVE_MODEL,
        )
        service._bot_text_buffer = "keep all accumulated response text"
        service._llm_output_buffer = "keep all accumulated output"

        first_phase = SimpleNamespace(
            server_content=SimpleNamespace(
                interaction_status=SimpleNamespace(value="IN_PROGRESS")
            )
        )
        second_phase = SimpleNamespace(
            server_content=SimpleNamespace(interaction_status="IN_PROGRESS")
        )
        terminal = SimpleNamespace(
            server_content=SimpleNamespace(interaction_status="REQUIRES_ACTION")
        )

        await service._handle_msg_turn_complete(first_phase)
        await service._handle_msg_turn_complete(second_phase)

        assert service.is_interaction_in_progress() is True
        assert service._bot_text_buffer == "keep all accumulated response text"
        assert service._llm_output_buffer == "keep all accumulated output"
        assert parent_messages == []

        await service._handle_msg_turn_complete(terminal)

        assert service.is_interaction_in_progress() is False
        assert parent_messages == [terminal]

    asyncio.run(scenario())


def test_async_live_ignores_terminal_while_explicit_input_is_open(monkeypatch):
    parent_messages = []

    async def record_parent_completion(_service, message):
        parent_messages.append(message)

    async def scenario():
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        _enable_interaction_status_sdk(monkeypatch)
        monkeypatch.setattr(
            GeminiLiveLLMService,
            "_handle_msg_turn_complete",
            record_parent_completion,
        )
        service = _async_live_pipeline()._create_llm(
            GeminiLiveLLMServiceWithReconnection,
            OPAQUE_LIVE_MODEL,
        )
        terminal = SimpleNamespace(
            server_content=SimpleNamespace(interaction_status="REQUIRES_ACTION")
        )

        service._explicit_input_activity_open = True
        await service._handle_msg_turn_complete(terminal)
        assert parent_messages == []

        service._explicit_input_activity_open = False
        await service._handle_msg_turn_complete(terminal)
        assert parent_messages == [terminal]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status", [None, "INTERACTION_STATUS_UNSPECIFIED", "unexpected"]
)
def test_async_live_rejects_non_definitive_interaction_status(monkeypatch, status):
    async def scenario():
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        _enable_interaction_status_sdk(monkeypatch)
        service = _async_live_pipeline()._create_llm(
            GeminiLiveLLMServiceWithReconnection,
            OPAQUE_LIVE_MODEL,
        )
        message = SimpleNamespace(
            server_content=SimpleNamespace(interaction_status=status)
        )

        with pytest.raises(RuntimeError, match="without a usable interaction_status"):
            await service._handle_msg_turn_complete(message)

    asyncio.run(scenario())


def test_private_sdk_interaction_status_enum_is_supported():
    interaction_status_type = getattr(genai_types, "InteractionStatus", None)
    if interaction_status_type is None:
        pytest.skip("Public google-genai SDK does not expose interaction_status")

    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            turn_complete=True,
            interaction_status=interaction_status_type.IN_PROGRESS,
        )
    )

    assert (
        GeminiLiveLLMServiceWithReconnection._get_interaction_status(message)
        == "IN_PROGRESS"
    )


def test_legacy_gemini_turn_complete_behavior_is_unchanged(monkeypatch):
    parent_messages = []

    async def record_parent_completion(_service, message):
        parent_messages.append(message)

    async def scenario():
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setattr(
            GeminiLiveLLMService,
            "_handle_msg_turn_complete",
            record_parent_completion,
        )
        service = _pipeline(MODEL)._create_llm(
            GeminiLiveLLMServiceWithReconnection,
            MODEL,
        )
        message = SimpleNamespace(server_content=SimpleNamespace())

        await service._handle_msg_turn_complete(message)

        assert service.is_interaction_in_progress() is False
        assert parent_messages == [message]

    asyncio.run(scenario())


@pytest.mark.parametrize("model", [OPAQUE_LIVE_MODEL, MODEL])
def test_no_audio_timeout_terminates_without_replay(tmp_path, model):
    async def scenario():
        pipeline = (
            _async_live_pipeline()
            if model == OPAQUE_LIVE_MODEL
            else _pipeline(model)
        )
        pipeline.model_name = model
        pipeline.turn_idx = 12
        pipeline.turn_gate = SimpleNamespace(
            _no_response_timeout=15.0,
            clear_pending=lambda: None,
        )
        pipeline.assistant_shim = SimpleNamespace(clear_buffer=lambda: None)
        pipeline.recorder = TranscriptRecorder(tmp_path, model)
        cancelled = asyncio.Event()

        class RecordingTask:
            async def cancel(self):
                cancelled.set()

        pipeline.task = RecordingTask()
        pipeline._on_empty_response("no_response")
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        pipeline.recorder.close()

        runtime = json.loads((tmp_path / "runtime.json").read_text())
        assert pipeline.done is True
        assert pipeline.needs_turn_retry is False
        assert pipeline._turn_retry_count == 0
        assert runtime["status"] == "failed"
        assert runtime["valid"] is False
        assert runtime["failure"] == {
            "reason": "no_audio_timeout",
            "turn": 12,
            "replayed": False,
            "timeout_seconds": 15.0,
        }

    asyncio.run(scenario())


def test_opaque_gemini_live_alias_queues_gemini_context_frame():
    class RecordingTask:
        def __init__(self):
            self.frames = []

        async def queue_frames(self, frames):
            self.frames.extend(frames)

    pipeline = _async_live_pipeline()
    pipeline.context = SimpleNamespace()
    pipeline.task = RecordingTask()

    asyncio.run(pipeline._queue_first_turn())

    assert len(pipeline.task.frames) == 1
    assert isinstance(pipeline.task.frames[0], LLMContextFrame)


def test_gemini_live_service_classifies_an_opaque_alias_as_live():
    pipeline = _pipeline("future-confidential-alias")
    pipeline.service_name = "gemini-live"

    assert pipeline._is_gemini_live() is True


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


def test_async_turn_gate_waits_for_provider_terminal_after_audio_drains():
    completed = []

    async def scenario():
        async def on_turn_ready(text):
            completed.append(text)

        gate = TurnGate(
            on_turn_ready=on_turn_ready,
            audio_drain_delay=0,
            require_terminal_signal=True,
        )
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        gate.set_pending_transcript("first async audio phase")
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert completed == []
        assert gate._pending_transcript == "first async audio phase"

        await gate.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        await gate._turn_end_task

    asyncio.run(scenario())
    assert completed == ["first async audio phase"]


def test_async_turn_gate_revokes_terminal_for_buffered_tts_continuation():
    completed = []

    async def scenario():
        async def on_turn_ready(text):
            completed.append(text)

        gate = TurnGate(
            on_turn_ready=on_turn_ready,
            audio_drain_delay=60,
            require_terminal_signal=True,
        )
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        gate.set_pending_transcript("first tool-response phase")
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await gate.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert gate._turn_end_task is not None
        assert gate._terminal_signal_received is True

        # The provider had already produced a second audio phase, but it reached
        # the client pipeline just after REQUIRES_ACTION and before the drain
        # window elapsed. It must remain part of the current benchmark turn.
        await gate.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert completed == []
        assert gate._terminal_signal_received is False
        assert gate._pending_transcript == "first tool-response phase"

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        gate.set_pending_transcript("second buffered phase")
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)
        assert completed == []

        gate._audio_drain_delay = 0
        await gate.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        await gate._turn_end_task

    asyncio.run(scenario())
    assert completed == ["first tool-response phase second buffered phase"]


def test_async_transcript_does_not_flush_on_intermediate_playback_stop():
    updates = []

    async def scenario():
        processor = TTSStoppedAssistantTranscriptProcessor(
            flush_on_bot_stopped=False
        )

        @processor.event_handler("on_transcript_update")
        async def on_update(_processor, frame):
            updates.extend(message.content for message in frame.messages)

        await processor.process_frame(
            TTSTextFrame("first phase ", aggregated_by="test"),
            FrameDirection.DOWNSTREAM,
        )
        await processor.process_frame(
            BotStoppedSpeakingFrame(),
            FrameDirection.DOWNSTREAM,
        )
        assert updates == []

        await processor.process_frame(
            TTSTextFrame("second phase", aggregated_by="test"),
            FrameDirection.DOWNSTREAM,
        )
        await processor.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    asyncio.run(scenario())
    assert updates == ["first phase second phase"]
