"""Regression coverage for bot turn tags in the null audio transport."""

import asyncio
import re

import numpy as np
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from multi_turn_eval.transports.null_audio_output import (
    AUDIO_TAG_DURATION_MS,
    NullAudioOutputTransport,
)


def _transport() -> NullAudioOutputTransport:
    return NullAudioOutputTransport(TransportParams(audio_out_enabled=True))


def test_short_bot_frames_receive_the_entire_turn_tag(monkeypatch):
    """A short first frame must carry the tag into following bot frames."""
    emitted = []

    async def capture(_self, frame, _direction):
        emitted.append(frame)

    monkeypatch.setattr(BaseOutputTransport, "process_frame", capture)
    monkeypatch.setattr("multi_turn_eval.transports.null_audio_output.time.time", lambda: 1.0)
    transport = _transport()
    transport._recording_start_time = 1.0
    transport._recording_sample_rate = 24_000
    transport._tag_next_bot_audio = True
    first = OutputAudioRawFrame(
        audio=bytes(120 * 2), sample_rate=24_000, num_channels=1
    )
    second = OutputAudioRawFrame(
        audio=bytes(240 * 2), sample_rate=24_000, num_channels=1
    )

    asyncio.run(
        transport._emit_bot_with_silence_fill(first, FrameDirection.DOWNSTREAM)
    )
    asyncio.run(
        transport._emit_bot_with_silence_fill(second, FrameDirection.DOWNSTREAM)
    )

    written = np.frombuffer(b"".join(frame.audio for frame in emitted), dtype=np.int16)
    expected = transport._generate_audio_tag(24_000)
    assert len(expected) == 24_000 * AUDIO_TAG_DURATION_MS // 1000
    np.testing.assert_array_equal(written[: len(expected)], expected)
    assert transport._pending_bot_tag is None


def test_tag_position_is_logged_when_tagged_samples_are_emitted(monkeypatch):
    """The logged position follows actual downstream output, not tag mixing."""
    messages = []

    async def discard(_self, _frame, _direction=FrameDirection.DOWNSTREAM):
        pass

    monkeypatch.setattr(BaseOutputTransport, "push_frame", discard)
    monkeypatch.setattr(
        "multi_turn_eval.transports.null_audio_output.logger.info", messages.append
    )
    transport = _transport()
    transport._recording_start_time = 1.0
    transport._recording_sample_rate = 24_000
    transport._bot_actual_samples = 500
    transport._start_tag_detection()
    tag = transport._generate_audio_tag(24_000)
    audio = np.concatenate(
        [np.zeros(150, dtype=np.int16), tag[:50]]
    )
    frame = OutputAudioRawFrame(
        audio=audio.tobytes(), sample_rate=24_000, num_channels=1
    )

    asyncio.run(transport.push_frame(frame, FrameDirection.DOWNSTREAM))
    asyncio.run(
        transport.push_frame(
            OutputAudioRawFrame(
                audio=tag[50:].tobytes(), sample_rate=24_000, num_channels=1
            ),
            FrameDirection.DOWNSTREAM,
        )
    )

    assert transport._bot_actual_samples == 500 + 150 + len(tag)
    assert messages == [
        "[NullAudioOutput] Bot turn tag: sample_pos=27ms, "
        "freq=2000Hz, duration=15ms"
    ]


def test_logged_tag_position_matches_where_the_tag_lands(monkeypatch):
    """The logged position must survive MediaSender buffering.

    Frames are queued at schedule time and only pushed downstream later. That
    delay is what made the old code report a position taken from a counter that
    had not yet advanced past the queued audio.
    """
    queued, emitted, messages = [], [], []

    async def queue_it(_self, frame, _direction):
        queued.append(frame)

    async def emit_it(_self, frame, _direction=FrameDirection.DOWNSTREAM):
        emitted.append(frame)

    monkeypatch.setattr(BaseOutputTransport, "process_frame", queue_it)
    monkeypatch.setattr(BaseOutputTransport, "push_frame", emit_it)
    monkeypatch.setattr("multi_turn_eval.transports.null_audio_output.time.time", lambda: 1.0)
    monkeypatch.setattr(
        "multi_turn_eval.transports.null_audio_output.logger.info", messages.append
    )

    transport = _transport()
    transport._recording_start_time = 1.0
    transport._recording_sample_rate = 24_000

    def frame(num_samples):
        return OutputAudioRawFrame(
            audio=bytes(num_samples * 2), sample_rate=24_000, num_channels=1
        )

    # Queue audio ahead of the tagged turn so scheduled runs ahead of emitted.
    asyncio.run(transport._emit_bot_with_silence_fill(frame(480), FrameDirection.DOWNSTREAM))
    asyncio.run(transport._emit_bot_with_silence_fill(frame(480), FrameDirection.DOWNSTREAM))
    transport._tag_next_bot_audio = True
    asyncio.run(transport._emit_bot_with_silence_fill(frame(120), FrameDirection.DOWNSTREAM))
    asyncio.run(transport._emit_bot_with_silence_fill(frame(480), FrameDirection.DOWNSTREAM))

    for queued_frame in queued:
        asyncio.run(transport.push_frame(queued_frame, FrameDirection.DOWNSTREAM))

    stream = np.frombuffer(b"".join(f.audio for f in emitted), dtype=np.int16)
    tag = transport._generate_audio_tag(24_000)
    lead = int(np.argmax(tag != 0))  # the tag opens with fade-in zeros
    actual_start = int(np.nonzero(stream)[0][0]) - lead
    np.testing.assert_array_equal(stream[actual_start : actual_start + len(tag)], tag)

    tag_logs = [m for m in messages if "Bot turn tag" in m]
    assert len(tag_logs) == 1
    logged_ms = int(re.search(r"sample_pos=(\d+)ms", tag_logs[0]).group(1))
    logged_sample = round(logged_ms * 24_000 / 1000)
    assert abs(logged_sample - actual_start) / 24_000 * 1000 < 2.0


def test_logged_tag_position_does_not_drift_from_resampled_stream(monkeypatch):
    """Measured output, not source ratio math, must determine every tag position.

    Each 1,001-sample source frame becomes a 1,000-sample emitted frame. Over
    500 seconds, source-ratio positions are more than 500ms ahead of the WAV.
    """
    queued, messages = [], []

    async def queue_source(_self, frame, _direction):
        queued.append(frame)

    async def discard(_self, _frame, _direction=FrameDirection.DOWNSTREAM):
        pass

    monkeypatch.setattr(BaseOutputTransport, "process_frame", queue_source)
    monkeypatch.setattr(BaseOutputTransport, "push_frame", discard)
    monkeypatch.setattr(
        "multi_turn_eval.transports.null_audio_output.time.time", lambda: 1.0
    )
    monkeypatch.setattr(
        "multi_turn_eval.transports.null_audio_output.logger.info", messages.append
    )

    transport = _transport()
    transport._recording_start_time = 1.0
    transport._recording_sample_rate = 24_000
    source_samples = 0
    true_tag_starts = []

    async def emit_resampled_source(source_frame):
        # Simulate SOXR emitting 0.1% fewer samples while preserving the tag
        # at the start of its tagged source frame.
        emitted = OutputAudioRawFrame(
            audio=source_frame.audio[: 1_000 * 2],
            sample_rate=24_000,
            num_channels=1,
        )
        await transport.push_frame(emitted, FrameDirection.DOWNSTREAM)

    async def scenario():
        nonlocal source_samples
        source_frame = OutputAudioRawFrame(
            audio=bytes(1_001 * 2), sample_rate=24_000, num_channels=1
        )
        for _ in range(100):
            for _ in range(120):
                await transport._emit_bot_with_silence_fill(
                    source_frame, FrameDirection.DOWNSTREAM
                )
                source_samples += 1_001
                await emit_resampled_source(queued.pop())

            true_tag_starts.append(transport._bot_actual_samples)
            transport._tag_next_bot_audio = True
            await transport._emit_bot_with_silence_fill(
                source_frame, FrameDirection.DOWNSTREAM
            )
            source_samples += 1_001
            await emit_resampled_source(queued.pop())

    asyncio.run(scenario())

    tag_logs = [message for message in messages if "Bot turn tag" in message]
    assert len(tag_logs) == len(true_tag_starts) == 100
    logged_samples = [
        round(int(re.search(r"sample_pos=(\d+)ms", message).group(1)) * 24)
        for message in tag_logs
    ]
    end_error_ms = abs(logged_samples[-1] - true_tag_starts[-1]) / 24
    assert end_error_ms < 20
    assert (source_samples - transport._bot_actual_samples) / 24 > 500


def test_pending_tag_is_warned_at_recording_end(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "multi_turn_eval.transports.null_audio_output.logger.warning", warnings.append
    )
    transport = _transport()
    transport._recording_start_time = 1.0
    transport._recording_sample_rate = 24_000
    transport._start_tag_detection()

    transport.log_recording_summary()

    assert warnings == [
        "[NullAudioOutput] Bot turn tag remains pending at recording end; "
        "its audio was not emitted downstream"
    ]
