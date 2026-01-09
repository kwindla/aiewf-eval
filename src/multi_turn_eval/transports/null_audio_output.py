#
# Null Audio Output Transport for Pipecat
#
# A minimal output transport that inherits BaseOutputTransport's speaking detection
# logic (BotStartedSpeakingFrame/BotStoppedSpeakingFrame) but discards audio output.
#
# This is useful for pipelines that need speaking detection without actual audio playback,
# such as evaluation/test pipelines or speech-to-speech model testing.
#
# IMPORTANT: This transport simulates real-time audio playback timing. Without this,
# BotStoppedSpeakingFrame would be generated based on when audio frames stop arriving
# (LLM generation time) rather than when audio would finish playing (content duration).
# LLMs generate audio faster than real-time, so we must pace frame consumption to match
# actual playback duration.
#
# AUDIO RECORDING FIX: This transport is the "source of truth" for silence insertion
# for BOTH user and bot audio streams. It tracks wall-clock time from recording start
# and inserts silence frames to fill any gaps > 10ms. This ensures the downstream
# AudioBufferProcessor receives continuous streams aligned to wall-clock time for both
# tracks, enabling accurate audio-based TTFB measurements.
#

import asyncio
import time

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

# Maximum gap (in seconds) before silence is inserted
# 10ms = 240 samples at 24kHz
MAX_GAP_SECS = 0.010

# RMS threshold for detecting speech onset in bot audio (dB)
# -30dB is closer to actual speech levels, avoiding pre-speech silent padding
BOT_SPEECH_RMS_THRESHOLD_DB = -30.0


class NullAudioOutputTransport(BaseOutputTransport):
    """Output transport that tracks audio for speaking detection but discards output.

    This transport extends BaseOutputTransport to inherit its MediaSender logic,
    which automatically generates BotStartedSpeakingFrame and BotStoppedSpeakingFrame
    based on audio output timing. However, it doesn't actually play or send the audio
    anywhere - it just discards it.

    CRITICAL: This transport simulates real-time audio playback by sleeping
    proportionally to each audio frame's duration. This is necessary because:
    - LLMs generate audio faster than real-time (e.g., 33s of audio in 10s)
    - BotStoppedSpeakingFrame is triggered when audio queue is empty for BOT_VAD_STOP_SECS
    - Without timing simulation, BotStoppedSpeakingFrame fires too early
    - This would cause turn advancement before audio "finishes playing"

    DUAL-TRACK SILENCE INSERTION: This transport is the "source of truth" for
    wall-clock aligned audio recording. It tracks both user (InputAudioRawFrame)
    and bot (OutputAudioRawFrame) audio streams separately, inserting silence for
    any gaps > 10ms. This ensures the downstream AudioBufferProcessor receives
    continuous streams for both tracks aligned to the same T0 baseline.

    This is useful for:
    - Test/evaluation pipelines where you don't need audio playback
    - Speech-to-speech model testing where you only need transcripts
    - Pipelines that need speaking state tracking without audio output hardware
    - Audio-based TTFB analysis requiring wall-clock aligned recordings

    The key mechanism inherited from BaseOutputTransport:
    - MediaSender tracks TTSAudioRawFrame timing
    - After BOT_VAD_STOP_SECS of no audio, generates BotStoppedSpeakingFrame
    - BotStoppedSpeakingFrame flows upstream to trigger response finalization
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the null audio output transport.

        Args:
            params: Transport configuration parameters. Should have audio_out_enabled=True
                    and appropriate sample rate settings.
            **kwargs: Additional arguments passed to BaseOutputTransport.
        """
        super().__init__(params, **kwargs)
        # Timing state for simulating real-time audio playback
        # Based on WebsocketServerOutputTransport's timing implementation
        self._next_send_time = 0.0
        self._total_audio_duration = 0.0
        self._total_sleep_time = 0.0
        self._frame_count = 0
        self._playback_start_time = 0.0

        # Recording baseline - shared by both user and bot tracks
        self._recording_start_time: float = 0.0
        self._recording_sample_rate: int = 0  # Sample rate for recording

        # Bot audio (OutputAudioRawFrame) tracking
        self._bot_actual_samples: int = 0  # Actual samples pushed downstream
        self._bot_sample_rate: int = 0  # Bot frame sample rate (for logging)
        self._bot_num_channels: int = 1
        self._bot_silence_frames_inserted: int = 0
        self._bot_silence_samples_inserted: int = 0
        self._bot_frame_count: int = 0

        # Bot speech onset detection (RMS-based)
        self._bot_audio_start_time: float = 0.0  # When first bot audio byte arrived for current turn
        self._bot_first_speech_logged: bool = False  # Whether we've logged speech onset for current turn

        # User audio (InputAudioRawFrame) tracking
        self._user_actual_samples: int = 0  # Actual samples pushed downstream
        self._user_sample_rate: int = 0  # User frame sample rate (for logging)
        self._user_num_channels: int = 1
        self._user_silence_frames_inserted: int = 0
        self._user_silence_samples_inserted: int = 0
        self._user_frame_count: int = 0

    def reset_recording_baseline(self, recording_sample_rate: int):
        """Reset the recording baseline for sample-accurate silence insertion.

        Call this when audio recording starts to synchronize the silence insertion
        timing with the AudioBufferProcessor's recording start.

        Args:
            recording_sample_rate: The sample rate used by AudioBufferProcessor for
                recording. This may differ from the audio frame sample rates
                (e.g., Ultravox outputs 48kHz but we record at 24kHz). Silence frames
                must be created at the recording sample rate to maintain correct timing.
        """
        # Use time.time() to match AudioBufferProcessor's clock source
        self._recording_start_time = time.time()
        self._recording_sample_rate = recording_sample_rate

        # Reset bot audio tracking
        self._bot_actual_samples = 0
        self._bot_sample_rate = 0
        self._bot_silence_frames_inserted = 0
        self._bot_silence_samples_inserted = 0
        self._bot_frame_count = 0

        # Reset bot speech onset tracking
        self._bot_audio_start_time = 0.0
        self._bot_first_speech_logged = False

        # Reset user audio tracking
        self._user_actual_samples = 0
        self._user_sample_rate = 0
        self._user_silence_frames_inserted = 0
        self._user_silence_samples_inserted = 0
        self._user_frame_count = 0

        logger.info(f"[NullAudioOutput] Recording baseline reset (recording_sample_rate={recording_sample_rate})")

    async def start(self, frame: StartFrame):
        """Start the transport and initialize the MediaSender.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # Call set_transport_ready to initialize MediaSender which handles
        # BotStartedSpeakingFrame/BotStoppedSpeakingFrame generation
        await self.set_transport_ready(frame)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push frame downstream, tracking actual output samples for both tracks.

        This override counts the actual samples being pushed to AudioBufferProcessor
        for both user (InputAudioRawFrame) and bot (OutputAudioRawFrame) audio,
        rather than predicting them based on input frame sizes. This eliminates drift
        caused by SOXR resampler filter delays and state resets.

        The key insight: MediaSender resamples and chunks audio frames before pushing
        them downstream. The SOXR stream resampler may produce slightly different
        output sample counts than simple ratio math predicts, especially after the
        stream is cleared (at turn boundaries with >0.2s gaps). By counting actual
        output here, we get exact sample counts.

        Args:
            frame: The frame to push.
            direction: The direction of frame flow (default DOWNSTREAM).
        """
        if direction == FrameDirection.DOWNSTREAM and self._recording_start_time > 0:
            # Track bot audio samples
            if isinstance(frame, OutputAudioRawFrame):
                raw_samples = len(frame.audio) // (frame.num_channels * 2)
                if frame.sample_rate == self._recording_sample_rate:
                    actual_samples = raw_samples
                else:
                    # Convert to recording sample rate
                    actual_samples = round(raw_samples * self._recording_sample_rate / frame.sample_rate)
                self._bot_actual_samples += actual_samples

            # Track user audio samples
            elif isinstance(frame, InputAudioRawFrame):
                raw_samples = len(frame.audio) // (frame.num_channels * 2)
                if frame.sample_rate == self._recording_sample_rate:
                    actual_samples = raw_samples
                else:
                    # Convert to recording sample rate
                    actual_samples = round(raw_samples * self._recording_sample_rate / frame.sample_rate)
                self._user_actual_samples += actual_samples

        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, inserting silence for gaps and handling interruptions.

        For both InputAudioRawFrame (user) and OutputAudioRawFrame (bot) flowing
        downstream, this method:
        1. Calculates the expected sample position based on wall-clock time
        2. If we're behind (gap in audio > 10ms), inserts a silence frame first
        3. Then processes the actual frame through the normal path

        This ensures the downstream AudioBufferProcessor receives continuous
        frames aligned to wall-clock time for BOTH tracks.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        # Handle bot audio (OutputAudioRawFrame) - insert silence for gaps
        if (
            isinstance(frame, OutputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._recording_start_time > 0  # Only if recording is active
        ):
            await self._emit_bot_with_silence_fill(frame, direction)
            return

        # Handle user audio (InputAudioRawFrame) - insert silence for gaps
        if (
            isinstance(frame, InputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._recording_start_time > 0  # Only if recording is active
        ):
            await self._emit_user_with_silence_fill(frame, direction)
            return

        await super().process_frame(frame, direction)

        # Reset playback timing on interruption (matches WebsocketServerOutputTransport behavior)
        if isinstance(frame, InterruptionFrame):
            self._next_send_time = 0.0
            # Note: Do NOT reset recording baseline (_recording_start_time, sample counters)
            # Recording is continuous - the AudioBufferProcessor buffer accumulates throughout
            # the session. Resetting counters mid-recording would break wall-clock alignment
            # and cause bot audio to overlap with user audio in the recorded file.
            # Only playback pacing timing needs to reset on interruption.
            logger.debug("[NullAudioOutput] Playback timing reset due to interruption")

    async def _emit_bot_with_silence_fill(self, frame: OutputAudioRawFrame, direction: FrameDirection):
        """Insert silence frame if needed for bot audio, then emit the actual frame.

        This method calculates the expected sample position based on elapsed
        wall-clock time since recording started. If we're behind by more than
        MAX_GAP_SECS (10ms), it creates and pushes a silence frame to fill the
        gap before processing the actual audio frame.

        Args:
            frame: The OutputAudioRawFrame to process.
            direction: The direction of frame flow (should be DOWNSTREAM).
        """
        current_time = time.time()

        # Initialize sample rate and channels from first bot frame
        if self._bot_sample_rate == 0:
            self._bot_sample_rate = frame.sample_rate
            self._bot_num_channels = frame.num_channels
            # Initialize speech onset tracking for first turn
            self._bot_audio_start_time = current_time
            self._bot_first_speech_logged = False
            logger.info(
                f"[NullAudioOutput] Bot audio initialized: sample_rate={self._bot_sample_rate}, "
                f"recording_sample_rate={self._recording_sample_rate}, "
                f"num_channels={self._bot_num_channels}"
            )

        # Calculate expected sample position based on wall-clock time
        elapsed_secs = current_time - self._recording_start_time
        expected_samples = round(elapsed_secs * self._recording_sample_rate)

        # Calculate gap using ACTUAL output samples (measured by push_frame override)
        gap_samples = expected_samples - self._bot_actual_samples
        gap_secs = gap_samples / self._recording_sample_rate if self._recording_sample_rate > 0 else 0

        # Only insert silence for gaps > MAX_GAP_SECS (10ms)
        if gap_secs > MAX_GAP_SECS:
            # Create silence frame to fill the gap at the recording sample rate
            silence_bytes = bytes(gap_samples * self._bot_num_channels * 2)
            silence_frame = OutputAudioRawFrame(
                audio=silence_bytes,
                sample_rate=self._recording_sample_rate,
                num_channels=self._bot_num_channels,
            )

            # Push silence frame downstream - push_frame override counts these samples
            await self.push_frame(silence_frame, direction)

            # Track silence statistics
            self._bot_silence_frames_inserted += 1
            self._bot_silence_samples_inserted += gap_samples

            # Log all silence insertions (helpful for debugging turn boundaries)
            gap_ms = gap_secs * 1000
            logger.info(
                f"[NullAudioOutput] Bot: Inserted {gap_ms:.0f}ms silence "
                f"({gap_samples} samples): elapsed={elapsed_secs:.2f}s, "
                f"expected={expected_samples}, actual_before={expected_samples - gap_samples}"
            )

            # Reset speech onset tracking for new turn (gap indicates turn boundary)
            # Record when bot audio started for this new turn
            self._bot_audio_start_time = current_time
            self._bot_first_speech_logged = False

        # Now process the actual frame through normal BaseOutputTransport path
        # This handles pacing and BotStarted/StoppedSpeakingFrame logic
        await super().process_frame(frame, direction)

        # Check for speech onset using RMS threshold
        if not self._bot_first_speech_logged:
            rms_db = self._calculate_rms_db(frame.audio)
            if rms_db > BOT_SPEECH_RMS_THRESHOLD_DB:
                # Log both wall-clock elapsed time AND actual sample position
                # These may differ due to MediaSender buffering/processing
                speech_onset_time = time.time()
                silent_padding_ms = (speech_onset_time - self._bot_audio_start_time) * 1000
                elapsed_ms = (speech_onset_time - self._recording_start_time) * 1000
                # Actual sample position is what's been pushed to recorder
                sample_position_ms = (self._bot_actual_samples / self._recording_sample_rate) * 1000
                logger.info(
                    f"[NullAudioOutput] Bot speech onset: T+{elapsed_ms:.0f}ms "
                    f"(sample_pos={sample_position_ms:.0f}ms, silent_padding={silent_padding_ms:.0f}ms, rms={rms_db:.1f}dB)"
                )
                self._bot_first_speech_logged = True

        # Periodic logging to track sample counting progression (every 500 frames)
        self._bot_frame_count += 1
        if self._bot_frame_count % 500 == 0:
            current_time = time.time()
            elapsed = current_time - self._recording_start_time
            expected = round(elapsed * self._recording_sample_rate)
            diff_samples = self._bot_actual_samples - expected
            diff_ms = (diff_samples / self._recording_sample_rate) * 1000
            logger.info(
                f"[NullAudioOutput] Bot frame {self._bot_frame_count}: "
                f"elapsed={elapsed:.2f}s, actual_samples={self._bot_actual_samples}, "
                f"expected={expected}, diff={diff_ms:+.0f}ms"
            )

    async def _emit_user_with_silence_fill(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Insert silence frame if needed for user audio, then emit the actual frame.

        This method calculates the expected sample position based on elapsed
        wall-clock time since recording started. If we're behind by more than
        MAX_GAP_SECS (10ms), it creates and pushes a silence frame to fill the
        gap before processing the actual audio frame.

        Args:
            frame: The InputAudioRawFrame to process.
            direction: The direction of frame flow (should be DOWNSTREAM).
        """
        current_time = time.time()

        # Initialize sample rate and channels from first user frame
        if self._user_sample_rate == 0:
            self._user_sample_rate = frame.sample_rate
            self._user_num_channels = frame.num_channels
            logger.info(
                f"[NullAudioOutput] User audio initialized: sample_rate={self._user_sample_rate}, "
                f"recording_sample_rate={self._recording_sample_rate}, "
                f"num_channels={self._user_num_channels}"
            )

        # Log timing diagnostic if frame has send time attached
        if hasattr(frame, '_paced_input_send_time'):
            import time as time_module
            arrival_time = time_module.monotonic()
            traversal_ms = (arrival_time - frame._paced_input_send_time) * 1000
            logger.info(
                f"[NullAudioOutput] Frame traversal time: {traversal_ms:.1f}ms "
                f"(sent={frame._paced_input_send_time:.3f}, arrived={arrival_time:.3f})"
            )

        # Calculate expected sample position based on wall-clock time
        elapsed_secs = current_time - self._recording_start_time
        expected_samples = round(elapsed_secs * self._recording_sample_rate)

        # Calculate gap using ACTUAL output samples (measured by push_frame override)
        gap_samples = expected_samples - self._user_actual_samples
        gap_secs = gap_samples / self._recording_sample_rate if self._recording_sample_rate > 0 else 0

        # Only insert silence for gaps > MAX_GAP_SECS (10ms)
        if gap_secs > MAX_GAP_SECS:
            # Create silence frame to fill the gap at the recording sample rate
            silence_bytes = bytes(gap_samples * self._user_num_channels * 2)
            silence_frame = InputAudioRawFrame(
                audio=silence_bytes,
                sample_rate=self._recording_sample_rate,
                num_channels=self._user_num_channels,
            )

            # Push silence frame downstream - push_frame override counts these samples
            await self.push_frame(silence_frame, direction)

            # Track silence statistics
            self._user_silence_frames_inserted += 1
            self._user_silence_samples_inserted += gap_samples

            # Log all silence insertions (helpful for debugging turn boundaries)
            gap_ms = gap_secs * 1000
            logger.info(
                f"[NullAudioOutput] User: Inserted {gap_ms:.0f}ms silence "
                f"({gap_samples} samples): elapsed={elapsed_secs:.2f}s, "
                f"expected={expected_samples}, actual_before={expected_samples - gap_samples}"
            )

        # Pass frame through to downstream processors
        await super().process_frame(frame, direction)

        # Periodic logging to track sample counting progression (every 500 frames)
        self._user_frame_count += 1
        if self._user_frame_count % 500 == 0:
            current_time = time.time()
            elapsed = current_time - self._recording_start_time
            expected = round(elapsed * self._recording_sample_rate)
            diff_samples = self._user_actual_samples - expected
            diff_ms = (diff_samples / self._recording_sample_rate) * 1000
            logger.info(
                f"[NullAudioOutput] User frame {self._user_frame_count}: "
                f"elapsed={elapsed:.2f}s, actual_samples={self._user_actual_samples}, "
                f"expected={expected}, diff={diff_ms:+.0f}ms"
            )

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Simulate audio playback timing, then discard the frame.

        This method sleeps to simulate real-time audio playback. Without this,
        the audio queue would drain at LLM generation speed (faster than real-time),
        causing BotStoppedSpeakingFrame to fire too early.

        The timing algorithm (from WebsocketServerOutputTransport):
        1. Calculate when this frame should finish "playing"
        2. Sleep until that time (if in the future)
        3. Advance the next send time by this frame's duration

        Args:
            frame: The audio frame to "write" (actually discarded after timing).

        Returns:
            True always, indicating the frame was "successfully written".
        """
        # Calculate this frame's duration in seconds
        # Formula: num_bytes / (sample_rate * num_channels * bytes_per_sample)
        # For 16-bit audio, bytes_per_sample = 2
        num_samples = len(frame.audio) // (frame.num_channels * 2)
        frame_duration = num_samples / frame.sample_rate

        self._frame_count += 1
        self._total_audio_duration += frame_duration

        # Log first frame and periodic updates
        if self._frame_count == 1:
            self._playback_start_time = time.monotonic()
            logger.info(
                f"[NullAudioOutput] First audio frame: {frame_duration*1000:.1f}ms, "
                f"samples={num_samples}, sr={frame.sample_rate}"
            )

        # Simulate real-time playback with pacing
        await self._simulate_playback_timing(frame_duration)

        # Log every 100 frames
        if self._frame_count % 100 == 0:
            elapsed = time.monotonic() - self._playback_start_time
            logger.info(
                f"[NullAudioOutput] Frame {self._frame_count}: "
                f"total_audio={self._total_audio_duration:.2f}s, "
                f"total_sleep={self._total_sleep_time:.2f}s, "
                f"wall_elapsed={elapsed:.2f}s"
            )

        # Don't actually play/send the audio - just discard it
        # Return True so MediaSender continues to track speaking state
        return True

    async def _simulate_playback_timing(self, duration: float):
        """Sleep to simulate real-time audio playback.

        This implements a pacing algorithm that ensures audio is "consumed"
        at real-time speed, not at LLM generation speed.

        Args:
            duration: The duration of the audio frame in seconds.
        """
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)

        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
            self._total_sleep_time += sleep_duration
            # We caught up to schedule, advance by frame duration
            self._next_send_time += duration
        else:
            # We're behind or just starting, reset baseline and advance
            self._next_send_time = time.monotonic() + duration

    def _log_playback_summary(self):
        """Log a summary of playback timing (called when speaking stops)."""
        if self._frame_count > 0:
            elapsed = time.monotonic() - self._playback_start_time
            logger.info(
                f"[NullAudioOutput] Playback summary: "
                f"frames={self._frame_count}, "
                f"audio_duration={self._total_audio_duration:.2f}s, "
                f"sleep_time={self._total_sleep_time:.2f}s, "
                f"wall_elapsed={elapsed:.2f}s"
            )
            # Reset for next speaking segment
            self._frame_count = 0
            self._total_audio_duration = 0.0
            self._total_sleep_time = 0.0

    def log_recording_summary(self):
        """Log a summary of silence insertion during recording.

        Call this when recording ends to see statistics about gap filling
        for both user and bot audio tracks.
        """
        if self._recording_start_time > 0:
            elapsed = time.time() - self._recording_start_time
            sr = self._recording_sample_rate if self._recording_sample_rate > 0 else 1

            # Bot audio summary
            bot_silence_secs = self._bot_silence_samples_inserted / sr
            bot_total_secs = self._bot_actual_samples / sr
            logger.info(
                f"[NullAudioOutput] Bot recording summary: "
                f"actual_samples={self._bot_actual_samples} ({bot_total_secs:.1f}s), "
                f"silence_inserted={self._bot_silence_samples_inserted} ({bot_silence_secs:.1f}s), "
                f"silence_frames={self._bot_silence_frames_inserted}"
            )

            # User audio summary
            user_silence_secs = self._user_silence_samples_inserted / sr
            user_total_secs = self._user_actual_samples / sr
            logger.info(
                f"[NullAudioOutput] User recording summary: "
                f"actual_samples={self._user_actual_samples} ({user_total_secs:.1f}s), "
                f"silence_inserted={self._user_silence_samples_inserted} ({user_silence_secs:.1f}s), "
                f"silence_frames={self._user_silence_frames_inserted}"
            )

            logger.info(f"[NullAudioOutput] Recording wall-clock elapsed: {elapsed:.1f}s")

    def _calculate_rms_db(self, audio_bytes: bytes) -> float:
        """Calculate RMS level in dB for an audio frame.

        Args:
            audio_bytes: Raw PCM audio data as bytes (16-bit signed integers).

        Returns:
            RMS level in dB (0 dB = full scale). Returns -100.0 for silence/empty frames.
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio) == 0:
            return -100.0
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1:
            return -100.0
        # dB relative to full scale (32768 for 16-bit audio)
        return 20 * np.log10(rms / 32768)
