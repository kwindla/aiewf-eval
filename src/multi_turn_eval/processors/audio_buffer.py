#
# Custom AudioBufferProcessor with wall-clock aligned track recording.
#
# This processor ensures both user and bot audio tracks are aligned to wall-clock
# time by inserting leading silence from recording_start_time to the first audio
# frame of each track.
#
# Background:
# - NullAudioOutputTransport._emit_with_silence_fill() inserts leading silence for
#   bot audio from recording_start_time to when bot audio arrives
# - But user audio (InputAudioRawFrame) doesn't go through NullAudioOutputTransport
# - This causes user and bot tracks to be misaligned by ~1 second
# - This processor adds equivalent leading silence for user audio
#
# The fix ensures:
# - User track position 0 = wall-clock recording_start_time
# - Bot track position 0 = wall-clock recording_start_time (via NullAudioOutputTransport)
# - Both tracks are properly aligned for audio-based TTFB analysis
#

import time

from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection


class WallClockAlignedAudioBufferProcessor(AudioBufferProcessor):
    """AudioBufferProcessor with wall-clock aligned user and bot tracks.

    This processor extends AudioBufferProcessor to ensure both tracks are aligned
    to wall-clock time. It:
    1. Tracks recording_start_time when start_recording() is called
    2. Inserts leading silence for the first user audio frame (from recording start)
    3. Disables the base class's automatic silence insertion (> 1s gap) to avoid
       double-counting with NullAudioOutputTransport's silence filling

    The result is that both user and bot tracks start at the same wall-clock time,
    enabling accurate audio-based TTFB measurements.
    """

    def __init__(self, **kwargs):
        """Initialize the wall-clock aligned audio buffer processor."""
        super().__init__(**kwargs)
        self._recording_start_time: float = 0.0
        self._first_user_frame_received: bool = False

    async def start_recording(self):
        """Start recording and capture the recording start time.

        This timestamp is used to calculate leading silence for the user track,
        matching NullAudioOutputTransport's behavior for the bot track.
        """
        self._recording_start_time = time.time()
        self._first_user_frame_received = False
        logger.info(f"[AudioBuffer] Recording started at {self._recording_start_time:.3f}")
        await super().start_recording()

    async def _process_recording(self, frame: Frame):
        """Process audio frames, adding leading silence for first user frame.

        For the first InputAudioRawFrame, calculates the gap from recording_start_time
        and inserts that much silence BEFORE the audio. This aligns the user track
        to wall-clock time, matching the bot track's alignment.

        Args:
            frame: The frame to process.
        """
        # Handle first user frame: insert leading silence
        if (
            isinstance(frame, InputAudioRawFrame)
            and self._recording
            and not self._first_user_frame_received
            and self._recording_start_time > 0
            and self._sample_rate > 0
        ):
            self._first_user_frame_received = True

            # Calculate leading silence duration
            elapsed = time.time() - self._recording_start_time
            if elapsed > 0:
                # bytes_per_sample = 2 (16-bit audio)
                num_samples = int(elapsed * self._sample_rate)
                silence_bytes = num_samples * 2
                silence = b"\x00" * silence_bytes

                # Insert leading silence into user buffer
                self._user_audio_buffer.extend(silence)

                logger.info(
                    f"[AudioBuffer] Inserted {elapsed*1000:.0f}ms leading silence "
                    f"for user track ({num_samples} samples)"
                )

        # Call parent implementation for normal processing
        await super()._process_recording(frame)

    def _compute_silence(self, from_time: float) -> bytes:
        """Override to disable automatic silence insertion.

        The base class inserts silence for gaps >= 1 second between frames.
        We disable this because:
        1. NullAudioOutputTransport handles bot audio silence filling
        2. We handle leading silence for user audio in _process_recording()

        Args:
            from_time: The timestamp of the last audio frame (unused).

        Returns:
            Empty bytes (no silence).
        """
        return b""


# Alias for backwards compatibility
NoSilenceAudioBufferProcessor = WallClockAlignedAudioBufferProcessor
