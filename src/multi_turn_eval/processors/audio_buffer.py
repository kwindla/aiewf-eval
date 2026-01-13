#
# Custom AudioBufferProcessor that disables base class silence insertion.
#
# The base AudioBufferProcessor._compute_silence() inserts silence for gaps >= 1 second.
# This causes double-counting with NullAudioOutputTransport's rigorous silence insertion.
#
# NullAudioOutputTransport is the "source of truth" for wall-clock aligned silence
# insertion for BOTH user and bot audio tracks, with < 10ms gap tolerance.
# This wrapper disables the base class's coarse 1-second gap filling.
#

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


class WallClockAlignedAudioBufferProcessor(AudioBufferProcessor):
    """AudioBufferProcessor with base class silence insertion disabled.

    NullAudioOutputTransport handles all silence insertion for both user and
    bot tracks with < 10ms gap tolerance. This subclass disables the base
    class's _compute_silence() which would insert additional silence for
    gaps >= 1 second, causing double-counting and timeline drift.
    """

    def _compute_silence(self, from_time: float) -> bytes:
        """Override to disable automatic silence insertion.

        The base class inserts silence for gaps >= 1 second between frames.
        We disable this because NullAudioOutputTransport handles all silence
        insertion with < 10ms precision.

        Args:
            from_time: The timestamp of the last audio frame (unused).

        Returns:
            Empty bytes (no silence).
        """
        return b""


# Backwards compatibility alias
NoSilenceAudioBufferProcessor = WallClockAlignedAudioBufferProcessor
