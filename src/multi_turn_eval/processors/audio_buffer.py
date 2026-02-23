#
# Custom AudioBufferProcessor that disables base class silence insertion.
#
# NullAudioOutputTransport is the "source of truth" for wall-clock aligned silence
# insertion for BOTH user and bot audio tracks, with < 10ms gap tolerance.
#
# This wrapper disables two base class behaviors that would double-count silence:
#
# 1. _compute_silence() (pipecat <= 0.0.98): Inserted silence for gaps >= 1 second
#    between frames of the same track.
#
# 2. _sync_buffer_to_position() (pipecat >= 0.0.99): Pads the opposite track's buffer
#    to match whenever audio arrives on either track. This causes exponential buffer
#    growth because bot silence from NullAudioOutputTransport and user-driven sync
#    padding compound each other.
#

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


class WallClockAlignedAudioBufferProcessor(AudioBufferProcessor):
    """AudioBufferProcessor with base class silence insertion disabled.

    NullAudioOutputTransport handles all silence insertion for both user and
    bot tracks with < 10ms gap tolerance. This subclass disables the base
    class's silence insertion methods to prevent double-counting.
    """

    def _compute_silence(self, from_time: float) -> bytes:
        """Override to disable automatic silence insertion.

        The base class (pipecat <= 0.0.98) inserts silence for gaps >= 1 second
        between frames. We disable this because NullAudioOutputTransport handles
        all silence insertion with < 10ms precision.

        Args:
            from_time: The timestamp of the last audio frame (unused).

        Returns:
            Empty bytes (no silence).
        """
        return b""

    def _sync_buffer_to_position(self, buffer: bytearray, target_position: int):
        """Override to disable cross-track buffer synchronization.

        The base class (pipecat >= 0.0.99) pads the opposite track's buffer
        whenever audio arrives on either track. This double-counts silence
        because NullAudioOutputTransport already inserts wall-clock aligned
        silence for both tracks independently.

        Args:
            buffer: The buffer that would be padded (ignored).
            target_position: The target byte position (ignored).
        """
        pass


# Backwards compatibility alias
NoSilenceAudioBufferProcessor = WallClockAlignedAudioBufferProcessor
