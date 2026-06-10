import time
from typing import List

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import (
    TextPartForConcatenation,
    concatenate_aggregated_text,
)
from pipecat.utils.time import time_now_iso8601

# Vendored: pipecat 1.x removed the transcript-processor subsystem these shims
# are built on. See multi_turn_eval/vendor/transcript_processor.py.
from multi_turn_eval.vendor.transcript_processor import (
    AssistantTranscriptProcessor,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    TranscriptProcessor,
)


class TTSStoppedAssistantTranscriptProcessor(AssistantTranscriptProcessor):
    """Assistant transcript shim that flushes on end-of-response and re-emits updates.

    - Aggregates TTSTextFrame fragments (AUDIO modality) and LLMTextFrame fragments (TEXT modality).
    - Emits a single assistant TranscriptionUpdateFrame when either TTSStoppedFrame (audio) or
      LLMFullResponseEndFrame (text) arrives.
    - Replays that update through the shared TranscriptProcessor event system so external handlers fire.
    - Avoids default flush triggers from AssistantTranscriptProcessor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_text_parts = []
        self._aggregation_start_time = None

    def clear_buffer(self):
        """Clear accumulated text buffer. Call this during reconnection to discard stale partial responses."""
        if self._current_text_parts:
            logger.info(f"[TRANSCRIPT] Clearing {len(self._current_text_parts)} accumulated text parts due to reconnection")
        self._current_text_parts = []
        self._aggregation_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Bypass AssistantTranscriptProcessor.process_frame to avoid its default
        # flush triggers. Call the base FrameProcessor implementation directly.
        await FrameProcessor.process_frame(self, frame, direction)

        if isinstance(frame, LLMTextFrame):
            # LLMTextFrame handling:
            # - Gemini: sends "thinking" content (skip_tts not explicitly set) - ignore
            # - Ultravox voice mode: sends transcript with skip_tts=True - AGGREGATE (no TTSTextFrame sent)
            # - Ultravox text mode: sends text with skip_tts=False - ignore (use TTSTextFrame instead)
            # - OpenAI Realtime: does NOT send LLMTextFrame
            text = getattr(frame, "text", "")
            skip_tts = getattr(frame, "skip_tts", None)

            # Only aggregate for Ultravox voice mode (skip_tts=True explicitly set)
            # Ultravox text mode has skip_tts=False and sends TTSTextFrame too
            if skip_tts is True:
                logger.info(
                    f"[TRANSCRIPT] Received LLMTextFrame (skip_tts={skip_tts}): {text[:100]}... ({len(text)} chars)"
                )
                if not self._aggregation_start_time:
                    self._aggregation_start_time = time_now_iso8601()
                self._current_text_parts.append(
                    TextPartForConcatenation(text, includes_inter_part_spaces=True)
                )
            else:
                logger.debug(
                    f"[TRANSCRIPT] Ignoring LLMTextFrame (skip_tts={skip_tts}): {text[:60]}..."
                )
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSTextFrame):
            # TTSTextFrame contains actual spoken content - aggregate for transcript
            text = getattr(frame, "text", "")
            logger.info(
                f"[TRANSCRIPT] Received text frame: {text[:100]}... ({len(text)} chars)"
            )
            if not self._aggregation_start_time:
                self._aggregation_start_time = time_now_iso8601()
                logger.info(
                    f"[TRANSCRIPT] Started aggregation at {self._aggregation_start_time}"
                )
            self._current_text_parts.append(
                TextPartForConcatenation(text, includes_inter_part_spaces=True)
            )
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
            # Flush aggregated text on audio stop or text response end
            logger.info(f"[TRANSCRIPT] Received flush frame: {type(frame).__name__}")

            # Only flush if we have accumulated text
            # This prevents double-flushing when both TTSStoppedFrame and LLMFullResponseEndFrame arrive
            if self._current_text_parts:
                # Simple join and normalize whitespace - OpenAI Realtime tokens may have
                # irregular spacing (e.g., " " tokens before numbers)
                raw_text = "".join(p.text for p in self._current_text_parts)
                # Normalize multiple spaces to single space
                import re

                total_text = re.sub(r" +", " ", raw_text).strip()
                logger.info(
                    f"[TRANSCRIPT] Flushing {len(total_text)} chars of aggregated text"
                )

                # Emit transcript update via the event system (triggers on_transcript_update handlers)
                if total_text:
                    message = TranscriptionMessage(
                        role="assistant",
                        content=total_text,
                        timestamp=self._aggregation_start_time,
                    )
                    await self._emit_update([message])

                # Reset aggregation state
                self._current_text_parts = []
                self._aggregation_start_time = None
            else:
                logger.info("[TRANSCRIPT] No text to flush (already flushed)")

            await self.push_frame(frame, direction)
        else:
            # Forward everything else without flushing
            await self.push_frame(frame, direction)
