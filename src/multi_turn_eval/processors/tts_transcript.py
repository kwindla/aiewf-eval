from typing import List
import time
from typing import List

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionMessage,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_response_universal import (
    TextPartForConcatenation,
    concatenate_aggregated_text,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import (
    AssistantTranscriptProcessor,
    TranscriptionUpdateFrame,
    TranscriptProcessor,
)
from pipecat.processors.aggregators.llm_response_universal import (
    TextPartForConcatenation,
    concatenate_aggregated_text,
)
from pipecat.utils.time import time_now_iso8601


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
            # LLMTextFrame contains "thinking" blocks from native audio models (e.g., Gemini 2.5 Flash)
            # These are internal reasoning, not spoken content - ignore for transcript
            text = getattr(frame, "text", "")
            logger.debug(f"[TRANSCRIPT] Ignoring LLMTextFrame (thinking): {text[:60]}...")
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSTextFrame):
            # TTSTextFrame contains actual spoken content - aggregate for transcript
            text = getattr(frame, "text", "")
            logger.info(f"[TRANSCRIPT] Received text frame: {text[:100]}... ({len(text)} chars)")
            if not self._aggregation_start_time:
                self._aggregation_start_time = time_now_iso8601()
                logger.info(f"[TRANSCRIPT] Started aggregation at {self._aggregation_start_time}")
            self._current_text_parts.append(
                TextPartForConcatenation(text, includes_inter_part_spaces=True)
            )
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
            # Flush aggregated text on audio stop or text response end
            logger.info(f"[TRANSCRIPT] Received flush frame: {type(frame).__name__}")
            # Simple join and normalize whitespace - OpenAI Realtime tokens may have
            # irregular spacing (e.g., " " tokens before numbers)
            raw_text = "".join(p.text for p in self._current_text_parts)
            # Normalize multiple spaces to single space
            import re
            total_text = re.sub(r' +', ' ', raw_text).strip()
            logger.info(f"[TRANSCRIPT] Flushing {len(total_text)} chars of aggregated text")

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

            await self.push_frame(frame, direction)
        else:
            # Forward everything else without flushing
            await self.push_frame(frame, direction)
