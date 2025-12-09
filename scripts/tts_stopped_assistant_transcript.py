from typing import List
import time
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSTextFrame,
    TTSStoppedFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    TranscriptionMessage,
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Bypass AssistantTranscriptProcessor.process_frame to avoid its default
        # flush triggers. Call the base FrameProcessor implementation directly.
        await FrameProcessor.process_frame(self, frame, direction)

        if isinstance(frame, (TTSTextFrame, LLMTextFrame)):
            # frame.text exists on TTSTextFrame/LLMTextFrame; guard defensively
            text = getattr(frame, "text", "")
            logger.info(f"[TRANSCRIPT] Received text frame: {text[:100]}... ({len(text)} chars)")
            if not getattr(self, "_aggregation_start_time", None):
                self._aggregation_start_time = time_now_iso8601()
                logger.info(f"[TRANSCRIPT] Started aggregation at {self._aggregation_start_time}")
            # AssistantTranscriptProcessor expects TextPartForConcatenation items
            self._current_text_parts.append(
                TextPartForConcatenation(text, includes_inter_part_spaces=False)
            )
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
            # Flush aggregated text on audio stop or text response end
            logger.info(f"[TRANSCRIPT] Received flush frame: {type(frame).__name__}")
            total_text = (
                concatenate_aggregated_text(self._current_text_parts)
                if hasattr(self, "_current_text_parts")
                else ""
            )
            logger.info(f"[TRANSCRIPT] Flushing {len(total_text)} chars of aggregated text")
            await self._emit_aggregated_text()
            # Also emit a TranscriptionUpdateFrame so external transcript handlers (e.g., in convo-test) fire
            if total_text:
                tuf = TranscriptionUpdateFrame(
                    messages=[TranscriptionMessage(role="assistant", content=total_text)]
                )
                await self.push_frame(tuf, direction)
            await self.push_frame(frame, direction)
        else:
            # Forward everything else without flushing
            await self.push_frame(frame, direction)
