"""Vendored pipecat transcript processors (removed upstream in pipecat 1.x).

pipecat 0.0.99 deprecated and 1.x removed ``TranscriptProcessor`` /
``AssistantTranscriptProcessor`` and the ``TranscriptionMessage`` /
``ThoughtTranscriptionMessage`` / ``TranscriptionUpdateFrame`` types in favor
of ``LLMUserAggregator`` / ``LLMAssistantAggregator`` events. Our realtime
pipeline and ``TTSStoppedAssistantTranscriptProcessor`` shim are built on the
removed API (turn gating waits for a single aggregated assistant transcript
per turn), so we vendor the final upstream implementation here — the same
approach used for ``multi_turn_eval.vendor.nemotron_omni`` in the
0.0.101 -> 1.1.0 bump.

Source: pipecat @ 345ccc0ab — src/pipecat/processors/transcript_processor.py
plus the three transcript types from src/pipecat/frames/frames.py, with the
deprecation warnings dropped. Behavior is otherwise unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    DataFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    format_pts,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from pipecat.utils.time import time_now_iso8601


@dataclass
class TranscriptionMessage:
    """A message in a conversation transcript.

    Parameters:
        role: The role of the message sender (user or assistant).
        content: The message content/text.
        user_id: Optional identifier for the user.
        timestamp: Optional timestamp when the message was created.
    """

    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ThoughtTranscriptionMessage:
    """An LLM thought message in a conversation transcript.

    Parameters:
        content: The thought content/text.
        timestamp: Optional timestamp when the thought was created.
    """

    role: Literal["assistant"] = field(default="assistant", init=False)
    content: str
    timestamp: Optional[str] = None


@dataclass
class TranscriptionUpdateFrame(DataFrame):
    """Frame containing new messages added to conversation transcript.

    Parameters:
        messages: List of new transcript messages that were added.
    """

    messages: List[TranscriptionMessage | ThoughtTranscriptionMessage]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, messages: {len(self.messages)})"


class BaseTranscriptProcessor(FrameProcessor):
    """Base class for processing conversation transcripts.

    Provides common functionality for handling transcript messages and updates.
    """

    def __init__(self, **kwargs):
        """Initialize processor with empty message store.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._processed_messages: List[TranscriptionMessage] = []
        self._register_event_handler("on_transcript_update")

    async def _emit_update(self, messages: List[TranscriptionMessage]):
        """Emit transcript updates for new messages.

        Args:
            messages: New messages to emit in update.
        """
        if messages:
            self._processed_messages.extend(messages)
            update_frame = TranscriptionUpdateFrame(messages=messages)
            await self._call_event_handler("on_transcript_update", update_frame)
            await self.push_frame(update_frame)


class UserTranscriptProcessor(BaseTranscriptProcessor):
    """Processes user transcription frames into timestamped conversation messages."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process TranscriptionFrames into user conversation messages.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            message = TranscriptionMessage(
                role="user", user_id=frame.user_id, content=frame.text, timestamp=frame.timestamp
            )
            await self._emit_update([message])

        await self.push_frame(frame, direction)


class AssistantTranscriptProcessor(BaseTranscriptProcessor):
    """Processes assistant TTS text frames and LLM thought frames into timestamped messages.

    This processor aggregates both TTS text frames and LLM thought frames into
    complete utterances and thoughts, emitting them as transcript messages.

    An assistant utterance is completed when:
    - The bot stops speaking (BotStoppedSpeakingFrame)
    - The bot is interrupted (InterruptionFrame)
    - The pipeline ends (EndFrame, CancelFrame)

    A thought is completed when:
    - The thought ends (LLMThoughtEndFrame)
    - The bot is interrupted (InterruptionFrame)
    - The pipeline ends (EndFrame, CancelFrame)
    """

    def __init__(self, *, process_thoughts: bool = False, **kwargs):
        """Initialize processor with aggregation state.

        Args:
            process_thoughts: Whether to process LLM thought frames. Defaults to False.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._process_thoughts = process_thoughts
        self._current_assistant_text_parts: List[TextPartForConcatenation] = []
        self._assistant_text_start_time: Optional[str] = None

        self._current_thought_parts: List[TextPartForConcatenation] = []
        self._thought_start_time: Optional[str] = None
        self._thought_active = False

    async def _emit_aggregated_assistant_text(self):
        """Aggregate and emit text fragments as a transcript message."""
        if self._current_assistant_text_parts and self._assistant_text_start_time:
            content = concatenate_aggregated_text(self._current_assistant_text_parts)
            if content:
                logger.trace(f"Emitting aggregated assistant message: {content}")
                message = TranscriptionMessage(
                    role="assistant",
                    content=content,
                    timestamp=self._assistant_text_start_time,
                )
                await self._emit_update([message])
            else:
                logger.trace("No content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_assistant_text_parts = []
            self._assistant_text_start_time = None

    async def _emit_aggregated_thought(self):
        """Aggregate and emit thought fragments as a thought transcript message."""
        if self._current_thought_parts and self._thought_start_time:
            content = concatenate_aggregated_text(self._current_thought_parts)
            if content:
                logger.trace(f"Emitting aggregated thought message: {content}")
                message = ThoughtTranscriptionMessage(
                    content=content,
                    timestamp=self._thought_start_time,
                )
                await self._emit_update([message])
            else:
                logger.trace("No thought content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_thought_parts = []
            self._thought_start_time = None
            self._thought_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames into assistant conversation messages and thought messages.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            # Push frame first otherwise our emitted transcription update frame
            # might get cleaned up.
            await self.push_frame(frame, direction)
            # Emit accumulated text and thought with interruptions
            await self._emit_aggregated_assistant_text()
            if self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
        elif isinstance(frame, LLMThoughtStartFrame):
            # Start a new thought
            if self._process_thoughts:
                self._thought_active = True
                self._thought_start_time = time_now_iso8601()
                self._current_thought_parts = []
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMThoughtTextFrame):
            # Aggregate thought text if we have an active thought
            if self._process_thoughts and self._thought_active:
                self._current_thought_parts.append(
                    TextPartForConcatenation(
                        frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                    )
                )
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMThoughtEndFrame):
            # Emit accumulated thought when thought ends
            if self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSTextFrame):
            # Start timestamp on first text part
            if not self._assistant_text_start_time:
                self._assistant_text_start_time = time_now_iso8601()

            self._current_assistant_text_parts.append(
                TextPartForConcatenation(
                    frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                )
            )
            await self.push_frame(frame, direction)
        elif isinstance(frame, (BotStoppedSpeakingFrame, EndFrame)):
            # Emit accumulated text when bot finishes speaking or pipeline ends.
            await self._emit_aggregated_assistant_text()
            # Emit accumulated thought at pipeline end if still active
            if isinstance(frame, EndFrame) and self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class TranscriptProcessor:
    """Factory for creating and managing transcript processors.

    Provides unified access to user and assistant transcript processors
    with shared event handling. The assistant processor handles both TTS text
    and LLM thought frames.
    """

    def __init__(self, *, process_thoughts: bool = False):
        """Initialize factory.

        Args:
            process_thoughts: Whether the assistant processor should handle LLM thought
                frames. Defaults to False.
        """
        self._process_thoughts = process_thoughts
        self._user_processor = None
        self._assistant_processor = None
        self._event_handlers = {}

    def user(self, **kwargs) -> UserTranscriptProcessor:
        """Get the user transcript processor.

        Args:
            **kwargs: Arguments specific to UserTranscriptProcessor.

        Returns:
            The user transcript processor instance.
        """
        if self._user_processor is None:
            self._user_processor = UserTranscriptProcessor(**kwargs)
            # Apply any registered event handlers
            for event_name, handler in self._event_handlers.items():

                @self._user_processor.event_handler(event_name)
                async def user_handler(processor, frame):
                    return await handler(processor, frame)

        return self._user_processor

    def assistant(self, **kwargs) -> AssistantTranscriptProcessor:
        """Get the assistant transcript processor.

        Args:
            **kwargs: Arguments specific to AssistantTranscriptProcessor.

        Returns:
            The assistant transcript processor instance.
        """
        if self._assistant_processor is None:
            self._assistant_processor = AssistantTranscriptProcessor(
                process_thoughts=self._process_thoughts, **kwargs
            )
            # Apply any registered event handlers
            for event_name, handler in self._event_handlers.items():

                @self._assistant_processor.event_handler(event_name)
                async def assistant_handler(processor, frame):
                    return await handler(processor, frame)

        return self._assistant_processor

    def event_handler(self, event_name: str):
        """Register event handler for both processors.

        Args:
            event_name: Name of event to handle.

        Returns:
            Decorator function that registers handler with both processors.
        """

        def decorator(handler):
            self._event_handlers[event_name] = handler

            # Apply handler to existing processors if they exist
            if self._user_processor:

                @self._user_processor.event_handler(event_name)
                async def user_handler(processor, frame):
                    return await handler(processor, frame)

            if self._assistant_processor:

                @self._assistant_processor.event_handler(event_name)
                async def assistant_handler(processor, frame):
                    return await handler(processor, frame)

            return handler

        return decorator
