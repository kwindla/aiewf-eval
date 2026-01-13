"""Nova Sonic pipeline components for AWS Bedrock Nova Sonic models."""

import asyncio
import json
import time
import wave
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from multi_turn_eval.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.transports.base_transport import TransportParams

from multi_turn_eval.transports.null_audio_output import NullAudioOutputTransport


class ResamplingSileroVAD(SileroVADAnalyzer):
    """SileroVADAnalyzer that resamples audio from input rate to 16kHz.

    Silero VAD only supports 16kHz or 8kHz. This subclass resamples incoming
    audio (e.g., 24kHz) to 16kHz before VAD processing. For Nova Sonic's
    16kHz input, no resampling is needed, but this provides consistency
    with the realtime pipeline.
    """

    def __init__(self, **kwargs):
        super().__init__(sample_rate=16000, **kwargs)
        self._input_sample_rate: int = 16000

    def set_sample_rate(self, sample_rate: int):
        # Store the input sample rate for resampling, but tell parent we're using 16kHz
        self._input_sample_rate = sample_rate
        super().set_sample_rate(16000)

    async def analyze_audio(self, buffer: bytes):
        # Resample before buffering/analysis if input rate differs from 16kHz
        if self._input_sample_rate != 16000:
            audio_int16 = np.frombuffer(buffer, np.int16).astype(np.float32)
            # Simple linear interpolation resampling
            ratio = 16000 / self._input_sample_rate
            new_length = int(len(audio_int16) * ratio)
            if new_length > 0:
                indices = np.linspace(0, len(audio_int16) - 1, new_length)
                resampled = np.interp(indices, np.arange(len(audio_int16)), audio_int16)
                buffer = resampled.astype(np.int16).tobytes()
        return await super().analyze_audio(buffer)


@dataclass
class NovaSonicCompletionEndFrame(DataFrame):
    """Signal that Nova Sonic has finished generating the complete response.

    This frame is emitted when Nova Sonic's `completionEnd` event is received,
    indicating that all text chunks should arrive soon. Use this to know when
    to start the final text collection timeout.
    """
    pass


@dataclass
class NovaSonicTextTurnEndFrame(DataFrame):
    """Signal that Nova Sonic has finished generating text for this turn.

    This frame is emitted when we receive a FINAL TEXT content with stopReason=END_TURN,
    indicating that the transcript for this assistant response is complete.
    """
    pass


class NovaSonicLLMServiceWithCompletionSignal(AWSNovaSonicLLMService):
    """Extended Nova Sonic service that emits frames for turn completion detection.

    The base AWSNovaSonicLLMService handles events but doesn't expose key signals.
    This subclass:
    1. Tracks the current content being received (type, role, generationStage)
    2. Emits NovaSonicTextTurnEndFrame when FINAL TEXT ends with END_TURN
    3. Emits NovaSonicCompletionEndFrame when the session's completionEnd arrives
    4. Emits TTFB metrics (time from trigger to first audio)
    5. Supports Nova 2 Sonic VAD configuration (endpointingSensitivity)
    6. Overrides reset_conversation() with retry limits to prevent infinite error cascade
    """

    def __init__(
        self,
        endpointing_sensitivity: str = None,
        max_reconnect_attempts: int = 3,
        max_context_turns: int = 15,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        on_retriggered: Optional[Callable[[], None]] = None,
        on_max_reconnects_exceeded: Optional[Callable[[], Any]] = None,
        **kwargs,
    ):
        """Initialize the Nova Sonic service.

        Args:
            endpointing_sensitivity: VAD sensitivity for Nova 2 Sonic only.
                Options: "HIGH" (quick cutoff), "MEDIUM" (default), "LOW" (longer wait).
                Only applicable to amazon.nova-2-sonic-v1:0 model.
                Nova Sonic v1 does not support this parameter.
            max_reconnect_attempts: Maximum reconnection attempts before giving up.
            max_context_turns: Maximum number of user/assistant turn pairs to keep during
                reconnection. Older turns are truncated to avoid exceeding Nova Sonic's
                context limits. System messages are always preserved.
            on_reconnecting: Callback when reconnection starts (pause audio input).
            on_reconnected: Callback when reconnection completes (resume audio input).
            on_retriggered: Callback after assistant response is re-triggered (signal turn detector).
            on_max_reconnects_exceeded: Async callback when max reconnects exceeded (cancel task).
        """
        super().__init__(**kwargs)
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None
        self._ttfb_started = False  # Track if we've started TTFB timing for this turn
        self._endpointing_sensitivity = endpointing_sensitivity

        # Reconnection handling
        self._max_reconnect_attempts = max_reconnect_attempts
        self._max_context_turns = max_context_turns
        self._reconnect_attempts = 0
        self._is_reconnecting = False
        self._need_retrigger_after_reconnect = False
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        self._on_retriggered = on_retriggered
        self._on_max_reconnects_exceeded = on_max_reconnects_exceeded

    def can_generate_metrics(self) -> bool:
        """Enable metrics generation for TTFB tracking.

        The base FrameProcessor returns False by default, which prevents
        start_ttfb_metrics() and stop_ttfb_metrics() from working.
        """
        return True

    def is_reconnecting(self) -> bool:
        """Check if currently reconnecting (for external coordination)."""
        return self._is_reconnecting

    def reset_reconnect_counter(self):
        """Reset the reconnection attempt counter (call on successful turn completion).

        Does NOT reset if currently reconnecting, to prevent race conditions where
        a turn completes during reconnection and resets the counter mid-cycle.
        """
        if self._is_reconnecting:
            logger.debug(
                f"Not resetting reconnect counter during reconnection (current: {self._reconnect_attempts})"
            )
            return
        if self._reconnect_attempts > 0:
            logger.info(f"Resetting reconnect counter (was {self._reconnect_attempts})")
        self._reconnect_attempts = 0

    def _truncate_context_for_reconnection(self):
        """Truncate context for reconnection to fit within Nova Sonic's limits.

        Nova Sonic has strict context limits during session reconnection (~5-10K chars total).
        Strategy: Use a minimal system prompt (just core instructions) + most recent 1 turn (2 messages).

        Based on testing:
        - 21.6K chars: "Chat history is over max limit" error
        - 10.8K chars: Still too large
        - Need to stay under ~5K chars total for reliable reconnection

        Returns the number of messages removed, or 0 if no truncation was needed.
        """
        if not self._context:
            return 0

        messages = self._context.get_messages()
        if not messages:
            return 0

        # Separate system messages from conversation messages
        system_messages = []
        conversation_messages = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)

        # Use a minimal system prompt for reconnection (~300 chars)
        # This preserves core behavior while fitting within strict limits
        minimal_system_prompt = """You are a helpful voice assistant for the AI Engineer World's Fair 2025 (June 3-5, San Francisco).
Answer questions about the conference schedule, sessions, and speakers.
Be conversational and concise. If you don't have specific information, say so politely."""

        if system_messages:
            original_content = str(system_messages[0].get("content", ""))
            original_len = len(original_content)
            system_messages = [{"role": "system", "content": minimal_system_prompt}]
            logger.warning(
                f"Using minimal system prompt for reconnection: {original_len} chars -> {len(minimal_system_prompt)} chars"
            )

        # Keep ZERO conversation messages - Nova Sonic's internal state may be corrupted
        # after 8-minute timeout, and any conversation history causes "over max limit" errors
        max_messages = 0
        if len(conversation_messages) > max_messages:
            messages_removed = len(conversation_messages)
            truncated_conversation = []  # Keep nothing
            logger.warning(
                f"Truncating conversation for reconnection: keeping last {max_messages} messages. "
                f"Removing {messages_removed} older messages."
            )
        else:
            messages_removed = 0
            truncated_conversation = conversation_messages
            logger.debug(
                f"Conversation truncation not needed: {len(conversation_messages)} messages "
                f"<= {max_messages} max"
            )

        # Rebuild: minimal system + recent conversation
        new_messages = system_messages + truncated_conversation

        # Update the context with truncated messages
        self._context.set_messages(new_messages)
        return messages_removed

    async def reset_conversation(self):
        """Override to add retry limits, context truncation, and preserve trigger state.

        The base class calls this automatically when errors occur in the receive task.
        Without retry limits, connection errors can cascade infinitely.

        Key improvements:
        1. Retry limits - gives up after max_reconnect_attempts
        2. Context truncation - removes old messages to fit within Nova Sonic limits
        3. Preserves trigger state - re-triggers assistant response after reconnection
        4. Callbacks - notifies external components to pause/resume audio input
        """
        # Check retry limit
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self._max_reconnect_attempts}) reached. "
                f"Giving up on reconnection."
            )
            await self.push_error(
                error_msg=f"Nova Sonic: Max reconnect attempts ({self._max_reconnect_attempts}) exceeded"
            )
            self._wants_connection = False

            # Call the max reconnects exceeded callback to terminate gracefully
            if self._on_max_reconnects_exceeded:
                try:
                    result = self._on_max_reconnects_exceeded()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.exception(f"Error in on_max_reconnects_exceeded callback: {e}")
            return

        self._reconnect_attempts += 1
        self._is_reconnecting = True

        logger.warning(
            f"Nova Sonic reset_conversation() attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}"
        )

        # Remember if we need to re-trigger after reconnection
        # This is lost in _disconnect() so we must capture it here
        self._need_retrigger_after_reconnect = (
            self._triggering_assistant_response or self._assistant_is_responding
        )
        logger.info(
            f"Nova Sonic: Will re-trigger after reconnect: {self._need_retrigger_after_reconnect} "
            f"(triggering={self._triggering_assistant_response}, responding={self._assistant_is_responding})"
        )

        # Truncate context to avoid exceeding Nova Sonic's limits during reconnection
        messages_removed = self._truncate_context_for_reconnection()
        if messages_removed > 0:
            logger.info(f"Nova Sonic: Removed {messages_removed} old messages before reconnection")

        # Notify external components to pause audio input
        if self._on_reconnecting:
            try:
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        # Call parent implementation (handles disconnect/reconnect/context reload)
        try:
            await super().reset_conversation()
        except Exception as e:
            logger.exception(f"Error in parent reset_conversation: {e}")
            self._is_reconnecting = False
            raise

        self._is_reconnecting = False

        # Notify external components reconnection is complete
        if self._on_reconnected:
            try:
                self._on_reconnected()
            except Exception as e:
                logger.warning(f"Error in on_reconnected callback: {e}")

        # Re-trigger assistant response if we were in the middle of one
        # NOTE: Disabled re-trigger after reconnection as it causes "Chat history over max limit"
        # errors. The user will need to re-send their audio to continue the conversation.
        if self._need_retrigger_after_reconnect:
            logger.warning(
                "Nova Sonic: Skipping re-trigger after reconnection (causes errors). "
                "User must re-send audio to continue."
            )
            self._need_retrigger_after_reconnect = False
            # Don't trigger - let the next user audio input restart the conversation

    async def _send_session_start_event(self):
        """Override to add endpointingSensitivity for Nova 2 Sonic VAD control.

        Nova 2 Sonic supports VAD configuration via endpointingSensitivity:
        - HIGH: Very sensitive to pauses (quick cutoff)
        - MEDIUM: Balanced sensitivity (default)
        - LOW: Less sensitive to pauses (longer wait before cutoff)

        Nova Sonic v1 does not support this parameter.
        """
        # Build inference configuration
        inference_config = {
            "maxTokens": self._params.max_tokens,
            "topP": self._params.top_p,
            "temperature": self._params.temperature,
        }

        # Add endpointingSensitivity for Nova 2 Sonic
        if self._endpointing_sensitivity:
            inference_config["endpointingSensitivity"] = self._endpointing_sensitivity
            logger.info(f"NovaSonicLLM: Using endpointingSensitivity={self._endpointing_sensitivity}")

        session_start = json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": inference_config
                }
            }
        })
        await self._send_client_event(session_start)

    async def start_ttfb_for_user_audio_complete(self):
        """Start TTFB timing when user audio delivery is complete.

        This should be called when the last byte of user audio has been
        delivered to the model. TTFB = time from this point to first model audio.
        """
        logger.info("NovaSonicLLM: Starting TTFB metrics (user audio complete)")
        await self.start_ttfb_metrics()
        self._ttfb_started = True
        self._audio_output_count = 0  # Reset for new turn

    async def trigger_assistant_response(self):
        """Override to trigger assistant response."""
        logger.info("NovaSonicLLM: Triggering assistant response")
        await super().trigger_assistant_response()

    async def _receive_task_handler(self):
        """Override to add custom event handling for turn end detection.

        This extends the parent's receive handler to:
        1. Track content metadata (type, role, generationStage)
        2. Emit NovaSonicTextTurnEndFrame when AUDIO ends with END_TURN
        3. Emit NovaSonicCompletionEndFrame when completionEnd arrives
        4. Capture TTFB metrics on first audio output
        """
        try:
            while self._stream and not self._disconnecting:
                output = await self._stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)

                    if "event" in json_data:
                        event_json = json_data["event"]

                        # Route events to handlers
                        if "completionStart" in event_json:
                            await self._handle_completion_start_event(event_json)
                        elif "contentStart" in event_json:
                            await self._handle_content_start_event(event_json)
                        elif "textOutput" in event_json:
                            await self._handle_text_output_event(event_json)
                        elif "audioOutput" in event_json:
                            await self._handle_audio_output_event(event_json)
                        elif "toolUse" in event_json:
                            await self._handle_tool_use_event(event_json)
                        elif "contentEnd" in event_json:
                            await self._handle_content_end_event(event_json)
                        elif "completionEnd" in event_json:
                            await self._handle_completion_end_event(event_json)
        except Exception as e:
            if self._disconnecting:
                logger.debug(f"NovaSonicLLM: _receive_task_handler exception during disconnect: {e}")
                return
            logger.error(f"NovaSonicLLM: Error in receive task: {e}")
            await self.push_error(error_msg=f"Error processing responses: {e}", exception=e)
            if self._wants_connection:
                await self.reset_conversation()

    async def _handle_completion_start_event(self, event_json):
        """Log when a new completion starts."""
        logger.debug("NovaSonicLLM: === completionStart ===")
        await super()._handle_completion_start_event(event_json)

    async def _handle_content_start_event(self, event_json):
        """Track content block info for detecting turn end."""
        content_start = event_json.get("contentStart", {})
        self._current_content_type = content_start.get("type")
        self._current_content_role = content_start.get("role")

        # Parse generationStage from additionalModelFields
        additional = content_start.get("additionalModelFields")
        if additional:
            try:
                fields = json.loads(additional) if isinstance(additional, str) else additional
                self._current_generation_stage = fields.get("generationStage")
            except:
                self._current_generation_stage = None
        else:
            self._current_generation_stage = None

        # Track content block depth
        if not hasattr(self, '_content_depth'):
            self._content_depth = 0
        self._content_depth += 1

        logger.info(
            f"NovaSonicLLM: >>> contentStart [{self._content_depth}] "
            f"type={self._current_content_type} role={self._current_content_role} "
            f"stage={self._current_generation_stage}"
        )
        await super()._handle_content_start_event(event_json)

    async def _handle_text_output_event(self, event_json):
        """Log text output events and emit SPECULATIVE text for transcription."""
        text_output = event_json.get("textOutput", {})
        content = text_output.get("content", "")

        # Log the text
        logger.debug(
            f"NovaSonicLLM:     textOutput type={self._current_content_type} "
            f"role={self._current_content_role} stage={self._current_generation_stage} "
            f"content={content[:80]!r}..."
        )

        # Emit SPECULATIVE ASSISTANT text as TTSTextFrame for transcription
        # This arrives in real-time with audio, unlike FINAL which is delayed 30+ seconds
        if (self._current_content_role == "ASSISTANT" and
            self._current_generation_stage == "SPECULATIVE" and
            content):
            from pipecat.frames.frames import AggregationType
            logger.info(f"NovaSonicLLM: Emitting SPECULATIVE text ({len(content)} chars): {content[:60]}...")
            frame = TTSTextFrame(content, aggregated_by=AggregationType.SENTENCE)
            await self.push_frame(frame)

        await super()._handle_text_output_event(event_json)

    async def _handle_audio_output_event(self, event_json):
        """Log audio output events and capture TTFB on first audio."""
        if not hasattr(self, '_audio_output_count'):
            self._audio_output_count = 0
        self._audio_output_count += 1

        # Stop TTFB metrics on first audio output (this is the "first byte" for speech-to-speech)
        if self._audio_output_count == 1 and self._ttfb_started:
            logger.info("NovaSonicLLM: Stopping TTFB metrics on first audio output")
            await self.stop_ttfb_metrics()
            self._ttfb_started = False

        if self._audio_output_count == 1 or self._audio_output_count % 50 == 0:
            logger.info(
                f"NovaSonicLLM:     audioOutput #{self._audio_output_count} "
                f"role={self._current_content_role}"
            )
        await super()._handle_audio_output_event(event_json)

    async def _handle_content_end_event(self, event_json):
        """Detect when AUDIO ends with END_TURN - this signals the turn is complete.

        Since we're using SPECULATIVE text (which arrives with audio), we use AUDIO END_TURN
        as the turn completion signal instead of waiting for FINAL text.
        """
        content_end = event_json.get("contentEnd", {})
        stop_reason = content_end.get("stopReason", "?")

        # Track content block depth
        if not hasattr(self, '_content_depth'):
            self._content_depth = 0
        depth_before = self._content_depth
        self._content_depth = max(0, self._content_depth - 1)

        logger.debug(
            f"NovaSonicLLM: <<< contentEnd [{depth_before}->{self._content_depth}] "
            f"type={self._current_content_type} role={self._current_content_role} "
            f"stage={self._current_generation_stage} stopReason={stop_reason}"
        )

        # Check for AUDIO with END_TURN - this means the assistant is done speaking
        # Since we capture SPECULATIVE text (which arrives with audio), this is our turn end signal
        if (self._current_content_type == "AUDIO" and
            self._current_content_role == "ASSISTANT" and
            stop_reason == "END_TURN"):
            logger.info(
                f"NovaSonicLLM: *** AUDIO TURN END *** Assistant audio complete - pushing signal"
            )
            await self.push_frame(NovaSonicTextTurnEndFrame())

        # Clear tracking
        self._current_content_type = None
        self._current_content_role = None
        self._current_generation_stage = None

        await super()._handle_content_end_event(event_json)

    async def _handle_completion_end_event(self, event_json):
        """Handle Nova Sonic's completionEnd event by pushing a signal frame."""
        logger.info("NovaSonicLLM: === completionEnd === pushing signal frame")
        await self.push_frame(NovaSonicCompletionEndFrame())


# ============================================================================
# Nova Sonic Turn Gate (Simplified Turn Detection)
# ============================================================================


class NovaSonicTurnGate(FrameProcessor):
    """Simplified turn gate for Nova Sonic using BotStoppedSpeakingFrame.

    This processor replaces the complex NovaSonicTurnEndDetector with a simpler
    approach that mirrors the realtime pipeline:

    1. Accumulates SPECULATIVE text from TTSTextFrame as it arrives
    2. Waits for BotStoppedSpeakingFrame (from NullAudioOutputTransport)
    3. Adds a small delay to ensure all audio has been processed
    4. Triggers the turn-end callback with accumulated text

    This works because:
    - NullAudioOutputTransport paces audio at real-time speed
    - BotStoppedSpeakingFrame fires after 2s of empty audio queue
    - SPECULATIVE text arrives with audio, so it's complete by the time
      BotStoppedSpeakingFrame fires
    """

    def __init__(
        self,
        on_turn_ready: Callable[[str], Any],
        audio_drain_delay: float = 0.5,
        response_timeout_sec: float = 60.0,
        metrics_callback: Optional[Callable[[MetricsFrame], None]] = None,
        on_greeting_started: Optional[Callable[[], None]] = None,
        on_greeting_done: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """Initialize the turn gate.

        Args:
            on_turn_ready: Async callback to invoke when turn is ready to advance.
                          Called with the assistant's response text.
            audio_drain_delay: Seconds to wait after BotStoppedSpeakingFrame before
                              triggering turn end. Default 0.5s.
            response_timeout_sec: Maximum time to wait for any response (fallback).
            metrics_callback: Optional callback for metrics frames.
            on_greeting_started: Optional callback when initial greeting starts.
            on_greeting_done: Optional callback when initial greeting completes.
        """
        super().__init__(**kwargs)
        self._on_turn_ready = on_turn_ready
        self._audio_drain_delay = audio_drain_delay
        self._response_timeout = response_timeout_sec
        self._metrics_callback = metrics_callback

        # Greeting detection
        self._on_greeting_started = on_greeting_started
        self._on_greeting_done = on_greeting_done
        self._greeting_started_signaled = False
        self._greeting_done_signaled = False

        # State tracking
        self._response_text = ""
        self._response_active = False
        self._waiting_for_response = False
        self._turn_end_task: Optional[asyncio.Task] = None
        self._response_timeout_task: Optional[asyncio.Task] = None
        self._processing_turn_end = False
        self._audio_frame_count = 0

    def signal_trigger_sent(self):
        """Called when assistant response is triggered - start response timeout."""
        self._waiting_for_response = True
        self._response_text = ""
        self._audio_frame_count = 0
        logger.info(
            f"[NovaSonicTurnGate] Trigger sent, waiting for response (timeout={self._response_timeout}s)"
        )
        if self._response_timeout_task:
            self._response_timeout_task.cancel()
        self._response_timeout_task = asyncio.create_task(self._check_response_timeout())

    def clear_pending(self):
        """Clear any pending state (e.g., on reconnection)."""
        self._response_text = ""
        self._response_active = False
        self._waiting_for_response = False
        self._processing_turn_end = False
        self._audio_frame_count = 0
        if self._turn_end_task and not self._turn_end_task.done():
            self._turn_end_task.cancel()
            self._turn_end_task = None
        if self._response_timeout_task and not self._response_timeout_task.done():
            self._response_timeout_task.cancel()
            self._response_timeout_task = None

    def reset_for_reconnection(self):
        """Reset state after reconnection."""
        logger.info("[NovaSonicTurnGate] Resetting state for reconnection")
        self.clear_pending()

    async def _delayed_turn_end(self, text: str):
        """Wait for audio to drain, then trigger turn end."""
        try:
            logger.info(f"[NovaSonicTurnGate] Waiting {self._audio_drain_delay}s for audio to drain...")
            await asyncio.sleep(self._audio_drain_delay)
            logger.info(f"[NovaSonicTurnGate] Triggering turn end with transcript ({len(text)} chars)")
            self._processing_turn_end = False
            self._response_active = False
            self._waiting_for_response = False
            await self._on_turn_ready(text)
        except asyncio.CancelledError:
            logger.info("[NovaSonicTurnGate] Turn end cancelled")

    async def _check_response_timeout(self):
        """Check if response started within timeout period."""
        try:
            await asyncio.sleep(self._response_timeout)

            # Guard against concurrent turn completions
            if self._processing_turn_end:
                logger.debug("[NovaSonicTurnGate] Response timeout but already processing, ignoring")
                return

            # If we get here, no response or no BotStoppedSpeakingFrame within timeout
            if self._waiting_for_response or self._response_active:
                self._processing_turn_end = True
                text = self._response_text or "[NO RESPONSE - TIMEOUT]"
                logger.warning(
                    f"[NovaSonicTurnGate] Timeout after {self._response_timeout}s - "
                    f"ending turn with {len(self._response_text)} chars"
                )
                self._response_active = False
                self._waiting_for_response = False
                await self._on_turn_ready(text)
                self._processing_turn_end = False
        except asyncio.CancelledError:
            pass  # Response completed normally

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, accumulate text, and watch for BotStoppedSpeakingFrame."""
        await super().process_frame(frame, direction)

        # Track metrics
        if isinstance(frame, MetricsFrame) and self._metrics_callback:
            self._metrics_callback(frame)

        # Greeting detection: signal greeting started on first BotStartedSpeakingFrame
        # This must happen before signal_trigger_sent() is called (before first turn)
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._greeting_started_signaled and self._on_greeting_started:
                self._greeting_started_signaled = True
                logger.info("[NovaSonicTurnGate] Initial greeting started")
                self._on_greeting_started()

        # Track response lifecycle
        if isinstance(frame, LLMFullResponseStartFrame):
            self._response_active = True
            self._waiting_for_response = False
            self._response_text = ""
            self._audio_frame_count = 0
            # Cancel response timeout since we got a response
            if self._response_timeout_task and not self._response_timeout_task.done():
                self._response_timeout_task.cancel()
                self._response_timeout_task = None
            logger.debug("[NovaSonicTurnGate] Response started")

        # Accumulate SPECULATIVE text from TTSTextFrame
        elif isinstance(frame, TTSTextFrame):
            text = getattr(frame, "text", None)
            if text and (self._waiting_for_response or self._response_active) and not self._processing_turn_end:
                logger.info(f"[NovaSonicTurnGate] Accumulating text ({len(text)} chars): {text[:60]}...")
                self._response_text += text

        # Track audio frames for logging
        elif isinstance(frame, TTSAudioRawFrame):
            self._audio_frame_count += 1

        # Watch for BotStoppedSpeakingFrame - this is the turn end signal
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(
                f"[NovaSonicTurnGate] BotStoppedSpeakingFrame received "
                f"(text={len(self._response_text)} chars, audio_frames={self._audio_frame_count})"
            )

            # Greeting detection: signal greeting done on first BotStoppedSpeakingFrame
            # This must happen before signal_trigger_sent() is called (before first turn)
            if not self._greeting_done_signaled and self._on_greeting_done:
                self._greeting_done_signaled = True
                logger.info("[NovaSonicTurnGate] Initial greeting complete")
                self._on_greeting_done()

            # Guard against concurrent turn completions
            if self._processing_turn_end:
                logger.debug("[NovaSonicTurnGate] Already processing turn end, ignoring")
            elif self._response_active or self._response_text:
                self._processing_turn_end = True
                text = self._response_text or "[audio response - no text captured]"
                # Cancel any existing turn end task
                if self._turn_end_task and not self._turn_end_task.done():
                    self._turn_end_task.cancel()
                # Cancel response timeout
                if self._response_timeout_task and not self._response_timeout_task.done():
                    self._response_timeout_task.cancel()
                    self._response_timeout_task = None
                # Schedule delayed turn end
                self._turn_end_task = asyncio.create_task(self._delayed_turn_end(text))

        await self.push_frame(frame, direction)


# ============================================================================
# Nova Sonic Pipeline
# ============================================================================


class NovaSonicPipeline:
    """Pipeline for AWS Nova Sonic speech-to-speech models.

    Nova Sonic has unique behavior that requires special handling:
    1. Speech-to-speech model: audio in, audio out
    2. Requires 16kHz audio input
    3. Text transcripts arrive AFTER audio (8+ seconds delay)
    4. Requires special "trigger" mechanism to start assistant response
    5. Uses AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION in system instruction
    6. Connection timeout after 8 minutes - handled via automatic reconnection

    This pipeline creates its own LLM service (requires_service=False).
    """

    requires_service = False  # We create our own LLM

    def __init__(self, benchmark):
        """Initialize the pipeline.

        Args:
            benchmark: A BenchmarkConfig instance with turns, tools, and system instruction.
        """
        import os

        self.benchmark = benchmark
        self.turns = benchmark.turns
        self.turn_idx = 0
        self.done = False
        self.recorder = None
        self.task = None
        self.context = None
        self.llm = None
        self.model_name = None
        self._turn_indices = None

        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()

        # Nova Sonic specific
        self.paced_input = None
        self.turn_gate = None  # Simplified turn gate using BotStoppedSpeakingFrame
        self.context_aggregator = None
        self.output_transport = None  # NullAudioOutputTransport for pacing
        self.audio_buffer = None  # AudioBufferProcessor for recording

        # AWS credentials (needed for LLM creation)
        self._aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self._aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        self._aws_region = os.getenv("AWS_REGION", "us-east-1")

    @property
    def effective_turns(self):
        """Get the turns to run (filtered by turn_indices if set)."""
        if self._turn_indices is not None:
            return [self.turns[i] for i in self._turn_indices if i < len(self.turns)]
        return self.turns

    def _get_actual_turn_index(self, effective_index: int) -> int:
        """Convert effective turn index to actual turn index."""
        if self._turn_indices is not None:
            return self._turn_indices[effective_index]
        return effective_index

    def _get_current_turn(self) -> dict:
        """Get the current turn data."""
        return self.effective_turns[self.turn_idx]

    def _get_audio_path_for_turn(self, turn_index: int):
        """Get the audio file path for a turn.

        Prefers benchmark.get_audio_path() if available, falls back to
        the turn's audio_file field.

        Args:
            turn_index: The effective turn index (index into effective_turns).

        Returns:
            Path to audio file as string, or None if not available.
        """
        from pathlib import Path

        # Try benchmark's get_audio_path method first (uses audio_dir)
        if hasattr(self.benchmark, "get_audio_path"):
            actual_index = self._get_actual_turn_index(turn_index)
            path = self.benchmark.get_audio_path(actual_index)
            if path and path.exists():
                return str(path)

        # Fall back to turn's audio_file field
        turn = self.effective_turns[turn_index]
        return turn.get("audio_file")

    async def run(
        self,
        recorder,
        model: str,
        service_class=None,
        service_name=None,
        turn_indices=None,
    ) -> None:
        """Run the complete benchmark.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: Ignored for Nova Sonic (we create our own LLM).
            service_name: Ignored for Nova Sonic (we create our own LLM).
            turn_indices: Optional list of turn indices to run (for debugging).
        """
        import os
        import soundfile as sf

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

        from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
        from multi_turn_eval.transports.paced_input import PacedInputTransport

        self.recorder = recorder
        self.model_name = model
        self._turn_indices = turn_indices

        # Validate AWS credentials
        if not (self._aws_access_key_id and self._aws_secret_access_key):
            raise EnvironmentError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for Nova Sonic"
            )

        # Get system instruction and tools from benchmark
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # For the old Nova Sonic model, we'd need to append AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION.
        # For Nova 2 Sonic, this is not needed - LLMRunFrame triggers response directly.
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

        # Check if we're using the old model (needs trigger instruction)
        is_old_model = "nova-sonic-v1" in model and "nova-2-sonic" not in model
        if is_old_model:
            nova_sonic_system_instruction = (
                f"{system_instruction} "
                f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
            )
        else:
            nova_sonic_system_instruction = system_instruction
        logger.info(f"Using full system instruction ({len(nova_sonic_system_instruction)} chars)")

        # Create Nova Sonic LLM service
        self.llm = NovaSonicLLMServiceWithCompletionSignal(
            secret_access_key=self._aws_secret_access_key,
            access_key_id=self._aws_access_key_id,
            session_token=self._aws_session_token,
            region=self._aws_region,
            model=model if ":" in model else "amazon.nova-sonic-v1:0",
            voice_id="tiffany",
            system_instruction=nova_sonic_system_instruction,
            tools=tools,
            endpointing_sensitivity="HIGH",  # Quick cutoff for faster responses
        )

        # Register function handler
        from pipecat.services.llm_service import FunctionCallParams

        async def function_catchall(params: FunctionCallParams):
            # Create a key for duplicate detection (function_name + args)
            call_key = (params.function_name, str(params.arguments or {}))

            # Check for duplicate tool call
            if call_key in self._seen_tool_calls:
                logger.warning(
                    f"Skipping duplicate tool call: {params.function_name} "
                    f"(tool_call_id={getattr(params, 'tool_call_id', 'unknown')})"
                )
                await params.result_callback({"status": "duplicate_skipped"})
                return

            # Track this call
            self._seen_tool_calls.add(call_key)

            logger.info(f"Function call: {params}")
            result = {"status": "success"}
            await params.result_callback(result)

        self.llm.register_function(None, function_catchall)

        # Create context with greeting trigger message
        # Nova 2 Sonic needs the context to end with a user message to trigger a response.
        # We add a simple greeting trigger that will cause the bot to greet when LLMRunFrame
        # is queued. This message will be transcribed and added to conversation history.
        greeting_trigger = "Hello!"
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": greeting_trigger},
        ]
        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

        # Metrics handler
        def handle_metrics(frame: MetricsFrame):
            from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData

            for md in frame.data:
                if isinstance(md, LLMUsageMetricsData):
                    self.recorder.record_usage_metrics(md.value, getattr(md, "model", None))
                elif isinstance(md, TTFBMetricsData):
                    self.recorder.record_ttfb(md.value)

        # End of turn callback
        async def end_of_turn(assistant_text: str):
            if self.done:
                logger.info("end_of_turn called but already done")
                return

            # Record this turn
            self.recorder.write_turn(
                user_text=self._get_current_turn().get("input", ""),
                assistant_text=assistant_text,
                reconnection_count=self._turn_reconnection_count,
            )

            # Reset reconnection counter after recording
            self._turn_reconnection_count = 0

            # Reset reconnect counter on successful turn completion
            self.llm.reset_reconnect_counter()

            self.turn_idx += 1

            # Reset tool call tracking for the new turn
            self._seen_tool_calls.clear()

            if self.turn_idx < len(self.effective_turns):
                actual_idx = self._get_actual_turn_index(self.turn_idx)
                self.recorder.start_turn(actual_idx)
                logger.info(f"Starting turn {self.turn_idx}: {self._get_current_turn()['input'][:50]}...")
                await self._queue_next_turn()
            else:
                logger.info("Conversation complete!")
                self.recorder.write_summary()
                self.done = True
                await self.task.cancel()

        # Greeting events - used to wait for initial greeting before sending user audio
        self._greeting_started: asyncio.Event = asyncio.Event()
        self._greeting_done: asyncio.Event = asyncio.Event()

        # Create simplified turn gate (replaces complex NovaSonicTurnEndDetector)
        self.turn_gate = NovaSonicTurnGate(
            on_turn_ready=end_of_turn,
            audio_drain_delay=0.5,
            response_timeout_sec=60.0,
            metrics_callback=handle_metrics,
            on_greeting_started=lambda: self._greeting_started.set(),
            on_greeting_done=lambda: self._greeting_done.set(),
        )

        # Create local VAD analyzer for user speech detection (measurement only)
        # This emits VADUserStartedSpeakingFrame/VADUserStoppedSpeakingFrame
        # which trigger audio tags in NullAudioOutputTransport for timing analysis.
        # Server-side VAD handles actual turn triggering.
        vad_params = VADParams(
            start_secs=0.2,  # Emit VADUserStartedSpeaking 0.2s after speech starts
            stop_secs=0.8,   # Emit VADUserStoppedSpeaking 0.8s after speech ends
        )
        user_vad = ResamplingSileroVAD(params=vad_params)
        logger.info(
            f"[NovaSonic] User VAD config: start_secs={vad_params.start_secs}, "
            f"stop_secs={vad_params.stop_secs}"
        )

        # Create paced input transport (Nova Sonic requires 16kHz)
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_in_passthrough=True,
            vad_analyzer=user_vad,
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
            wait_for_ready=True,  # Wait for LLM to be ready before sending audio
        )

        # Set BOT_VAD_STOP_SECS to 2.0s for reliable end-of-response detection
        # This must be set BEFORE creating NullAudioOutputTransport
        import pipecat.transports.base_output as base_output_module
        base_output_module.BOT_VAD_STOP_SECS = 2.0
        logger.info("[NovaSonic] Set BOT_VAD_STOP_SECS to 2.0s for reliable turn detection")

        # Create null output transport to generate BotStoppedSpeakingFrame
        # Nova Sonic outputs 24kHz audio
        self.output_transport = NullAudioOutputTransport(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=24000,  # Nova Sonic output is 24kHz
            )
        )

        # Create audio buffer processor for recording
        # Use 24kHz to match Nova Sonic output (user audio will be upsampled from 16kHz)
        # NullAudioOutputTransport is the "source of truth" for wall-clock aligned recording.
        # It inserts silence for any gaps > 10ms in BOTH user and bot audio tracks.
        # WallClockAlignedAudioBufferProcessor just accumulates the continuous streams.
        logger.info("[NovaSonic] Creating WallClockAlignedAudioBufferProcessor with sample_rate=24000, num_channels=2")
        self.audio_buffer = WallClockAlignedAudioBufferProcessor(
            sample_rate=24000,
            num_channels=2,  # Stereo: user on left channel, bot on right channel
        )

        # Register event handler to save audio when track data is ready
        @self.audio_buffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            processor, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
        ):
            """Save conversation audio with user and bot on separate channels."""
            logger.info(
                f"[NovaSonic AudioRecording] on_track_audio_data triggered: "
                f"user={len(user_audio)} bytes, bot={len(bot_audio)} bytes, "
                f"{sample_rate}Hz, {num_channels}ch"
            )

            # Get run directory from recorder
            if not self.recorder or not hasattr(self.recorder, "run_dir"):
                logger.error("[NovaSonic AudioRecording] Cannot save audio: no recorder or run_dir available")
                return

            # Convert to numpy for processing
            user_np = np.frombuffer(user_audio, dtype=np.int16)
            bot_np = np.frombuffer(bot_audio, dtype=np.int16)

            # Pad shorter track to match longer
            max_len = max(len(user_np), len(bot_np))
            if len(user_np) < max_len:
                user_np = np.concatenate([user_np, np.zeros(max_len - len(user_np), dtype=np.int16)])
            if len(bot_np) < max_len:
                bot_np = np.concatenate([bot_np, np.zeros(max_len - len(bot_np), dtype=np.int16)])

            # Interleave for stereo: user=left, bot=right
            stereo = np.zeros(max_len * 2, dtype=np.int16)
            stereo[0::2] = user_np
            stereo[1::2] = bot_np

            output_path = self.recorder.run_dir / "conversation.wav"
            logger.info(f"[NovaSonic AudioRecording] Saving conversation audio to {output_path}")

            try:
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(2)  # Stereo
                    wf.setsampwidth(2)  # 16-bit audio = 2 bytes per sample
                    wf.setframerate(sample_rate)
                    wf.writeframes(stereo.tobytes())

                # Calculate duration for logging
                duration_secs = max_len / sample_rate
                file_size_mb = (max_len * 2 * 2) / (1024 * 1024)
                logger.info(
                    f"[NovaSonic AudioRecording] Saved conversation audio: {output_path} "
                    f"({duration_secs:.1f}s, {file_size_mb:.2f}MB)"
                )
            except Exception as e:
                logger.exception(f"[NovaSonic AudioRecording] Failed to save audio: {e}")

        # Track interrupted turn state for reconnection handling
        self._interrupted_turn_text = ""
        self._was_responding_at_disconnect = False
        self._turn_reconnection_count = 0

        # Set up reconnection callbacks
        def on_reconnecting():
            logger.info("Reconnection starting: pausing audio input and resetting turn gate")
            self.paced_input.pause()

            # Capture accumulated text BEFORE reset - capture if ANY text accumulated,
            # not just when response_active is True (fixes text loss during early reconnection)
            accumulated_text = self.turn_gate._response_text or ""
            self._was_responding_at_disconnect = (
                self.turn_gate._response_active
                or self.turn_gate._waiting_for_response
                or len(accumulated_text) > 0
            )
            self._interrupted_turn_text = accumulated_text

            if accumulated_text:
                logger.warning(
                    f"Turn {self.turn_idx} interrupted. "
                    f"Captured {len(accumulated_text)} chars before reset. "
                    f"response_active={self.turn_gate._response_active}, "
                    f"waiting={self.turn_gate._waiting_for_response}"
                )

            self.turn_gate.reset_for_reconnection()
            self._turn_reconnection_count += 1

        def on_reconnected():
            logger.info("Reconnection complete: waiting 2s before resuming audio")
            import threading
            import asyncio

            def delayed_resume():
                import time

                time.sleep(2.0)
                logger.info("Delayed audio resume: signaling ready now")
                self.paced_input.signal_ready()

                # If we were mid-response when disconnected, we need to:
                # 1. Complete the interrupted turn (with whatever text was collected)
                # 2. This will trigger _queue_next_turn() to queue the next turn's audio
                if self._was_responding_at_disconnect:
                    logger.warning(
                        f"Handling interrupted turn {self.turn_idx} after reconnection. "
                        f"Text captured: {len(self._interrupted_turn_text)} chars"
                    )
                    time.sleep(0.5)  # Small delay to let signal_ready settle

                    # Bridge to async: run end_of_turn in a new event loop
                    text = self._interrupted_turn_text or "[Turn interrupted by 8-minute reconnection]"

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(end_of_turn(text))
                        logger.info(f"Successfully advanced to turn {self.turn_idx} after interrupted turn")
                    except Exception as e:
                        logger.error(f"Error handling interrupted turn: {e}")
                    finally:
                        loop.close()

                    # Reset state
                    self._was_responding_at_disconnect = False
                    self._interrupted_turn_text = ""
                else:
                    # We weren't mid-response, but we may have queued audio that was lost
                    # during reconnection (paced_input.pause() clears the queue).
                    # Re-queue the current turn's audio if we haven't completed it yet.
                    if self.turn_idx < len(self.turns):
                        logger.info(
                            f"Re-queuing turn {self.turn_idx} after reconnection "
                            f"(was not mid-response)"
                        )
                        time.sleep(0.5)  # Small delay to let signal_ready settle

                        # Bridge to async: run _queue_next_turn in a new event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self._requeue_current_turn())
                            logger.info(
                                f"Successfully re-queued turn {self.turn_idx} after reconnection"
                            )
                        except Exception as e:
                            logger.error(f"Error re-queuing turn after reconnection: {e}")
                        finally:
                            loop.close()

            threading.Thread(target=delayed_resume, daemon=True).start()

        def on_retriggered():
            logger.info("Assistant response re-triggered after reconnection")
            self.turn_gate.signal_trigger_sent()

        async def on_max_reconnects_exceeded():
            logger.error("Max reconnect attempts exceeded - terminating pipeline")
            self.done = True
            self.recorder.write_summary()
            await self.task.cancel()

        self.llm._on_reconnecting = on_reconnecting
        self.llm._on_reconnected = on_reconnected
        self.llm._on_retriggered = on_retriggered
        self.llm._on_max_reconnects_exceeded = on_max_reconnects_exceeded

        # Recorder accessor for ToolCallRecorder
        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Build pipeline
        # Structure mirrors the realtime pipeline:
        # - turn_gate accumulates text and waits for BotStoppedSpeakingFrame
        # - output_transport paces audio and generates BotStoppedSpeakingFrame
        # - audio_buffer records the audio after pacing
        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.llm,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.turn_gate,
                self.context_aggregator.assistant(),
                self.output_transport,
                self.audio_buffer,
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=120,  # Longer timeout for Nova Sonic's delayed responses and reconnection
            idle_timeout_frames=(TTSAudioRawFrame, TTSTextFrame, InputAudioRawFrame, MetricsFrame),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        # Initialize first turn
        actual_first_idx = self._get_actual_turn_index(0)
        self.recorder.start_turn(actual_first_idx)

        # Queue first turn
        asyncio.create_task(self._queue_first_turn())

        # Run pipeline
        runner = PipelineRunner(handle_sigint=True)
        await runner.run(self.task)

    async def _queue_first_turn(self, delay: float = 1.0):
        """Queue the first turn - send user question as AUDIO, then trigger."""
        import soundfile as sf

        await asyncio.sleep(delay)

        # Start audio recording
        logger.info("[NovaSonic] Starting audio recording")
        await self.audio_buffer.start_recording()

        # Set recording baselines to unblock the paced_input feeder thread
        # This must happen after start_recording() to ensure all components
        # use the same T=0 baseline for wall-clock synchronized audio
        self.output_transport.reset_recording_baseline(
            recording_sample_rate=self.audio_buffer._init_sample_rate
        )
        self.paced_input.set_recording_baseline()
        logger.info("[NovaSonic] Recording baselines set")

        # Enable tagging for the initial greeting audio.
        # Normally tags are triggered by VADUserStoppedSpeakingFrame, but the
        # greeting happens before any user speech, so we enable it explicitly.
        self.output_transport.enable_greeting_tag()

        # Queue LLMRunFrame to trigger initial greeting
        # The context has a "Hello!" user message that triggers Nova Sonic to greet
        logger.info("Queuing LLMRunFrame to trigger initial greeting...")
        await self.task.queue_frames([LLMRunFrame()])

        # Wait for connection to establish
        await asyncio.sleep(1.0)

        # Signal LLM ready to receive audio
        logger.info("Signaling LLM ready for audio...")
        self.paced_input.signal_ready()

        # Wait for initial greeting to complete before playing user audio
        # This ensures the greeting isn't interrupted by user audio
        greeting_start_timeout = 8.0  # seconds to wait for bot to start speaking
        greeting_complete_timeout = 30.0  # seconds to wait for bot to stop speaking

        logger.info(f"[NovaSonic] Waiting up to {greeting_start_timeout}s for initial greeting to start...")
        greeting_occurred = False
        try:
            await asyncio.wait_for(self._greeting_started.wait(), timeout=greeting_start_timeout)
            logger.info(f"[NovaSonic] Bot started greeting, waiting up to {greeting_complete_timeout}s for completion...")
            try:
                await asyncio.wait_for(self._greeting_done.wait(), timeout=greeting_complete_timeout)
                logger.info("[NovaSonic] Initial greeting complete, proceeding with user audio")
                greeting_occurred = True
            except asyncio.TimeoutError:
                logger.error(
                    f"[NovaSonic] Greeting did not complete within {greeting_complete_timeout}s timeout. "
                    "Bot started speaking but never stopped."
                )
                greeting_occurred = True
        except asyncio.TimeoutError:
            logger.warning("[NovaSonic] No greeting started within timeout, model doesn't greet - proceeding with user audio")

        # If a greeting occurred, clear the turn gate state
        # The greeting should not be recorded as part of turn 0's response
        if greeting_occurred:
            logger.info("[NovaSonic] Clearing TurnGate state after greeting")
            self.turn_gate.clear_pending()

        # Queue user's question as AUDIO
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        if audio_path:
            # Calculate audio duration
            data, sr = sf.read(audio_path, dtype="int16")
            audio_duration_sec = len(data) / sr
            logger.info(f"Audio duration: {audio_duration_sec:.2f}s")

            self.paced_input.enqueue_wav_file(audio_path)
            logger.info(f"Queued user question audio: {audio_path}")

            # Signal trigger as soon as we start sending audio.
            # This tells the turn detector to start accepting text.
            # We don't send audio until previous turn ended, so we can safely
            # clear buffers and accept any incoming text from this point.
            self.turn_gate.signal_trigger_sent()

            # Wait for audio to finish streaming (plus small buffer)
            wait_time = audio_duration_sec + 0.5
            logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
            await asyncio.sleep(wait_time)

            # Start TTFB timing now that user audio is complete
            self.recorder.reset_ttfb()
            await self.llm.start_ttfb_for_user_audio_complete()

            # Trigger assistant response
            # Nova Sonic v1 needs explicit audio trigger, Nova 2 Sonic auto-triggers via VAD
            logger.info("Triggering assistant response after user audio...")
            if self.llm._is_assistant_response_trigger_needed():
                await self.llm.trigger_assistant_response()
            else:
                # Nova 2 Sonic - send LLMRunFrame to trigger context push
                logger.info("Using LLMRunFrame for Nova 2 Sonic")
                await self.task.queue_frames([LLMRunFrame()])
            logger.info("Triggered assistant response")
        else:
            logger.error("No audio file for first turn - Nova Sonic requires audio input!")
            await self.task.cancel()

    async def _queue_next_turn(self):
        """Queue audio for the next turn."""
        import soundfile as sf
        from pathlib import Path

        from pipecat.frames.frames import LLMMessagesAppendFrame

        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)

        if audio_path:
            try:
                # Calculate audio duration
                data, sr = sf.read(audio_path, dtype="int16")
                audio_duration_sec = len(data) / sr
                logger.info(f"Audio duration for turn {self.turn_idx}: {audio_duration_sec:.2f}s")

                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued audio for turn {self.turn_idx}")

                # Signal trigger as soon as we start sending audio.
                # This tells the turn detector to start accepting text.
                self.turn_gate.signal_trigger_sent()

                # Wait for audio to finish streaming
                wait_time = audio_duration_sec + 0.5
                logger.info(f"Waiting {wait_time:.2f}s for audio to finish streaming...")
                await asyncio.sleep(wait_time)

                # Start TTFB timing
                self.recorder.reset_ttfb()
                await self.llm.start_ttfb_for_user_audio_complete()

                # Trigger assistant response
                # Nova Sonic v1 needs explicit audio trigger, Nova 2 Sonic auto-triggers via VAD
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    # Nova 2 Sonic - send LLMRunFrame to trigger context push
                    logger.info("Using LLMRunFrame for Nova 2 Sonic")
                    await self.task.queue_frames([LLMRunFrame()])
                logger.info(f"Triggered assistant response for turn {self.turn_idx}")
            except Exception as e:
                logger.exception(f"Failed to queue audio for turn {self.turn_idx}: {e}")
                # Fall back to text
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
                await asyncio.sleep(0.5)
                # Trigger assistant response (model-specific)
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    await self.task.queue_frames([LLMRunFrame()])
                self.turn_gate.signal_trigger_sent()
        else:
            # No audio file, use text
            await self.task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": turn["input"]}],
                        run_llm=False,
                    )
                ]
            )
            await asyncio.sleep(0.5)
            # Trigger assistant response (model-specific)
            if self.llm._is_assistant_response_trigger_needed():
                await self.llm.trigger_assistant_response()
            else:
                await self.task.queue_frames([LLMRunFrame()])
            self.turn_gate.signal_trigger_sent()

    async def _requeue_current_turn(self):
        """Re-queue audio for the current turn after reconnection.

        This is called when reconnection happens while waiting for a response
        (but not mid-response). The audio queue was cleared during reconnection,
        so we need to re-send the audio for the current turn.

        Unlike _queue_next_turn, this skips the 5-second initial wait since
        we're recovering from a reconnection and want to resume quickly.
        """
        import soundfile as sf

        from pipecat.frames.frames import LLMMessagesAppendFrame

        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)

        if audio_path:
            try:
                # Calculate audio duration
                data, sr = sf.read(audio_path, dtype="int16")
                audio_duration_sec = len(data) / sr
                logger.info(
                    f"[Reconnect] Audio duration for turn {self.turn_idx}: "
                    f"{audio_duration_sec:.2f}s"
                )

                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"[Reconnect] Queued audio for turn {self.turn_idx}")

                # Signal trigger as soon as we start sending audio
                self.turn_gate.signal_trigger_sent()

                # Wait for audio to finish streaming
                wait_time = audio_duration_sec + 0.5
                logger.info(
                    f"[Reconnect] Waiting {wait_time:.2f}s for audio to finish streaming..."
                )
                await asyncio.sleep(wait_time)

                # Start TTFB timing
                self.recorder.reset_ttfb()
                await self.llm.start_ttfb_for_user_audio_complete()

                # Trigger assistant response
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    logger.info("[Reconnect] Using LLMRunFrame for Nova 2 Sonic")
                    await self.task.queue_frames([LLMRunFrame()])
                logger.info(f"[Reconnect] Triggered assistant response for turn {self.turn_idx}")
            except Exception as e:
                logger.exception(
                    f"[Reconnect] Failed to queue audio for turn {self.turn_idx}: {e}"
                )
                # Fall back to text
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
                await asyncio.sleep(0.5)
                if self.llm._is_assistant_response_trigger_needed():
                    await self.llm.trigger_assistant_response()
                else:
                    await self.task.queue_frames([LLMRunFrame()])
                self.turn_gate.signal_trigger_sent()
        else:
            # No audio file, use text
            logger.info(f"[Reconnect] No audio file for turn {self.turn_idx}, using text")
            await self.task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": turn["input"]}],
                        run_llm=False,
                    )
                ]
            )
            await asyncio.sleep(0.5)
            if self.llm._is_assistant_response_trigger_needed():
                await self.llm.trigger_assistant_response()
            else:
                await self.task.queue_frames([LLMRunFrame()])
            self.turn_gate.signal_trigger_sent()
