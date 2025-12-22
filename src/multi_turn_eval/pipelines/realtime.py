"""Realtime pipeline for OpenAI Realtime and Gemini Live models.

This pipeline works with speech-to-speech models that use audio input/output:
- OpenAI Realtime (gpt-realtime)
- Gemini Live (gemini-*-native-audio-*)
- Ultravox (ultravox-v0.7)

Pipeline:
    paced_input → context_aggregator.user() → transcript.user() →
    llm → ToolCallRecorder → assistant_shim → audio_buffer → context_aggregator.assistant()
"""

import asyncio
import os
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionMessage,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from multi_turn_eval.processors.audio_buffer import WallClockAlignedAudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.ultravox.llm import OneShotInputParams
from pipecat.transports.base_transport import TransportParams

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
from multi_turn_eval.processors.tts_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)
from multi_turn_eval.transports.null_audio_output import NullAudioOutputTransport
from multi_turn_eval.transports.paced_input import PacedInputTransport


class TurnGate(FrameProcessor):
    """Gates turn advancement until bot finishes speaking.

    This processor coordinates between transcript completion and audio playback:
    1. Stores the pending assistant transcript when received (on TTSStoppedFrame)
    2. Waits for BotStoppedSpeakingFrame (from NullAudioOutputTransport)
    3. Adds a small delay to ensure all audio has been processed
    4. Only then triggers the turn-end callback

    This prevents the next turn's audio from being sent while the bot is
    still "speaking" (outputting audio frames).
    """

    def __init__(self, on_turn_ready: Callable[[str], Any], audio_drain_delay: float = 0.5, **kwargs):
        """Initialize the turn gate.

        Args:
            on_turn_ready: Async callback to invoke when turn is ready to advance.
                          Called with the assistant's response text.
            audio_drain_delay: Seconds to wait after BotStoppedSpeakingFrame before
                              triggering turn end. This allows remaining audio frames
                              to drain through the pipeline. Default 0.5s works well
                              when BOT_VAD_STOP_SECS is increased to 2s.
        """
        super().__init__(**kwargs)
        self._on_turn_ready = on_turn_ready
        self._audio_drain_delay = audio_drain_delay
        self._pending_transcript: Optional[str] = None
        self._bot_speaking = False
        self._turn_end_task: Optional[asyncio.Task] = None

    def set_pending_transcript(self, text: str):
        """Store transcript received from assistant_shim.

        Called by the transcript handler when assistant response is complete.
        The turn won't advance until BotStoppedSpeakingFrame is received.
        """
        logger.info(f"[TurnGate] Storing pending transcript ({len(text)} chars)")
        self._pending_transcript = text

    def clear_pending(self):
        """Clear any pending transcript (e.g., on reconnection)."""
        self._pending_transcript = None
        if self._turn_end_task and not self._turn_end_task.done():
            self._turn_end_task.cancel()
            self._turn_end_task = None

    async def _delayed_turn_end(self, text: str):
        """Wait for audio to drain, then trigger turn end."""
        try:
            logger.info(f"[TurnGate] Waiting {self._audio_drain_delay}s for audio to drain...")
            await asyncio.sleep(self._audio_drain_delay)
            logger.info(f"[TurnGate] Triggering turn end with transcript ({len(text)} chars)")
            await self._on_turn_ready(text)
        except asyncio.CancelledError:
            logger.info("[TurnGate] Turn end cancelled (likely bot started speaking again)")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Watch for BotStoppedSpeakingFrame to trigger turn advancement."""
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[TurnGate] BotStoppedSpeakingFrame received")
            self._bot_speaking = False

            # If we have a pending transcript, schedule turn end after delay
            if self._pending_transcript is not None:
                text = self._pending_transcript
                self._pending_transcript = None
                # Cancel any existing turn end task
                if self._turn_end_task and not self._turn_end_task.done():
                    self._turn_end_task.cancel()
                # Schedule delayed turn end
                self._turn_end_task = asyncio.create_task(self._delayed_turn_end(text))

        await self.push_frame(frame, direction)


class GeminiLiveLLMServiceWithReconnection(GeminiLiveLLMService):
    """Extended Gemini Live service that exposes reconnection events.

    The base GeminiLiveLLMService handles reconnection internally when the
    10-minute session timeout occurs, but doesn't expose events for external
    coordination. This subclass:

    1. Calls on_reconnecting callback before disconnecting
    2. Calls on_reconnected callback after reconnecting
    3. Tracks whether we were in the middle of receiving a response

    This allows the test harness to:
    - Pause audio input during reconnection
    - Re-queue the interrupted turn's audio after reconnection
    - Reset turn tracking state
    """

    def __init__(
        self,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """Initialize with optional reconnection callbacks.

        Args:
            on_reconnecting: Called before disconnecting during reconnection.
                            Use this to pause audio input and save state.
            on_reconnected: Called after reconnection completes.
                           Use this to resume audio input and re-queue interrupted turn.
        """
        super().__init__(**kwargs)
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        self._reconnecting = False

    def is_reconnecting(self) -> bool:
        """Check if currently in the middle of a reconnection."""
        return self._reconnecting

    async def _reconnect(self):
        """Override to call callbacks before/after reconnection."""
        self._reconnecting = True

        # Call on_reconnecting callback
        if self._on_reconnecting:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnecting callback")
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        # Call parent reconnect implementation
        try:
            await super()._reconnect()
        finally:
            self._reconnecting = False

        # Call on_reconnected callback
        if self._on_reconnected:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnected callback")
                self._on_reconnected()
            except Exception as e:
                logger.warning(f"Error in on_reconnected callback: {e}")


class LLMFrameLogger(FrameProcessor):
    """Logs every frame emitted by the LLM stage and captures TTFB metrics."""

    def __init__(self, recorder_accessor):
        super().__init__()
        self._recorder_accessor = recorder_accessor

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"[LLM→] {frame.__class__.__name__} ({direction})")
        # Capture TTFB from MetricsFrame for realtime/live models
        if isinstance(frame, MetricsFrame):
            for md in frame.data:
                if isinstance(md, TTFBMetricsData):
                    recorder = self._recorder_accessor()
                    if recorder:
                        recorder.record_ttfb(md.value)
        await self.push_frame(frame, direction)


class RealtimePipeline(BasePipeline):
    """Pipeline for OpenAI Realtime and Gemini Live models.

    This pipeline handles speech-to-speech models with:
    - Paced audio input at realtime pace
    - Server-side VAD for turn detection
    - Transcript-based end-of-turn detection
    - Reconnection handling for Gemini Live 10-minute timeout
    """

    requires_service = True

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.context_aggregator = None
        self.paced_input = None
        self.transcript = None
        self.assistant_shim = None
        self.audio_buffer: Optional[AudioBufferProcessor] = None
        self.turn_gate: Optional[TurnGate] = None
        self.output_transport: Optional[NullAudioOutputTransport] = None
        self.current_turn_audio_path: Optional[str] = None
        self.needs_turn_retry: bool = False
        self.reconnection_grace_until: float = 0

    def _is_gemini_live(self) -> bool:
        """Check if current model is Gemini Live."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return (m.startswith("gemini") or m.startswith("models/gemini")) and (
            "live" in m or "native-audio" in m
        )

    def _is_openai_realtime(self) -> bool:
        """Check if current model is OpenAI Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "realtime" in m and m.startswith("gpt")

    def _is_ultravox_realtime(self) -> bool:
        """Check if current model is Ultravox Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "ultravox" in m

    def _get_audio_path_for_turn(self, turn_index: int) -> Optional[str]:
        """Get the audio file path for a turn.

        Prefers benchmark.get_audio_path() if available, falls back to
        the turn's audio_file field.

        Args:
            turn_index: The effective turn index (index into effective_turns).

        Returns:
            Path to audio file as string, or None if not available.
        """
        # Try benchmark's get_audio_path method first (uses audio_dir)
        if hasattr(self.benchmark, "get_audio_path"):
            actual_index = self._get_actual_turn_index(turn_index)
            path = self.benchmark.get_audio_path(actual_index)
            if path and path.exists():
                return str(path)

        # Fall back to turn's audio_file field
        turn = self.effective_turns[turn_index]
        return turn.get("audio_file")

    def _create_llm(self, service_class: Optional[type], model: str) -> FrameProcessor:
        """Create LLM service with proper configuration for realtime models.

        For OpenAI Realtime, we must pass session_properties with turn_detection
        config at construction time. The server-side VAD settings prevent
        client-side interruptions from truncating responses.

        For Gemini Live, we use GeminiLiveLLMServiceWithReconnection and pass
        VAD parameters through the input params.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        class_name = service_class.__name__
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        if "OpenAIRealtime" in class_name:
            # OpenAI Realtime: Configure server-side VAD to prevent interruptions
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")

            # Configure audio input with VAD settings
            # turn_detection must be inside audio.input, not at SessionProperties top level
            # COMMENTED OUT FOR TESTING - comparing with/without custom VAD settings
            # audio_config = rt_events.AudioConfiguration(
            #     input=rt_events.AudioInput(
            #         turn_detection=rt_events.TurnDetection(
            #             type="server_vad",
            #             threshold=0.5,
            #             prefix_padding_ms=300,
            #             silence_duration_ms=1500,
            #         )
            #     )
            # )

            session_props = rt_events.SessionProperties(
                instructions=system_instruction,
                tools=tools,
                # audio=audio_config,  # Using default VAD settings
            )
            return service_class(
                api_key=api_key,
                model=model,
                system_instruction=system_instruction,
                session_properties=session_props,
            )
        elif "UltravoxRealtime" in class_name:
            # Ultravox Realtime: Use OneShotInputParams
            api_key = os.getenv("ULTRAVOX_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ULTRAVOX_API_KEY environment variable is required"
                )

            params = OneShotInputParams(
                api_key=api_key,
                system_prompt=system_instruction,
                temperature=1.0,
                model=model,
            )
            return service_class(
                params=params,
                one_shot_selected_tools=tools,
            )
        else:
            # For Gemini Live and others, use base class implementation
            return super()._create_llm(service_class, model)

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # Both OpenAI Realtime and Gemini Live read the system instruction from
        # an LLMContextFrame. The pipecat service extracts the system message
        # and applies it via session properties (OpenAI) or context (Gemini).
        messages = [{"role": "system", "content": system_instruction}]

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

    def _setup_llm(self) -> None:
        """Configure LLM and set up reconnection callbacks for Gemini Live."""
        self.llm.register_function(None, self._function_catchall)

        # Set up reconnection callbacks for Gemini Live
        if self._is_gemini_live() and isinstance(self.llm, GeminiLiveLLMServiceWithReconnection):
            self.llm._on_reconnecting = self._on_gemini_reconnecting
            self.llm._on_reconnected = self._on_gemini_reconnected

    def _on_gemini_reconnecting(self):
        """Called when Gemini Live starts reconnecting due to session timeout."""
        logger.info(f"Gemini reconnecting: pausing audio, turn {self.turn_idx} will be retried")
        self.needs_turn_retry = True
        # Pause audio input to avoid sending audio during reconnection
        self.paced_input.pause()
        # Clear the transcript buffer to discard partial responses from before reconnection
        self.assistant_shim.clear_buffer()
        # Set grace period to ignore stale TTSStoppedFrame events that arrive after reconnection
        self.reconnection_grace_until = time.monotonic() + 10.0
        logger.info(
            f"Set reconnection grace period until {self.reconnection_grace_until}"
        )

    def _on_gemini_reconnected(self):
        """Called when Gemini Live reconnection completes."""
        logger.info(f"Gemini reconnected: scheduling turn {self.turn_idx} retry")
        # Resume audio input
        self.paced_input.signal_ready()
        # Schedule a task to re-queue the current turn's audio after a short delay
        asyncio.create_task(self._retry_current_turn_after_reconnection())

    async def _retry_current_turn_after_reconnection(self):
        """Re-queue the current turn's audio after reconnection."""
        if not self.needs_turn_retry:
            logger.info("No turn retry needed")
            return

        logger.info(f"Waiting 2s for connection to stabilize before retrying turn {self.turn_idx}")
        await asyncio.sleep(2.0)

        # Check if we still need to retry
        if not self.needs_turn_retry:
            logger.info("Turn retry cancelled (turn completed normally)")
            return

        # Get the audio path for the current turn
        audio_path = self.current_turn_audio_path or self._get_audio_path_for_turn(self.turn_idx)
        if audio_path:
            logger.info(f"Re-queuing audio for turn {self.turn_idx}: {audio_path}")
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                self.needs_turn_retry = False
                logger.info(f"Successfully re-queued audio for turn {self.turn_idx}")
                # Wait for audio to finish then clear grace period
                await asyncio.sleep(5.0)
                self.reconnection_grace_until = 0
                logger.info("Cleared reconnection grace period - accepting new transcript updates")
            except Exception as e:
                logger.exception(f"Failed to re-queue audio for turn {self.turn_idx}: {e}")
        else:
            logger.warning(f"No audio path available for turn {self.turn_idx}, falling back to text")
            # Fall back to text
            await self.task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": turn["input"]}],
                        run_llm=False,
                    )
                ]
            )
            self.needs_turn_retry = False
            await asyncio.sleep(3.0)
            self.reconnection_grace_until = 0
            logger.info("Cleared reconnection grace period (text fallback)")

    def _build_task(self) -> None:
        """Build the pipeline with paced input and transcript processors."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Determine sample rate from first audio file
        default_sr = 24000
        t0_audio = self._get_audio_path_for_turn(0)
        if t0_audio:
            try:
                _, t0_sr = sf.read(t0_audio, dtype="int16", always_2d=True)
                default_sr = int(t0_sr)
            except Exception as e:
                logger.warning(f"Could not read sample rate from {t0_audio}: {e}")

        # Create paced input transport
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=default_sr,
            audio_in_channels=1,
            audio_in_passthrough=True,
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
        )

        # Create transcript processors
        self.transcript = TranscriptProcessor()
        self.assistant_shim = TTSStoppedAssistantTranscriptProcessor()

        # Create audio buffer processor for recording both user and bot audio
        # WallClockAlignedAudioBufferProcessor ensures both tracks are aligned to wall-clock time:
        # - User track gets leading silence from recording_start to first user audio
        # - Bot track gets leading silence via NullAudioOutputTransport
        logger.info(f"[AudioRecording] Creating WallClockAlignedAudioBufferProcessor with sample_rate={default_sr}, num_channels=2")
        self.audio_buffer = WallClockAlignedAudioBufferProcessor(
            sample_rate=default_sr,
            num_channels=2,  # Stereo: user on left channel, bot on right channel
        )

        # Register event handler to save audio when track data is ready
        @self.audio_buffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            processor, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
        ):
            """Save conversation audio with user and bot on separate channels."""
            logger.info(
                f"[AudioRecording] on_track_audio_data triggered: "
                f"user={len(user_audio)} bytes, bot={len(bot_audio)} bytes, "
                f"{sample_rate}Hz, {num_channels}ch"
            )

            # Get run directory from recorder
            if not self.recorder or not hasattr(self.recorder, "run_dir"):
                logger.error("[AudioRecording] Cannot save audio: no recorder or run_dir available")
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
            logger.info(f"[AudioRecording] Saving conversation audio to {output_path}")

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
                    f"[AudioRecording] Saved conversation audio: {output_path} "
                    f"({duration_secs:.1f}s, {file_size_mb:.2f}MB)"
                )

                # Log silence insertion statistics from output transport
                if self.output_transport is not None and hasattr(self.output_transport, 'log_recording_summary'):
                    self.output_transport.log_recording_summary()
            except Exception as e:
                logger.exception(f"[AudioRecording] Failed to save audio: {e}")

        # Register event handler for transcript updates
        # Note: We store the transcript but wait for BotStoppedSpeakingFrame before advancing turn
        @self.assistant_shim.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            # Check grace period
            if time.monotonic() < self.reconnection_grace_until:
                logger.warning(
                    f"Ignoring transcript update during reconnection grace period "
                    f"(until {self.reconnection_grace_until})"
                )
                return

            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage) and getattr(msg, "role", None) == "assistant":
                    timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    line = f"{timestamp}{msg.role}: {msg.content}"
                    logger.info(f"Transcript: {line}")
                    # Clear retry flag - turn completed successfully
                    self.needs_turn_retry = False
                    # Store transcript in turn_gate; it will trigger _on_turn_end
                    # when BotStoppedSpeakingFrame is received
                    self.turn_gate.set_pending_transcript(msg.content)

        # Create TurnGate to coordinate transcript with audio playback completion
        self.turn_gate = TurnGate(on_turn_ready=self._on_turn_end)

        # Create null output transport to generate BotStoppedSpeakingFrame
        # This tracks when the bot finishes "speaking" (outputting audio)
        # Increase the silence threshold from 0.35s to 2s to handle LLM pauses during generation
        import pipecat.transports.base_output as base_output_module

        base_output_module.BOT_VAD_STOP_SECS = 2.0
        logger.info("[AudioRecording] Set BOT_VAD_STOP_SECS to 2.0s for more reliable turn detection")

        self.output_transport = NullAudioOutputTransport(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=default_sr,
            )
        )

        llm_logger = LLMFrameLogger(recorder_accessor)

        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.transcript.user(),
                self.llm,
                llm_logger,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.assistant_shim,
                self.turn_gate,  # Wait for BotStoppedSpeakingFrame before advancing turn
                self.context_aggregator.assistant(),
                self.output_transport,  # Paces OutputAudioRawFrame via write_audio_frame()
                self.audio_buffer,  # Record audio AFTER output_transport pacing
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=45,
            idle_timeout_frames=(InputAudioRawFrame, OutputAudioRawFrame, MetricsFrame),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

    async def _queue_first_turn(self) -> None:
        """Queue audio for the first turn."""
        # Start audio recording
        logger.info("[AudioRecording] Starting audio recording")
        await self.audio_buffer.start_recording()

        # Synchronize output transport's silence insertion timing with recording start
        # This ensures bot audio gaps are filled with silence aligned to wall-clock time
        # Pass the recording sample rate (from AudioBufferProcessor) so silence is created
        # at the correct rate, even if output audio frames are at a different rate
        if self.output_transport is not None:
            self.output_transport.reset_recording_baseline(
                recording_sample_rate=self.audio_buffer._init_sample_rate
            )

        # For Gemini Live, push context frame to initialize the LLM with system
        # instruction and tools. This triggers ONE reconnect at startup.
        # For OpenAI Realtime and Ultravox Realtime, DO NOT send a context frame -
        # they get their config via session_properties/params at construction time.
        if self._is_gemini_live():
            await self.task.queue_frames([LLMContextFrame(self.context)])

        # Give the pipeline a moment to start
        await asyncio.sleep(1.0)

        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued paced audio for first turn: {audio_path}")
            except Exception as e:
                logger.exception(f"Failed to queue audio from {audio_path}: {e}")
                self.current_turn_audio_path = None
                # Fall back to text
                if self._is_gemini_live():
                    await self.task.queue_frames(
                        [
                            LLMMessagesAppendFrame(
                                messages=[{"role": "user", "content": turn["input"]}]
                            )
                        ]
                    )
                else:
                    await self.task.queue_frames([LLMRunFrame()])
        else:
            # No audio file, use text
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}]
                        )
                    ]
                )
            else:
                await self.task.queue_frames([LLMRunFrame()])

    async def _queue_next_turn(self) -> None:
        """Queue audio or text for the next turn."""
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued paced audio for turn {self.turn_idx}: {audio_path}")
            except Exception as e:
                logger.exception(f"Failed to queue audio for turn {self.turn_idx}: {e}")
                audio_path = None

        if not audio_path:
            self.current_turn_audio_path = None
            # Fall back to text
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
            else:
                # OpenAI Realtime fallback
                self.context.add_messages([{"role": "user", "content": turn["input"]}])
                await self.task.queue_frames([LLMRunFrame()])
