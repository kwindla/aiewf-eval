import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from loguru import logger
from dotenv import load_dotenv

from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    CancelFrame,
    LLMFullResponseEndFrame,
    TranscriptionMessage,
    InputAudioRawFrame,
    LLMRunFrame,
    LLMMessagesAppendFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    LLMTokenUsage,
    TTFBMetricsData,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiLiveLLMService,
)
from pipecat.services.google.gemini_live.llm import (
    InputParams as GeminiLiveInputParams,
    GeminiVADParams,
)
from google.genai import types as genai_types
from pipecat.services.openai.base_llm import BaseOpenAILLMService
import pipecat.services.openai.realtime.events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMContextAssistantTimestampFrame
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair


from system_instruction_short import system_instruction
from turns import turns
from tools_schema import ToolsSchemaForTest
import soundfile as sf
from pipecat.transports.base_transport import TransportParams
from scripts.paced_input_transport import PacedInputTransport
from pipecat.utils.time import time_now_iso8601
from scripts.tts_stopped_assistant_transcript import (
    TTSStoppedAssistantTranscriptProcessor,
)
from scripts.tool_call_recorder import ToolCallRecorder

load_dotenv()

logger.info("Starting conversation test...")


# -------------------------
# Utilities for persistence
# -------------------------


def now_iso() -> str:
    # Use timezone-aware UTC to avoid deprecation warnings
    try:
        from datetime import UTC

        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except Exception:
        # Fallback for older Python versions
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class RunRecorder:
    """Accumulates per-turn data and writes JSONL + summary."""

    def __init__(self, model_name: str):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self.run_dir = Path("runs") / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.run_dir / "transcript.jsonl"
        self.fp = self.out_path.open("a", encoding="utf-8")
        self.model_name = model_name

        # per-turn working state
        self.turn_start_monotonic: Optional[float] = None
        self.turn_usage: Dict[str, Any] = {}
        self.turn_calls: List[Dict[str, Any]] = []
        self.turn_results: List[Dict[str, Any]] = []
        self.turn_index: int = 0
        self.turn_ttfb_ms: Optional[int] = None

        # simple turn counter; judging happens post-run
        self.total_turns_scored = 0

    def start_turn(self, turn_index: int):
        self.turn_index = turn_index
        self.turn_start_monotonic = time.monotonic()
        self.turn_usage = {}
        self.turn_calls = []
        self.turn_results = []
        self.turn_ttfb_ms = None

    def record_ttfb(self, ttfb_seconds: float):
        # Store TTFB in milliseconds (only keep first TTFB per turn)
        if self.turn_ttfb_ms is None:
            self.turn_ttfb_ms = int(ttfb_seconds * 1000)

    def record_usage_metrics(self, m: LLMTokenUsage, model: Optional[str] = None):
        # store last seen usage; fine for turn-local
        self.turn_usage = {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
            "cache_read_input_tokens": m.cache_read_input_tokens,
            "cache_creation_input_tokens": m.cache_creation_input_tokens,
        }
        if model:
            self.model_name = model

    def record_tool_call(self, name: str, args: Dict[str, Any]):
        self.turn_calls.append({"name": name, "args": args})

    def record_tool_result(self, name: str, response: Dict[str, Any]):
        self.turn_results.append({"name": name, "response": response})

    def write_turn(self, *, user_text: str, assistant_text: str):
        latency_ms = None
        if self.turn_start_monotonic is not None:
            latency_ms = int((time.monotonic() - self.turn_start_monotonic) * 1000)

        rec = {
            "ts": now_iso(),
            "turn": self.turn_index,
            "model_name": self.model_name,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tool_calls": self.turn_calls,
            "tool_results": self.turn_results,
            "tokens": self.turn_usage or None,
            "ttfb_ms": self.turn_ttfb_ms,
            "latency_ms": latency_ms,
        }
        self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fp.flush()
        self.total_turns_scored += 1

    def write_summary(self):
        runtime = {
            "model_name": self.model_name,
            "turns": self.total_turns_scored,
            "note": "runtime-only; scoring is performed post-run (see alt_summary.json)",
        }
        (self.run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


# -------------------------
# Tool call stub (records)
# -------------------------

recorder: Optional[RunRecorder] = None


async def function_catchall(params: FunctionCallParams):
    logger.info(f"Function call: {params}")
    # Note: Tool call recording is handled by ToolCallRecorder in the pipeline.
    # We only need to return the result here.
    result = {"status": "success"}
    await params.result_callback(result)


class NextTurn(FrameProcessor):
    def __init__(
        self, end_of_turn_callback: Callable, metrics_callback: Callable[[MetricsFrame], None]
    ):
        super().__init__()
        self.end_of_turn_callback = end_of_turn_callback
        self.metrics_callback = metrics_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            self.metrics_callback(frame)

        # Treat assistant timestamp frame as end-of-turn marker
        if isinstance(frame, LLMContextAssistantTimestampFrame):
            logger.info("EOT (timestamp)")
            await self.end_of_turn_callback()


class RealtimeEOTShim(FrameProcessor):
    """Shim for OpenAI Realtime: inject assistant message + timestamp when
    the service omits LLMFullResponseStartFrame (so the assistant aggregator
    won't aggregate tokens or emit a timestamp).

    Sits between the LLM and the assistant context aggregator.
    """

    def __init__(self):
        super().__init__()

    async def push_eot_frames(self):
        await asyncio.sleep(0.5)
        ts = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(ts)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            asyncio.create_task(self.push_eot_frames())

        await self.push_frame(frame, direction)


async def main():
    turn_idx = 0

    # Configure model routing + user field for caching experiments
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")

    def is_google_model(name: str) -> bool:
        n = (name or "").lower()
        return n.startswith("gemini") or n.startswith("gemma")

    def is_gemini_live_model(name: str) -> bool:
        n = (name or "").lower()
        return (n.startswith("gemini") or n.startswith("models/gemini")) and (
            "live" in n or "native-audio" in n
        )

    def is_openrouter_model(name: str) -> bool:
        n = (name or "").lower()
        return (
            n.startswith("meta-llama")
            or n.startswith("openrouter")
            or n.startswith("openai/")
            or n.startswith("qwen/")
        )

    def is_bedrock_model(name: str) -> bool:
        n = (name or "").lower()
        return "amazon" in n

    def is_openai_realtime_model(name: str) -> bool:
        n = (name or "").lower()
        return "gpt-realtime" in n

    extra = {"user": os.getenv("EVAL_USER", "eval-runner")}
    llm = None

    if is_gemini_live_model(model_name):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for Gemini Live models")
        # Gemini Live expects fully-qualified model ids like "models/<id>"
        m = model_name if model_name.startswith("models/") else f"models/{model_name}"
        # Disable thinking by setting budget to 0, and configure VAD to match gpt-realtime
        gemini_live_params = GeminiLiveInputParams(
            thinking=genai_types.ThinkingConfig(thinking_budget=0),
            vad=GeminiVADParams(
                prefix_padding_ms=300,  # Audio to include before VAD detects speech
                silence_duration_ms=1500,  # Silence duration to detect speech stop
            ),
        )
        llm = GeminiLiveLLMService(
            api_key=api_key,
            model=m,
            system_instruction=system_instruction,
            tools=ToolsSchemaForTest,
            params=gemini_live_params,
        )
        llm.register_function(None, function_catchall)
    elif is_google_model(model_name):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is required for Google models")
        # Google service handles its own adapter and can upgrade OpenAI context under the hood
        llm = GoogleLLMService(api_key=api_key, model=model_name)
        llm.register_function(None, function_catchall)
    elif is_openai_realtime_model(model_name):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI Realtime models")
        # New OpenAI Realtime LLM Service (non-beta)
        # Configure server-side turn detection to automatically create a response
        # when speech stops. This prevents the pipeline from hanging waiting for
        # a manual response.create after audio has been appended.
        #
        # This is working but not with the full-length system instruction. We also need to
        # add tools to the session properties.
        # Use default session properties - let the API use its default server_vad
        # Note: Explicitly setting audio.input.turn_detection seems to break VAD
        # in some cases, while the defaults work. See debugging notes below.
        session_props = rt_events.SessionProperties(
            instructions=system_instruction,
            tools=ToolsSchemaForTest,
            turn_detection={
                "type": "server_vad",  # Use server-side Voice Activity Detection
                "threshold": 0.5,  # Activation threshold (0.0-1.0), higher requires louder audio
                "prefix_padding_ms": 300,  # Audio to include before VAD detects speech
                "silence_duration_ms": 1500,  # Silence duration to detect speech stop; lower values lead to quicker responses
            },
        )
        llm = OpenAIRealtimeLLMService(
            api_key=api_key,
            model=model_name,
            system_instruction=system_instruction,
            session_properties=session_props,
        )
        llm.register_function(None, function_catchall)
    elif is_bedrock_model(model_name):
        try:
            # Import lazily to avoid requiring AWS deps for non-Bedrock runs
            from pipecat.services.aws.llm import AWSBedrockLLMService  # type: ignore
        except Exception as e:  # pragma: no cover - surfaced at runtime only
            raise RuntimeError(
                "Amazon Bedrock support requires installing 'pipecat-ai[aws]'."
            ) from e
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        if not (aws_access_key_id and aws_secret_access_key and aws_region):
            raise EnvironmentError(
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required for Amazon Bedrock models"
            )
        llm = AWSBedrockLLMService(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            model=model_name,
        )
        llm.register_function(None, function_catchall)
    elif is_openrouter_model(model_name):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required for OpenRouter models")
        # params = BaseOpenAILLMService.InputParams(extra=extra)
        llm = OpenAILLMService(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            params=params,
        )
        llm.register_function(None, function_catchall)
    else:
        # Default to OpenAI-compatible endpoint using OPENAI_API_KEY
        if model_name.startswith("gpt-5"):
            extra["service_tier"] = "priority"
            if model_name == "gpt-5.1":
                logger.info("Setting reasoning_effort to none for gpt-5.1")
                extra["reasoning_effort"] = "none"
            else:
                logger.info("Setting reasoning_effort to minimal for gpt-5")
                extra["reasoning_effort"] = "minimal"
        params = BaseOpenAILLMService.InputParams(extra=extra)
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=model_name, params=params)
        llm.register_function(None, function_catchall)

    # Set up recorder
    global recorder
    recorder = RunRecorder(model_name=model_name)
    recorder.start_turn(turn_idx)

    if is_openai_realtime_model(model_name) or is_gemini_live_model(model_name):
        messages = []
    else:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": turns[turn_idx]["input"]},
        ]

    context = LLMContext(messages, tools=ToolsSchemaForTest)
    context_aggregator = LLMContextAggregatorPair(context)

    # Track index into context messages to extract per-turn assistant text
    last_msg_idx = len(messages)

    def handle_metrics(frame: MetricsFrame):
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                recorder.record_usage_metrics(md.value, getattr(md, "model", None))
            elif isinstance(md, TTFBMetricsData):
                recorder.record_ttfb(md.value)

    done = False

    async def end_of_turn(direct_assistant_text: str = None):
        nonlocal turn_idx, last_msg_idx, done
        if done:
            logger.info("!!!! EOT top (done)")
            await llm.push_frame(CancelFrame())
            return

        # For realtime models, use the direct text passed from transcript handler
        # For text models, context_aggregator.assistant() has already added the
        # complete assistant message to context by the time we get here
        if direct_assistant_text is not None:
            assistant_text = direct_assistant_text
        else:
            # context_aggregator.assistant() adds the assistant message as the last entry
            msgs = context.get_messages()
            assistant_text = ""
            if msgs and msgs[-1].get("role") == "assistant":
                content = msgs[-1].get("content", "")
                assistant_text = content if isinstance(content, str) else ""

        recorder.write_turn(
            user_text=turns[turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        # Update last_msg_idx to current position so next turn only captures new messages
        msgs = context.get_messages()
        last_msg_idx = len(msgs)

        turn_idx += 1
        if turn_idx < len(turns):
            recorder.start_turn(turn_idx)
            await asyncio.sleep(0.10)
            logger.info(f"!!!!!! User input: {turns[turn_idx]['input']}")
            logger.info(f"!!!!!! Context: {context}")

            # For non-live models, keep local context in sync and trigger a run.
            # For Gemini Live, do NOT pre-add the user message here; instead send
            # an LLMMessagesAppendFrame so the service generates a single response.
            if not (is_openai_realtime_model(model_name) or is_gemini_live_model(model_name)):
                context.add_messages([{"role": "user", "content": turns[turn_idx]["input"]}])
                last_msg_idx = len(context.get_messages())
                await task.queue_frames([LLMRunFrame()])
            else:
                # Live speech models: prefer audio if available; else fall back to
                # appropriate text triggering per service.
                audio_path = turns[turn_idx].get("audio_file")
                if audio_path:
                    try:
                        assert paced_input is not None
                        paced_input.enqueue_wav_file(audio_path)
                    except Exception as e:
                        logger.exception(
                            f"Failed to queue audio for turn {turn_idx} from {audio_path}: {e}"
                        )
                        # Fall through to text path if audio fails
                        audio_path = None

                if not audio_path:
                    if is_gemini_live_model(model_name):
                        # Append the user message and let Gemini Live create a single response.
                        await task.queue_frames(
                            [
                                LLMMessagesAppendFrame(
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": turns[turn_idx]["input"],
                                        }
                                    ],
                                    run_llm=False,
                                )
                            ]
                        )
                        # The aggregator will add one user message to context internally.
                        # We keep last_msg_idx as-is; next EOT will capture assistant-only deltas.
                    else:
                        # OpenAI Realtime fallback: trigger a run with current context
                        context.add_messages(
                            [{"role": "user", "content": turns[turn_idx]["input"]}]
                        )
                        last_msg_idx = len(context.get_messages())
                        await task.queue_frames([LLMRunFrame()])
        else:
            logger.info("Conversation complete")
            recorder.write_summary()
            done = True
            logger.info("!!!! EOT conversation complete (done)")
            await llm.push_frame(CancelFrame())

    next_turn = NextTurn(end_of_turn, handle_metrics)

    # Used for speech-to-speech models because LLMTextFrame doesn't work the way it should with the speech-to-speech APIs.
    # But transcript delivery is flaky. It's hard to tell if this is a Pipecat issue or an API issue.
    transcript = TranscriptProcessor()
    assistant_shim = TTSStoppedAssistantTranscriptProcessor()

    paced_input = None

    # Recorder accessor for ToolCallRecorder
    def current_recorder():
        global recorder
        return recorder

    if is_openai_realtime_model(model_name) or is_gemini_live_model(model_name):
        # Create a paced input transport set to match the first audio file's sample rate (default 24000 Hz)
        default_sr = 24000
        t0_audio = turns[0].get("audio_file")
        try:
            if t0_audio:
                _, t0_sr = sf.read(t0_audio, dtype="int16", always_2d=True)
                default_sr = int(t0_sr)
        except Exception:
            pass
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=default_sr,
            audio_in_channels=1,
            audio_in_passthrough=True,
        )
        paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
        )

        class LLMFrameLogger(FrameProcessor):
            """Logs every frame emitted by the LLM stage and captures TTFB metrics."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if not isinstance(frame, InputAudioRawFrame):
                    logger.info(f"[LLM→] {frame.__class__.__name__} ({direction})")
                # Capture TTFB from MetricsFrame for realtime/live models
                if isinstance(frame, MetricsFrame):
                    for md in frame.data:
                        if isinstance(md, TTFBMetricsData):
                            recorder.record_ttfb(md.value)
                await self.push_frame(frame, direction)

        class PreLLMFrameLogger(FrameProcessor):
            """Logs every frame emitted by the LLM stage."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if not isinstance(frame, InputAudioRawFrame):
                    logger.info(f"[PreLLM→] {frame.__class__.__name__} ({direction})")
                await self.push_frame(frame, direction)

        pre_llm_logger = PreLLMFrameLogger()
        llm_logger = LLMFrameLogger()

        pipeline = Pipeline(
            [
                paced_input,  # paced audio input at realtime pace
                context_aggregator.user(),  # User text context
                transcript.user(),
                pre_llm_logger,
                llm,  # LLM
                llm_logger,  # debug: log all frames from LLM
                ToolCallRecorder(current_recorder),
                assistant_shim,  # flushes only on TTSStoppedFrame - MUST be before context_aggregator.assistant() to see TTSTextFrames
                context_aggregator.assistant(),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                context_aggregator.user(),
                llm,
                ToolCallRecorder(current_recorder),
                context_aggregator.assistant(),
                next_turn,
            ]
        )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=45,
        idle_timeout_frames=(MetricsFrame,),
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Register event handler only for assistant transcript updates
    @assistant_shim.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage) and getattr(msg, "role", None) == "assistant":
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"
                logger.info(f"Transcript: {line}")
                # Note: Don't add to context here - context_aggregator.assistant() already does that
                # Small delay to let downstream settle, then next turn
                await asyncio.sleep(1.0)
                # Pass the assistant text directly to avoid race with context_aggregator
                await end_of_turn(direct_assistant_text=msg.content)

    async def queue_audio_for_first_turn(delay=1.0):
        # Give the pipeline a moment to start, then enqueue paced audio
        await asyncio.sleep(delay)
        audio_path = turns[0].get("audio_file")
        try:
            assert paced_input is not None
            paced_input.enqueue_wav_file(audio_path)
            logger.info("!!!!!! OpenAI Realtime model - queued paced audio for first turn")
        except Exception as e:
            logger.exception(f"Failed to queue audio from {audio_path}: {e}")
            # Fall back to a text-triggered run
            if is_gemini_live_model(model_name):
                await task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turns[0]["input"]}]
                        )
                    ]
                )
            else:
                await task.queue_frames([LLMRunFrame()])

    # Queue first user turn
    last_msg_idx = len(messages)
    if is_openai_realtime_model(model_name) or is_gemini_live_model(model_name):
        # For Realtime, avoid sending a context frame that would force an early
        # response.create; let audio + explicit stop drive turn boundaries.
        asyncio.create_task(queue_audio_for_first_turn())
    else:
        await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
