"""Text-based pipeline for synchronous LLM services.

This pipeline works with text-in/text-out LLM services:
- OpenAI (GPT-4o, GPT-4.1, etc.)
- Anthropic (Claude Sonnet, Claude Haiku, etc.)
- Google (Gemini Flash, etc.)
- AWS Bedrock (Claude, Llama, etc.)
- OpenRouter (various models)

Pipeline: UserAggregator → LLM → ToolCallRecorder → AssistantAggregator → NextTurn
"""

import asyncio
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTimestampFrame,
    LLMRunFrame,
    MetricsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.frames import ToolResultTurnCompleteFrame
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder


class NextTurn(FrameProcessor):
    """Frame processor that detects end-of-turn and handles metrics.

    Watches for LLMContextAssistantTimestampFrame which signals that the
    assistant's response is complete and has been added to context.
    """

    def __init__(
        self,
        end_of_turn_callback,
        metrics_callback,
        recorder_accessor,
        tool_result_run_llm_accessor,
    ):
        super().__init__()
        self.end_of_turn_callback = end_of_turn_callback
        self.metrics_callback = metrics_callback
        self.recorder_accessor = recorder_accessor
        self.tool_result_run_llm_accessor = tool_result_run_llm_accessor
        self._awaiting_tool_completion = False
        self._tool_wait_task: asyncio.Task | None = None
        self._ending_turn = False
        self._active_turn_sig: tuple[int | None, float | None] | None = None
        self._finalized_turn_sig: tuple[int | None, float | None] | None = None
        self._saw_timestamp_this_turn = False

    def _pending_tool_calls(self) -> int:
        """Return number of tool calls started but not yet recorded as results."""
        try:
            rec = self.recorder_accessor() if callable(self.recorder_accessor) else None
            if rec is None:
                return 0
            pending = len(rec.turn_calls) - len(rec.turn_results)
            return pending if pending > 0 else 0
        except Exception:
            return 0

    def _tool_result_runs_llm(self) -> bool:
        try:
            if callable(self.tool_result_run_llm_accessor):
                return bool(self.tool_result_run_llm_accessor())
            return True
        except Exception:
            return True

    def _current_turn_index(self) -> int | None:
        try:
            rec = self.recorder_accessor() if callable(self.recorder_accessor) else None
            if rec is None:
                return None
            return getattr(rec, "turn_index", None)
        except Exception:
            return None

    def _current_turn_signature(self) -> tuple[int | None, float | None] | None:
        try:
            rec = self.recorder_accessor() if callable(self.recorder_accessor) else None
            if rec is None:
                return None
            return (getattr(rec, "turn_index", None), getattr(rec, "turn_start_monotonic", None))
        except Exception:
            return None

    def _sync_turn_state(self) -> None:
        sig = self._current_turn_signature()
        if sig != self._active_turn_sig:
            self._active_turn_sig = sig
            self._finalized_turn_sig = None
            self._saw_timestamp_this_turn = False

    async def _finalize_turn(self, reason: str) -> None:
        if self._ending_turn:
            return
        sig = self._current_turn_signature()
        if self._finalized_turn_sig is not None and sig == self._finalized_turn_sig:
            logger.debug(f"Ignoring duplicate turn-finalize signal: {reason}")
            return
        self._ending_turn = True
        logger.info(reason)
        self._awaiting_tool_completion = False
        try:
            await self.end_of_turn_callback()
        finally:
            self._finalized_turn_sig = sig
            self._ending_turn = False

    async def _wait_for_tool_completion(self) -> None:
        """Wait briefly for tool results, then end the turn."""
        start = time.monotonic()
        timeout_s = 5.0
        try:
            while self._awaiting_tool_completion:
                if self._pending_tool_calls() == 0:
                    await self._finalize_turn("EOT deferred: tool calls complete")
                    return
                if (time.monotonic() - start) >= timeout_s:
                    logger.warning(
                        "EOT deferred timeout waiting for tool results; forcing turn end"
                    )
                    await self._finalize_turn("EOT deferred timeout")
                    return
                await asyncio.sleep(0.02)
        finally:
            self._tool_wait_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        self._sync_turn_state()

        if isinstance(frame, MetricsFrame):
            self.metrics_callback(frame)

        # Treat assistant timestamp frame as end-of-turn marker
        if isinstance(frame, LLMContextAssistantTimestampFrame):
            self._saw_timestamp_this_turn = True
            if self._pending_tool_calls() > 0:
                # Defer turn completion until tool results are returned to the model.
                self._awaiting_tool_completion = True
                pending = self._pending_tool_calls()
                logger.info(
                    f"EOT deferred: waiting for {pending} pending tool call result(s)"
                )
                if self._tool_wait_task is None:
                    self._tool_wait_task = asyncio.create_task(
                        self._wait_for_tool_completion()
                    )
            else:
                await self._finalize_turn("EOT (timestamp)")
            return

        # If we previously deferred EOT and tool results are now recorded, finish turn.
        if self._awaiting_tool_completion and self._pending_tool_calls() == 0:
            await self._finalize_turn("EOT deferred: tool calls complete")

        # Tool-only assistant turns can complete without an assistant timestamp
        # when run_llm is disabled on tool results. We emit and consume an
        # explicit completion signal for those cases.
        if isinstance(frame, ToolResultTurnCompleteFrame) and not self._tool_result_runs_llm():
            current_sig = self._current_turn_signature()
            frame_sig = (frame.turn_index, frame.turn_start_monotonic)
            if current_sig is not None and frame_sig != current_sig:
                logger.debug(
                    "Ignoring stale tool-complete signal "
                    f"(frame_sig={frame_sig}, current_sig={current_sig})"
                )
                return
            if self._saw_timestamp_this_turn:
                logger.debug(
                    "Ignoring tool-complete signal because timestamp already ended "
                    f"turn {frame.turn_index}"
                )
                return
            if self._pending_tool_calls() == 0:
                await self._finalize_turn("EOT (tool_result_complete signal)")
            else:
                self._awaiting_tool_completion = True
                if self._tool_wait_task is None:
                    self._tool_wait_task = asyncio.create_task(
                        self._wait_for_tool_completion()
                    )
            return


class TextPipeline(BasePipeline):
    """Pipeline for text-based (synchronous) LLM services.

    This is the simplest pipeline type:
    1. User message is added to context
    2. LLMRunFrame triggers the LLM
    3. LLM responds with text
    4. Context aggregator captures response
    5. NextTurn detects end-of-turn and advances
    """

    requires_service = True
    supports_recovery = True
    default_tool_result_run_llm = False

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.context_aggregator = None
        self.last_msg_idx = 0

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt, tools, and first user message."""
        # Get system instruction from benchmark
        system_instruction = getattr(self.benchmark, "system_instruction", "")

        # Initial messages: system + first user turn
        first_turn = self._get_current_turn()
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": first_turn["input"]},
        ]

        # Get tools schema from benchmark
        tools = getattr(self.benchmark, "tools_schema", None)

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
        self.last_msg_idx = len(messages)

    def _setup_llm(self) -> None:
        """Register the function handler for all tools."""
        self.llm.register_function(None, self._function_catchall)

    def _build_task(self) -> None:
        """Build the pipeline with context aggregators and turn detector."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        def extract_text(content) -> str:
            """Extract plain text from assistant content (string or block list)."""
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # Anthropic-style content blocks may mix text + tool_use.
                # Keep text blocks in transcript while ignoring non-text blocks.
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                        continue
                    if isinstance(block, dict):
                        block_text = block.get("text")
                        if isinstance(block_text, str):
                            text_parts.append(block_text)
                        continue
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str):
                        text_parts.append(block_text)
                return "\n".join(part for part in text_parts if part).strip()
            return ""

        def unwrap_message(msg):
            """Return a plain message payload from service-specific wrappers."""
            return getattr(msg, "message", msg)

        def message_role(msg) -> str:
            payload = unwrap_message(msg)
            if isinstance(payload, dict):
                return payload.get("role", "")
            return getattr(payload, "role", "") or ""

        def message_content(msg):
            payload = unwrap_message(msg)
            if isinstance(payload, dict):
                return payload.get("content", "")
            return getattr(payload, "content", "")

        # Create the end-of-turn handler
        async def end_of_turn():
            if self.done:
                return

            # Extract assistant text from messages added after the current user turn.
            # This is robust even when tool_result user messages are appended after
            # assistant tool_use blocks.
            msgs = self.context.get_messages()
            assistant_text = ""
            start_idx = min(self.last_msg_idx, len(msgs))
            turn_msgs = msgs[start_idx:]
            assistant_parts = []
            for msg in turn_msgs:
                if message_role(msg) != "assistant":
                    continue
                text = extract_text(message_content(msg))
                if text:
                    assistant_parts.append(text)
            assistant_text = "\n\n".join(assistant_parts).strip()

            await self._on_turn_end(assistant_text)

        def tool_result_run_llm_accessor():
            return self._tool_result_run_llm

        next_turn = NextTurn(
            end_of_turn,
            self._handle_metrics,
            recorder_accessor,
            tool_result_run_llm_accessor,
        )

        pipeline = Pipeline(
            [
                self.context_aggregator.user(),
                self.llm,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.context_aggregator.assistant(),
                next_turn,
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=45,
            idle_timeout_frames=(MetricsFrame,),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

    async def _queue_first_turn(self) -> None:
        """Queue LLMRunFrame to start the first turn."""
        # The first user message is already in context from _setup_context
        logger.debug(
            "queue_turn: reason=first "
            f"turn_idx={self.turn_idx} "
            f"user={self._get_current_turn().get('input', '')[:80]!r}"
        )
        await self.task.queue_frames([LLMRunFrame()])

    async def _queue_next_turn(self) -> None:
        """Add next user message to context and trigger LLM."""
        turn = self._get_current_turn()
        self.context.add_messages([{"role": "user", "content": turn["input"]}])
        self.last_msg_idx = len(self.context.get_messages())
        logger.debug(
            "queue_turn: reason=next "
            f"turn_idx={self.turn_idx} "
            f"user={turn.get('input', '')[:80]!r} "
            f"context_messages={self.last_msg_idx}"
        )
        await self.task.queue_frames([LLMRunFrame()])

    async def _queue_recovery_turn(self) -> None:
        """Queue a synthetic user nudge to recover a missed required tool call."""
        self.context.add_messages([{"role": "user", "content": "Please go ahead."}])
        self.last_msg_idx = len(self.context.get_messages())
        logger.debug(
            "queue_turn: reason=recovery "
            f"turn_idx={self.turn_idx} "
            f"context_messages={self.last_msg_idx}"
        )
        await self.task.queue_frames([LLMRunFrame()])
