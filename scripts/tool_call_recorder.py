from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ToolCallRecorder(FrameProcessor):
    """Records tool calls and results into the global RunRecorder.

    This processor is transport-agnostic and relies on LLMService emitting
    FunctionCallInProgressFrame and FunctionCallResultFrame while executing
    tool calls. We append minimal details to the active `recorder`.
    """

    def __init__(self, recorder_ref: Any):
        super().__init__()
        # `recorder_ref` is a zero-arg callable returning the current recorder,
        # so we don't capture a stale reference across turns.
        self._recorder_ref = recorder_ref
        # Track last recorded call to prevent duplicates
        self._last_recorded_call = None

    def _rec(self):
        try:
            return self._recorder_ref() if callable(self._recorder_ref) else None
        except Exception:
            return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only process DOWNSTREAM frames to avoid duplication
        # (LLM service emits frames in both directions)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            rec = self._rec()
            if rec is not None:
                try:
                    # Create a tuple for comparison (function_name, args)
                    current_call = (frame.function_name, str(frame.arguments or {}))

                    # Skip if this is a duplicate of the last recorded call
                    if current_call == self._last_recorded_call:
                        logger.debug(
                            f"ToolCallRecorder: skipping duplicate call to {frame.function_name}"
                        )
                    else:
                        # Keep tool_call_id separate from arguments
                        rec.record_tool_call(
                            frame.function_name,
                            frame.arguments or {},
                        )
                        logger.info(
                            f"[TOOL_RECORDER] FunctionCallInProgressFrame name={frame.function_name} args={frame.arguments} tool_call_id={getattr(frame,'tool_call_id',None)}"
                        )
                        # Update last recorded call
                        self._last_recorded_call = current_call
                except Exception as e:
                    logger.debug(f"ToolCallRecorder: failed to record call: {e}")

        elif isinstance(frame, FunctionCallResultFrame):
            rec = self._rec()
            if rec is not None:
                try:
                    rec.record_tool_result(
                        frame.function_name,
                        {
                            "tool_call_id": frame.tool_call_id,
                            "result": frame.result,
                            "properties": getattr(frame, "properties", None),
                        },
                    )
                    logger.info(
                        f"[TOOL_RECORDER] FunctionCallResultFrame name={frame.function_name} tool_call_id={frame.tool_call_id} result_keys={list((frame.result or {}).keys())}"
                    )
                except Exception as e:
                    logger.debug(f"ToolCallRecorder: failed to record result: {e}")

        await self.push_frame(frame, direction)
