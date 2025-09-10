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

    def _rec(self):
        try:
            return self._recorder_ref() if callable(self._recorder_ref) else None
        except Exception:
            return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallInProgressFrame):
            rec = self._rec()
            if rec is not None:
                try:
                    rec.record_tool_call(
                        frame.function_name,
                        {
                            "tool_call_id": frame.tool_call_id,
                            **(frame.arguments or {}),
                        },
                    )
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
                except Exception as e:
                    logger.debug(f"ToolCallRecorder: failed to record result: {e}")

        await self.push_frame(frame, direction)

