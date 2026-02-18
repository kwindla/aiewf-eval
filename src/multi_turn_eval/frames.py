"""Custom frames used by the evaluation pipeline."""

from dataclasses import dataclass
from typing import Optional

from pipecat.frames.frames import DataFrame


@dataclass
class ToolResultTurnCompleteFrame(DataFrame):
    """Signal emitted when a tool result completes and no auto LLM rerun is requested.

    This lets turn-finalization logic handle tool-only turns where the assistant
    aggregator may not emit a timestamp frame.
    """

    turn_index: Optional[int]
    turn_start_monotonic: Optional[float]
    function_name: str
    tool_call_id: str
