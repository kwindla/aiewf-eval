"""Transcript recording and metrics collection for evaluation runs."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from pipecat.metrics.metrics import LLMTokenUsage


def now_iso() -> str:
    """Return current UTC time in ISO format with milliseconds."""
    try:
        from datetime import UTC
        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except ImportError:
        # Fallback for older Python versions
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class TranscriptRecorder:
    """Accumulates per-turn data and writes JSONL + summary.

    This class records turn-by-turn conversation data including:
    - User and assistant text
    - Tool calls and results
    - Token usage metrics
    - Time to first byte (TTFB)
    - Total turn latency

    Output files:
    - transcript.jsonl: One JSON record per turn
    - runtime.json: Summary metadata
    """

    def __init__(self, run_dir: Path, model_name: str):
        """Initialize the transcript recorder.

        Args:
            run_dir: Directory to write output files. Will be created if it doesn't exist.
            model_name: Name of the model being evaluated.
        """
        self.run_dir = run_dir
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
        """Start recording a new turn.

        Args:
            turn_index: Zero-based index of the turn being started.
        """
        self.turn_index = turn_index
        self.turn_start_monotonic = time.monotonic()
        self.turn_usage = {}
        self.turn_calls = []
        self.turn_results = []
        self.turn_ttfb_ms = None

    def record_ttfb(self, ttfb_seconds: float):
        """Record time to first byte for the current turn.

        Only the first TTFB value per turn is recorded; subsequent calls are ignored.

        Args:
            ttfb_seconds: Time to first byte in seconds.
        """
        ttfb_ms = int(ttfb_seconds * 1000)
        logger.debug(
            f"TranscriptRecorder: record_ttfb called with {ttfb_seconds:.3f}s ({ttfb_ms}ms), "
            f"current={self.turn_ttfb_ms}"
        )
        if self.turn_ttfb_ms is None:
            self.turn_ttfb_ms = ttfb_ms
            logger.debug(f"TranscriptRecorder: set turn_ttfb_ms = {ttfb_ms}")
        else:
            logger.debug(f"TranscriptRecorder: IGNORING - already set to {self.turn_ttfb_ms}")

    def reset_ttfb(self):
        """Reset TTFB to None, allowing it to be set again.

        Call this when starting TTFB timing for a new turn to ensure
        spurious TTFB values from pipeline initialization don't interfere.
        """
        if self.turn_ttfb_ms is not None:
            logger.debug(f"TranscriptRecorder: Resetting TTFB (was {self.turn_ttfb_ms})")
        self.turn_ttfb_ms = None

    def record_usage_metrics(self, m: LLMTokenUsage, model: Optional[str] = None):
        """Record token usage metrics for the current turn.

        Args:
            m: Token usage data from the LLM.
            model: Optional model name to update (if different from initial).
        """
        self.turn_usage = {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
            "cache_read_input_tokens": m.cache_read_input_tokens,
            "cache_creation_input_tokens": m.cache_creation_input_tokens,
        }
        if model:
            self.model_name = model

    def record_tool_call(self, name: str, args: Dict[str, Any], is_duplicate: bool = False):
        """Record a tool call for the current turn.

        Args:
            name: Name of the tool/function being called.
            args: Arguments passed to the tool.
            is_duplicate: Whether this is a duplicate call (same function + args).
        """
        call_record = {"name": name, "args": args}
        if is_duplicate:
            call_record["is_duplicate"] = True
        self.turn_calls.append(call_record)

    def record_tool_result(self, name: str, response: Dict[str, Any]):
        """Record a tool result for the current turn.

        Args:
            name: Name of the tool/function that was called.
            response: Response from the tool.
        """
        self.turn_results.append({"name": name, "response": response})

    def write_turn(
        self, *, user_text: str, assistant_text: str, reconnection_count: int = 0
    ):
        """Write the completed turn to the transcript file.

        Args:
            user_text: The user's input text for this turn.
            assistant_text: The assistant's response text for this turn.
            reconnection_count: Number of reconnections during this turn (0 = no reconnection).
        """
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
            "reconnection_count": reconnection_count,
        }
        self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fp.flush()
        self.total_turns_scored += 1
        logger.info(f"Recorded turn {self.turn_index}: {assistant_text[:100]}...")

    def write_summary(self):
        """Write the runtime summary file."""
        runtime = {
            "model_name": self.model_name,
            "turns": self.total_turns_scored,
            "note": "runtime-only; scoring is performed post-run",
        }
        (self.run_dir / "runtime.json").write_text(
            json.dumps(runtime, indent=2), encoding="utf-8"
        )

    def close(self):
        """Close the transcript file handle."""
        if self.fp and not self.fp.closed:
            self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Alias for backward compatibility
RunRecorder = TranscriptRecorder
