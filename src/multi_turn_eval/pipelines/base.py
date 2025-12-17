"""Base pipeline class for multi-turn evaluation.

The pipeline owns EVERYTHING - the CLI/runner just calls pipeline.run().
Each pipeline type (text, realtime, nova-sonic) handles its own specifics.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder


class BasePipeline(ABC):
    """Base class for all pipelines. Owns all execution state and logic.

    The pipeline is responsible for:
    1. Creating and configuring the LLM service
    2. Setting up context with system prompt and tools
    3. Building the pipeline with all processors
    4. Managing turn flow (queuing turns, detecting end-of-turn)
    5. Recording transcripts and metrics

    Subclasses implement the abstract methods to customize behavior.
    """

    # Set to False for pipelines that create their own LLM (e.g., Nova Sonic)
    requires_service = True

    def __init__(self, benchmark):
        """Initialize the pipeline.

        Args:
            benchmark: A BenchmarkConfig instance with turns, tools, and system instruction.
        """
        self.benchmark = benchmark
        self.turns = benchmark.turns
        self.turn_idx = 0
        self.done = False
        self.recorder: Optional[TranscriptRecorder] = None
        self.task: Optional[PipelineTask] = None
        self.context: Optional[LLMContext] = None
        self.llm: Optional[FrameProcessor] = None
        self.model_name: Optional[str] = None
        self._turn_indices: Optional[List[int]] = None
        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()

    @property
    def effective_turns(self) -> List[dict]:
        """Get the turns to run (filtered by turn_indices if set)."""
        if self._turn_indices is not None:
            return [self.turns[i] for i in self._turn_indices if i < len(self.turns)]
        return self.turns

    async def run(
        self,
        recorder: TranscriptRecorder,
        model: str,
        service_class: Optional[type] = None,
        turn_indices: Optional[List[int]] = None,
    ) -> None:
        """Run the complete benchmark. Pipeline handles everything internally.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: LLM service class (required unless pipeline sets requires_service=False).
            turn_indices: Optional list of turn indices to run (for debugging).
        """
        self.recorder = recorder
        self.model_name = model
        self._turn_indices = turn_indices

        # Create LLM service
        self.llm = self._create_llm(service_class, model)

        # Setup (pipeline-specific)
        self._setup_context()
        self._setup_llm()
        self._build_task()

        # Initialize first turn BEFORE queueing
        self.recorder.start_turn(self._get_actual_turn_index(0))

        # Queue first turn and run
        await self._queue_first_turn()
        runner = PipelineRunner(handle_sigint=True)
        await runner.run(self.task)

    def _get_actual_turn_index(self, effective_index: int) -> int:
        """Convert effective turn index to actual turn index."""
        if self._turn_indices is not None:
            return self._turn_indices[effective_index]
        return effective_index

    def _get_current_turn(self) -> dict:
        """Get the current turn data."""
        return self.effective_turns[self.turn_idx]

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create LLM service. Override for pipelines that create their own.

        Args:
            service_class: LLM service class to instantiate.
            model: Model name/identifier.

        Returns:
            Configured LLM service instance.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        # Build kwargs with API keys based on service class name
        kwargs: Dict[str, Any] = {"model": model}
        class_name = service_class.__name__
        model_lower = model.lower()

        if "Anthropic" in class_name:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "OpenAI" in class_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gpt-5 series models: set reasoning effort and priority tier
            if model_lower.startswith("gpt-5"):
                from pipecat.services.openai.llm import OpenAILLMService
                # gpt-5.1 and gpt-5.2 use "none", other gpt-5 models use "minimal"
                if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt-5.2"):
                    reasoning_effort = "none"
                else:
                    reasoning_effort = "minimal"
                kwargs["params"] = OpenAILLMService.InputParams(
                    service_tier="priority",
                    extra={"reasoning_effort": reasoning_effort},
                )
                logger.info(f"Configured {model} with reasoning_effort={reasoning_effort}, service_tier=priority")

        elif "Google" in class_name or "Gemini" in class_name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gemini-3 series models: use minimal thinking
            if "gemini-3" in model_lower:
                from google.genai import types
                from pipecat.services.google.llm import GoogleLLMService
                kwargs["params"] = GoogleLLMService.InputParams(
                    extra={
                        "thinking_config": types.ThinkingConfig(
                            thinking_level="MINIMAL",
                            include_thoughts=True,
                        )
                    }
                )
                logger.info(f"Configured {model} with thinking_level=MINIMAL")

        elif "Bedrock" in class_name:
            # AWS Bedrock uses AWS credentials from environment
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if not (access_key and secret_key):
                raise EnvironmentError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required"
                )
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            session_token = os.getenv("AWS_SESSION_TOKEN")
            if session_token:
                kwargs["aws_session_token"] = session_token
            region = os.getenv("AWS_REGION", "us-east-1")
            kwargs["region"] = region

        return service_class(**kwargs)

    async def _on_turn_end(self, assistant_text: str) -> None:
        """Called when assistant finishes. Base handles common logic.

        Args:
            assistant_text: The assistant's response text.
        """
        if self.done:
            return

        # Get the actual turn index for recording
        actual_turn_idx = self._get_actual_turn_index(self.turn_idx)

        # Record turn (common)
        self.recorder.write_turn(
            user_text=self.effective_turns[self.turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        # Advance (common)
        self.turn_idx += 1

        # Reset tool call tracking for the new turn
        self._seen_tool_calls.clear()
        self._duplicate_tool_call_ids.clear()

        if self.turn_idx < len(self.effective_turns):
            # Start next turn
            actual_next_idx = self._get_actual_turn_index(self.turn_idx)
            self.recorder.start_turn(actual_next_idx)
            await self._queue_next_turn()
        else:
            # All turns complete
            logger.info("Conversation complete")
            self.done = True
            self.recorder.write_summary()
            await self.task.cancel()

    def _handle_metrics(self, frame: MetricsFrame) -> None:
        """Common metrics handling."""
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                self.recorder.record_usage_metrics(md.value, getattr(md, "model", None))
            elif isinstance(md, TTFBMetricsData):
                self.recorder.record_ttfb(md.value)

    async def _function_catchall(self, params: FunctionCallParams) -> None:
        """Common function handler - returns success, handles end_session.

        Tool call recording is handled by ToolCallRecorder in the pipeline.
        This just returns the result and handles the special end_session tool.

        Duplicate tool calls (same function + args) are detected and skipped
        to prevent context pollution that can confuse the model.
        """
        # Create a key for duplicate detection (function_name + args)
        call_key = (params.function_name, str(params.arguments or {}))

        # Check for duplicate tool call
        if call_key in self._seen_tool_calls:
            tool_call_id = getattr(params, 'tool_call_id', None)
            logger.warning(
                f"Skipping duplicate tool call: {params.function_name} "
                f"(tool_call_id={tool_call_id})"
            )
            # Track this tool_call_id as a duplicate so ToolCallRecorder can filter it
            if tool_call_id:
                self._duplicate_tool_call_ids.add(tool_call_id)
            # Return a result to satisfy the API, but mark it as skipped
            await params.result_callback({"status": "duplicate_skipped"})
            return

        # Track this call
        self._seen_tool_calls.add(call_key)

        result = {"status": "success"}
        await params.result_callback(result)

        # end_session tool: gracefully terminate
        if params.function_name == "end_session":
            logger.info("end_session tool called - gracefully ending run")
            self.done = True
            # Small delay to let tool call frames propagate through ToolCallRecorder
            await asyncio.sleep(0.05)
            # Write the final turn (assistant response may be empty since it's just a tool call)
            if self.turn_idx < len(self.effective_turns):
                self.recorder.write_turn(
                    user_text=self.effective_turns[self.turn_idx].get("input", ""),
                    assistant_text="",
                )
            self.recorder.write_summary()
            # Cancel the pipeline task to exit cleanly
            await self.task.cancel()

    # ---- Abstract methods (pipeline-specific) ----

    @abstractmethod
    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        pass

    @abstractmethod
    def _setup_llm(self) -> None:
        """Configure LLM (register functions, set callbacks)."""
        pass

    @abstractmethod
    def _build_task(self) -> None:
        """Build Pipeline and PipelineTask with all processors."""
        pass

    @abstractmethod
    async def _queue_first_turn(self) -> None:
        """Queue the first turn to start the conversation."""
        pass

    @abstractmethod
    async def _queue_next_turn(self) -> None:
        """Queue the next turn (called from _on_turn_end)."""
        pass
