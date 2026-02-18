"""Base pipeline class for multi-turn evaluation.

The pipeline owns EVERYTHING - the CLI/runner just calls pipeline.run().
Each pipeline type (text, realtime, nova-sonic) handles its own specifics.
"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    # Set to True in subclasses that support synthetic recovery turns.
    supports_recovery = False
    # Whether tool results should trigger immediate LLM inference by default.
    default_tool_result_run_llm = True

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
        self.service_name: Optional[str] = None
        self._turn_indices: Optional[List[int]] = None
        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()
        # True while evaluating the synthetic "Please go ahead." turn.
        self._in_recovery_turn: bool = False
        # Global runtime switches for controlled A/B testing.
        self._enable_recovery_nudges: bool = _env_bool("MTE_ENABLE_RECOVERY", True)
        self._enable_tool_call_dedupe: bool = _env_bool("MTE_DEDUPE_TOOL_CALLS", True)
        # Whether a tool result should trigger immediate follow-up LLM inference.
        self._tool_result_run_llm: bool = _env_bool(
            "MTE_TOOL_RESULT_RUN_LLM",
            self.default_tool_result_run_llm,
        )

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
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
    ) -> None:
        """Run the complete benchmark. Pipeline handles everything internally.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: LLM service class (required unless pipeline sets requires_service=False).
            service_name: Service name/alias (e.g., "openai", "openrouter").
            turn_indices: Optional list of turn indices to run (for debugging).
        """
        self.recorder = recorder
        self.model_name = model
        self.service_name = service_name  # Store for use in _create_llm overrides
        self._turn_indices = turn_indices

        logger.info(f"Recovery nudges enabled={self._enable_recovery_nudges}")
        logger.info(f"Tool call dedupe enabled={self._enable_tool_call_dedupe}")
        logger.info(f"Tool result run_llm enabled={self._tool_result_run_llm}")

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

        Note:
            Subclasses can access self.service_name if needed for service-specific config.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        # Build kwargs with API keys based on service class name
        kwargs: Dict[str, Any] = {"model": model}
        class_name = service_class.__name__
        model_lower = model.lower()
        service_name_lower = (self.service_name or "").lower()

        # Handle OpenRouter (uses OpenAI-compatible API with different base URL and API key)
        if service_name_lower == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
            logger.info(f"Using OpenRouter with base_url={kwargs['base_url']}")
            return service_class(**kwargs)

        if "Anthropic" in class_name:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
            from pipecat.services.anthropic.llm import AnthropicLLMService
            enable_prompt_caching = _env_bool("MTE_ANTHROPIC_PROMPT_CACHING", True)
            kwargs["params"] = AnthropicLLMService.InputParams(
                enable_prompt_caching=enable_prompt_caching,
            )
            logger.info(
                f"Configured {model} with enable_prompt_caching={enable_prompt_caching}"
            )
        elif "Groq" in class_name:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "Cerebras" in class_name:
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise EnvironmentError("CEREBRAS_API_KEY environment variable is required")
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

        is_recovery_turn = self._in_recovery_turn
        logger.debug(
            "on_turn_end start: "
            f"turn_idx={self.turn_idx} "
            f"recovery={is_recovery_turn} "
            f"tool_calls={len(self.recorder.turn_calls) if self.recorder else 'n/a'} "
            f"tool_results={len(self.recorder.turn_results) if self.recorder else 'n/a'}"
        )

        # Record turn (common)
        self.recorder.write_turn(
            user_text=self.effective_turns[self.turn_idx].get("input", ""),
            assistant_text=assistant_text,
            recovery_turn=is_recovery_turn,
        )

        # Recovery attempt complete: always advance to next scripted turn.
        if is_recovery_turn:
            self._in_recovery_turn = False
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
            return

        # Normal scripted turn: optionally inject one recovery attempt.
        should_recover = self._should_recover()
        logger.debug(
            "on_turn_end decision: "
            f"turn_idx={self.turn_idx} should_recover={should_recover}"
        )
        if should_recover:
            self._in_recovery_turn = True

            # Recovery is a new turn attempt; clear duplicate tracking first.
            self._seen_tool_calls.clear()
            self._duplicate_tool_call_ids.clear()

            # Re-start recorder state at the same actual turn index.
            actual_turn_idx = self._get_actual_turn_index(self.turn_idx)
            self.recorder.start_turn(actual_turn_idx)

            await self._queue_recovery_turn()
            return

        # Normal advancement (common)
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

        # Optional duplicate suppression to avoid context pollution.
        if self._enable_tool_call_dedupe:
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
                await params.result_callback(
                    {"status": "duplicate_skipped"},
                    properties=FunctionCallResultProperties(
                        run_llm=self._tool_result_run_llm
                    ),
                )
                return

        # Track this call
        self._seen_tool_calls.add(call_key)

        result = {"status": "success"}
        await params.result_callback(
            result,
            properties=FunctionCallResultProperties(
                run_llm=self._tool_result_run_llm
            ),
        )

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
                    recovery_turn=self._in_recovery_turn,
                )
            self.recorder.write_summary()
            # Cancel the pipeline task to exit cleanly
            await self.task.cancel()

    def _has_required_call(self) -> bool:
        """Check whether current turn's required function call is present."""
        if self.recorder is None or self.turn_idx >= len(self.effective_turns):
            return True

        turn = self.effective_turns[self.turn_idx]
        required = turn.get("required_function_call")
        if not required:
            return True

        expected_name = required["name"]
        expected_args = required.get("args", {}) or {}

        # Ignore duplicate calls when checking whether requirement was met.
        actual_calls = [c for c in self.recorder.turn_calls if not c.get("is_duplicate")]
        for call in actual_calls:
            if call.get("name") != expected_name:
                continue
            if self._args_semantically_match(expected_args, call.get("args") or {}):
                return True
        return False

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Normalize short argument text for lightweight semantic matching."""
        t = (s or "").lower().strip()
        t = t.replace("opentelemetry", "open telemetry")
        t = t.replace("can't", "cannot").replace("cant", "cannot")
        t = t.replace("can not", "cannot")
        t = t.replace("unable to", "cannot")
        t = re.sub(r"[^a-z0-9\s]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def _string_semantically_matches(cls, expected: str, actual: str) -> bool:
        exp_norm = cls._normalize_text(expected)
        act_norm = cls._normalize_text(actual)
        if exp_norm == act_norm:
            return True
        if exp_norm and act_norm:
            # Accept direct containment so concise expected text can match
            # richer user-authored tool arguments.
            if exp_norm in act_norm or act_norm in exp_norm:
                return True

        stopwords = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "about",
            "is",
            "are",
            "was",
            "were",
            "my",
            "your",
            "our",
        }
        exp_tokens = [t for t in exp_norm.split() if t and t not in stopwords]
        act_tokens = [t for t in act_norm.split() if t and t not in stopwords]
        if exp_tokens and act_tokens:
            exp_set = set(exp_tokens)
            act_set = set(act_tokens)
            intersection = len(exp_set & act_set)
            union = len(exp_set | act_set)
            min_len = min(len(exp_set), len(act_set))
            exp_len = len(exp_set)
            act_len = len(act_set)
            if union > 0 and min_len > 0:
                jaccard = intersection / union
                # Treat expected text as the anchor; a verbose actual should
                # still match when it fully contains expected semantics.
                expected_containment = intersection / exp_len if exp_len else 0.0
                actual_containment = intersection / act_len if act_len else 0.0
                if expected_containment >= 0.8:
                    return True
                if jaccard >= 0.6 and expected_containment >= 0.7:
                    return True
                if expected_containment >= 0.6 and actual_containment >= 0.6:
                    return True

        return SequenceMatcher(None, exp_norm, act_norm).ratio() >= 0.82

    @classmethod
    def _args_semantically_match(cls, expected: Any, actual: Any) -> bool:
        """Recursive semantic equivalence for required tool args."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for key, exp_val in expected.items():
                if key not in actual:
                    return False
                if not cls._args_semantically_match(exp_val, actual[key]):
                    return False
            return True

        if isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                return False
            return all(cls._args_semantically_match(e, a) for e, a in zip(expected, actual))

        if isinstance(expected, str):
            if not isinstance(actual, str):
                return False
            return cls._string_semantically_matches(expected, actual)

        return expected == actual

    def _should_recover(self) -> bool:
        """Return True if we should inject a synthetic recovery turn."""
        if not self._enable_recovery_nudges:
            return False
        if not self.supports_recovery or self._in_recovery_turn:
            return False
        if self.turn_idx >= len(self.effective_turns):
            return False
        turn = self.effective_turns[self.turn_idx]
        required = turn.get("required_function_call")
        return bool(required) and not self._has_required_call()

    async def _queue_recovery_turn(self) -> None:
        """Queue a synthetic recovery turn.

        Subclasses that opt into recovery (supports_recovery=True) must override.
        """
        raise NotImplementedError

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
