"""Base pipeline class for multi-turn evaluation.

The pipeline owns EVERYTHING - the CLI/runner just calls pipeline.run().
Each pipeline type (text, realtime, nova-sonic) handles its own specifics.
"""

import asyncio
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections import deque
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
        # Recovery turn bookkeeping.
        self._recovery_for_actual_turn: Optional[int] = None
        self._recovery_turn_index_base: int = 0
        self._recovery_turn_counter: int = 0

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

        # Keep synthetic recovery turns on unique transcript turn IDs.
        if self.effective_turns:
            max_actual_turn = self._get_actual_turn_index(len(self.effective_turns) - 1)
            self._recovery_turn_index_base = max_actual_turn + 1
        else:
            self._recovery_turn_index_base = 0
        self._recovery_turn_counter = 0

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

    def _sanitize_openai_context_tool_ids(self) -> None:
        """Normalize missing tool-call IDs for OpenAI-compatible services.

        Some custom OpenAI-compatible endpoints can emit tool calls/results
        with null IDs, or emit empty assistant text stubs immediately before
        a tool-call message. The official OpenAI client validates these IDs as
        required strings when re-sending context on the next turn, and some
        backends reject empty assistant content outright.
        """
        service_name = (self.service_name or "").lower()
        if service_name not in {"openai", "openrouter", "mistral", "nemotron", "modal"}:
            return
        if self.context is None:
            return

        messages = self.context.get_messages()
        if not messages:
            return

        pending_tool_call_ids: deque[str] = deque()
        patched = 0
        dropped_empty_assistant = 0
        drop_indexes: list[int] = []

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                content = msg.get("content")
                has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
                content_empty = (
                    content is None
                    or (isinstance(content, str) and not content.strip())
                    or (isinstance(content, list) and len(content) == 0)
                )

                # Some OpenAI-compatible backends emit a blank assistant text
                # message followed by a proper assistant tool_calls message.
                if content_empty and not has_tool_calls:
                    drop_indexes.append(idx)
                    dropped_empty_assistant += 1
                    continue

                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get("id")
                    if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                        tool_call["id"] = tool_call_id
                        patched += 1
                    pending_tool_call_ids.append(tool_call_id)
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                    if pending_tool_call_ids:
                        msg["tool_call_id"] = pending_tool_call_ids.popleft()
                    else:
                        msg["tool_call_id"] = f"call_{uuid.uuid4().hex[:24]}"
                    patched += 1
                    continue

                # Keep our pending queue aligned when IDs already exist.
                if pending_tool_call_ids and pending_tool_call_ids[0] == tool_call_id:
                    pending_tool_call_ids.popleft()
                elif tool_call_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(tool_call_id)

        for idx in reversed(drop_indexes):
            del messages[idx]

        if patched:
            logger.warning(
                f"Patched {patched} missing OpenAI tool-call IDs in context"
            )
        if dropped_empty_assistant:
            logger.warning(
                "Dropped {} empty assistant message(s) from OpenAI-compatible context".format(
                    dropped_empty_assistant
                )
            )

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

        # Handle Mistral (uses an OpenAI-compatible chat API with its own base URL)
        if service_name_lower == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise EnvironmentError("MISTRAL_API_KEY environment variable is required")
            base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
            kwargs["api_key"] = api_key
            kwargs["base_url"] = base_url
            logger.info(f"Using Mistral with base_url={base_url}")
            return service_class(**kwargs)

        # Handle Modal (uses OpenAI-compatible API with custom endpoint)
        if service_name_lower == "modal":
            api_key = os.getenv("MODAL_API_KEY")
            if not api_key:
                raise EnvironmentError("MODAL_API_KEY environment variable is required")
            base_url = os.getenv("MODAL_BASE_URL", "https://api.us-west-2.modal.direct/v1")
            kwargs["api_key"] = api_key
            kwargs["base_url"] = base_url

            # Disable reasoning by default (instruct mode); override with
            # MTE_MODAL_THINKING=1 to re-enable thinking mode.
            from pipecat.services.openai.llm import OpenAILLMService
            enable_thinking = _env_bool("MTE_MODAL_THINKING", False)
            extra: Dict[str, Any] = {}
            if not enable_thinking:
                extra["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            kwargs["params"] = OpenAILLMService.InputParams(extra=extra)
            logger.info(
                f"Using Modal with base_url={base_url}, thinking={enable_thinking}"
            )
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
        elif "OpenAI" in class_name or service_name_lower == "nemotron":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Route selected OpenAI text models to Responses API without changing
            # benchmark pipelines.
            if service_name_lower == "openai" and (
                model_lower.startswith("gpt-4.1") or model_lower.startswith("gpt-5.4")
            ):
                from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService

                service_class = OpenAIResponsesLLMService
                class_name = service_class.__name__
                logger.info(f"Configured {model} to use {class_name}")

            # Configure gpt-5 series models: set reasoning effort and priority tier
            if model_lower.startswith("gpt-5"):
                from pipecat.services.openai.llm import OpenAILLMService
                # gpt-5.1 and gpt-5.2 use "none"; most other gpt-5 models use "minimal".
                #
                # gpt-5.4 rejects reasoning_effort with tools on
                # /v1/chat/completions; use Responses API with reasoning.effort when
                # routed, otherwise omit reasoning_effort for chat.completions.
                if model_lower.startswith("gpt-5.4"):
                    if class_name == "OpenAIResponsesLLMService":
                        reasoning_effort = os.getenv(
                            "MTE_OPENAI_RESPONSES_REASONING_EFFORT", "low"
                        ).strip().lower()
                        # gpt-5.4 model docs expose these effort levels.
                        allowed_efforts = {"none", "low", "medium", "high", "xhigh"}
                        if reasoning_effort not in allowed_efforts:
                            logger.warning(
                                "Invalid MTE_OPENAI_RESPONSES_REASONING_EFFORT='{}'; defaulting to low".format(
                                    reasoning_effort
                                )
                            )
                            reasoning_effort = "low"
                        kwargs["params"] = OpenAILLMService.InputParams(
                            service_tier="priority",
                            extra={"reasoning": {"effort": reasoning_effort}},
                        )
                        logger.info(
                            f"Configured {model} with reasoning.effort={reasoning_effort}, service_tier=priority (Responses API)"
                        )
                    else:
                        kwargs["params"] = OpenAILLMService.InputParams(
                            service_tier="priority",
                        )
                        logger.info(
                            f"Configured {model} with service_tier=priority (reasoning_effort omitted for chat.completions tools compatibility)"
                        )
                else:
                    if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt-5.2"):
                        reasoning_effort = "none"
                    else:
                        reasoning_effort = "minimal"
                    kwargs["params"] = OpenAILLMService.InputParams(
                        service_tier="priority",
                        extra={"reasoning_effort": reasoning_effort},
                    )
                    logger.info(
                        f"Configured {model} with reasoning_effort={reasoning_effort}, service_tier=priority"
                    )

        elif "Google" in class_name or "Gemini" in class_name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure Gemini 3 series models with an explicit thinking mode
            # so benchmark sweeps can compare disabled vs minimal reasoning.
            if "gemini-3" in model_lower:
                from pipecat.services.google.llm import GoogleLLMService
                thinking_mode = os.getenv("MTE_GOOGLE_THINKING_MODE", "minimal").strip().lower()

                if thinking_mode in {"disabled", "disable", "off", "none", "budget0", "0"}:
                    kwargs["params"] = GoogleLLMService.InputParams(
                        thinking=GoogleLLMService.ThinkingConfig(
                            thinking_budget=0,
                        )
                    )
                    logger.info(
                        f"Configured {model} with thinking_budget=0 (disabled)"
                    )
                elif thinking_mode in {"minimal", "min"}:
                    kwargs["params"] = GoogleLLMService.InputParams(
                        thinking=GoogleLLMService.ThinkingConfig(
                            thinking_level="minimal",
                            include_thoughts=True,
                        )
                    )
                    logger.info(
                        f"Configured {model} with thinking_level=minimal"
                    )
                elif thinking_mode in {"low", "medium", "high"}:
                    kwargs["params"] = GoogleLLMService.InputParams(
                        thinking=GoogleLLMService.ThinkingConfig(
                            thinking_level=thinking_mode,
                            include_thoughts=True,
                        )
                    )
                    logger.info(
                        f"Configured {model} with thinking_level={thinking_mode}"
                    )
                elif thinking_mode in {"default", "auto"}:
                    logger.info(
                        f"Configured {model} with provider default thinking behavior"
                    )
                else:
                    raise ValueError(
                        "Unsupported MTE_GOOGLE_THINKING_MODE={!r}; expected disabled, minimal, low, medium, high, or default".format(
                            thinking_mode
                        )
                    )

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

        user_text = (
            "Please go ahead."
            if is_recovery_turn
            else self.effective_turns[self.turn_idx].get("input", "")
        )
        recovery_for_turn = self._recovery_for_actual_turn if is_recovery_turn else None

        # Record turn (common)
        self.recorder.write_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            recovery_turn=is_recovery_turn,
            recovery_for_turn=recovery_for_turn,
        )

        # Recovery attempt complete: always advance to next scripted turn.
        if is_recovery_turn:
            self._in_recovery_turn = False
            self._recovery_for_actual_turn = None
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

            # Re-start recorder state at a unique synthetic turn index to avoid
            # duplicate "turn" IDs in transcript output.
            self._recovery_for_actual_turn = self._get_actual_turn_index(self.turn_idx)
            recovery_turn_idx = (
                self._recovery_turn_index_base + self._recovery_turn_counter
            )
            self._recovery_turn_counter += 1
            self.recorder.start_turn(recovery_turn_idx)

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

    def _tool_result_properties(self) -> Optional[FunctionCallResultProperties]:
        """Return explicit tool-result properties only when overriding defaults.

        `run_llm=False` is a behavioral override we need for text pipelines that
        manage turn advancement explicitly. For `run_llm=True`, allow Pipecat's
        default behavior (run only after the last in-progress tool call) by
        returning `None`.
        """
        if self._tool_result_run_llm:
            return None
        return FunctionCallResultProperties(run_llm=False)

    async def _emit_tool_result_callback(
        self, params: FunctionCallParams, result: Dict[str, Any]
    ) -> None:
        properties = self._tool_result_properties()
        if properties is None:
            await params.result_callback(result)
        else:
            await params.result_callback(result, properties=properties)

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
                await self._emit_tool_result_callback(
                    params, {"status": "duplicate_skipped"}
                )
                return

        # Track this call
        self._seen_tool_calls.add(call_key)

        result = {"status": "success"}
        await self._emit_tool_result_callback(params, result)

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
        return cls._args_semantically_match_with_key(expected, actual, key=None)

    @classmethod
    def _args_semantically_match_with_key(
        cls,
        expected: Any,
        actual: Any,
        *,
        key: Optional[str],
    ) -> bool:
        """Recursive semantic equivalence for required tool args.

        Identifier-like values (for keys such as *_id) are matched strictly.
        """
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for k, exp_val in expected.items():
                if k not in actual:
                    return False
                child_key = str(k)
                if not cls._args_semantically_match_with_key(
                    exp_val, actual[k], key=child_key
                ):
                    return False
            return True

        if isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                return False
            return all(
                cls._args_semantically_match_with_key(e, a, key=key)
                for e, a in zip(expected, actual)
            )

        if isinstance(expected, str):
            if not isinstance(actual, str):
                return False
            if cls._is_identifier_key(key):
                return expected.strip() == actual.strip()
            return cls._string_semantically_matches(expected, actual)

        return expected == actual

    @staticmethod
    def _is_identifier_key(key: Optional[str]) -> bool:
        if not key:
            return False
        normalized = key.strip().lower()
        if normalized in {"id", "session_id", "tool_call_id", "conversation_id"}:
            return True
        return normalized.endswith("_id")

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
