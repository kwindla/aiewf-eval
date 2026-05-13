"""Audio-input pipeline for Nemotron audio-in/text-out services.

The LLM service is vendored from `../nemotron-nano-omni` — see
`multi_turn_eval/vendor/README.md` for source / refresh procedure. This
pipeline owns audio-file loading and the four `TextPipeline` overrides; the
cache state machine lives entirely in the vendored service.
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameProcessor

from multi_turn_eval.pipelines.text import TextPipeline
from multi_turn_eval.vendor.nemotron_omni import (
    NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS,
    NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE,
    NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K,
    NemotronAssistantAggregator,
    NemotronOmniAudioLLMService,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _audio_prompt() -> str | None:
    value = os.getenv("MTE_AUDIO_IN_PROMPT")
    if value is None:
        return "Listen to the audio and respond to the spoken instruction."

    if value.strip().lower() in {"", "0", "false", "none"}:
        return None

    return value


class AudioInPipeline(TextPipeline):
    """Pipeline that sends benchmark user turns as WAV audio content parts.

    Built on top of the vendored Nemotron Omni service. The conversation cache
    is owned by the service: when ``MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1``
    we generate a per-run ``conversation_id`` and let the upstream contract
    drive suffix-only attach automatically (there is no longer a separate
    ``suffix_only`` knob in the new API).
    """

    def _build_audio_user_message(self, turn_index: int) -> dict[str, Any]:
        actual_index = self._get_actual_turn_index(turn_index)
        audio_path = self.benchmark.get_audio_path(actual_index)
        if audio_path is None:
            raise FileNotFoundError(
                "No audio path returned for "
                f"turn_index={turn_index} actual_turn_index={actual_index}"
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                "Audio file not found for "
                f"turn_index={turn_index} actual_turn_index={actual_index}: {audio_path}"
            )

        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        content: list[dict[str, Any]] = []

        prompt = _audio_prompt()
        if prompt:
            content.append({"type": "text", "text": prompt})

        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            }
        )

        return {"role": "user", "content": content}

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt, tools, and first audio turn."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")

        messages = [
            {"role": "system", "content": system_instruction},
            self._build_audio_user_message(0),
        ]

        tools = getattr(self.benchmark, "tools_schema", None)

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
        self.last_msg_idx = len(messages)

    async def _queue_next_turn(self) -> None:
        """Add next audio user message to context and trigger LLM."""
        self.context.add_messages([self._build_audio_user_message(self.turn_idx)])
        self._sanitize_openai_context_tool_ids()
        self.last_msg_idx = len(self.context.get_messages())
        logger.debug(
            "queue_turn: reason=next "
            f"turn_idx={self.turn_idx} "
            f"context_messages={self.last_msg_idx}"
        )
        await self.task.queue_frames([LLMRunFrame()])

    async def _queue_recovery_turn(self) -> None:
        """Queue a synthetic text nudge to recover a missed required tool call."""
        self.context.add_messages([{"role": "user", "content": "Please go ahead."}])
        self._sanitize_openai_context_tool_ids()
        self.last_msg_idx = len(self.context.get_messages())
        logger.debug(
            "queue_turn: reason=recovery "
            f"turn_idx={self.turn_idx} "
            f"context_messages={self.last_msg_idx}"
        )
        await self.task.queue_frames([LLMRunFrame()])

    def _build_assistant_aggregator(self) -> FrameProcessor:
        """Use the upstream NemotronAssistantAggregator so the local context
        matches what vLLM commits across cache-on / cache-off paths.
        """
        return NemotronAssistantAggregator(
            self.context,
            interrupted_tool_pass_signal=None,
            conversation_commit_boundary_tracker=self.llm.conversation_commit_boundary_tracker,
        )

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        if service_class is not None and not issubclass(
            service_class, NemotronOmniAudioLLMService
        ):
            alias = self.service_name or getattr(
                service_class, "__name__", repr(service_class)
            )
            raise ValueError(
                "AudioInPipeline requires service alias 'nemotron-audio-in'; "
                f"got {alias!r}"
            )

        llm_class = service_class or NemotronOmniAudioLLMService

        # Deprecation warning for the now-defunct suffix-only knob.
        if os.getenv("MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY") is not None:
            logger.warning(
                "MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY is no longer honored — the "
                "upstream contract makes suffix-only implicit whenever "
                "conversation_id is set. Use MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE "
                "to toggle the cache."
            )

        conv_cache = _env_bool("MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE", False)
        conversation_id = f"mte-{uuid.uuid4().hex}" if conv_cache else None

        enable_thinking = _env_bool("MTE_NEMOTRON_AUDIO_IN_THINKING", False)

        settings = NemotronOmniAudioLLMService.Settings(
            model=model,
            system_instruction=getattr(self.benchmark, "system_instruction", "") or None,
            temperature=float(
                os.getenv(
                    "MTE_NEMOTRON_AUDIO_IN_TEMPERATURE",
                    str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE),
                )
            ),
            max_tokens=int(
                os.getenv(
                    "MTE_NEMOTRON_AUDIO_IN_MAX_TOKENS",
                    str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS),
                )
            ),
            top_p=None,
            top_k=int(
                os.getenv(
                    "MTE_NEMOTRON_AUDIO_IN_TOP_K",
                    str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K),
                )
            ),
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

        return llm_class(
            api_key=os.getenv("NEMOTRON_AUDIO_IN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None,
            base_url=os.getenv(
                "MTE_NEMOTRON_AUDIO_IN_BASE_URL",
                "http://192.168.7.228:8000/v1",
            ),
            conversation_id=conversation_id,
            request_timeout_secs=float(
                os.getenv("MTE_NEMOTRON_AUDIO_IN_TIMEOUT_SECS", "180")
            ),
            enable_bash_tool=False,
            settings=settings,
        )
