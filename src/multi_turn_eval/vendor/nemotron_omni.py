#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nemotron Omni audio-input LLM service.

This service sends buffered user audio to a local OpenAI-compatible vLLM
endpoint and streams text deltas back as Pipecat LLM frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    is_given as context_is_given,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallRunnerItem,
    LLMService,
)
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven
from pipecat.utils.time import time_now_iso8601

_TRACE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

# Request fields the service owns; `settings.extra` must not override these.
_RESERVED_PAYLOAD_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "conversation_id",
        "conversation_require_cache",
        "tools",
        "tool_choice",
    }
)

DEFAULT_VOICE_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond in plain text only. Keep answers "
    "brief, direct, and conversational, usually one or two short sentences. Your "
    "implementation uses the Nemotron Nano Omni LLM and the Kyutai Pocket TTS "
    "voice model. Your "
    "replies will be read aloud by a text-to-speech system, so write natural "
    "spoken language rather than visual formatting. Do not use Markdown, bullet "
    "points, numbered lists, code blocks, tables, emojis, emoticons, decorative "
    "symbols, or special formatting. Avoid long lists. Do not mention these "
    "formatting rules unless asked. When you use a tool, treat the "
    "latest sync tool message as ground truth. Sync tool messages contain a JSON "
    "observation object with keys ok, status, summary, command, exit_code, "
    "timed_out, stdout, and stderr. Trust ok, status, exit_code, and timed_out. "
    "Use stdout and stderr as evidence. Non-empty stderr with exit_code 0 is "
    "normal evidence, not automatic failure. If status is duplicate_suppressed "
    "or round_limit_reached, that is still ground truth, so answer from it and "
    "do not re-call the tool. The client separately displays raw bash commands "
    "and raw terminal output, so do not read ASCII art, borders, terminal markup, "
    "or long command output literally unless the user explicitly asks you to. "
    "Interpret the result and explain the useful meaning briefly instead of "
    "reciting raw JSON. "
    "For cowthink or cowsay-style output, focus on the message inside the bubble "
    "and say that the command rendered it as ASCII art. Only use the bash tool "
    "when the user's latest request explicitly asks you to inspect or operate on "
    "the local machine, run a command, or when local inspection is genuinely "
    "necessary to answer correctly. If the user explicitly asks you to use the "
    "bash tool, run a command, or report command output, you must call the bash "
    "tool and answer from that result rather than from memory. After you receive "
    "a bash tool result, do not call the same command again unless the user asks "
    "for a rerun or the situation has changed. Do not use the bash tool to echo, "
    "printf, paraphrase, or draft an answer you could simply say directly. The "
    "tool is for real command execution and explicit user-requested command "
    "output, not for generating prose. The user may ask about the Unix tool "
    "cowthink; use the bash tool to inspect it when needed."
)

BASH_TOOL_NAME = "run_bash"
DEFAULT_AUDIO_PROMPT = (
    "Listen to the audio and respond to the spoken instruction. If the user "
    "explicitly asks you to use the bash tool, run a command, inspect the local "
    "machine, or report command output, you must call run_bash and answer from "
    "the tool result. Sync tool messages contain a JSON observation object; "
    "trust ok, status, exit_code, and timed_out, and use stdout and stderr as "
    "evidence. Non-empty stderr with exit_code 0 is normal evidence, not "
    "automatic failure. If status is duplicate_suppressed or round_limit_reached, "
    "answer from that observation and do not re-call the tool. Explain the "
    "meaning of the result instead of reciting raw JSON. Otherwise answer "
    "directly without tools unless local inspection is genuinely needed. Never "
    "use bash just to echo or paraphrase an answer you could say directly."
)
NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE = 0.2
NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS = 1024
NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K = 1
BASH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": BASH_TOOL_NAME,
        "description": (
            "Execute arbitrary bash code in the local project workspace and return "
            "a JSON observation object with exactly these fields: ok, status, "
            "summary, command, exit_code, timed_out, stdout, stderr. Trust ok, "
            "status, exit_code, and timed_out as ground truth, and use stdout and "
            "stderr as supporting evidence. Non-empty stderr with exit_code 0 is "
            "normal evidence, not automatic failure. Reserved status values that "
            "can appear without re-running the command include "
            "`duplicate_suppressed` (the exact same command was already run in "
            "this assistant turn, so the prior result was reused) and "
            "`round_limit_reached` (the tool-round budget was already exhausted, "
            "so answer from that observation rather than calling again). Use this "
            "when the user asks you to inspect or operate on the local machine. "
            "If the user explicitly asks you to use bash, run a command, or report "
            "command output, call this tool instead of answering from memory. Do "
            "not call the exact same command again in the same assistant turn "
            "unless the tool result shows that a rerun is required. Do not use "
            "this tool to echo, printf, paraphrase, or draft natural-language "
            "answers that you could say directly. Examples: use "
            "`git branch --show-current` to see the current git branch; use "
            "`find . -maxdepth 1 -type f | wc -l` to count files in this directory; "
            "use `find . -type f | wc -l` to count files total in this project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Bash code to run with `bash -lc`.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class NemotronExactAssistantMessageFrame(DataFrame):
    message: dict[str, Any]
    batch_id: str | None = None


@dataclass
class InterruptedToolPassSignal:
    replace_interrupted_tool_pass: bool = False


@dataclass
class ConversationCommitBoundaryTracker:
    _provisional_batch_user_keys: dict[str, str | None] = field(default_factory=dict)

    def mark_provisional_batch(self, *, batch_id: str, user_turn_key: str | None) -> None:
        self._provisional_batch_user_keys[batch_id] = user_turn_key

    def clear_provisional_batch(self, batch_id: str | None) -> None:
        if batch_id is None:
            return
        self._provisional_batch_user_keys.pop(batch_id, None)

    def provisional_user_turn_keys(self) -> tuple[str, ...]:
        return tuple(
            user_turn_key
            for user_turn_key in self._provisional_batch_user_keys.values()
            if user_turn_key is not None
        )


@dataclass
class ChatCompletionPassResult:
    output_text: str
    tool_calls: list[dict[str, Any]]
    first_token: bool


class ConversationCacheMissError(RuntimeError):
    pass


@dataclass
class NormalizedRequestSnapshot:
    context: LLMContext
    universal_messages: list[Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tool_choice: Any | None
    cache_shape_fingerprint: str


@dataclass
class SerialFunctionCallRunnerItem(FunctionCallRunnerItem):
    batch_id: str = ""


@dataclass
class ToolBatchState:
    batch_id: str
    user_turn_key: str | None
    runner_items: list[SerialFunctionCallRunnerItem]
    started_ids: set[str] = field(default_factory=set)
    pending_signature_keys: set[tuple[str, str]] = field(default_factory=set)
    has_real_execution: bool = False
    round_counted: bool = False
    completed: bool = False


@dataclass
class NemotronOmniAudioLLMSettings(LLMSettings):
    """Settings for :class:`NemotronOmniAudioLLMService`.

    Parameters:
        audio_prompt: Text instruction sent alongside each captured audio turn.
        chat_template_kwargs: Extra chat-template kwargs forwarded to vLLM.
    """

    audio_prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    chat_template_kwargs: dict[str, Any] | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


@dataclass
class _NemotronProvisionalSyncToolPass:
    assistant_message: dict[str, Any]
    batch_id: str | None = None
    assistant_row: dict[str, Any] | None = None
    tool_rows_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    provisional_rows: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_call_ids: set[str] = field(default_factory=set)


class NemotronAssistantAggregator(LLMAssistantAggregator):
    def __init__(
        self,
        context: LLMContext,
        *,
        interrupted_tool_pass_signal: InterruptedToolPassSignal | None = None,
        conversation_commit_boundary_tracker: ConversationCommitBoundaryTracker | None = None,
        **kwargs,
    ):
        super().__init__(context, **kwargs)
        self._interrupted_tool_pass_signal = interrupted_tool_pass_signal
        self._conversation_commit_boundary_tracker = conversation_commit_boundary_tracker
        self._pending_exact_assistant_message: dict[str, Any] | None = None
        self._pending_exact_assistant_batch_id: str | None = None
        self._current_exact_response_suppressed = False
        self._active_exact_sync_pass: _NemotronProvisionalSyncToolPass | None = None
        self._provisional_exact_sync_passes: list[_NemotronProvisionalSyncToolPass] = []
        self._response_commit_candidates: list[_NemotronProvisionalSyncToolPass] = []
        self._current_response_interrupted = False
        self._current_response_had_error = False
        self._suppress_interrupted_aggregation_commit = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, NemotronExactAssistantMessageFrame):
            await self._handle_exact_assistant_message(frame)
            return
        if isinstance(frame, ErrorFrame):
            self._current_response_had_error = True
        await super().process_frame(frame, direction)

    async def push_aggregation(self) -> str:
        if not self._aggregation:
            return ""
        if not self._should_suppress_aggregation_commit():
            return await super().push_aggregation()

        aggregation = self.aggregation_string()
        await super().reset()
        return aggregation

    async def _handle_exact_assistant_message(
        self, frame: NemotronExactAssistantMessageFrame
    ) -> None:
        tool_calls = frame.message.get("tool_calls") or []
        if not tool_calls:
            return

        pending_tool_call_ids = {
            str(tool_call["id"])
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and tool_call.get("id") is not None
        }
        exact_message = copy.deepcopy(frame.message)
        self._pending_exact_assistant_message = exact_message
        self._pending_exact_assistant_batch_id = frame.batch_id
        self._current_exact_response_suppressed = True
        self._active_exact_sync_pass = _NemotronProvisionalSyncToolPass(
            assistant_message=copy.deepcopy(exact_message),
            batch_id=frame.batch_id,
            pending_tool_call_ids=pending_tool_call_ids,
        )

    async def _handle_llm_start(self, frame: LLMFullResponseStartFrame):
        self._current_response_interrupted = False
        self._current_response_had_error = False
        self._response_commit_candidates = list(self._provisional_exact_sync_passes)
        await super()._handle_llm_start(frame)

    async def _handle_llm_end(self, frame: LLMFullResponseEndFrame):
        try:
            await super()._handle_llm_end(frame)
        finally:
            if not self._current_response_interrupted and not self._current_response_had_error:
                self._commit_response_candidates()
            self._response_commit_candidates = []
            self._current_response_interrupted = False
            self._current_response_had_error = False
            self._current_exact_response_suppressed = False

    async def _handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        if not frame.cancel_on_interruption:
            await super()._handle_function_call_in_progress(frame)
            return

        sync_pass = self._active_exact_sync_pass
        if sync_pass is None or self._pending_exact_assistant_message is None:
            logger.warning(
                f"{self}: missing exact assistant message for sync tool call "
                f"[{frame.function_name}:{frame.tool_call_id}]; falling back to stock path"
            )
            await super()._handle_function_call_in_progress(frame)
            return

        logger.debug(
            f"{self} FunctionCallInProgressFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        sync_pass.pending_tool_call_ids.add(frame.tool_call_id)
        if sync_pass.assistant_row is None:
            assistant_row = copy.deepcopy(sync_pass.assistant_message)
            sync_pass.assistant_row = assistant_row
            sync_pass.provisional_rows.append(assistant_row)
            self._ensure_provisional_sync_pass(sync_pass)
            self._context.add_message(assistant_row)
            await self.push_frame(LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601()))

        if frame.tool_call_id not in sync_pass.tool_rows_by_id:
            tool_row = {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
            sync_pass.tool_rows_by_id[frame.tool_call_id] = tool_row
            sync_pass.provisional_rows.append(tool_row)
            self._ensure_provisional_sync_pass(sync_pass)
            self._context.add_message(tool_row)

        self._function_calls_in_progress[frame.tool_call_id] = frame

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        in_progress_frame = self._function_calls_in_progress.get(frame.tool_call_id)
        is_sync_exact = (
            in_progress_frame is not None
            and in_progress_frame.cancel_on_interruption
            and self._find_provisional_sync_pass(frame.tool_call_id) is not None
        )
        is_final = frame.properties.is_final if frame.properties else True

        await super()._handle_function_call_result(frame)

        if not is_sync_exact or not is_final:
            return

        sync_pass = self._find_provisional_sync_pass(frame.tool_call_id)
        if sync_pass is None:
            return
        sync_pass.pending_tool_call_ids.discard(frame.tool_call_id)
        self._maybe_clear_active_exact_sync_pass(sync_pass)

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        logger.debug(
            f"{self} FunctionCallCancelFrame: [{frame.function_name}:{frame.tool_call_id}]"
        )
        function_call = self._function_calls_in_progress.get(frame.tool_call_id)
        sync_pass = self._find_provisional_sync_pass(frame.tool_call_id)

        if function_call is not None and not function_call.cancel_on_interruption:
            await super()._handle_function_call_cancel(frame)
            return

        if frame.tool_call_id in self._function_calls_in_progress:
            if sync_pass is None and function_call and function_call.cancel_on_interruption:
                self._update_function_call_result(
                    frame.function_name,
                    frame.tool_call_id,
                    "CANCELLED",
                )
            del self._function_calls_in_progress[frame.tool_call_id]

        if sync_pass is None:
            return

        tool_row = sync_pass.tool_rows_by_id.pop(frame.tool_call_id, None)
        if tool_row is not None:
            self._remove_provisional_rows([tool_row])
            if tool_row in sync_pass.provisional_rows:
                sync_pass.provisional_rows.remove(tool_row)
        sync_pass.pending_tool_call_ids.discard(frame.tool_call_id)
        self._maybe_clear_active_exact_sync_pass(sync_pass)
        # If every sync-tool row of this pass has now been cancelled (no live
        # tool rows, nothing still pending), the provisional `assistant(tool_calls)`
        # row is orphaned — a `tool_calls` row with no matching `tool` rows is an
        # invalid transcript shape. Drop it and the pass. This covers the
        # new-user-turn supersession path (step 3) that emits per-tool cancels
        # without broadcasting an InterruptionFrame; the InterruptionFrame path
        # already drops everything via `_drop_all_provisional_sync_tool_rows`.
        # (If a *started* sibling already completed with a result before the rest
        # were cancelled — only reachable via a new LLMContextFrame with no
        # preceding InterruptionFrame, which this bot never produces — that
        # surviving tool row keeps the assistant row, and the standard
        # InterruptionFrame cleanup would still drop the whole pass.)
        if not sync_pass.tool_rows_by_id and not sync_pass.pending_tool_call_ids:
            leftover_rows = list(sync_pass.provisional_rows)
            if leftover_rows:
                self._remove_provisional_rows(leftover_rows)
            sync_pass.provisional_rows.clear()
            sync_pass.assistant_row = None
            if sync_pass in self._provisional_exact_sync_passes:
                self._provisional_exact_sync_passes.remove(sync_pass)
            if self._conversation_commit_boundary_tracker is not None:
                self._conversation_commit_boundary_tracker.clear_provisional_batch(
                    sync_pass.batch_id
                )

    async def _handle_interruptions(self, frame: InterruptionFrame):
        removed_rows = self._drop_all_provisional_sync_tool_rows()
        if removed_rows and self._interrupted_tool_pass_signal is not None:
            self._interrupted_tool_pass_signal.replace_interrupted_tool_pass = True

        self._current_response_interrupted = True
        self._suppress_interrupted_aggregation_commit = removed_rows
        try:
            await super()._handle_interruptions(frame)
        finally:
            self._suppress_interrupted_aggregation_commit = False
            self._current_exact_response_suppressed = False
            self._clear_active_exact_sync_pass_state()

    def _should_suppress_aggregation_commit(self) -> bool:
        return (
            self._suppress_interrupted_aggregation_commit
            or self._current_exact_response_suppressed
        )

    def _ensure_provisional_sync_pass(self, sync_pass: _NemotronProvisionalSyncToolPass) -> None:
        if sync_pass not in self._provisional_exact_sync_passes:
            self._provisional_exact_sync_passes.append(sync_pass)

    def _find_provisional_sync_pass(
        self, tool_call_id: str
    ) -> _NemotronProvisionalSyncToolPass | None:
        if (
            self._active_exact_sync_pass is not None
            and tool_call_id in self._active_exact_sync_pass.pending_tool_call_ids
        ):
            return self._active_exact_sync_pass
        for sync_pass in self._provisional_exact_sync_passes:
            if tool_call_id in sync_pass.tool_rows_by_id:
                return sync_pass
            if tool_call_id in sync_pass.pending_tool_call_ids:
                return sync_pass
        return None

    def _commit_response_candidates(self) -> None:
        for sync_pass in self._response_commit_candidates:
            if sync_pass not in self._provisional_exact_sync_passes:
                continue
            self._provisional_exact_sync_passes.remove(sync_pass)
            sync_pass.provisional_rows.clear()
            if self._conversation_commit_boundary_tracker is not None:
                self._conversation_commit_boundary_tracker.clear_provisional_batch(
                    sync_pass.batch_id
                )

    def _drop_all_provisional_sync_tool_rows(self) -> bool:
        rows_to_remove: list[dict[str, Any]] = []
        for sync_pass in self._provisional_exact_sync_passes:
            rows_to_remove.extend(sync_pass.provisional_rows)
            sync_pass.provisional_rows.clear()
            sync_pass.tool_rows_by_id.clear()
            sync_pass.assistant_row = None
            if self._conversation_commit_boundary_tracker is not None:
                self._conversation_commit_boundary_tracker.clear_provisional_batch(
                    sync_pass.batch_id
                )
        self._provisional_exact_sync_passes = []
        if not rows_to_remove:
            return False
        self._remove_provisional_rows(rows_to_remove)
        return True

    def _remove_provisional_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._context.transform_messages(
            lambda messages: [message for message in messages if not any(message is row for row in rows)]
        )

    def _maybe_clear_active_exact_sync_pass(
        self, sync_pass: _NemotronProvisionalSyncToolPass
    ) -> None:
        if self._active_exact_sync_pass is not sync_pass:
            return
        if sync_pass.pending_tool_call_ids:
            return
        self._clear_active_exact_sync_pass_state()

    def _clear_active_exact_sync_pass_state(self) -> None:
        self._active_exact_sync_pass = None
        self._pending_exact_assistant_message = None
        self._pending_exact_assistant_batch_id = None


class NemotronOmniAudioLLMService(LLMService):
    """Audio-input LLM service for Nemotron Omni on vLLM.

    The service consumes ``LLMContextFrame`` frames, translates Pipecat's
    universal ``input_audio`` context parts into vLLM ``audio_url`` content
    parts, submits the full conversation to ``/v1/chat/completions``, and emits
    streamed ``LLMTextFrame`` output bounded by full-response frames.
    """

    Settings = NemotronOmniAudioLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        settings: Settings | None = None,
        audio_passthrough: bool = False,
        conversation_id: str | None = None,
        request_timeout_secs: float = 180.0,
        enable_bash_tool: bool = False,
        bash_tool_cwd: str | None = None,
        bash_tool_timeout_secs: float = 20.0,
        bash_tool_max_output_chars: int = 12000,
        bash_tool_max_rounds: int = 3,
        bash_tool_event_sender: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        conversation_commit_boundary_tracker: ConversationCommitBoundaryTracker | None = None,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            api_key: Optional bearer token for the OpenAI-compatible endpoint.
            base_url: Endpoint base URL, usually ``http://127.0.0.1:8000/v1``.
            model: Model name exposed by vLLM.
            settings: Runtime-updatable LLM settings.
            audio_passthrough: Whether to pass input audio frames downstream.
            conversation_id: Optional stable id sent to vLLM for
                client-authoritative conversation-cache reuse.
            request_timeout_secs: Total HTTP timeout for one streamed request.
            enable_bash_tool: Whether to expose the local ``run_bash`` tool.
            bash_tool_cwd: Working directory for bash tool calls.
            bash_tool_timeout_secs: Maximum runtime for one bash tool call.
            bash_tool_max_output_chars: Maximum stdout or stderr characters returned.
            bash_tool_max_rounds: Maximum tool-call iterations for one model response.
            bash_tool_event_sender: Optional async callback for publishing
                structured bash tool events to clients.
            **kwargs: Additional arguments for ``LLMService``.
        """
        default_settings = self.Settings(
            model="nemotron_3_nano_omni",
            system_instruction=DEFAULT_VOICE_SYSTEM_INSTRUCTION,
            temperature=NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE,
            max_tokens=NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS,
            top_p=None,
            top_k=NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            audio_prompt=DEFAULT_AUDIO_PROMPT,
            chat_template_kwargs={"enable_thinking": False},
            extra={},
        )

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, run_in_parallel=False, **kwargs)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_completions_url = f"{self._base_url}/chat/completions"
        self._audio_passthrough = audio_passthrough
        self._conversation_id = conversation_id
        self._conversation_cache_committed = False
        self.committed_messages: list[dict[str, Any]] = []
        self._committed_cache_shape_fingerprint: str | None = None
        self._request_timeout_secs = request_timeout_secs
        self._enable_bash_tool = enable_bash_tool
        self._bash_tool_cwd = bash_tool_cwd or os.getcwd()
        self._bash_tool_timeout_secs = bash_tool_timeout_secs
        self._bash_tool_max_output_chars = bash_tool_max_output_chars
        self._bash_tool_max_rounds = bash_tool_max_rounds
        self._bash_tool_event_sender = bash_tool_event_sender
        self._conversation_commit_boundary_tracker = (
            conversation_commit_boundary_tracker
            if conversation_commit_boundary_tracker is not None
            else ConversationCommitBoundaryTracker()
        )
        # Default OFF. The plan (step 5) called for flipping this on, but that
        # premise was wrong: historical-audio stripping is computed *relative to
        # the latest user row*, so a user row that was kept (with audio) in the
        # turn where it was latest gets stripped in every later turn — which
        # makes `committed_messages` (a snapshot of one turn's transcript) never
        # a prefix of a later turn's transcript, so the cache rotates every turn
        # and cross-turn prefix reuse is lost entirely. With stripping OFF the
        # transcript is monotonic, the committed prefix matches, suffix
        # projection works, and the historical audio is processed once and then
        # served from the engine's prefix cache. Set to "1" only if you also
        # disable conversation caching (no `conversation_id`).
        self._strip_historical_audio_from_payload = (
            os.getenv("NEMOTRON_OMNI_STRIP_HISTORICAL_AUDIO_FROM_PAYLOAD", "0") != "0"
        )

        self._session: aiohttp.ClientSession | None = None
        self._generation_task: asyncio.Task | None = None
        self._round_limit_result_task: asyncio.Task | None = None
        self._current_turn_user_key: str | None = None
        self._current_turn_round_count = 0
        self._round_limit_closure_issued = False
        self._turn_tool_results: dict[tuple[str, str], dict[str, Any]] = {}
        self._current_batch_id: str | None = None
        self._running_batch_id: str | None = None
        self._pending_followup_batch_id: str | None = None
        self._pending_exact_assistant_message_for_batch: dict[str, Any] | None = None
        self._stale_batch_ids: set[str] = set()
        self._tool_batch_states: dict[str, ToolBatchState] = {}
        trace_dir = os.getenv("NEMOTRON_OMNI_TRACE_DIR")
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_redact_audio = os.getenv("NEMOTRON_OMNI_TRACE_REDACT_AUDIO", "1") != "0"
        self._top_level_request_seq = 0
        if self._enable_bash_tool:
            self.register_function(
                BASH_TOOL_NAME,
                self._handle_run_bash_function_call,
                cancel_on_interruption=True,
            )

    @property
    def conversation_commit_boundary_tracker(self) -> ConversationCommitBoundaryTracker:
        return self._conversation_commit_boundary_tracker

    def can_generate_metrics(self) -> bool:
        """Return whether the service emits processing, TTFB, and usage metrics."""
        return True

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run out-of-band text inference.

        The service is audio-input oriented and does not currently support
        out-of-band text-only inference.
        """
        raise NotImplementedError(f"run_inference() not supported by {self.__class__.__name__}")

    async def start(self, frame: StartFrame):
        """Start the service and initialize the HTTP client."""
        await super().start(frame)
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout_secs)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self, frame: EndFrame):
        """Stop the service and close active HTTP resources."""
        await super().stop(frame)
        await self._cancel_round_limit_result_task()
        await self._cancel_generation_task()
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close active HTTP resources."""
        await super().cancel(frame)
        await self._cancel_round_limit_result_task()
        await self._cancel_generation_task()
        await self._close_session()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio, VAD, lifecycle, and pass-through frames."""
        await super().process_frame(frame, direction)

        # LLMService.process_frame handles interruptions and settings updates,
        # but it does not forward arbitrary frames for us.
        if isinstance(frame, LLMContextFrame):
            latest_user_key = self._latest_user_turn_key_from_context(frame.context)
            if latest_user_key and latest_user_key != self._current_turn_user_key:
                superseded_sync_batch = (
                    self._current_batch_id is not None or self._pending_followup_batch_id is not None
                )
                # Order matters. Mark the in-flight batch stale first
                # (synchronously) so the sequential runner discards its queued
                # siblings the moment it is unblocked; then cancel the running
                # tool task; only then pop the batch state, reset per-turn
                # dedup/round state, and broadcast cancels for the queued ids.
                # If we popped/reset before cancelling the running tool, a tool
                # that finished in the await gap could repopulate per-turn
                # `_turn_tool_results` for the superseding turn and emit a stale
                # run_llm=True followup that supersedes it.
                self._mark_current_batch_stale()
                await self._cancel_round_limit_result_task()
                await self._cancel_generation_task()
                if superseded_sync_batch:
                    await self._cancel_running_sync_function_calls()
                queued_cancellations = self._prepare_batch_supersession()
                self._reset_turn_tool_state(latest_user_key)
                if queued_cancellations:
                    await self._broadcast_queued_function_call_cancellations(queued_cancellations)
            else:
                self._maybe_credit_pending_followup_batch(latest_user_key)
                await self._cancel_generation_task()

            self._generation_task = self.create_task(
                self._run_context_generation_task(frame.context),
                name="nemotron_omni_context_completion",
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            queued_cancellations = self._prepare_batch_supersession()
            if self._generation_task:
                self._generation_task.cancel()
            await self.push_frame(frame, direction)
            await self._cancel_round_limit_result_task()
            await self._cancel_generation_task()
            if queued_cancellations:
                await self._broadcast_queued_function_call_cancellations(queued_cancellations)
        elif isinstance(frame, LLMRunFrame):
            logger.debug(f"{self}: ignoring {frame.name}; LLMContextFrame triggers inference")
        else:
            await self.push_frame(frame, direction)

    async def _run_context_generation_task(self, context: LLMContext) -> None:
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"{self}: completion failed: {exc}")
            await self.push_frame(ErrorFrame(error=str(exc)))
        finally:
            await self._finalize_generation_task()
            if self._generation_task is asyncio.current_task():
                self._generation_task = None

    async def _finalize_generation_task(self) -> None:
        try:
            await asyncio.shield(self.stop_processing_metrics())
        except Exception as exc:
            logger.warning(f"{self}: failed stopping processing metrics: {exc}")

        try:
            await asyncio.shield(self.push_frame(LLMFullResponseEndFrame()))
        except Exception as exc:
            logger.warning(f"{self}: failed pushing LLMFullResponseEndFrame: {exc}")

    async def _cancel_generation_task(self):
        if self._generation_task:
            await self.cancel_task(self._generation_task)
            self._generation_task = None

    async def _cancel_round_limit_result_task(self):
        if self._round_limit_result_task:
            await self.cancel_task(self._round_limit_result_task)
            self._round_limit_result_task = None

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _latest_user_turn_key_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            return json.dumps(
                copy.deepcopy(message),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        return None

    def _latest_user_turn_key_from_context(self, context: LLMContext) -> str | None:
        snapshot = self._normalized_request_snapshot(copy.deepcopy(context))
        if snapshot is None:
            return None
        return self._latest_user_turn_key_from_messages(self._with_system_message(snapshot.messages))

    def _committable_messages_after_success(
        self,
        full_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        provisional_user_turn_keys = (
            self._conversation_commit_boundary_tracker.provisional_user_turn_keys()
        )
        if provisional_user_turn_keys:
            provisional_user_turn_key_set = set(provisional_user_turn_keys)
            for index, message in enumerate(full_messages):
                if message.get("role") != "user":
                    continue
                message_key = json.dumps(
                    copy.deepcopy(message),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if message_key in provisional_user_turn_key_set:
                    return copy.deepcopy(full_messages[: index + 1])

        # Advance the durable conversation-cache boundary only through the
        # latest user row. Rows appended after that user belong to the active
        # assistant turn; sync-tool re-entry can still rewrite or delete them
        # before the next user turn settles the transcript.
        latest_user_index = -1
        for index in range(len(full_messages) - 1, -1, -1):
            if full_messages[index].get("role") == "user":
                latest_user_index = index
                break
        if latest_user_index < 0:
            return []
        return copy.deepcopy(full_messages[: latest_user_index + 1])

    def _reset_turn_tool_state(self, latest_user_key: str | None) -> None:
        self._current_turn_user_key = latest_user_key
        self._current_turn_round_count = 0
        self._round_limit_closure_issued = False
        self._turn_tool_results = {}
        self._pending_followup_batch_id = None

    def _mark_current_batch_stale(self) -> str | None:
        if self._current_batch_id is not None:
            self._stale_batch_ids.add(self._current_batch_id)
        return self._current_batch_id

    def _prepare_batch_supersession(self) -> list[FunctionCallFromLLM]:
        batch_id = self._mark_current_batch_stale()
        if batch_id is None:
            self._pending_followup_batch_id = None
            return []

        batch_state = self._tool_batch_states.pop(batch_id, None)
        if batch_state is None:
            if self._pending_followup_batch_id == batch_id:
                self._pending_followup_batch_id = None
            if self._current_batch_id == batch_id:
                self._current_batch_id = None
            return []

        for signature in batch_state.pending_signature_keys:
            self._turn_tool_results.pop(signature, None)

        if self._pending_followup_batch_id == batch_id:
            self._pending_followup_batch_id = None
        if self._current_batch_id == batch_id:
            self._current_batch_id = None

        queued_calls: list[FunctionCallFromLLM] = []
        for runner_item in batch_state.runner_items:
            if runner_item.tool_call_id in batch_state.started_ids:
                continue
            queued_calls.append(
                FunctionCallFromLLM(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    context=runner_item.context,
                )
            )
        return queued_calls

    def _maybe_credit_pending_followup_batch(self, latest_user_key: str | None) -> None:
        batch_id = self._pending_followup_batch_id
        if batch_id is None:
            return
        batch_state = self._tool_batch_states.get(batch_id)
        if batch_state is None:
            self._pending_followup_batch_id = None
            return
        if batch_state.user_turn_key != latest_user_key:
            return

        if batch_state.has_real_execution and not batch_state.round_counted:
            self._current_turn_round_count += 1
            batch_state.round_counted = True
        self._pending_followup_batch_id = None
        self._tool_batch_states.pop(batch_id, None)
        if self._current_batch_id == batch_id:
            self._current_batch_id = None

    def _mark_batch_ready_for_followup(self, tool_call_id: str) -> bool:
        batch_id = self._running_batch_id
        if batch_id is None:
            return False

        batch_state = self._tool_batch_states.get(batch_id)
        if batch_state is None:
            return False

        final_runner_item = next(
            (runner_item for runner_item in batch_state.runner_items if runner_item.run_llm),
            None,
        )
        if final_runner_item is None or final_runner_item.tool_call_id != tool_call_id:
            return False

        batch_state.completed = True
        self._pending_followup_batch_id = batch_id
        if batch_state.has_real_execution and not batch_state.round_counted:
            self._current_turn_round_count += 1
            batch_state.round_counted = True
        return True

    async def _await_generation_task_before_tool_followup(self) -> None:
        task = self._generation_task
        if task is None or task.done() or task is asyncio.current_task():
            return
        await asyncio.shield(task)

    async def _cancel_running_sync_function_calls(self) -> None:
        for function_name, entry in list(self._functions.items()):
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)

    async def _broadcast_queued_function_call_cancellations(
        self,
        function_calls: list[FunctionCallFromLLM],
    ) -> None:
        if not function_calls:
            return

        for function_call in function_calls:
            await self.broadcast_frame(
                FunctionCallCancelFrame,
                function_name=function_call.function_name,
                tool_call_id=function_call.tool_call_id,
            )
        await self._call_event_handler("on_function_calls_cancelled", function_calls)

    def _messages_for_adapter_boundary(
        self,
        context: LLMContext,
    ) -> list[Any]:
        adapter_llm = self.get_llm_adapter().id_for_llm_specific_messages
        filtered_messages: list[Any] = []
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) and message.llm != adapter_llm:
                continue
            filtered_messages.append(copy.deepcopy(message))
        return filtered_messages

    def _provider_messages_from_universal_messages(
        self,
        messages: list[Any],
    ) -> list[dict[str, Any]]:
        provider_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                candidate = copy.deepcopy(message.message)
            else:
                candidate = copy.deepcopy(message)

            if isinstance(candidate, dict) and candidate.get("role") == "developer":
                candidate["role"] = "user"

            converted = self._convert_context_message(candidate)
            if converted is not None:
                provider_messages.append(converted)
        return provider_messages

    def _provider_tools_from_context(
        self,
        context: LLMContext,
    ) -> list[dict[str, Any]] | None:
        provider_tools = self.get_llm_adapter().from_standard_tools(context.tools)
        if not context_is_given(provider_tools) or not provider_tools:
            return None
        return copy.deepcopy(provider_tools)

    def _provider_tool_choice_from_context(
        self,
        context: LLMContext,
        *,
        provider_tools: list[dict[str, Any]] | None,
    ) -> Any | None:
        if not provider_tools or not context_is_given(context.tool_choice):
            return None
        return copy.deepcopy(context.tool_choice)

    def _cache_shape_fingerprint(
        self,
        *,
        provider_tools: list[dict[str, Any]] | None,
        provider_tool_choice: Any | None,
    ) -> str:
        # `model` is part of cache-attach semantics — the engine checkpoint is
        # rendered with the model's tokenizer + chat template, so a model change
        # must rotate the conversation_id (an append-only suffix under the old id
        # would otherwise look cache-compatible).
        prompt_shape: dict[str, Any] = {"model": self._settings.model}
        if provider_tools is not None:
            prompt_shape["tools"] = copy.deepcopy(provider_tools)
        if provider_tool_choice is not None:
            prompt_shape["tool_choice"] = copy.deepcopy(provider_tool_choice)
        if self._settings.chat_template_kwargs:
            prompt_shape["chat_template_kwargs"] = copy.deepcopy(
                self._settings.chat_template_kwargs
            )
        if self._settings.extra:
            prompt_shape["extra"] = copy.deepcopy(self._settings.extra)
        return json.dumps(prompt_shape, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _normalized_request_snapshot(
        self,
        context: LLMContext,
    ) -> NormalizedRequestSnapshot | None:
        universal_messages = self._messages_for_adapter_boundary(context)
        if not universal_messages:
            logger.debug(f"{self}: ignoring empty LLM context")
            return None

        provider_messages = self._provider_messages_from_universal_messages(universal_messages)
        if not provider_messages:
            logger.debug(f"{self}: ignoring LLM context without provider-visible messages")
            return None

        provider_tools = self._provider_tools_from_context(context)
        provider_tool_choice = self._provider_tool_choice_from_context(
            context,
            provider_tools=provider_tools,
        )
        return NormalizedRequestSnapshot(
            context=context,
            universal_messages=universal_messages,
            messages=provider_messages,
            tools=provider_tools,
            tool_choice=provider_tool_choice,
            cache_shape_fingerprint=self._cache_shape_fingerprint(
                provider_tools=provider_tools,
                provider_tool_choice=provider_tool_choice,
            ),
        )

    def _build_payload(
        self,
        snapshot: NormalizedRequestSnapshot,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        current_full = self._with_system_message(snapshot.messages)
        if self._strip_historical_audio_from_payload:
            current_full = self._strip_historical_audio_from_messages(current_full)
        if not current_full or self._latest_user_turn_key_from_messages(current_full) is None:
            logger.debug(f"{self}: ignoring LLM context without a latest user message")
            return None

        request_messages = copy.deepcopy(current_full)
        if self._conversation_id is not None:
            committed_prefix_matches = current_full[: len(self.committed_messages)] == self.committed_messages
            fingerprint_changed = (
                self._conversation_cache_committed
                and self._committed_cache_shape_fingerprint is not None
                and snapshot.cache_shape_fingerprint != self._committed_cache_shape_fingerprint
            )
            non_append_rewrite = self._conversation_cache_committed and not committed_prefix_matches

            if fingerprint_changed or non_append_rewrite:
                rotation_reason = (
                    "cache-shape change"
                    if fingerprint_changed
                    else "non-append committed-prefix rewrite"
                )
                if non_append_rewrite:
                    logger.warning(
                        f"{self}: committed-prefix diverged before rotation: "
                        f"{self._describe_committed_prefix_divergence(current_full)}"
                    )
                self._rotate_conversation_cache_projection(reason=rotation_reason)
                request_messages = copy.deepcopy(current_full)
            elif (
                self._conversation_cache_committed
                and self._committed_cache_shape_fingerprint == snapshot.cache_shape_fingerprint
                and committed_prefix_matches
            ):
                request_messages = copy.deepcopy(current_full[len(self.committed_messages) :])

        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "_cache_shape_fingerprint": snapshot.cache_shape_fingerprint,
        }
        if self._conversation_id is not None:
            payload["conversation_id"] = self._conversation_id
            payload["conversation_committed_message_count"] = len(
                self._committable_messages_after_success(current_full)
            )
            if (
                self._conversation_cache_committed
                and self._committed_cache_shape_fingerprint == snapshot.cache_shape_fingerprint
                and current_full[: len(self.committed_messages)] == self.committed_messages
            ):
                payload["conversation_require_cache"] = True
        if snapshot.tools is not None:
            payload["tools"] = copy.deepcopy(snapshot.tools)
            if snapshot.tool_choice is not None:
                payload["tool_choice"] = copy.deepcopy(snapshot.tool_choice)

        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens
        if self._settings.temperature is not None:
            payload["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            payload["top_p"] = self._settings.top_p
        if self._settings.top_k is not None:
            payload["top_k"] = self._settings.top_k
        if self._settings.frequency_penalty is not None:
            payload["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty is not None:
            payload["presence_penalty"] = self._settings.presence_penalty
        if self._settings.seed is not None:
            payload["seed"] = self._settings.seed
        if self._settings.chat_template_kwargs:
            payload["chat_template_kwargs"] = self._settings.chat_template_kwargs

        if self._settings.extra:
            # `extra` carries model-specific knobs; it must not be allowed to
            # clobber the protocol-owned fields (messages / conversation_id /
            # conversation_require_cache / tools / tool_choice / model / stream),
            # because doing so would silently desync the cache contract while
            # `_process_context` still promotes `committed_messages` as if the
            # original transcript went over the wire.
            for key, value in self._settings.extra.items():
                if key in _RESERVED_PAYLOAD_KEYS:
                    logger.warning(
                        f"{self}: ignoring reserved key {key!r} in settings.extra; "
                        "it would override a protocol-owned request field"
                    )
                    continue
                payload[key] = value

        return payload, copy.deepcopy(current_full)

    @staticmethod
    def _compact_message_repr(message: Any) -> str:
        if not isinstance(message, dict):
            return f"<non-dict {type(message).__name__}>"
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            content_repr = repr(content[:80])
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        parts.append(f"text:{(part.get('text') or '')[:40]!r}")
                    else:
                        parts.append(str(ptype))
                else:
                    parts.append(f"<{type(part).__name__}>")
            content_repr = "[" + ",".join(parts) + "]"
        elif content is None:
            content_repr = "None"
        else:
            content_repr = f"<{type(content).__name__}>"
        extra = ""
        if message.get("tool_calls"):
            tc_names = [
                (tc.get("function") or {}).get("name")
                for tc in message["tool_calls"]
                if isinstance(tc, dict)
            ]
            extra += f" tool_calls={tc_names}"
        if message.get("tool_call_id"):
            extra += f" tool_call_id={message.get('tool_call_id')!r}"
        if "name" in message:
            extra += f" name={message.get('name')!r}"
        return f"{{role={role!r} content={content_repr}{extra}}}"

    def _describe_committed_prefix_divergence(
        self, current_full: list[dict[str, Any]]
    ) -> str:
        committed = self.committed_messages
        prefix = current_full[: len(committed)]
        diff_idx: int | None = None
        for idx in range(min(len(committed), len(prefix))):
            if committed[idx] != prefix[idx]:
                diff_idx = idx
                break
        if diff_idx is None and len(committed) != len(prefix):
            diff_idx = min(len(committed), len(prefix))
        head = (
            f"len(committed)={len(committed)} len(current_full)={len(current_full)} "
            f"first_diff_idx={diff_idx}"
        )
        if diff_idx is None:
            return head + " (no element diff -- prefix already matches?)"
        committed_at = (
            self._compact_message_repr(committed[diff_idx])
            if diff_idx < len(committed)
            else "<missing>"
        )
        current_at = (
            self._compact_message_repr(current_full[diff_idx])
            if diff_idx < len(current_full)
            else "<missing>"
        )
        return f"{head} committed[{diff_idx}]={committed_at} current_full[{diff_idx}]={current_at}"

    def _rotate_conversation_cache_projection(self, *, reason: str) -> None:
        if self._conversation_id is None:
            return

        old_conversation_id = self._conversation_id
        self._conversation_id = f"pipecat-{uuid.uuid4().hex}"
        self._conversation_cache_committed = False
        self.committed_messages = []
        self._committed_cache_shape_fingerprint = None
        self._turn_tool_results = {}
        logger.info(
            f"{self}: rotating conversation_id from {old_conversation_id} "
            f"to {self._conversation_id} after {reason}"
        )

    def _strip_historical_audio_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest_user_index = -1
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                latest_user_index = index
                break
        if latest_user_index < 0:
            return copy.deepcopy(messages)

        stripped_messages = copy.deepcopy(messages)
        stripped_audio_parts = 0
        for index, message in enumerate(stripped_messages):
            if index == latest_user_index or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            filtered_content = []
            for item in content:
                if not isinstance(item, dict):
                    filtered_content.append(item)
                    continue
                item_type = item.get("type")
                if item_type in {"audio_url", "input_audio"}:
                    stripped_audio_parts += 1
                    continue
                filtered_content.append(item)
            if filtered_content:
                message["content"] = filtered_content

        if stripped_audio_parts:
            logger.debug(
                f"{self}: stripped {stripped_audio_parts} historical audio part(s) "
                "from payload messages"
            )
        return stripped_messages

    def _convert_context_message(self, message: Any) -> dict[str, Any] | None:
        if isinstance(message, LLMSpecificMessage):
            logger.debug(f"{self}: skipping LLM-specific context message for {message.llm}")
            return None
        if not isinstance(message, dict):
            logger.debug(f"{self}: skipping unsupported context message: {message!r}")
            return None

        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            logger.debug(f"{self}: skipping context message with unsupported role {role!r}")
            return None

        converted: dict[str, Any] = {"role": role}
        if "tool_calls" in message:
            converted["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            converted["tool_call_id"] = message["tool_call_id"]
        if "name" in message:
            converted["name"] = message["name"]

        content = message.get("content")
        if isinstance(content, str) or content is None:
            converted["content"] = content
            return converted

        if not isinstance(content, list):
            logger.debug(f"{self}: skipping unsupported content in context message: {content!r}")
            return None

        converted_content: list[dict[str, Any]] = []
        for item in content:
            converted_item = self._convert_context_content_part(item)
            if converted_item:
                converted_content.append(converted_item)

        if not converted_content:
            return None

        converted["content"] = converted_content
        return converted

    def _convert_context_content_part(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")
        if item_type == "text":
            return {"type": "text", "text": item.get("text", "")}
        if item_type == "audio_url":
            return item
        if item_type == "input_audio":
            input_audio = item.get("input_audio") or {}
            data = input_audio.get("data") or item.get("audio")
            if not data:
                return None
            audio_format = input_audio.get("format") or item.get("format") or "wav"
            url = data if str(data).startswith("data:") else f"data:audio/{audio_format};base64,{data}"
            return {"type": "audio_url", "audio_url": {"url": url}}
        if item_type == "image_url":
            return item
        if "text" in item:
            return {"type": "text", "text": item["text"]}

        logger.debug(f"{self}: skipping unsupported context content part: {item!r}")
        return None

    def _count_audio_parts(self, messages: list[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            count += sum(1 for item in content if item.get("type") == "audio_url")
        return count

    def _message_role_summary(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "?")
            audio_parts = self._count_audio_parts([message])
            if audio_parts:
                parts.append(f"{role}[audio={audio_parts}]")
            elif message.get("tool_calls"):
                parts.append(f"{role}[tool_calls]")
            elif role == "tool":
                parts.append(f"{role}[result]")
            else:
                parts.append(role)
        return ",".join(parts)

    def _with_system_message(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_messages = copy.deepcopy(messages)
        if not self._settings.system_instruction:
            return normalized_messages

        system_message = {
            "role": "system",
            "content": self._settings.system_instruction,
        }
        if normalized_messages and normalized_messages[0].get("role") == "system":
            normalized_messages[0] = system_message
            return normalized_messages
        return [system_message, *normalized_messages]

    def _trace_request_id(self, top_level_request_seq: int, attempt_num: int) -> str:
        conversation_part = self._conversation_id or "no-conversation"
        return (
            f"nemotron-{conversation_part}-turn-{top_level_request_seq:03d}-"
            f"attempt-{attempt_num:02d}"
        )

    def _trace_json_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if (
                self._trace_redact_audio
                and
                set(value.keys()) == {"url"}
                and isinstance(value["url"], str)
                and value["url"].startswith("data:audio/")
            ):
                url = value["url"]
                _, _, encoded = url.partition(",")
                digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
                return {
                    "url": f"<data-audio-base64 sha256={digest} chars={len(encoded)}>",
                }
            return {
                str(key): self._trace_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._trace_json_value(item) for item in value]
        return value

    def _write_trace_file(
        self,
        *,
        trace_id: str,
        phase: str,
        payload: dict[str, Any],
    ) -> None:
        if self._trace_dir is None:
            return
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            safe_trace_id = _TRACE_NAME_RE.sub("_", Path(str(trace_id)).name)
            safe_phase = _TRACE_NAME_RE.sub("_", Path(str(phase)).name)
            if not safe_trace_id or not safe_phase:
                return
            trace_path = self._trace_dir / f"{safe_trace_id}.{safe_phase}.json"
            trace_payload = {
                "trace_id": trace_id,
                "phase": phase,
                "timestamp": time.time(),
                **payload,
            }
            trace_path.write_text(
                json.dumps(
                    self._trace_json_value(trace_payload),
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"{self}: failed to write trace file for {trace_id}: {exc}")

    async def _process_context(self, context: LLMContext):
        snapshot_context = copy.deepcopy(context)
        snapshot = self._normalized_request_snapshot(snapshot_context)
        if snapshot is None:
            return

        payload_info = self._build_payload(snapshot)
        if payload_info is None:
            return

        payload, full_messages = payload_info
        result = await self._run_completion_payload(
            payload,
            full_messages=full_messages,
            request_description=(
                f"context with {len(payload['messages'])} messages and "
                f"{self._count_audio_parts(payload['messages'])} audio parts"
            ),
            start_ttfb=True,
        )
        if result is None:
            return
        if self._conversation_id is not None:
            committed_message_count = payload.get("conversation_committed_message_count")
            if isinstance(committed_message_count, int):
                self.committed_messages = copy.deepcopy(
                    full_messages[:committed_message_count]
                )
            else:
                self.committed_messages = self._committable_messages_after_success(
                    full_messages
                )
            self._conversation_cache_committed = True
            self._committed_cache_shape_fingerprint = snapshot.cache_shape_fingerprint
        if not result.tool_calls:
            return

        function_calls, surviving_tool_calls = self._function_calls_from_tool_calls(
            result.tool_calls,
            context=context,
        )
        if not function_calls:
            return
        # Build the exact assistant row from the SURVIVING tool calls only, so a
        # dropped malformed call can't leave an assistant(tool_calls) row that
        # names more ids than there will be tool rows for.
        exact_assistant_message = self._build_exact_assistant_message(
            result.output_text,
            surviving_tool_calls,
        )

        if self._round_limit_closure_issued:
            logger.warning(
                f"{self}: tool closure pass for user turn requested more sync tools; "
                "executing no additional tool work"
            )
            return

        if self._current_turn_round_count >= self._bash_tool_max_rounds:
            logger.warning(
                f"{self}: reached bash tool round limit ({self._bash_tool_max_rounds}); "
                "synthesizing terminal tool results for closure pass"
            )
            self._round_limit_closure_issued = True
            await self._emit_round_limit_results(
                function_calls,
                exact_assistant_message=exact_assistant_message,
            )
            return

        self._pending_exact_assistant_message_for_batch = copy.deepcopy(
            exact_assistant_message
        )
        await self.run_function_calls(function_calls)

    async def _run_completion_payload(
        self,
        payload: dict[str, Any],
        *,
        full_messages: list[dict[str, Any]],
        request_description: str,
        start_ttfb: bool,
    ) -> ChatCompletionPassResult | None:
        started_at = time.perf_counter()
        self._top_level_request_seq += 1
        top_level_request_seq = self._top_level_request_seq

        try:
            if start_ttfb:
                await self.start_ttfb_metrics()

            headers = {"Accept": "text/event-stream"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            if not self._session:
                timeout = aiohttp.ClientTimeout(total=self._request_timeout_secs)
                self._session = aiohttp.ClientSession(timeout=timeout)

            cache_info = (
                f" with conversation_id={self._conversation_id}"
                if self._conversation_id
                else ""
            )
            logger.debug(f"{self}: sending {request_description}{cache_info}")
            current_payload = copy.deepcopy(payload)
            retried_after_cache_miss = False
            first_token = True
            attempt_num = 0
            while True:
                attempt_num += 1
                trace_id = self._trace_request_id(top_level_request_seq, attempt_num)
                attempt_messages = current_payload.get("messages")
                if not isinstance(attempt_messages, list):
                    attempt_messages = []
                logger.debug(
                    f"{self}: completion attempt {attempt_num} "
                    f"messages={len(attempt_messages)} "
                    f"roles={self._message_role_summary(attempt_messages)} "
                    f"audio_parts={self._count_audio_parts(attempt_messages)} "
                    f"require_cache={bool(current_payload.get('conversation_require_cache'))}"
                )
                self._write_trace_file(
                    trace_id=trace_id,
                    phase="client-request",
                    payload={
                        "request_description": request_description,
                        "conversation_id": current_payload.get("conversation_id"),
                        "conversation_cache_committed": self._conversation_cache_committed,
                        "committed_messages": self.committed_messages,
                        "committed_cache_shape_fingerprint": self._committed_cache_shape_fingerprint,
                        "messages_role_summary": self._message_role_summary(attempt_messages),
                        "http_payload": self._http_payload(current_payload),
                        "conversation_full_messages": full_messages,
                    },
                )
                try:
                    result = await self._stream_completion_pass(
                        current_payload,
                        headers={**headers, "X-Request-Id": trace_id},
                        first_token=first_token,
                        trace_id=trace_id,
                    )
                except ConversationCacheMissError:
                    if retried_after_cache_miss:
                        raise
                    current_payload = self._build_cache_rebase_payload(
                        payload=current_payload,
                        full_messages=full_messages,
                    )
                    retried_after_cache_miss = True
                    logger.info(
                        f"{self}: conversation cache miss for "
                        f"{self._conversation_id}; retrying with full context"
                    )
                    continue
                first_token = result.first_token
                break
            logger.debug(
                f"{self}: completed response in {time.perf_counter() - started_at:.3f}s: "
                f"{result.output_text!r}"
            )
            return result
        except asyncio.CancelledError:
            logger.debug(f"{self}: completion cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: completion failed: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
            return None

    async def _stream_completion_pass(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        first_token: bool,
        trace_id: str,
    ) -> ChatCompletionPassResult:
        output_text = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async with self._session.post(
            self._chat_completions_url,
            json=self._http_payload(payload),
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                self._write_trace_file(
                    trace_id=trace_id,
                    phase="client-error",
                    payload={
                        "status": response.status,
                        "response_text": error_text,
                    },
                )
                if self._is_conversation_cache_miss(response.status, error_text):
                    raise ConversationCacheMissError(error_text)
                raise RuntimeError(
                    f"vLLM request failed with {response.status}: {error_text}"
                )

            async for event in self._iter_sse_events(response):
                if event == "[DONE]":
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    logger.debug(f"{self}: skipping malformed SSE event: {event!r}")
                    continue

                usage = chunk.get("usage")
                if usage:
                    await self.start_llm_usage_metrics(
                        LLMTokenUsage(
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        )
                    )

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    tool_call_deltas = delta.get("tool_calls") or []
                    for tool_call_delta in tool_call_deltas:
                        self._merge_tool_call_delta(tool_calls_by_index, tool_call_delta)

                    text = delta.get("content") or ""
                    if not text:
                        continue
                    if first_token:
                        first_token = False
                        await self.stop_ttfb_metrics()
                    output_text += text
                    await self._push_llm_text(text)

        result = ChatCompletionPassResult(
            output_text=output_text,
            tool_calls=self._finalize_tool_calls(tool_calls_by_index),
            first_token=first_token,
        )
        self._write_trace_file(
            trace_id=trace_id,
            phase="client-response",
            payload={
                "output_text": result.output_text,
                "tool_calls": result.tool_calls,
                "first_token_pending": result.first_token,
            },
        )
        return result

    @staticmethod
    def _http_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if not key.startswith("_")
        }

    @staticmethod
    def _is_conversation_cache_miss(status: int, error_text: str) -> bool:
        if status != 409:
            return False
        try:
            data = json.loads(error_text)
        except json.JSONDecodeError:
            return False
        error = data.get("error")
        if not isinstance(error, dict):
            return False
        return error.get("type") == "ConversationCacheMissError"

    def _build_cache_rebase_payload(
        self,
        *,
        payload: dict[str, Any],
        full_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        retry_payload = copy.deepcopy(payload)
        retry_payload["messages"] = copy.deepcopy(full_messages)
        retry_payload.pop("conversation_require_cache", None)
        return retry_payload

    def _function_calls_from_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        context: LLMContext,
    ) -> tuple[list[FunctionCallFromLLM], list[dict[str, Any]]]:
        """Build runnable items plus the surviving provider tool-call dicts.

        Returns ``(function_calls, surviving_tool_calls)`` where ``surviving_tool_calls``
        is the subset of the provider's ``tool_calls`` that produced a runnable item,
        each guaranteed to carry a stable ``id`` matching the corresponding
        ``FunctionCallFromLLM`` / ``tool`` row. The exact assistant row MUST be built
        from ``surviving_tool_calls`` (not the raw ``tool_calls``) so a dropped malformed
        call can never leave an ``assistant(tool_calls)`` row that names more ids than the
        runner will produce ``tool`` rows for.
        """
        function_calls: list[FunctionCallFromLLM] = []
        surviving_tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            function_name = function.get("name") or ""
            arguments_text = function.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError:
                logger.warning(
                    f"{self}: failed to parse function call arguments for "
                    f"{function_name or '<missing-name>'}: {arguments_text}"
                )
                continue
            if not isinstance(arguments, dict):
                logger.warning(
                    f"{self}: function call arguments for {function_name or '<missing-name>'} "
                    "did not decode to an object"
                )
                continue
            tool_call_id = tool_call.get("id") or f"call_{index}"
            function_calls.append(
                FunctionCallFromLLM(
                    context=context,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    arguments=arguments,
                )
            )
            surviving = copy.deepcopy(tool_call)
            surviving["id"] = tool_call_id
            surviving_tool_calls.append(surviving)
        return function_calls, surviving_tool_calls

    @staticmethod
    def _build_exact_assistant_message(
        output_text: str,
        surviving_tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": output_text,
            "tool_calls": copy.deepcopy(surviving_tool_calls),
        }

    async def _emit_round_limit_results(
        self,
        function_calls: list[FunctionCallFromLLM],
        *,
        exact_assistant_message: dict[str, Any],
    ) -> None:
        if not function_calls:
            return

        await self.push_frame(
            NemotronExactAssistantMessageFrame(message=copy.deepcopy(exact_assistant_message))
        )
        await self._call_event_handler("on_function_calls_started", function_calls)
        await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=function_calls)
        await self._cancel_round_limit_result_task()
        self._round_limit_result_task = self.create_task(
            self._emit_round_limit_result_frames(function_calls),
            name="nemotron_round_limit_result_frames",
        )

    async def _emit_round_limit_result_frames(
        self,
        function_calls: list[FunctionCallFromLLM],
    ) -> None:
        try:
            await self._await_generation_task_before_tool_followup()
            for index, function_call in enumerate(function_calls):
                await self.broadcast_frame(
                    FunctionCallInProgressFrame,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    cancel_on_interruption=True,
                    group_id=None,
                )
                result = self._build_bash_tool_result(
                    command=str(function_call.arguments.get("code") or ""),
                    command_started=False,
                    exit_code=None,
                    stdout_text="",
                    stderr_text="",
                    timed_out=False,
                    status_override="round_limit_reached",
                    summary_override=(
                        "The per-user-turn sync tool round limit was reached. "
                        "Do not call the tool again; answer from this observation."
                    ),
                )
                await self.broadcast_frame(
                    FunctionCallResultFrame,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    result=result,
                    run_llm=index == len(function_calls) - 1,
                )
        finally:
            if self._round_limit_result_task is asyncio.current_task():
                self._round_limit_result_task = None

    @staticmethod
    def _merge_tool_call_delta(
        tool_calls_by_index: dict[int, dict[str, Any]],
        tool_call_delta: dict[str, Any],
    ) -> None:
        index = tool_call_delta.get("index")
        if index is None:
            index = len(tool_calls_by_index)

        entry = tool_calls_by_index.setdefault(
            index,
            {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if tool_call_delta.get("id"):
            entry["id"] = tool_call_delta["id"]
        if tool_call_delta.get("type"):
            entry["type"] = tool_call_delta["type"]

        function_delta = tool_call_delta.get("function") or {}
        function = entry.setdefault("function", {"name": "", "arguments": ""})
        if function_delta.get("name"):
            function["name"] = f"{function.get('name') or ''}{function_delta['name']}"
        if function_delta.get("arguments"):
            function["arguments"] = (
                f"{function.get('arguments') or ''}{function_delta['arguments']}"
            )

    @staticmethod
    def _finalize_tool_calls(
        tool_calls_by_index: dict[int, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = copy.deepcopy(tool_calls_by_index[index])
            tool_call["id"] = tool_call.get("id") or f"call_{index}"
            tool_call["type"] = tool_call.get("type") or "function"
            function = tool_call.setdefault("function", {})
            function["name"] = function.get("name") or ""
            function["arguments"] = function.get("arguments") or "{}"
            tool_calls.append(tool_call)
        return tool_calls

    async def _run_sequential_function_calls(
        self,
        runner_items: list[FunctionCallRunnerItem] | tuple[FunctionCallRunnerItem, ...],
    ):
        batch_id = uuid.uuid4().hex
        serial_runner_items: list[SerialFunctionCallRunnerItem] = []
        for index, runner_item in enumerate(runner_items):
            serial_runner_items.append(
                SerialFunctionCallRunnerItem(
                    registry_item=runner_item.registry_item,
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    context=runner_item.context,
                    run_llm=index == len(runner_items) - 1,
                    group_id=runner_item.group_id,
                    batch_id=batch_id,
                )
            )

        self._current_batch_id = batch_id
        self._tool_batch_states[batch_id] = ToolBatchState(
            batch_id=batch_id,
            user_turn_key=self._current_turn_user_key,
            runner_items=serial_runner_items,
        )
        self._conversation_commit_boundary_tracker.mark_provisional_batch(
            batch_id=batch_id,
            user_turn_key=self._current_turn_user_key,
        )
        if self._pending_exact_assistant_message_for_batch is not None:
            await self.push_frame(
                NemotronExactAssistantMessageFrame(
                    message=copy.deepcopy(self._pending_exact_assistant_message_for_batch),
                    batch_id=batch_id,
                )
            )
            self._pending_exact_assistant_message_for_batch = None
        await super()._run_sequential_function_calls(serial_runner_items)

    async def _sequential_runner_handler(self):
        while True:
            runner_item = await self._sequential_runner_queue.get()
            batch_id = getattr(runner_item, "batch_id", "")
            if batch_id and batch_id in self._stale_batch_ids:
                logger.debug(
                    f"{self}: discarding stale queued tool call "
                    f"[{runner_item.function_name}:{runner_item.tool_call_id}]"
                )
                continue

            batch_state = self._tool_batch_states.get(batch_id)
            if batch_state is not None:
                batch_state.started_ids.add(runner_item.tool_call_id)

            self._running_batch_id = batch_id or None
            task = self.create_task(self._run_function_call(runner_item))
            self._function_call_tasks[task] = runner_item
            try:
                await task
                if batch_state is not None and runner_item.run_llm:
                    batch_state.completed = True
                    self._pending_followup_batch_id = batch_id
            except asyncio.CancelledError:
                pass
            finally:
                self._function_call_tasks.pop(task, None)
                if self._running_batch_id == batch_id:
                    self._running_batch_id = None

    async def _handle_interruptions(self, frame: InterruptionFrame):
        self._mark_current_batch_stale()
        await super()._handle_interruptions(frame)

    async def _cancel_function_call(self, function_name: str | None):
        cancelled_tasks = set()
        cancelled_items = []
        for task, runner_item in list(self._function_call_tasks.items()):
            if runner_item.registry_item.function_name != function_name:
                continue

            name = runner_item.function_name
            tool_call_id = runner_item.tool_call_id
            logger.debug(f"{self} Cancelling function call [{name}:{tool_call_id}]...")

            if task:
                task.remove_done_callback(self._function_call_task_finished)
                await self.cancel_task(task)
                cancelled_tasks.add(task)

            await self.broadcast_frame(
                FunctionCallCancelFrame,
                function_name=name,
                tool_call_id=tool_call_id,
            )
            cancelled_items.append(
                FunctionCallFromLLM(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    context=runner_item.context,
                )
            )
            logger.debug(f"{self} Function call [{name}:{tool_call_id}] has been cancelled")

        for task in cancelled_tasks:
            self._function_call_task_finished(task)

        if cancelled_items:
            await self._call_event_handler("on_function_calls_cancelled", cancelled_items)

    async def _handle_run_bash_function_call(self, params: FunctionCallParams) -> None:
        result = await self._execute_bash_tool_request(
            arguments=params.arguments,
            tool_call_id=params.tool_call_id,
            function_name=params.function_name,
        )
        if self._mark_batch_ready_for_followup(params.tool_call_id):
            await self._await_generation_task_before_tool_followup()
        await params.result_callback(result)

    async def _execute_bash_tool_request(
        self,
        *,
        arguments: Mapping[str, Any] | Any,
        tool_call_id: str,
        function_name: str,
    ) -> dict[str, Any]:
        code = arguments.get("code") if isinstance(arguments, Mapping) else None
        if not isinstance(code, str) or not code.strip():
            return self._build_bash_tool_result(
                command=code if isinstance(code, str) else "",
                command_started=False,
                exit_code=None,
                stdout_text="",
                stderr_text="",
                timed_out=False,
                status_override="invalid_arguments",
                summary_override="Missing required string argument: code",
            )

        normalized_code = code.strip()
        if self._is_prose_echo_command(normalized_code):
            logger.debug(
                f"{self}: suppressing non-instrumental bash tool call: {normalized_code!r}"
            )
            await self._send_bash_tool_event(
                {
                    "phase": "policy_rejected",
                    "guardrail_triggered": True,
                    "guardrail_kind": "non_instrumental_bash_tool_use",
                    "guardrail_reason": "echo_or_printf_prose",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "code": normalized_code,
                }
            )
            return self._build_bash_tool_result(
                command=normalized_code,
                command_started=False,
                exit_code=None,
                stdout_text="",
                stderr_text="",
                timed_out=False,
                status_override="policy_rejected",
                summary_override=(
                    "Do not use bash to echo, printf, or paraphrase an answer you "
                    "could say directly. The bash tool is only for real command "
                    "execution or explicit user-requested command output. Answer "
                    "the user directly without further tool use."
                ),
            )

        signature = (function_name, normalized_code)
        previous_result = self._turn_tool_results.get(signature)
        if previous_result is not None:
            result = {
                "ok": bool(previous_result.get("ok")),
                "status": "duplicate_suppressed",
                "summary": (
                    "Exact duplicate command suppressed in this assistant turn. "
                    "Reused the prior result instead of re-running it. "
                    f"Original status: {previous_result.get('status') or 'unknown'}."
                ),
                "command": str(previous_result.get("command") or ""),
                "exit_code": previous_result.get("exit_code"),
                "timed_out": bool(previous_result.get("timed_out")),
                "stdout": str(previous_result.get("stdout") or ""),
                "stderr": str(previous_result.get("stderr") or ""),
            }
            logger.debug(
                f"{self}: suppressing duplicate tool call within one user turn: "
                f"{function_name} {normalized_code!r}"
            )
            await self._send_bash_tool_event(
                {
                    "phase": "duplicate_suppressed",
                    "guardrail_triggered": True,
                    "guardrail_kind": "duplicate_tool_call",
                    "guardrail_reason": "exact_duplicate_command",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "signature": normalized_code,
                    "result": result,
                }
            )
            return result

        result = await self._run_bash_tool(normalized_code, tool_call_id=tool_call_id)
        self._turn_tool_results[signature] = copy.deepcopy(result)
        if self._running_batch_id is not None:
            batch_state = self._tool_batch_states.get(self._running_batch_id)
            if batch_state is not None:
                batch_state.pending_signature_keys.add(signature)
                batch_state.has_real_execution = True
        return result

    @staticmethod
    def _is_prose_echo_command(code: str) -> bool:
        stripped = code.strip()
        if not stripped or any(token in stripped for token in ("&&", "||", "|", ";", "$(", "`")):
            return False
        try:
            argv = shlex.split(stripped)
        except ValueError:
            return False
        if not argv or argv[0] not in {"echo", "printf"}:
            return False
        payload_tokens = argv[1:]
        if not payload_tokens:
            return False
        payload_text = " ".join(token.rstrip("\\n") for token in payload_tokens).strip()
        if not payload_text:
            return False
        word_count = len(re.findall(r"[A-Za-z0-9]+", payload_text))
        has_sentence_punctuation = any(char in payload_text for char in ".!,?:;")
        return word_count >= 12 or (word_count >= 8 and has_sentence_punctuation)

    async def _run_bash_tool(self, code: str, *, tool_call_id: str) -> dict[str, Any]:
        logger.debug(
            f"{self}: running bash tool in {self._bash_tool_cwd!r}: {code!r}"
        )
        await self._send_bash_tool_event(
            {
                "phase": "start",
                "tool_call_id": tool_call_id,
                "code": code,
                "cwd": self._bash_tool_cwd,
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                code,
                cwd=self._bash_tool_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            result = self._build_bash_tool_result(
                command=code,
                command_started=False,
                exit_code=None,
                stdout_text="",
                stderr_text="",
                timed_out=False,
                summary_override=f"Failed to start bash: {exc}",
            )
            self._log_bash_tool_result(tool_call_id=tool_call_id, code=code, result=result)
            await self._send_bash_tool_event(
                {
                    "phase": "result",
                    "tool_call_id": tool_call_id,
                    "code": code,
                    "cwd": self._bash_tool_cwd,
                    "result": result,
                }
            )
            return result

        stdout, stderr, timed_out = await self._communicate_bash_process(process)

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        result = self._build_bash_tool_result(
            command=code,
            command_started=True,
            exit_code=process.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            timed_out=timed_out,
        )
        self._log_bash_tool_result(tool_call_id=tool_call_id, code=code, result=result)
        await self._send_bash_tool_event(
            {
                "phase": "result",
                "tool_call_id": tool_call_id,
                "code": code,
                "cwd": self._bash_tool_cwd,
                "result": result,
            }
        )
        return result

    async def _communicate_bash_process(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes, bool]:
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._bash_tool_timeout_secs,
            )
            return stdout, stderr, False
        except asyncio.TimeoutError:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            stdout, stderr = await process.communicate()
            return stdout, stderr, True
        except asyncio.CancelledError:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            with contextlib.suppress(Exception):
                await asyncio.shield(process.communicate())
            raise

    def _build_bash_tool_result(
        self,
        *,
        command: str,
        command_started: bool,
        exit_code: int | None,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool,
        status_override: str | None = None,
        summary_override: str | None = None,
        ok_override: bool | None = None,
    ) -> dict[str, Any]:
        stdout_truncated = len(stdout_text) > self._bash_tool_max_output_chars
        stderr_truncated = len(stderr_text) > self._bash_tool_max_output_chars
        command_not_found = command_started and exit_code == 127
        if status_override is not None:
            ok = ok_override if ok_override is not None else False
            status = status_override
            summary = summary_override or ""
        elif command_started and exit_code == 0 and not timed_out:
            ok = True
            status = "success"
            summary = "Command completed successfully."
            if stderr_text:
                summary += (
                    " The command wrote output to stderr, but exit_code is 0 so "
                    "that stderr output should not by itself be treated as failure."
                )
        elif timed_out:
            ok = False
            status = "timed_out"
            summary = f"Command timed out after {self._bash_tool_timeout_secs:g} seconds."
        elif not command_started:
            ok = False
            status = "failed_to_start"
            summary = summary_override or "Command could not be started."
        elif command_not_found:
            ok = False
            status = "command_not_found"
            summary = "Command exited with 127, which usually means the shell could not find it."
        else:
            ok = False
            status = "nonzero_exit"
            summary = (
                f"Command exited with status {exit_code}. Inspect stdout and stderr; "
                "a nonzero exit code can still include useful command output."
            )

        if stdout_truncated or stderr_truncated:
            truncated_parts: list[str] = []
            if stdout_truncated:
                truncated_parts.append("stdout")
            if stderr_truncated:
                truncated_parts.append("stderr")
            summary += (
                " Returned "
                + " and ".join(truncated_parts)
                + f" were truncated to {self._bash_tool_max_output_chars} chars."
            )

        return {
            "ok": ok,
            "status": status,
            "summary": summary,
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": self._truncate_tool_output(stdout_text),
            "stderr": self._truncate_tool_output(stderr_text),
        }

    def _log_bash_tool_result(
        self,
        *,
        tool_call_id: str,
        code: str,
        result: dict[str, Any],
    ) -> None:
        logger.debug(
            f"{self}: model-facing bash tool result for {tool_call_id} "
            f"({code!r}): {self._tool_result_json(result)}"
        )

    async def _send_bash_tool_event(self, payload: dict[str, Any]) -> None:
        if not self._bash_tool_event_sender:
            return

        message = {
            "type": "bash-tool",
            "timestamp": time.time(),
            **payload,
        }
        try:
            await self._bash_tool_event_sender(message)
        except Exception as exc:
            logger.warning(f"{self}: failed to send bash tool RTVI event: {exc}")

    def _truncate_tool_output(self, text: str) -> str:
        if len(text) <= self._bash_tool_max_output_chars:
            return text
        omitted = len(text) - self._bash_tool_max_output_chars
        return (
            text[: self._bash_tool_max_output_chars]
            + f"\n...[truncated {omitted} chars]"
        )

    @staticmethod
    def _tool_result_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True)

    async def _iter_sse_events(self, response: aiohttp.ClientResponse):
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                yield line[len("data:") :].strip()

        line = buffer.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()
