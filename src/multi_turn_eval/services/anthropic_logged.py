"""Anthropic service wrapper with payload logging and content-aware TTFB.

The upstream ``AnthropicLLMService`` calls ``stop_ttfb_metrics()`` immediately
after the message stream is *created* — before any stream events arrive, let
alone the first user-visible token. For no-thinking runs that is roughly the
first-token time, but with adaptive thinking enabled (claude-fable-5 effort
sweeps) the model can think for seconds after the stream opens, so the upstream
metric would report a near-constant TTFB regardless of effort level.

``LoggedCerebrasLLMService`` / ``LoggedGoogleLLMService`` fix the analogous
problem by mirroring upstream's ``_process_context``. Anthropic's is large, so
we instead hook the narrower ``_create_message_stream`` seam this subclass
already overrides for payload logging:

- upstream's premature ``stop_ttfb_metrics()`` call is swallowed while a gate
  flag is set;
- the returned stream is wrapped in a generator that stops TTFB on the first
  user-visible event (text delta, tool_use block start, or tool-args JSON
  delta — thinking/signature deltas do not count) and emits
  ``RawTTFBMetricsData`` on the first event of any kind.

This preserves the project convention (see ``multi_turn_eval.metrics``):

- ``ttfb_ms``     — time to first non-thinking token (voice-agent latency)
- ``raw_ttfb_ms`` — time to first stream event of any kind
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator

from loguru import logger
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.frames.frames import MetricsFrame
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.services.anthropic.llm import AnthropicLLMService

from multi_turn_eval.metrics import RawTTFBMetricsData
from multi_turn_eval.services.filler import filler_suffix

_filler_logged = False


def _filler_position() -> str:
    position = os.getenv("MTE_FILLER_POSITION", "suffix").strip().lower()
    return position if position in {"suffix", "prefix", "system"} else "suffix"


def _weave_filler(text: str, filler: str, position: str) -> str:
    if position == "prefix":
        return filler + " " + text.lstrip()
    return text.rstrip() + " " + filler


def _log_filler(filler: str, position: str) -> None:
    global _filler_logged
    if _filler_logged:
        return
    _filler_logged = True
    parts = filler.split()
    logger.info(
        f"MTE_FILLER_DOTS active: {len(parts)} x {parts[0]!r} filler tokens, "
        f"position={position} (history left filler-free)"
    )


def _apply_filler_anthropic(messages: list) -> list:
    """Return a filled copy of Anthropic-format ``messages``.

    Anthropic puts tool results in ``role:"user"`` messages whose content is a
    list of ``tool_result`` blocks; those are skipped so the filler lands on the
    real user question — matching the OpenAI path, where tool results are
    ``role:"tool"`` and never receive filler. Applied on a copy at request-build
    time, so the persisted context stays filler-free. ``system`` placement is
    handled separately because Anthropic sends it outside ``messages``.
    """
    filler = filler_suffix()
    if not filler or not messages:
        return messages
    position = _filler_position()
    if position == "system":
        return messages
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        new_content: Any
        if isinstance(content, str):
            new_content = _weave_filler(content, filler, position)
        elif isinstance(content, list):
            # Find the last text block; skip messages that carry only
            # tool_result (or other non-text) blocks — do NOT graft a text
            # block onto them.
            text_idx = next(
                (
                    j
                    for j in range(len(content) - 1, -1, -1)
                    if isinstance(content[j], dict)
                    and content[j].get("type") == "text"
                    and isinstance(content[j].get("text"), str)
                ),
                None,
            )
            if text_idx is None:
                continue
            new_content = list(content)
            part = dict(new_content[text_idx])
            part["text"] = _weave_filler(part["text"], filler, position)
            new_content[text_idx] = part
        else:
            continue
        new_messages = list(messages)
        nm = dict(msg)
        nm["content"] = new_content
        new_messages[i] = nm
        _log_filler(filler, position)
        return new_messages
    return messages


def _apply_filler_anthropic_system(system: Any) -> Any:
    """Return a filled copy of Anthropic's string or text-block system prompt."""
    filler = filler_suffix()
    if not filler or _filler_position() != "system":
        return system
    if isinstance(system, str):
        _log_filler(filler, "system")
        return _weave_filler(system, filler, "system")
    if not isinstance(system, list):
        return system
    for i in range(len(system) - 1, -1, -1):
        block = system[i]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            copied = list(system)
            copied_block = dict(block)
            copied_block["text"] = _weave_filler(block["text"], filler, "system")
            copied[i] = copied_block
            _log_filler(filler, "system")
            return copied
    return system


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
    return ""


def _is_visible_event(event: Any) -> bool:
    """True for stream events that carry user-visible output.

    Text deltas, tool_use block starts, and tool-args JSON deltas count.
    Thinking blocks, thinking/signature deltas, message_start/delta, and pings
    do not. Attribute checks mirror upstream's own event dispatch (Anthropic
    delta models only define their own fields, so ``hasattr`` is reliable).
    """
    etype = getattr(event, "type", None)
    if etype == "content_block_start":
        block = getattr(event, "content_block", None)
        return getattr(block, "type", None) == "tool_use"
    if etype == "content_block_delta":
        delta = getattr(event, "delta", None)
        return hasattr(delta, "text") or hasattr(delta, "partial_json")
    return False


class PatchedAnthropicLLMAdapter(AnthropicLLMAdapter):
    """Adapter that round-trips thought messages whose thinking text is empty.

    With ``thinking.display: "omitted"`` (claude-fable-5's default), thinking
    blocks stream with an empty ``thinking`` field plus a signature. Upstream's
    ``_from_anthropic_specific_message`` only rebuilds a thought message into a
    thinking block when BOTH text and signature are truthy, so an empty-text
    thought falls through to a role-less dict and the next request crashes
    (KeyError 'role'). The Anthropic docs explicitly support replaying empty
    thinking blocks: "pass each thinking block back to the API exactly as
    received, including blocks whose thinking field is empty."
    """

    def _from_universal_context_messages(self, universal_context_messages, **kwargs):
        # Signature-less thoughts (stream interrupted before the
        # signature_delta arrived) carry nothing replayable. Drop them before
        # conversion: converted 1:1 they would surface as phantom "(empty)"
        # assistant turns via upstream's empty-content fix whenever they don't
        # sit adjacent to another assistant message (Codex review finding).
        filtered = [
            m
            for m in universal_context_messages
            if not self._is_signatureless_thought(m)
        ]
        return super()._from_universal_context_messages(filtered, **kwargs)

    @staticmethod
    def _is_signatureless_thought(message) -> bool:
        return (
            isinstance(message, LLMSpecificMessage)
            and isinstance(message.message, dict)
            and message.message.get("type") == "thought"
            and not message.message.get("signature")
        )

    def _from_anthropic_specific_message(self, message):
        msg = message.message
        if isinstance(msg, dict) and msg.get("type") == "thought":
            signature = msg.get("signature")
            if signature:
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": msg.get("text") or "",
                            "signature": signature,
                        }
                    ],
                }
            # Belt and suspenders: signature-less thoughts are filtered out in
            # _from_universal_context_messages above and shouldn't reach here.
            # An empty assistant message merges into an adjacent assistant
            # turn; upstream's empty-content fix covers the standalone edge.
            return {"role": "assistant", "content": []}
        return super()._from_anthropic_specific_message(message)


class LoggedAnthropicLLMService(AnthropicLLMService):
    """Anthropic service with payload logging and content-aware TTFB."""

    adapter_class = PatchedAnthropicLLMAdapter

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # While True, stop_ttfb_metrics() calls are swallowed; the stream
        # wrapper clears the flag and stops TTFB on first visible content.
        self._mte_gate_ttfb = False

    async def start_ttfb_metrics(self, *, start_time: float | None = None) -> None:
        # New measurement cycle: clear any stale gate, e.g. if the prior turn
        # was cancelled between stream creation and wrapper entry, where the
        # generator's finally never ran (Codex review finding).
        self._mte_gate_ttfb = False
        await super().start_ttfb_metrics(start_time=start_time)

    async def stop_ttfb_metrics(self, *, end_time: float | None = None) -> None:
        if self._mte_gate_ttfb:
            return
        await super().stop_ttfb_metrics(end_time=end_time)

    async def _create_message_stream(self, api_call, params):  # type: ignore[override]
        # Inject filler into the outgoing request only; persisted context stays
        # untouched. Anthropic sends its system prompt outside ``messages``.
        msgs = params.get("messages")
        if isinstance(msgs, list):
            params["messages"] = _apply_filler_anthropic(msgs)
        if "system" in params:
            params["system"] = _apply_filler_anthropic_system(params["system"])

        if _env_bool("MTE_LOG_ANTHROPIC_PAYLOADS", False):
            safe_params = _json_safe(params)
            messages = safe_params.get("messages", [])
            last_user = (
                _extract_last_user_text(messages) if isinstance(messages, list) else ""
            )
            logger.debug(
                f"{self}: Anthropic exact request payload "
                f"(message_count={len(messages) if isinstance(messages, list) else 'n/a'}, "
                f"last_user_text={last_user!r}) | "
                f"{json.dumps(safe_params, ensure_ascii=False)}"
            )

        raw_t0 = time.monotonic()
        response = await super()._create_message_stream(api_call, params)
        if not params.get("stream"):
            return response

        # Gate the premature stop_ttfb_metrics() upstream makes right after
        # this method returns; the wrapper below stops TTFB on real content.
        self._mte_gate_ttfb = True
        return self._gate_ttfb_stream(response, raw_t0)

    async def _gate_ttfb_stream(self, response, raw_t0: float) -> AsyncIterator[Any]:
        # No try/finally gate reset here: an abandoned generator's finally runs
        # at async finalization time (GC/aclose), which after a cancellation
        # can land mid-way through the NEXT turn and clear a gate that turn
        # just set — accepting its premature TTFB stop. start_ttfb_metrics()
        # resets the gate at the start of each measurement cycle instead; a
        # stream that errors before any visible event simply leaves the turn
        # without a ttfb_ms (intentional — missing beats wrong).
        raw_emitted = False
        async for event in response:
            if not raw_emitted:
                raw_emitted = True
                await self.push_frame(
                    MetricsFrame(
                        data=[
                            RawTTFBMetricsData(
                                processor=self.name,
                                value=time.monotonic() - raw_t0,
                            )
                        ]
                    )
                )
            if self._mte_gate_ttfb and _is_visible_event(event):
                self._mte_gate_ttfb = False
                await self.stop_ttfb_metrics()
            yield event
