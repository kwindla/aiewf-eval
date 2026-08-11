"""Sanitized raw-WebSocket event tracing for Gemini Live diagnostics."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

import websockets
from google.genai import _live_converters, errors, types
from google.genai.live import AsyncSession
from loguru import logger


_ORIGINAL_SEND_REALTIME_INPUT = AsyncSession.send_realtime_input


def _text_field_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": value is not None}
    text = value.get("text")
    return {
        "present": True,
        "text_chars": len(text) if isinstance(text, str) else 0,
        "finished": value.get("finished"),
    }


def _part_summary(part: Any) -> Dict[str, Any]:
    if not isinstance(part, dict):
        return {"type": type(part).__name__}
    if "inlineData" in part:
        inline_data = part.get("inlineData") or {}
        data = inline_data.get("data")
        return {
            "type": "inline_data",
            "mime_type": inline_data.get("mimeType"),
            "encoded_chars": len(data) if isinstance(data, str) else 0,
        }
    if "text" in part:
        text = part.get("text")
        return {
            "type": "text",
            "text_chars": len(text) if isinstance(text, str) else 0,
            "thought": bool(part.get("thought")),
        }
    if "functionCall" in part:
        call = part.get("functionCall") or {}
        args = call.get("args")
        return {
            "type": "function_call",
            "name": call.get("name"),
            "id_present": bool(call.get("id")),
            "arg_keys": sorted(args) if isinstance(args, dict) else [],
        }
    return {"type": "other", "keys": sorted(part)}


def summarize_raw_live_event(response: Any, *, raw_length: int) -> Dict[str, Any]:
    """Summarize a raw Live event without retaining content or credentials."""
    if not isinstance(response, dict):
        return {
            "raw_length": raw_length,
            "response_type": type(response).__name__,
        }

    server_content = response.get("serverContent") or {}
    if not isinstance(server_content, dict):
        server_content = {}
    model_turn = server_content.get("modelTurn") or {}
    parts = model_turn.get("parts") if isinstance(model_turn, dict) else None
    part_summaries = (
        [_part_summary(part) for part in parts] if isinstance(parts, list) else []
    )

    summary: Dict[str, Any] = {
        "raw_length": raw_length,
        "top_level_keys": sorted(response),
        "server_content_keys": sorted(server_content),
        "turn_complete": server_content.get("turnComplete"),
        "generation_complete": server_content.get("generationComplete"),
        "interaction_status": server_content.get("interactionStatus"),
        "interrupted": server_content.get("interrupted"),
        "waiting_for_input": server_content.get("waitingForInput"),
        "model_turn_parts": part_summaries,
    }

    if "inputTranscription" in server_content:
        summary["input_transcription"] = _text_field_summary(
            server_content.get("inputTranscription")
        )
    if "outputTranscription" in server_content:
        summary["output_transcription"] = _text_field_summary(
            server_content.get("outputTranscription")
        )

    tool_call = response.get("toolCall")
    if isinstance(tool_call, dict):
        calls = tool_call.get("functionCalls")
        if isinstance(calls, list):
            summary["tool_calls"] = [
                {
                    "name": call.get("name") if isinstance(call, dict) else None,
                    "id_present": bool(call.get("id"))
                    if isinstance(call, dict)
                    else False,
                    "arg_keys": sorted(call.get("args"))
                    if isinstance(call, dict) and isinstance(call.get("args"), dict)
                    else [],
                }
                for call in calls
            ]

    cancellation = response.get("toolCallCancellation")
    if isinstance(cancellation, dict):
        ids = cancellation.get("ids")
        summary["tool_call_cancellation"] = {
            "count": len(ids) if isinstance(ids, list) else 0
        }

    resumption = response.get("sessionResumptionUpdate")
    if isinstance(resumption, dict):
        summary["session_resumption"] = {
            "resumable": resumption.get("resumable"),
            "handle_present": bool(resumption.get("newHandle")),
        }

    go_away = response.get("goAway")
    if isinstance(go_away, dict):
        summary["go_away"] = {"time_left": go_away.get("timeLeft")}

    usage = response.get("usageMetadata")
    if isinstance(usage, dict):
        summary["usage"] = {
            key: value
            for key, value in usage.items()
            if key.endswith("TokenCount") or key in {"trafficType"}
        }

    if "code" in response or "error" in response:
        error = response.get("error")
        summary["error"] = {
            "code": response.get("code")
            or (error.get("code") if isinstance(error, dict) else None),
            "status": error.get("status") if isinstance(error, dict) else None,
        }

    return summary


async def _receive_with_raw_event_trace(self: AsyncSession) -> types.LiveServerMessage:
    parameter_model = types.LiveServerMessage()
    sequence = getattr(self, "_mte_raw_event_sequence", 0) + 1
    wait_started_ns = time.monotonic_ns()
    try:
        raw_response = await self._ws.recv(decode=False)
    except TypeError:
        raw_response = await self._ws.recv()
    except asyncio.CancelledError:
        logger.info(
            "[GEMINI_RAW_RECV_CANCELLED] next_sequence={} wait_ms={:.3f}",
            sequence,
            (time.monotonic_ns() - wait_started_ns) / 1_000_000,
        )
        raise
    except websockets.exceptions.ConnectionClosed as error:
        if error.rcvd:
            code = error.rcvd.code
            reason = error.rcvd.reason
        else:
            code = 1006
            reason = websockets.frames.CLOSE_CODE_EXPLANATIONS.get(
                code, "Abnormal closure."
            )
        logger.error(
            "[GEMINI_RAW_CLOSE] code={} reason_chars={}", code, len(reason or "")
        )
        errors.APIError.raise_error(code, reason, None)

    if raw_response:
        try:
            response = json.loads(raw_response)
        except json.decoder.JSONDecodeError as error:
            logger.error(
                "[GEMINI_RAW_DECODE_ERROR] raw_length={}", len(raw_response)
            )
            raise ValueError("Failed to parse Gemini Live response") from error
    else:
        response = {}

    self._mte_raw_event_sequence = sequence
    summary = summarize_raw_live_event(response, raw_length=len(raw_response or b""))
    summary["sequence"] = sequence
    summary["received_monotonic_ns"] = time.monotonic_ns()
    summary["receive_wait_ms"] = (
        summary["received_monotonic_ns"] - wait_started_ns
    ) / 1_000_000
    logger.info(
        "[GEMINI_RAW_EVENT] {}",
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
    )

    if self._api_client.vertexai:
        response_dict = _live_converters._LiveServerMessage_from_vertex(response)
    else:
        response_dict = _live_converters._LiveServerMessage_from_mldev(response)

    if not response_dict and response:
        errors.APIError.raise_error(response.get("code"), response, None)
    return types.LiveServerMessage._from_response(
        response=response_dict, kwargs=parameter_model.model_dump()
    )


def _blob_size(blob: Any) -> int | None:
    if blob is None:
        return None
    data = blob.get("data") if isinstance(blob, dict) else getattr(blob, "data", None)
    if isinstance(data, (bytes, bytearray, memoryview, str)):
        return len(data)
    return None


async def _send_realtime_input_with_trace(
    self: AsyncSession,
    *,
    media: Any = None,
    audio: Any = None,
    audio_stream_end: bool | None = None,
    video: Any = None,
    text: str | None = None,
    activity_start: Any = None,
    activity_end: Any = None,
) -> None:
    """Trace successful SDK sends without recording input content."""
    sequence = getattr(self, "_mte_raw_send_sequence", 0) + 1
    wait_started_ns = time.monotonic_ns()
    try:
        await _ORIGINAL_SEND_REALTIME_INPUT(
            self,
            media=media,
            audio=audio,
            audio_stream_end=audio_stream_end,
            video=video,
            text=text,
            activity_start=activity_start,
            activity_end=activity_end,
        )
    except asyncio.CancelledError:
        logger.info(
            "[GEMINI_RAW_SEND_CANCELLED] next_sequence={} wait_ms={:.3f}",
            sequence,
            (time.monotonic_ns() - wait_started_ns) / 1_000_000,
        )
        raise
    except Exception as error:
        logger.error(
            "[GEMINI_RAW_SEND_ERROR] next_sequence={} error_type={} wait_ms={:.3f}",
            sequence,
            type(error).__name__,
            (time.monotonic_ns() - wait_started_ns) / 1_000_000,
        )
        raise

    sent_ns = time.monotonic_ns()
    self._mte_raw_send_sequence = sequence
    summary = {
        "sequence": sequence,
        "sent_monotonic_ns": sent_ns,
        "send_wait_ms": (sent_ns - wait_started_ns) / 1_000_000,
        "audio_bytes": _blob_size(audio),
        "media_bytes": _blob_size(media),
        "video_bytes": _blob_size(video),
        "text_chars": len(text) if isinstance(text, str) else None,
        "audio_stream_end": audio_stream_end,
        "activity_start": activity_start is not None,
        "activity_end": activity_end is not None,
    }
    logger.info(
        "[GEMINI_RAW_SEND] {}",
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
    )


def install_raw_live_event_trace() -> None:
    """Install sanitized receive and send tracing once for this process."""
    if not getattr(AsyncSession, "_mte_raw_event_trace_installed", False):
        AsyncSession._receive = _receive_with_raw_event_trace
        AsyncSession._mte_raw_event_trace_installed = True
    if not getattr(AsyncSession, "_mte_raw_send_trace_installed", False):
        AsyncSession.send_realtime_input = _send_realtime_input_with_trace
        AsyncSession._mte_raw_send_trace_installed = True
