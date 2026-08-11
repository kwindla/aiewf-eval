#!/usr/bin/env python3
"""Standalone Gemini Live turn-boundary probe for Async Gemini Live.

Run this with the private interaction-status SDK overlay. The probe bypasses
Pipecat and records only event structure, timings, transcriptions, and output
sizes; it never records audio bytes or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


REPO_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = os.environ.get("MTE_ASYNC_GEMINI_LIVE_MODEL")


def enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def load_audio(turn: int) -> tuple[bytes, int]:
    path = REPO_DIR / "benchmarks/_shared/audio" / f"turn_{turn:03d}.wav"
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM: {path}")
        sample_rate = wav.getframerate()
        audio = wav.readframes(wav.getnframes())
    return audio, sample_rate


def benchmark_turns() -> list[dict[str, Any]]:
    from benchmarks._shared import turns

    return turns


def benchmark_tools() -> list[dict[str, Any]]:
    from benchmarks._shared import ToolsSchemaForTest
    from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter

    return GeminiLLMAdapter().to_provider_tools_format(ToolsSchemaForTest)


def benchmark_system_instruction() -> str:
    from benchmarks.aiwf_medium_context.config import system_instruction

    return system_instruction


def build_config(args: argparse.Namespace) -> types.LiveConnectConfig:
    system_instruction = (
        benchmark_system_instruction()
        if args.system == "benchmark"
        else "Answer the user's request accurately and briefly."
    )
    config = types.LiveConnectConfig(
        generation_config=types.GenerationConfig(
            max_output_tokens=4096,
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                ),
                language_code="en-US",
            ),
        ),
        system_instruction=system_instruction,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
            include_thoughts=False,
        ),
        history_config=types.HistoryConfig(initial_history_in_client_content=True),
    )
    if args.tools == "benchmark":
        config.tools = benchmark_tools()
    if args.boundary == "explicit_vad":
        config.realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        )
    elif args.boundary == "auto_tuned":
        config.realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                silence_duration_ms=1200,
            )
        )
    return config


def build_seed_history(target_turn: int) -> list[types.Content]:
    history: list[types.Content] = []
    for turn in benchmark_turns()[:target_turn]:
        history.extend(
            [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=turn["input"])],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=turn["golden_text"])],
                ),
            ]
        )
    return history


def summarize_message(message: types.LiveServerMessage, received_ns: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"received_monotonic_ns": received_ns}
    server_content = message.server_content
    if server_content:
        summary.update(
            {
                "turn_complete": server_content.turn_complete,
                "generation_complete": server_content.generation_complete,
                "interaction_status": enum_value(
                    getattr(server_content, "interaction_status", None)
                ),
                "interrupted": server_content.interrupted,
                "waiting_for_input": server_content.waiting_for_input,
            }
        )
        if server_content.input_transcription:
            summary["input_transcription"] = server_content.input_transcription.text
        if server_content.output_transcription:
            summary["output_transcription"] = server_content.output_transcription.text
        if server_content.model_turn and server_content.model_turn.parts:
            parts: list[dict[str, Any]] = []
            for part in server_content.model_turn.parts:
                if part.inline_data:
                    parts.append(
                        {
                            "type": "audio",
                            "bytes": len(part.inline_data.data or b""),
                            "mime_type": part.inline_data.mime_type,
                        }
                    )
                elif part.text is not None:
                    parts.append(
                        {
                            "type": "text",
                            "characters": len(part.text),
                            "thought": bool(getattr(part, "thought", False)),
                        }
                    )
                else:
                    parts.append({"type": "other"})
            summary["parts"] = parts
    if message.tool_call and message.tool_call.function_calls:
        summary["tool_calls"] = [
            {
                "name": call.name,
                "id_present": bool(call.id),
                "argument_keys": sorted((call.args or {}).keys()),
            }
            for call in message.tool_call.function_calls
        ]
    if message.usage_metadata:
        summary["usage"] = {
            "prompt_tokens": message.usage_metadata.prompt_token_count,
            "response_tokens": message.usage_metadata.response_token_count,
            "thought_tokens": message.usage_metadata.thoughts_token_count,
            "total_tokens": message.usage_metadata.total_token_count,
        }
    if message.go_away:
        summary["go_away"] = True
    if message.session_resumption_update:
        summary["session_resumption"] = True
    if getattr(message, "voice_activity_detection_signal", None):
        summary["voice_activity_detection_signal"] = str(
            message.voice_activity_detection_signal
        )
    if getattr(message, "voice_activity", None):
        summary["voice_activity"] = str(message.voice_activity)
    return summary


def is_meaningful(event: dict[str, Any]) -> bool:
    return len(event) > 1


async def receive_loop(
    session: Any,
    result: dict[str, Any],
    terminal: asyncio.Event,
    first_output: asyncio.Event,
    require_interaction_status: bool,
) -> None:
    try:
        while True:
            async for message in session.receive():
                received_ns = time.monotonic_ns()
                event = summarize_message(message, received_ns)
                result["raw_typed_messages"] += 1
                if not is_meaningful(event):
                    result["empty_typed_messages"] += 1
                    continue
                result["events"].append(event)

                if event.get("parts") or event.get("output_transcription") or event.get(
                    "tool_calls"
                ):
                    if result.get("first_output_ns") is None:
                        result["first_output_ns"] = received_ns
                    first_output.set()

                status = event.get("interaction_status")
                if status == "REQUIRES_ACTION":
                    result["terminal_ns"] = received_ns
                    result["terminal_signal"] = "REQUIRES_ACTION"
                    terminal.set()
                elif status == "IN_PROGRESS":
                    result["in_progress_count"] += 1
                elif event.get("turn_complete") and not require_interaction_status:
                    result["terminal_ns"] = received_ns
                    result["terminal_signal"] = "turn_complete"
                    terminal.set()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        result["receive_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        terminal.set()


async def send_audio_chunks(
    session: Any,
    audio: bytes,
    sample_rate: int,
    result: dict[str, Any],
) -> None:
    chunk_bytes = int(sample_rate * 2 * 0.02)
    for offset in range(0, len(audio), chunk_bytes):
        chunk = audio[offset : offset + chunk_bytes]
        started_ns = time.monotonic_ns()
        await session.send_realtime_input(
            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={sample_rate}")
        )
        result["audio_send_calls"] += 1
        result["audio_bytes_sent"] += len(chunk)
        result["maximum_send_ms"] = max(
            result["maximum_send_ms"],
            (time.monotonic_ns() - started_ns) / 1_000_000,
        )
        await asyncio.sleep(0.02)


async def send_silence_until_stopped(
    session: Any,
    sample_rate: int,
    stop: asyncio.Event,
    result: dict[str, Any],
) -> None:
    silence = bytes(int(sample_rate * 2 * 0.02))
    while not stop.is_set():
        started_ns = time.monotonic_ns()
        await session.send_realtime_input(
            audio=types.Blob(data=silence, mime_type=f"audio/pcm;rate={sample_rate}")
        )
        result["silence_send_calls"] += 1
        result["silence_bytes_sent"] += len(silence)
        result["maximum_send_ms"] = max(
            result["maximum_send_ms"],
            (time.monotonic_ns() - started_ns) / 1_000_000,
        )
        await asyncio.sleep(0.02)


async def send_target(
    session: Any,
    args: argparse.Namespace,
    result: dict[str, Any],
    silence_stop: asyncio.Event,
) -> asyncio.Task | None:
    turn = benchmark_turns()[args.turn]
    if args.boundary in {"client_content", "client_content_nudge"}:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=turn["input"])],
            ),
            turn_complete=True,
        )
        if args.boundary == "client_content_nudge":
            await session.send_realtime_input(text=" ")
            result["realtime_text_nudge_sent"] = True
        result["boundary_ns"] = time.monotonic_ns()
        return None

    if args.boundary == "realtime_text":
        await session.send_realtime_input(text=turn["input"])
        result["boundary_ns"] = time.monotonic_ns()
        return None

    audio, sample_rate = load_audio(args.turn)
    result["sample_rate"] = sample_rate
    result["wav_bytes"] = len(audio)

    if args.boundary == "explicit_vad":
        await session.send_realtime_input(activity_start=types.ActivityStart())
        result["activity_start_sent"] = True
    elif args.pre_roll_ms:
        pre_roll = bytes(int(sample_rate * 2 * args.pre_roll_ms / 1000))
        await send_audio_chunks(session, pre_roll, sample_rate, result)

    await send_audio_chunks(session, audio, sample_rate, result)

    if args.boundary == "explicit_vad":
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        result["activity_end_sent"] = True
    elif args.boundary == "auto_stream_end":
        await session.send_realtime_input(audio_stream_end=True)
        result["audio_stream_end_sent"] = True

    result["boundary_ns"] = time.monotonic_ns()
    if args.boundary in {"auto_silence", "auto_tuned"}:
        return asyncio.create_task(
            send_silence_until_stopped(session, sample_rate, silence_stop, result)
        )
    return None


def finalize_result(result: dict[str, Any]) -> None:
    boundary_ns = result.get("boundary_ns")
    first_output_ns = result.pop("first_output_ns", None)
    terminal_ns = result.pop("terminal_ns", None)
    result["first_output_ms"] = (
        (first_output_ns - boundary_ns) / 1_000_000
        if first_output_ns is not None and boundary_ns is not None
        else None
    )
    result["terminal_ms"] = (
        (terminal_ns - boundary_ns) / 1_000_000
        if terminal_ns is not None and boundary_ns is not None
        else None
    )
    result.pop("boundary_ns", None)
    output_text = "".join(
        event.get("output_transcription") or "" for event in result["events"]
    )
    input_text = "".join(
        event.get("input_transcription") or "" for event in result["events"]
    )
    result["output_transcription"] = output_text
    result["input_transcription"] = input_text
    result["audio_output_bytes"] = sum(
        part.get("bytes", 0)
        for event in result["events"]
        for part in event.get("parts", [])
        if part.get("type") == "audio"
    )
    result["tool_calls"] = [
        call
        for event in result["events"]
        for call in event.get("tool_calls", [])
    ]
    if result.get("receive_error"):
        result["outcome"] = "receive_error"
    elif result.get("terminal_signal"):
        result["outcome"] = "terminal"
    elif result["tool_calls"]:
        result["outcome"] = "tool_call_without_terminal"
    elif result["audio_output_bytes"]:
        result["outcome"] = "audio_without_terminal"
    else:
        result["outcome"] = "timeout_no_output"


async def run_trial(args: argparse.Namespace, trial: int) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required")

    result: dict[str, Any] = {
        "trial": trial,
        "model": args.model,
        "boundary": args.boundary,
        "turn": args.turn,
        "system": args.system,
        "tools": args.tools,
        "history": args.history,
        "deadline_seconds": args.deadline,
        "started_unix": time.time(),
        "raw_typed_messages": 0,
        "empty_typed_messages": 0,
        "events": [],
        "in_progress_count": 0,
        "audio_send_calls": 0,
        "audio_bytes_sent": 0,
        "silence_send_calls": 0,
        "silence_bytes_sent": 0,
        "maximum_send_ms": 0.0,
    }
    config = build_config(args)
    client = genai.Client(api_key=api_key)
    terminal = asyncio.Event()
    first_output = asyncio.Event()
    silence_stop = asyncio.Event()
    silence_task: asyncio.Task | None = None

    async with client.aio.live.connect(model=args.model, config=config) as session:
        receiver = asyncio.create_task(
            receive_loop(
                session,
                result,
                terminal,
                first_output,
                args.require_interaction_status,
            )
        )
        try:
            if args.history == "golden" and args.turn:
                await session.send_client_content(
                    turns=build_seed_history(args.turn),
                    turn_complete=False,
                )
                result["history_messages"] = args.turn * 2
            else:
                result["history_messages"] = 0

            silence_task = await send_target(session, args, result, silence_stop)
            try:
                await asyncio.wait_for(terminal.wait(), timeout=args.deadline)
            except TimeoutError:
                result["deadline_expired"] = True
        finally:
            silence_stop.set()
            if silence_task:
                silence_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await silence_task
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

    finalize_result(result)
    result["finished_unix"] = time.time()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        required=DEFAULT_MODEL is None,
        help="Gemini Live model ID (or set MTE_ASYNC_GEMINI_LIVE_MODEL)",
    )
    parser.add_argument(
        "--require-interaction-status",
        action="store_true",
        help="Ignore turn_complete until interaction_status is REQUIRES_ACTION",
    )
    parser.add_argument(
        "--boundary",
        choices=(
            "auto_silence",
            "auto_tuned",
            "auto_stop",
            "auto_stream_end",
            "explicit_vad",
            "client_content",
            "client_content_nudge",
            "realtime_text",
        ),
        required=True,
    )
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--system", choices=("minimal", "benchmark"), default="minimal")
    parser.add_argument("--tools", choices=("none", "benchmark"), default="none")
    parser.add_argument("--history", choices=("none", "golden"), default="none")
    parser.add_argument("--deadline", type=float, default=60.0)
    parser.add_argument("--pre-roll-ms", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(1, args.trials + 1):
        result = await run_trial(args, trial)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "trial": trial,
                    "outcome": result["outcome"],
                    "first_output_ms": result["first_output_ms"],
                    "terminal_ms": result["terminal_ms"],
                    "input_transcription": result["input_transcription"],
                    "output_chars": len(result["output_transcription"]),
                    "tool_calls": [call["name"] for call in result["tool_calls"]],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def main() -> None:
    load_dotenv(REPO_DIR / ".env")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
