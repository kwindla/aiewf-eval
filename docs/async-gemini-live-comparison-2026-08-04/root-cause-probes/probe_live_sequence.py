#!/usr/bin/env python3
"""Run the opening benchmark sequence in one raw Gemini Live session."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from probe_live_boundaries import (
    DEFAULT_MODEL,
    REPO_DIR,
    benchmark_turns,
    build_config,
    load_audio,
    send_audio_chunks,
    send_silence_until_stopped,
    summarize_message,
)


async def receive_interaction(
    session: Any,
    boundary_ns: int,
    deadline: float,
    require_interaction_status: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "events": [],
        "typed_messages": 0,
        "empty_typed_messages": 0,
        "first_output_ms": None,
        "terminal_ms": None,
        "terminal_signal": None,
        "in_progress_count": 0,
        "tool_calls": [],
    }
    started = time.monotonic()
    while time.monotonic() - started < deadline:
        remaining = deadline - (time.monotonic() - started)
        try:
            async with asyncio.timeout(remaining):
                async for message in session.receive():
                    received_ns = time.monotonic_ns()
                    event = summarize_message(message, received_ns)
                    result["typed_messages"] += 1
                    if len(event) == 1:
                        result["empty_typed_messages"] += 1
                        continue
                    result["events"].append(event)
                    has_output = bool(
                        event.get("parts")
                        or event.get("output_transcription")
                        or event.get("tool_calls")
                    )
                    if has_output and result["first_output_ms"] is None:
                        result["first_output_ms"] = (
                            received_ns - boundary_ns
                        ) / 1_000_000
                    result["tool_calls"].extend(event.get("tool_calls", []))

                    status = event.get("interaction_status")
                    if status == "IN_PROGRESS":
                        result["in_progress_count"] += 1
                    if status == "REQUIRES_ACTION":
                        result["terminal_ms"] = (
                            received_ns - boundary_ns
                        ) / 1_000_000
                        result["terminal_signal"] = "REQUIRES_ACTION"
                        return result
                    if event.get("turn_complete") and not require_interaction_status:
                        result["terminal_ms"] = (
                            received_ns - boundary_ns
                        ) / 1_000_000
                        result["terminal_signal"] = "turn_complete"
                        return result
                    if result["tool_calls"]:
                        result["terminal_signal"] = "tool_call"
                        return result
        except TimeoutError:
            break
    result["deadline_expired"] = True
    return result


def compact_interaction(interaction: dict[str, Any]) -> None:
    interaction["input_transcription"] = "".join(
        event.get("input_transcription") or "" for event in interaction["events"]
    )
    interaction["output_transcription"] = "".join(
        event.get("output_transcription") or "" for event in interaction["events"]
    )
    interaction["audio_output_bytes"] = sum(
        part.get("bytes", 0)
        for event in interaction["events"]
        for part in event.get("parts", [])
        if part.get("type") == "audio"
    )


async def send_audio_turn(
    session: Any,
    args: argparse.Namespace,
    turn_index: int,
) -> tuple[int, asyncio.Event, asyncio.Task | None, dict[str, Any]]:
    audio, sample_rate = load_audio(turn_index)
    send_result = {
        "audio_send_calls": 0,
        "audio_bytes_sent": 0,
        "silence_send_calls": 0,
        "silence_bytes_sent": 0,
        "maximum_send_ms": 0.0,
    }
    stop_silence = asyncio.Event()
    if args.boundary == "explicit_vad":
        await session.send_realtime_input(activity_start=types.ActivityStart())
    else:
        pre_roll = bytes(int(sample_rate * 2 * args.pre_roll_ms / 1000))
        await send_audio_chunks(session, pre_roll, sample_rate, send_result)

    await send_audio_chunks(session, audio, sample_rate, send_result)
    if args.boundary == "explicit_vad":
        await session.send_realtime_input(activity_end=types.ActivityEnd())
    boundary_ns = time.monotonic_ns()
    silence_task = None
    if args.boundary in {"auto_silence", "auto_tuned"}:
        silence_task = asyncio.create_task(
            send_silence_until_stopped(
                session,
                sample_rate,
                stop_silence,
                send_result,
            )
        )
    return boundary_ns, stop_silence, silence_task, send_result


async def run_session(args: argparse.Namespace, trial: int) -> dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    client = genai.Client(api_key=api_key)
    result: dict[str, Any] = {
        "trial": trial,
        "model": args.model,
        "boundary": args.boundary,
        "deadline_seconds": args.deadline,
        "max_turn": args.max_turn,
        "turns": [],
        "started_unix": time.time(),
    }
    config = build_config(args)

    async with client.aio.live.connect(model=args.model, config=config) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Greet the user briefly.")],
            ),
            turn_complete=True,
        )
        await session.send_realtime_input(text=" ")
        greeting = await receive_interaction(
            session,
            time.monotonic_ns(),
            args.deadline,
            args.require_interaction_status,
        )
        compact_interaction(greeting)
        result["greeting"] = greeting
        if not greeting.get("terminal_signal") or greeting["terminal_signal"] == "tool_call":
            result["stop_reason"] = "greeting_not_terminal"
            return result

        for turn_index in range(args.max_turn + 1):
            boundary_ns, stop_silence, silence_task, send_result = await send_audio_turn(
                session, args, turn_index
            )
            interaction = await receive_interaction(
                session,
                boundary_ns,
                args.deadline,
                args.require_interaction_status,
            )
            stop_silence.set()
            if silence_task:
                silence_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await silence_task
            compact_interaction(interaction)
            interaction.update(
                {
                    "turn": turn_index,
                    "prompt": benchmark_turns()[turn_index]["input"],
                    "send": send_result,
                    "exceeds_15s_watchdog": interaction["first_output_ms"] is None
                    or interaction["first_output_ms"] > 15_000,
                }
            )
            result["turns"].append(interaction)
            args.output.write_text(json.dumps([result], indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "turn": turn_index,
                        "first_output_ms": interaction["first_output_ms"],
                        "terminal_ms": interaction["terminal_ms"],
                        "terminal_signal": interaction["terminal_signal"],
                        "input_transcription": interaction["input_transcription"],
                        "output_chars": len(interaction["output_transcription"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not interaction.get("terminal_signal") or interaction["terminal_signal"] == "tool_call":
                result["stop_reason"] = (
                    "unexpected_tool_call"
                    if interaction["terminal_signal"] == "tool_call"
                    else "turn_not_terminal"
                )
                break
        else:
            result["stop_reason"] = "target_reached"
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
        choices=("auto_silence", "auto_tuned", "explicit_vad"),
        required=True,
    )
    parser.add_argument("--max-turn", type=int, default=8)
    parser.add_argument("--deadline", type=float, default=60.0)
    parser.add_argument("--pre-roll-ms", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=("minimal", "benchmark"), default="benchmark")
    parser.add_argument("--tools", choices=("none", "benchmark"), default="benchmark")
    parser.add_argument("--history", choices=("none",), default="none")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        result = await run_session(args, trial)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "trial": trial,
                    "turns": len(result["turns"]),
                    "stop_reason": result.get("stop_reason"),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    load_dotenv(REPO_DIR / ".env")
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
