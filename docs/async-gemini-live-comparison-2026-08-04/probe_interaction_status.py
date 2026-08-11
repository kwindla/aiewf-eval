#!/usr/bin/env python3
"""Run the normal CLI with a sanitized Gemini Live raw-event trace.

The trace runs before the SDK converter and records event structure, status,
part types and sizes, tools, usage, resumption, GoAway, and errors. It never
records prompts, transcript text, audio bytes, arguments, credentials, or
resumption handles. Benchmark behavior is otherwise unchanged.
"""

from __future__ import annotations

from multi_turn_eval.services.gemini_raw_trace import install_raw_live_event_trace


def main() -> None:
    install_raw_live_event_trace()
    from multi_turn_eval.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
