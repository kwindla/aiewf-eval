#!/usr/bin/env python3
"""Frozen mechanical outcome scorer for AIWF turns 12 and 15."""

from __future__ import annotations

import json
import re
from typing import Any


FALSE_COMPLETION = re.compile(
    r"\b(i(?:'ve| have)|we(?:'ve| have))\s+(?:now\s+)?(?:submitted|added|recorded)|"
    r"\b(?:has been|is now)\s+(?:submitted|added|recorded)|\btaken care of\b",
    re.IGNORECASE,
)
CONFIRMATION = re.compile(
    r"\b(?:confirm|confirmation|would you like|shall i|should i|want me to|go ahead|"
    r"please say|please confirm|one moment)\b|\?",
    re.IGNORECASE,
)


def normalize(value: Any) -> str:
    text = str(value or "").casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        result.append(
            {
                "id": raw.get("id"),
                "name": function.get("name"),
                "args": parse_arguments(function.get("arguments", function.get("args"))),
                "raw": raw,
            }
        )
    return result


def transcript_message(row: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for index, call in enumerate(row.get("tool_calls") or []):
        calls.append(
            {
                "id": call.get("id", f"historical-{index}"),
                "type": "function",
                "function": {
                    "name": call.get("name"),
                    "arguments": json.dumps(call.get("args"), sort_keys=True),
                },
            }
        )
    return {"role": "assistant", "content": row.get("assistant_text") or "", "tool_calls": calls}


def expected_call_is_correct(turn: int, call: dict[str, Any]) -> bool:
    args = call.get("args")
    if not isinstance(args, dict):
        return False
    if turn == 12:
        if call.get("name") != "submit_session_suggestion":
            return False
        # Freeze the two semantically exact variants observed in the 150-run
        # historical corpus.  Token-presence scoring would incorrectly accept
        # negations, reordered decoys, or calls with invented extra arguments.
        if set(args) != {"name", "suggestion_text"}:
            return False
        return normalize(args["name"]) == "jennifer smith" and normalize(
            args["suggestion_text"]
        ) in {
            "a session on state machine abstractions for complex workflows",
            "state machine abstractions for complex workflows",
        }
    if turn == 15:
        if call.get("name") != "submit_dietary_request":
            return False
        if set(args) != {"name", "dietary_preference"}:
            return False
        return (
            normalize(args["name"]) == "jennifer smith"
            and normalize(args["dietary_preference"]) == "vegan"
        )
    raise ValueError(f"unsupported target turn: {turn}")


def score_message(
    turn: int,
    message: dict[str, Any] | None,
    *,
    request_error: str | None = None,
) -> dict[str, Any]:
    if request_error:
        return {"success": False, "category": "request_server_failure", "detail": request_error}
    if not isinstance(message, dict):
        return {"success": False, "category": "response_parser_failure", "detail": "missing message"}

    calls = message_tool_calls(message)
    if len(calls) > 1:
        return {
            "success": False,
            "category": "duplicate_or_multiple_tool_calls",
            "detail": f"{len(calls)} parsed calls",
            "calls": calls,
        }
    if len(calls) == 1:
        call = calls[0]
        if call["args"] is None:
            # A structurally delivered tool call with invalid or absent arguments
            # is a model outcome, not evidence that the response parser failed.
            return {"success": False, "category": "malformed_tool_call", "calls": calls}
        if expected_call_is_correct(turn, call):
            return {"success": True, "category": "correct_tool_and_arguments", "calls": calls}
        expected_name = (
            "submit_session_suggestion" if turn == 12 else "submit_dietary_request"
        )
        category = "correct_tool_wrong_or_missing_argument" if call["name"] == expected_name else "wrong_tool"
        return {"success": False, "category": category, "calls": calls}

    content = str(message.get("content") or "")
    if FALSE_COMPLETION.search(content):
        category = "no_tool_false_claim_of_completion"
    elif CONFIRMATION.search(content):
        category = "no_tool_redundant_confirmation_or_question"
    else:
        category = "no_tool_other"
    return {"success": False, "category": category, "content": content}
