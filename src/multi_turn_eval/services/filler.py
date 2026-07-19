"""Filler-token ("dots") prompt-suffix injection — latent-scratchpad probe.

Implements the technique from arxiv 2607.03502: appending content-free filler
tokens (space-separated dots — "one token per dot") to the prompt gives a frozen
model extra prefill/latent compute before it answers, WITHOUT emitting reasoning
tokens (so time-to-first-answer-token stays flat).

`MTE_FILLER_DOTS=<n>` appends `n` dots to the FINAL user-role message of each
outgoing request. Applied on a COPY at request-build time, so the persisted
context / in-context history stays filler-free by design — we don't want to
teach the model the filler pattern across turns. Off by default (unset).

Used by the Responses service (gpt-5.4/5.6) and the BaseTen service (inkling).
"""

import os
from typing import Any, List

from loguru import logger

_logged = False


def filler_suffix() -> str | None:
    """Return the filler string from MTE_FILLER_DOTS (# of dots), or None if off."""
    raw = os.getenv("MTE_FILLER_DOTS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return " ".join(["."] * n) if n > 0 else None


def apply_filler_to_last_user(messages: List[Any]) -> List[Any]:
    """Return a copy of `messages` with the filler suffix appended to the last
    user-role message. The input list and its dicts are never mutated, so the
    caller's persisted context is unaffected. No-op when the knob is unset.
    """
    global _logged
    filler = filler_suffix()
    if not filler or not messages:
        return messages
    new_messages = list(messages)
    for i in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_content: Any = content.rstrip() + " " + filler
        elif isinstance(content, list):
            new_content = list(content)
            for j in range(len(new_content) - 1, -1, -1):
                part = new_content[j]
                if (
                    isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                    and isinstance(part.get("text"), str)
                ):
                    p = dict(part)
                    p["text"] = p["text"].rstrip() + " " + filler
                    new_content[j] = p
                    break
            else:
                new_content.append({"type": "text", "text": filler})
        else:
            return messages  # unknown content shape — leave untouched
        nm = dict(msg)
        nm["content"] = new_content
        new_messages[i] = nm
        if not _logged:
            _logged = True
            logger.info(
                f"MTE_FILLER_DOTS active: appending {filler.count('.')} dots to the "
                "final user turn of each request (history left filler-free)"
            )
        return new_messages
    return messages  # no user message found
