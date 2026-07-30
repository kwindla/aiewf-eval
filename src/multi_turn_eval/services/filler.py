"""Filler-token ("dots") prompt-suffix injection — latent-scratchpad probe.

Implements an API-level probe motivated by arxiv 2607.03502: append nominally
content-free, space-separated glyphs to the prompt and measure behavior. Glyph counts
are not tokenizer-normalized, internal computation is not observed, and latency must
be measured rather than assumed.

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
    """Return the filler string from MTE_FILLER_DOTS (# of repeated glyphs), or
    None if off. MTE_FILLER_TOKEN overrides the repeated token (default ".") —
    e.g. "-" for dashes — for probing whether a pattern that doesn't read as
    end-of-conversation avoids the spurious-end_session failure mode."""
    raw = os.getenv("MTE_FILLER_DOTS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    token = os.getenv("MTE_FILLER_TOKEN", ".").strip() or "."
    return " ".join([token] * n)


def _weave(text: str, filler: str, position: str) -> str:
    if position == "prefix":
        return filler + " " + text.lstrip()
    return text.rstrip() + " " + filler


def apply_filler_to_last_user(messages: List[Any]) -> List[Any]:
    """Return a copy of `messages` with the filler applied. The input list and
    its dicts are never mutated, so the caller's persisted context is
    unaffected. No-op when the knob is unset.

    MTE_FILLER_POSITION selects where the filler lands (position ablation):
    - suffix (default): appended after the last user message's text
    - prefix: inserted before the last user message's text (same message)
    - system: appended to the first system/developer message; user untouched
    """
    global _logged
    filler = filler_suffix()
    if not filler or not messages:
        return messages
    position = os.getenv("MTE_FILLER_POSITION", "suffix").strip().lower()
    if position not in ("suffix", "prefix", "system"):
        position = "suffix"
    new_messages = list(messages)
    if position == "system":
        indices = range(len(new_messages))
        want_roles = ("system", "developer")
    else:
        indices = range(len(new_messages) - 1, -1, -1)
        want_roles = ("user",)
    for i in indices:
        msg = new_messages[i]
        if not isinstance(msg, dict) or msg.get("role") not in want_roles:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_content: Any = _weave(content, filler, position)
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
                    p["text"] = _weave(p["text"], filler, position)
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
            parts = filler.split()
            logger.info(
                f"MTE_FILLER_DOTS active: {len(parts)} x {parts[0]!r} filler tokens, "
                f"position={position} (history left filler-free)"
            )
        return new_messages
    return messages  # no target message found
