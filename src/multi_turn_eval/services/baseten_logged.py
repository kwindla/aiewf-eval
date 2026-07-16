"""BaseTen (OpenAI-compatible) reasoning LLM service with raw + content-aware TTFB.

BaseTen's Model API serves reasoning models (e.g. ``thinkingmachines/inkling``)
that stream chain-of-thought in a separate ``reasoning_content`` delta before the
answer ``content``. The stock ``OpenAILLMService`` stops its TTFB metric on the
first ``choices`` chunk — i.e. the first *reasoning* token — which badly
understates time-to-first-answer-token for a thinking model.

The behavior we need is identical to :class:`LoggedLilacLLMService`, whose
``_process_context`` is provider-agnostic: it stops the (content-aware) TTFB
metric on the first ``content`` / ``tool_calls`` delta — reasoning-only chunks do
not count — and separately emits ``RawTTFBMetricsData`` on the first chunk of any
kind. So every turn records BOTH:

- ``ttfb_ms``     (TTFAT) — time to first non-thinking (answer) token
- ``raw_ttfb_ms``         — time to first stream chunk (first reasoning token)

With thinking effort ``none`` (no separate reasoning stream) the two coincide;
at ``low``..``max`` they diverge, and ``raw_ttfb → ttfb`` is the per-turn thinking
delay. Reasoning length rides along as ``reasoning_tokens`` (recorded as
``thinking_tokens`` in the transcript).

Subclassed rather than reusing :class:`LoggedLilacLLMService` directly so run
logs and ``--service baseten`` resolution read ``baseten``, not ``lilac``. The
base class carries no Lilac-specific logic; if that ever changes, inline the
``_process_context`` here.
"""

from multi_turn_eval.services.lilac_logged import LoggedLilacLLMService


class LoggedBaseTenLLMService(LoggedLilacLLMService):
    """OpenAI-compatible BaseTen reasoning service recording raw + TTFAT TTFB."""

    pass
