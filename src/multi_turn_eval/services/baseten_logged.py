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

import asyncio
import math
import os
import time

from loguru import logger
from openai import RateLimitError

from multi_turn_eval.services.lilac_logged import LoggedLilacLLMService


class LoggedBaseTenLLMService(LoggedLilacLLMService):
    """OpenAI-compatible BaseTen reasoning service recording raw + TTFAT TTFB.

    Optional transport pacing happens before latency measurement. Explicit HTTP
    429 responses before streaming can be retried; retry delays remain measured.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_interval = float(os.getenv("MTE_BASETEN_MIN_REQUEST_INTERVAL_SECS", "0"))
        self._rate_retries = int(os.getenv("MTE_BASETEN_RATE_LIMIT_RETRIES", "0"))
        self._rate_backoff = float(os.getenv("MTE_BASETEN_RATE_LIMIT_BACKOFF_SECS", "10"))
        if (not math.isfinite(self._min_interval) or self._min_interval < 0
                or not 0 <= self._rate_retries <= 3
                or not math.isfinite(self._rate_backoff) or not 0 < self._rate_backoff <= 10):
            raise ValueError("Invalid Baseten request pacing/retry configuration")
        self._last_request_start = None

    async def _process_context(self, context):
        if self._last_request_start is not None:
            delay = self._min_interval - (time.monotonic() - self._last_request_start)
            if delay > 0:
                logger.info(f"Baseten request pacing: waiting {delay:.3f}s before latency measurement")
                await asyncio.sleep(delay)
        self._last_request_start = time.monotonic()
        return await super()._process_context(context)

    async def get_chat_completions(self, context):
        for attempt in range(self._rate_retries + 1):
            try:
                return await super().get_chat_completions(context)
            except RateLimitError:
                if attempt == self._rate_retries:
                    raise
                delay = self._rate_backoff * 2 ** attempt
                logger.warning(
                    f"Baseten HTTP 429 before stream: retry {attempt + 1}/{self._rate_retries} "
                    f"after {delay:g}s; wait included in request latency"
                )
                await asyncio.sleep(delay)
