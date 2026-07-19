"""OpenRouter (OpenAI-compatible) LLM service with raw + content-aware TTFB.

OpenRouter (https://openrouter.ai/api/v1) proxies many open-weights models
(Qwen3 dense 8B/14B/32B, DeepSeek-V3.x, ...). For hybrid-reasoning models it can
stream a ``reasoning`` delta before the answer ``content``; the stock
``OpenAILLMService`` would stop TTFB on that first reasoning token. This subclass
reuses :class:`LoggedLilacLLMService`'s provider-agnostic ``_process_context``
(content-aware TTFB + RawTTFBMetricsData) and inherits MTE_FILLER_DOTS injection.

Subclassed so run logs and ``--service openrouter`` resolution read
``openrouter``. Thinking is disabled per-run via OpenRouter's unified
``reasoning: {enabled: false}`` control (see the openrouter branch in
pipelines/base.py, gated by MTE_OPENROUTER_REASONING_OFF).
"""

from multi_turn_eval.services.lilac_logged import LoggedLilacLLMService


class LoggedOpenRouterLLMService(LoggedLilacLLMService):
    """OpenAI-compatible OpenRouter service recording raw + TTFAT TTFB.

    Filler-token injection and content-aware/raw TTFB are inherited from
    LoggedLilacLLMService; nothing OpenRouter-specific is needed here.
    """

    pass
