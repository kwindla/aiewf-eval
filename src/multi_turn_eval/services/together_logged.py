"""Together (OpenAI-compatible) reasoning LLM service with raw + content-aware TTFB.

Together's inference API (https://api.together.xyz/v1) is OpenAI-compatible and
serves open-weights reasoning models (Qwen3, DeepSeek-V3.x, ...) that stream
chain-of-thought in a separate ``reasoning_content`` delta before the answer
``content``. The behavior we need is identical to :class:`LoggedLilacLLMService`:
stop the content-aware TTFB metric on the first ``content`` / ``tool_calls``
delta (reasoning-only chunks don't count) and separately emit
``RawTTFBMetricsData`` on the first chunk of any kind. Filler-token injection
(MTE_FILLER_DOTS) is inherited from the base too.

Subclassed rather than reusing :class:`LoggedLilacLLMService` directly so run
logs and ``--service together`` resolution read ``together``, not ``lilac``.
"""

from multi_turn_eval.services.lilac_logged import LoggedLilacLLMService


class LoggedTogetherLLMService(LoggedLilacLLMService):
    """OpenAI-compatible Together reasoning service recording raw + TTFAT TTFB.

    Filler-token injection and content-aware/raw TTFB are inherited from
    LoggedLilacLLMService; nothing Together-specific is needed here.
    """

    pass
