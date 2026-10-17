"""OpenRouter (OpenAI-compatible) LLM service with raw + content-aware TTFB.

OpenRouter (https://openrouter.ai/api/v1) proxies many open-weights models
(Qwen3 dense 8B/14B/32B, DeepSeek-V3.x, ...). For hybrid-reasoning models it can
stream a ``reasoning`` delta before the answer ``content``; the stock
``OpenAILLMService`` would stop TTFB on that first reasoning token. This subclass
reuses :class:`LoggedLilacLLMService`'s provider-agnostic ``_process_context``
(content-aware TTFB + RawTTFBMetricsData) and inherits MTE_FILLER_DOTS injection.

Some OpenRouter providers accept a streaming request immediately (SSE keep-alives
flow) but then stall for minutes before the first real chunk. A timeout around
``create()`` cannot catch that — the call returns as soon as the connection
opens — so ``MTE_OPENROUTER_RETRY_TIMEOUT_SECS`` here bounds time to the FIRST
CHUNK and reissues the request once on stall. The second attempt is unbounded;
the pipeline idle timeout remains the backstop.
"""

import asyncio
import os

from loguru import logger

from multi_turn_eval.services.lilac_logged import LoggedLilacLLMService


class LoggedOpenRouterLLMService(LoggedLilacLLMService):
    """OpenAI-compatible OpenRouter service recording raw + TTFAT TTFB."""

    async def get_chat_completions(self, context):
        first_chunk_timeout = float(os.getenv("MTE_OPENROUTER_RETRY_TIMEOUT_SECS", "0") or 0)
        stream = await super().get_chat_completions(context)
        if first_chunk_timeout <= 0:
            return stream
        return self._first_chunk_guard(stream, context, first_chunk_timeout)

    async def _first_chunk_guard(self, stream, context, timeout_secs):
        """Yield the stream's chunks, reissuing the request once if the first
        chunk does not arrive within ``timeout_secs``."""

        async def close(obj):
            for name in ("aclose", "close"):
                fn = getattr(obj, name, None)
                if fn is not None:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        await result
                    return

        chunk_iter = stream.__aiter__()
        try:
            first = await asyncio.wait_for(chunk_iter.__anext__(), timeout_secs)
        except StopAsyncIteration:
            return
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                f"{self}: no first chunk within {timeout_secs}s; reissuing request"
            )
            await close(chunk_iter)
            await close(stream)
            stream = await super().get_chat_completions(context)
            chunk_iter = stream.__aiter__()
            try:
                first = await chunk_iter.__anext__()
            except StopAsyncIteration:
                return
        yield first
        async for chunk in chunk_iter:
            yield chunk
