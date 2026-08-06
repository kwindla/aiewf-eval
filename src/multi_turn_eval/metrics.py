"""Project-local metrics types.

We extend pipecat's metrics with a parallel "raw TTFB" measurement that fires
on the first stream event of any kind, regardless of whether it carries
reasoning, content, tool-calls, or transcript. Pipecat's standard
``TTFBMetricsData`` in this project is stopped on first user-visible content
(see ``LoggedCerebrasLLMService`` / ``LoggedGoogleLLMService`` /
``LoggedAnthropicLLMService``), so the two together give:

- ``ttfb_ms``     — time to first non-thinking token (voice-agent latency)
- ``raw_ttfb_ms`` — time to first chunk of any kind (network+prefill floor)

Per-service trigger nuance for the raw value: Cerebras fires on the first
chunk-with-choices, Google on the first chunk-with-candidates, and Anthropic
on the first stream event (``message_start``, which can precede the first
content block). Treat millisecond-level cross-provider deltas accordingly.
(``LoggedVLLMOpenAILLMService`` gates ``ttfb_ms`` but does not emit a raw
value.)

We carry the raw value as its own ``MetricsData`` subclass so it has a distinct
isinstance check downstream and never collides with the standard TTFB metric.
"""

from pipecat.metrics.metrics import MetricsData


class RawTTFBMetricsData(MetricsData):
    """Time to first stream chunk (raw), in seconds.

    Parameters:
        value: Raw TTFB measurement in seconds (start of request to first
            non-empty ``choices`` chunk, irrespective of content vs reasoning).
    """

    value: float
