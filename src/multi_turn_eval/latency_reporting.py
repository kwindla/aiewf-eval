"""Sample selection for ttfat-first-response-excluded-v2.

Callers retain their existing recovery, validity, and missing-value filters.
This helper excludes scripted index zero, not the first available measurement.
"""

DEFINITION_ID = "ttfat-first-response-excluded-v2"


def include_response_latency(turn: int | None, *, recovery: bool = False) -> bool:
    """Keep later responses and any recovery already eligible in the caller."""
    return recovery or turn != 0
