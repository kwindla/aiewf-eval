#!/usr/bin/env python3
"""Compare BaseTen Kimi K2.6 thinking-on and thinking-off N=30 cohorts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
OFF_CAMPAIGN = CAMPAIGN.parent / "aiewf-medium-none-n30-20260806"
ON_AGGREGATE = HERE / "aggregates.json"
OFF_AGGREGATE = OFF_CAMPAIGN / "analysis/aggregates.json"
ON_COLLECTION = CAMPAIGN / "collection/summary.json"
OFF_COLLECTION = OFF_CAMPAIGN / "collection/summary.json"
OUTPUT_JSON = HERE / "comparison.json"
OUTPUT_MD = HERE / "COMPARISON.md"
BOOTSTRAPS = 20_000
SEED = 20260806
ON_SIGNATURE = {
    "endpoint": "https://inference.baseten.co/v1",
    "reasoning_effort": "omit",
    "chat_template_args": {"enable_thinking": True},
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 8192,
    "filler": None,
}
OFF_TRANSMITTED_SIGNATURE = {
    "endpoint": "https://inference.baseten.co/v1",
    "reasoning_effort": "none",
    "temperature": 0.6,
    "max_tokens": 8192,
    "filler": None,
}
OFF_CONTROL_INTERPRETATION = (
    "chat_template_args.enable_thinking was omitted, leaving BaseTen's Kimi "
    "default thinking-off behavior; the transmitted reasoning_effort=none is "
    "unsupported for this model and ignored. Zero thinking tokens on all 900 "
    "scripted turns empirically confirm the effective off state."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def independent_cluster_difference_ci(
    on_correct: list[int], off_correct: list[int]
) -> list[float]:
    """Bootstrap on-minus-off strict pass-rate difference by conversation."""

    if len(on_correct) != 30 or len(off_correct) != 30:
        raise RuntimeError("difference CI requires 30 conversation clusters per arm")
    rng = random.Random(SEED)
    estimates = []
    for _ in range(BOOTSTRAPS):
        on = sum(on_correct[rng.randrange(30)] for _ in range(30)) / 900 * 100
        off = sum(off_correct[rng.randrange(30)] for _ in range(30)) / 900 * 100
        estimates.append(on - off)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def main() -> int:
    on = load(ON_AGGREGATE)
    off = load(OFF_AGGREGATE)
    on_collection = load(ON_COLLECTION)
    off_collection = load(OFF_COLLECTION)
    if on.get("arm") != "thinking" or off.get("arm") != "none":
        raise RuntimeError("unexpected arm identity")
    if on["strict_pass"]["total"] != 900 or off["strict_pass"]["total"] != 900:
        raise RuntimeError("comparison requires fixed 900-turn denominators")
    if on.get("request_signature") != ON_SIGNATURE:
        raise RuntimeError("thinking-on request signature mismatch")
    if off_collection.get("request_signature") != OFF_TRANSMITTED_SIGNATURE:
        raise RuntimeError("thinking-off request signature mismatch")
    if off["billed_token_totals_all_canonical_rows"]["thinking_tokens"] != 0:
        raise RuntimeError("thinking-off cohort unexpectedly reports thinking tokens")

    on_pass = float(on["strict_pass"]["rate_percent"])
    off_pass = float(off["strict_pass"]["rate_percent"])
    difference = on_pass - off_pass
    difference_ci = independent_cluster_difference_ci(
        [int(row["strict_passes"]) for row in on["runs"]],
        [int(row["strict_passes"]) for row in off["runs"]],
    )
    result = {
        "schema_version": 1,
        "model": "moonshotai/Kimi-K2.6",
        "provider": "BaseTen",
        "benchmark": "aiwf_medium_context",
        "fixed_denominator_per_arm": "30 conversations x 30 scripted turns = 900",
        "protocol_note": (
            "This reproduces vendor-mode protocols, not a pure one-knob causal "
            "comparison: thinking on uses temperature 1.0/top_p 0.95 and explicit "
            "chat_template_args.enable_thinking=true; thinking off uses temperature "
            "0.6/default top_p and omits chat_template_args.enable_thinking."
        ),
        "thinking_off_control_interpretation": OFF_CONTROL_INTERPRETATION,
        "thinking_on": on,
        "thinking_off": off,
        "strict_pass_difference_percentage_points_on_minus_off": difference,
        "strict_pass_difference_cluster_bootstrap_95_percent": difference_ci,
        "ttfat_ratio_on_over_off": on["ttfat_ms"]["p50"] / off["ttfat_ms"]["p50"],
        "input_hashes": {
            str(ON_AGGREGATE.relative_to(CAMPAIGN)): sha256(ON_AGGREGATE),
            str(OFF_AGGREGATE.relative_to(CAMPAIGN.parent)): sha256(OFF_AGGREGATE),
            str(ON_COLLECTION.relative_to(CAMPAIGN)): sha256(ON_COLLECTION),
            str(OFF_COLLECTION.relative_to(CAMPAIGN.parent)): sha256(OFF_COLLECTION),
            "analysis/compare_arms.py": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def ci(metric: dict[str, Any]) -> str:
        low, high = metric["conversation_cluster_bootstrap_95_percent"]
        return f"{low:.1f}–{high:.1f}%"

    on_end = on["end_session_outcomes"]["counts"]
    off_end = off["end_session_outcomes"]["counts"]
    report = f"""# BaseTen Kimi K2.6 thinking comparison

Both arms use 30 complete, valid conversations and the same fixed 900 scripted
turn denominator. Recovery turns are excluded from accuracy and TTFAT.

| Measure | Thinking on | Thinking off |
|---|---:|---:|
| Strict pass rate | {on_pass:.1f}% | {off_pass:.1f}% |
| Conversation-cluster bootstrap 95% CI | {ci(on['strict_pass'])} | {ci(off['strict_pass'])} |
| Any error | {on['strict_pass']['error_percent']:.1f}% | {off['strict_pass']['error_percent']:.1f}% |
| Tool error | {on['components']['tool_use_correct']['error_percent']:.1f}% | {off['components']['tool_use_correct']['error_percent']:.1f}% |
| Instruction error | {on['components']['instruction_following']['error_percent']:.1f}% | {off['components']['instruction_following']['error_percent']:.1f}% |
| KB error | {on['components']['kb_grounding']['error_percent']:.1f}% | {off['components']['kb_grounding']['error_percent']:.1f}% |
| TTFAT P50 | {on['ttfat_ms']['p50']:.0f} ms | {off['ttfat_ms']['p50']:.0f} ms |
| TTFAT P95 | {on['ttfat_ms']['p95']:.0f} ms | {off['ttfat_ms']['p95']:.0f} ms |
| TTFAT max | {on['ttfat_ms']['max']:.0f} ms | {off['ttfat_ms']['max']:.0f} ms |
| `end_session` on scripted turn | {on_end.get('scripted', 0)}/30 | {off_end.get('scripted', 0)}/30 |
| `end_session` on recovery turn | {on_end.get('recovery', 0)}/30 | {off_end.get('recovery', 0)}/30 |
| Missing `end_session` | {on_end.get('missing', 0)}/30 | {off_end.get('missing', 0)}/30 |
| Complete-conversation yield | {on['collection_reliability']['canonical_yield_per_conversation_attempt_percent']:.1f}% | {off['collection_reliability']['canonical_yield_per_conversation_attempt_percent']:.1f}% |

Thinking on minus off is {difference:+.1f} percentage points in strict pass rate;
the independent conversation-cluster bootstrap 95% interval for that difference
is {difference_ci[0]:+.1f} to {difference_ci[1]:+.1f} points. Median TTFAT is
{result['ttfat_ratio_on_over_off']:.2f}× the thinking-off value.

The thinking-on timing is content/tool TTFAT, not time to the first reasoning
token. Its raw first-chunk P50 is {on['raw_ttfb_ms']['p50']:.0f} ms and its
median measured reasoning delay is {on['reasoning_delay_ms']['p50']:.0f} ms;
{on['thinking']['scripted_rows_with_positive_thinking_tokens']}/900 scripted
rows report positive thinking tokens.

Important protocol caveat: this is a reproduction of the two vendor-mode
signatures, not a pure one-setting causal experiment. Thinking on uses
temperature 1.0 and top-p 0.95; thinking off used temperature 0.6 and the
provider-default top-p. Thinking on explicitly sends
`chat_template_args.enable_thinking=true`; thinking off omitted that argument,
leaving BaseTen's default off behavior. Although the off request transmitted
`reasoning_effort=none`, Kimi K2.6 is not a BaseTen `reasoning_effort` model and
that field is ignored. The off cohort's zero reported thinking tokens on all
900 scripted turns confirms its effective state.

The completion-yield row is operational campaign provenance, not a clean model
reliability effect. The earlier thinking-off campaign's 41 recorded attempts
include its initial concurrency-2 429 failures, provider stream/502 failures,
and two out-of-cohort duplicate/interrupted attempts; the thinking-on campaign
started with the stabilized serial 30-second-cooldown protocol.
"""
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(
        f"comparison complete: on={on_pass:.1f}% off={off_pass:.1f}% "
        f"difference={difference:+.1f}pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
