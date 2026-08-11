#!/usr/bin/env python3
"""Estimate model cost per conversation minute from the freshness usage sample.

The script is deliberately dependency-free so the calculation can be rerun
from the two versioned JSON inputs alone.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
MILLION = Decimal("1000000")


def decimal(value: Any) -> Decimal:
    """Convert JSON numbers without introducing binary floating-point noise."""

    return Decimal(str(value))


def rounded(value: Decimal, places: str = "0.00000001") -> float:
    return float(value.quantize(Decimal(places)))


def token_components(
    usage: dict[str, Any], pricing: dict[str, Any]
) -> tuple[dict[str, int], dict[str, Decimal]]:
    prompt = int(usage["prompt_tokens"])
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage["completion_tokens"])
    semantics = pricing["token_accounting"]

    if semantics == "prompt_excludes_cache_reads_and_writes":
        uncached = prompt
    elif semantics == "prompt_includes_cache_reads":
        uncached = prompt - cache_read
    else:
        raise ValueError(f"unsupported token accounting semantics: {semantics}")

    if uncached < 0:
        raise ValueError(
            f"cached input exceeds prompt total for {usage['key']}: "
            f"prompt={prompt}, cached={cache_read}"
        )

    rates = pricing["rates_usd_per_million_tokens"]
    if cache_write and "cache_write_5m" not in rates:
        raise ValueError(
            f"{usage['key']} has cache-write tokens but no cache-write price"
        )

    costs = {
        "uncached_input": decimal(uncached) * decimal(rates["input"]) / MILLION,
        "cached_input": decimal(cache_read)
        * decimal(rates["cached_input"])
        / MILLION,
        "cache_write_5m": decimal(cache_write)
        * decimal(rates.get("cache_write_5m", 0))
        / MILLION,
        # Provider completion counters already include billed reasoning tokens.
        "output": decimal(output) * decimal(rates["output"]) / MILLION,
    }
    tokens = {
        "uncached_input": uncached,
        "cached_input": cache_read,
        "cache_write_5m": cache_write,
        "output": output,
    }
    return tokens, costs


def estimate_row(
    usage: dict[str, Any], pricing: dict[str, Any]
) -> dict[str, Any]:
    minutes = decimal(usage["estimated_speech_minutes_150wpm"])
    if minutes <= 0:
        raise ValueError(f"non-positive conversation minutes for {usage['key']}")

    mode = pricing["billing_mode"]
    if mode == "tokens":
        tokens, component_decimals = token_components(usage, pricing)
        total = sum(component_decimals.values(), Decimal(0))
        components = {key: rounded(value) for key, value in component_decimals.items()}
        utilization = None
    elif mode == "active_gpu_conversation_minute":
        utilization = pricing["utilization_assumption"]
        active_conversations = int(utilization["active_conversations"])
        if active_conversations != 1:
            raise ValueError(
                "this artifact requires the explicit one-active-conversation "
                f"assumption, got {active_conversations} for {usage['key']}"
            )
        rate = decimal(pricing["active_compute_usd_per_minute"])
        total = minutes * rate / decimal(active_conversations)
        components = {"active_gpu_compute": rounded(total)}
        tokens = {
            "uncached_input": None,
            "cached_input": None,
            "cache_write_5m": None,
            "output": None,
        }
    else:
        raise ValueError(f"unsupported billing mode for {usage['key']}: {mode}")

    result = {
        "key": usage["key"],
        "label": usage["label"],
        "provider": usage["provider"],
        "model": usage["model"],
        "run_dir": usage["run_dir"],
        "billing_mode": mode,
        "estimated_conversation_minutes_150wpm": float(minutes),
        "billed_token_counts": tokens,
        "cost_components_usd": components,
        "estimated_sample_cost_usd": rounded(total),
        "estimated_cost_per_conversation_minute_usd": rounded(total / minutes),
        "pricing_source": pricing["source"],
        "pricing_notes": pricing.get("notes", ""),
    }
    if utilization is not None:
        result["utilization_assumption"] = utilization
    return result


def estimate_all(
    usage_rows: list[dict[str, Any]], pricing_document: dict[str, Any]
) -> dict[str, Any]:
    prices = pricing_document["models"]
    complete = [row for row in usage_rows if row["status"] == "complete"]
    missing = sorted(row["key"] for row in complete if row["key"] not in prices)
    if missing:
        raise ValueError(f"missing pricing for complete usage rows: {', '.join(missing)}")

    forbidden = sorted(key for key in prices if "pro" in key.casefold().split("_"))
    if forbidden:
        raise ValueError(f"OpenAI Pro pricing entries are forbidden: {', '.join(forbidden)}")

    estimates = [estimate_row(row, prices[row["key"]]) for row in complete]
    return {
        "schema_version": 1,
        "usage_sample_date": pricing_document["usage_sample_date"],
        "pricing_as_of": pricing_document["pricing_as_of"],
        "currency": pricing_document["currency"],
        "conversation_minutes_method": pricing_document[
            "conversation_minutes_method"
        ],
        "included_completed_conversations": len(estimates),
        "estimates": estimates,
    }


def markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Estimated text-model cost per conversation minute — "
        f"{document['pricing_as_of']}",
        "",
        "One completed freshness conversation per configuration; conversation "
        "minutes are actual user + assistant words at 150 words/minute.",
        "",
        "| Model configuration | Provider | Billing basis | Est. sample cost | Est. cost / conversation min |",
        "|---|---|---|---:|---:|",
    ]
    for row in document["estimates"]:
        if row["billing_mode"] == "tokens":
            basis = "tokens"
        else:
            utilization = row["utilization_assumption"]
            basis = (
                f"{utilization['accelerator']}, "
                f"{utilization['active_conversations']} active conversation"
            )
        lines.append(
            f"| {row['label']} | {row['provider']} | {basis} | "
            f"${row['estimated_sample_cost_usd']:.4f} | "
            f"${row['estimated_cost_per_conversation_minute_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Dedicated BaseTen values assume one active conversation has exclusive use "
            "of the listed GPU. They exclude cold-start and idle scale-down overhead; "
            "they are utilization estimates, not token prices.",
            "",
            "See `COST-METHODOLOGY.md` and `pricing-2026-08-06.json` for cache "
            "accounting, exclusions, assumptions, rates, and official sources.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage", type=Path, default=HERE / "usage-results.json")
    parser.add_argument(
        "--pricing", type=Path, default=HERE / "pricing-2026-08-06.json"
    )
    parser.add_argument(
        "--output-json", type=Path, default=HERE / "cost-results-2026-08-06.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=HERE / "cost-results-2026-08-06.md"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    usage = json.loads(args.usage.read_text(encoding="utf-8"))
    prices = json.loads(args.pricing.read_text(encoding="utf-8"))
    result = estimate_all(usage, prices)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown(result), encoding="utf-8")
    print(
        f"Wrote {len(result['estimates'])} completed-conversation estimates to "
        f"{args.output_json} and {args.output_markdown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
