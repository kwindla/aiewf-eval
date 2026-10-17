#!/usr/bin/env python3
"""Analyze a frozen Grok 4.6 AIEWF medium-context cohort."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "artifacts/canonical.tsv"
OUTPUT = HERE / "analysis"
MODEL = "grok-4.6"
REASONING_EFFORT = "low"
CONVERSATIONS = 30
TURNS = 30
DENOMINATOR = CONVERSATIONS * TURNS
COMPONENTS = ("tool_use_correct", "instruction_following", "kb_grounding")
PRICE = {"input": 2.0, "cached_input": 0.5, "output": 6.0}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def cluster_ci(per_conversation_correct: list[int]) -> list[float]:
    rng = random.Random(20260813)
    estimates = []
    for _ in range(20_000):
        correct = sum(
            per_conversation_correct[rng.randrange(CONVERSATIONS)]
            for _ in range(CONVERSATIONS)
        )
        estimates.append(correct / DENOMINATOR * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    return function.get("name") if isinstance(function, dict) else None


def main() -> int:
    with CANONICAL.open(newline="", encoding="utf-8") as handle:
        manifests = list(csv.DictReader(handle, delimiter="\t"))
    if [int(row["slot"]) for row in manifests] != list(range(1, CONVERSATIONS + 1)):
        raise RuntimeError("canonical manifest must contain slots 1 through 30")

    component_counts = {name: 0 for name in COMPONENTS}
    strict_counts: list[int] = []
    ttfat: list[float] = []
    raw_ttft: list[float] = []
    reasoning_gaps: list[float] = []
    cache_shares: list[float] = []
    token_totals = {
        name: 0
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "thinking_tokens",
        )
    }
    completed = recovery_responses = observed_responses = tool_turns = 0
    end_session = {"scripted": 0, "recovery": 0, "missing": 0}
    maximum_prompt_tokens = 0

    for manifest in manifests:
        run_dir = (ROOT / manifest["run_dir"]).resolve()
        run_dir.relative_to(ROOT)
        transcript = read_jsonl(run_dir / "transcript.jsonl")
        if any(row.get("model_name") != MODEL for row in transcript):
            raise RuntimeError(f"model identity mismatch: {run_dir}")
        scheduled = [row for row in transcript if row.get("recovery_turn") is not True]
        recovery = [row for row in transcript if row.get("recovery_turn") is True]
        if [row.get("turn") for row in scheduled] != list(range(len(scheduled))):
            raise RuntimeError(f"scheduled turns are not a contiguous prefix: {run_dir}")
        completed += len(scheduled) == TURNS
        recovery_responses += len(recovery)
        observed_responses += len(transcript)

        judged = read_jsonl(run_dir / "claude_judged.jsonl")
        by_turn = {int(row["turn"]): row for row in judged}
        per_component = {name: [False] * TURNS for name in COMPONENTS}
        for turn, row in by_turn.items():
            if not 0 <= turn < TURNS:
                raise RuntimeError(f"invalid judged turn {turn}: {run_dir}")
            scores = row.get("scores") or {}
            for name in COMPONENTS:
                per_component[name][turn] = scores.get(name) is True
        strict = [all(per_component[name][turn] for name in COMPONENTS) for turn in range(TURNS)]
        strict_counts.append(sum(strict))
        for name in COMPONENTS:
            component_counts[name] += sum(per_component[name])

        for row in transcript:
            include_latency = not (
                row.get("recovery_turn") is not True and row.get("turn") == 0
            )
            visible = row.get("ttfb_ms")
            raw = row.get("raw_ttfb_ms")
            if include_latency and isinstance(visible, (int, float)):
                ttfat.append(float(visible))
            if include_latency and isinstance(raw, (int, float)):
                raw_ttft.append(float(raw))
            if (
                include_latency
                and isinstance(visible, (int, float))
                and isinstance(raw, (int, float))
            ):
                reasoning_gaps.append(max(0.0, float(visible) - float(raw)))
            calls = row.get("tool_calls") or []
            tool_turns += bool(calls)
            tokens = row.get("tokens") or {}
            prompt_tokens = int(tokens.get("prompt_tokens") or 0)
            cached_tokens = int(tokens.get("cache_read_input_tokens") or 0)
            maximum_prompt_tokens = max(maximum_prompt_tokens, prompt_tokens)
            if include_latency and prompt_tokens:
                cache_shares.append(cached_tokens / prompt_tokens * 100)
            for name in token_totals:
                token_totals[name] += int(tokens.get(name) or 0)

        scripted_end = any(
            tool_name(call) == "end_session"
            for row in scheduled
            for call in (row.get("tool_calls") or [])
        )
        recovery_end = any(
            tool_name(call) == "end_session"
            for row in recovery
            for call in (row.get("tool_calls") or [])
        )
        end_session["scripted" if scripted_end else "recovery" if recovery_end else "missing"] += 1

    if not ttfat or not raw_ttft:
        raise RuntimeError("latency samples are missing")
    strict_correct = sum(strict_counts)
    uncached_input = token_totals["prompt_tokens"] - token_totals["cache_read_input_tokens"]
    # Responses API output_tokens already includes billed reasoning tokens.
    billed_output = token_totals["completion_tokens"]
    total_cost = (
        uncached_input * PRICE["input"]
        + token_totals["cache_read_input_tokens"] * PRICE["cached_input"]
        + billed_output * PRICE["output"]
    ) / 1_000_000
    result = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "conversations": CONVERSATIONS,
        "complete_conversations": completed,
        "fixed_scheduled_turn_denominator": DENOMINATOR,
        "strict": {
            "correct": strict_correct,
            "rate_percent": strict_correct / DENOMINATOR * 100,
            "conversation_cluster_bootstrap_95_percent": cluster_ci(strict_counts),
        },
        "components": {
            name: {
                "correct": count,
                "error_percent": (DENOMINATOR - count) / DENOMINATOR * 100,
            }
            for name, count in component_counts.items()
        },
        "latency": {
            "scope": "exclude_scripted_turn_0_include_later_scripted_and_recovery",
            "samples": len(ttfat),
            "ttfat_p50_ms": statistics.median(ttfat),
            "ttfat_p95_ms": percentile(ttfat, 0.95),
            "ttfat_max_ms": max(ttfat),
            "raw_ttft_p50_ms": statistics.median(raw_ttft),
            "raw_ttft_p95_ms": percentile(raw_ttft, 0.95),
            "reasoning_to_answer_gap_p50_ms": statistics.median(reasoning_gaps),
        },
        "cache": {
            "median_warm_cached_input_percent": statistics.median(cache_shares),
            "maximum_request_prompt_tokens": maximum_prompt_tokens,
        },
        "responses": {
            "observed": observed_responses,
            "recovery": recovery_responses,
            "tool_call_turns": tool_turns,
            "tool_call_turn_percent": tool_turns / observed_responses * 100,
            "end_session": end_session,
        },
        "tokens": token_totals,
        "pricing": {
            "as_of": "2026-08-13",
            "rates_per_million": PRICE,
            "tier": "under_200k_input_tokens_per_request",
            "estimated_total_usd": total_cost,
            "estimated_cost_per_observed_turn_usd": total_cost / observed_responses,
            "estimated_cost_per_conversation_usd": total_cost / CONVERSATIONS,
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    strict = result["strict"]
    latency = result["latency"]
    pricing = result["pricing"]
    cache = result["cache"]
    report = f"""# Grok 4.6 ({REASONING_EFFORT}) — AIEWF medium context

- Cohort: {CONVERSATIONS} canonical conversations; {completed} reached all {TURNS} scripted turns.
- End-session disposition: {end_session['scripted']} scripted, {end_session['recovery']} recovery, {end_session['missing']} missing; all missing future turns remain fixed-denominator failures.
- Strict pass rate: {strict['correct']}/{DENOMINATOR} ({strict['rate_percent']:.1f}%), conversation-cluster bootstrap 95% CI {strict['conversation_cluster_bootstrap_95_percent'][0]:.1f}–{strict['conversation_cluster_bootstrap_95_percent'][1]:.1f}%.
- Errors: tool {(DENOMINATOR - component_counts['tool_use_correct']) / DENOMINATOR * 100:.1f}%, instruction {(DENOMINATOR - component_counts['instruction_following']) / DENOMINATOR * 100:.1f}%, KB {(DENOMINATOR - component_counts['kb_grounding']) / DENOMINATOR * 100:.1f}%.
- Warm TTFAT: {latency['ttfat_p50_ms']:.0f}ms P50, {latency['ttfat_p95_ms']:.0f}ms P95, {latency['ttfat_max_ms']:.0f}ms max across {latency['samples']} later scripted and recovery responses; scripted turn 0 is excluded.
- Warm raw TTFT: {latency['raw_ttft_p50_ms']:.0f}ms P50, {latency['raw_ttft_p95_ms']:.0f}ms P95. Median reasoning-to-answer gap: {latency['reasoning_to_answer_gap_p50_ms']:.0f}ms.
- Stateful Responses caching: {cache['median_warm_cached_input_percent']:.1f}% median warm cached-input share; maximum request input was {cache['maximum_request_prompt_tokens']:,} tokens, below xAI's 200K higher-price threshold.
- Standard-context price estimate: ${pricing['estimated_cost_per_observed_turn_usd']:.4f} per observed turn (${pricing['estimated_total_usd']:.2f} total), using $2/M uncached input, $0.50/M cached input, and $6/M output including reasoning.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(OUTPUT / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
