#!/usr/bin/env python3
"""Analyze the completed Muse Glimmer reasoning-strength sweep."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "runs/muse-glimmer-reasoning-strength-n30-20260811"
ARMS = ("low", "medium", "high", "xhigh")
METRICS = (
    "turn_taking",
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
)
BOOTSTRAPS = 20_000
SEED = 20260811
TURN12_REDUNDANT = "no_tool_redundant_confirmation_or_question"
FALSE_COMPLETION = re.compile(
    r"\b(i(?:'ve| have)|we(?:'ve| have))\s+(?:now\s+)?(?:submitted|added|recorded)|"
    r"\b(?:has been|is now)\s+(?:submitted|added|recorded)|\btaken care of\b",
    re.I,
)
QUESTION_OR_CONFIRM = re.compile(
    r"\b(?:confirm|confirmation|would you like|shall i|should i|want me to|go ahead|"
    r"please say|please confirm|one moment)\b|\?",
    re.I,
)


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def interval(values: list[float]) -> list[float]:
    return [percentile(values, 0.025) or 0.0, percentile(values, 0.975) or 0.0]


def bootstrap_rate(scores: list[int], rng: random.Random) -> list[float]:
    n = len(scores)
    samples = [
        sum(scores[rng.randrange(n)] for _ in range(n)) / (n * 30)
        for _ in range(BOOTSTRAPS)
    ]
    return interval(samples)


def bootstrap_difference(
    left: list[int], right: list[int], rng: random.Random
) -> list[float]:
    nl = len(left)
    nr = len(right)
    samples = []
    for _ in range(BOOTSTRAPS):
        lrate = sum(left[rng.randrange(nl)] for _ in range(nl)) / (nl * 30)
        rrate = sum(right[rng.randrange(nr)] for _ in range(nr)) / (nr * 30)
        samples.append(lrate - rrate)
    return interval(samples)


def bootstrap_score_rate(
    scores: list[int], opportunities_per_score: int, rng: random.Random
) -> list[float]:
    n = len(scores)
    return interval(
        [
            sum(scores[rng.randrange(n)] for _ in range(n))
            / (n * opportunities_per_score)
            for _ in range(BOOTSTRAPS)
        ]
    )


def bootstrap_score_difference(
    left: list[int], right: list[int], opportunities_per_score: int, rng: random.Random
) -> list[float]:
    nl = len(left)
    nr = len(right)
    return interval(
        [
            sum(left[rng.randrange(nl)] for _ in range(nl))
            / (nl * opportunities_per_score)
            - sum(right[rng.randrange(nr)] for _ in range(nr))
            / (nr * opportunities_per_score)
            for _ in range(BOOTSTRAPS)
        ]
    )


def bootstrap_binary_rate(values: list[int], rng: random.Random) -> list[float]:
    n = len(values)
    return interval(
        [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAPS)]
    )


def bootstrap_binary_difference(
    left: list[int], right: list[int], rng: random.Random
) -> list[float]:
    nl = len(left)
    nr = len(right)
    return interval(
        [
            sum(left[rng.randrange(nl)] for _ in range(nl)) / nl
            - sum(right[rng.randrange(nr)] for _ in range(nr)) / nr
            for _ in range(BOOTSTRAPS)
        ]
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def correct_turn11(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    calls = row.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("name") != "submit_session_suggestion":
        return False
    args = calls[0].get("args") or {}
    suggestion = norm(args.get("suggestion_text"))
    return (
        isinstance(args, dict)
        and norm(args.get("name")) == "jennifer smith"
        and all(token in suggestion.split() for token in ("open", "telemetry", "tracing"))
    )


def classify_turn12(row: dict[str, Any]) -> str:
    calls = row.get("tool_calls") or []
    if len(calls) > 1:
        return "duplicate_or_multiple_tool_calls"
    if len(calls) == 1:
        call = calls[0]
        args = call.get("args") or {}
        suggestion = norm(args.get("suggestion_text"))
        if call.get("name") != "submit_session_suggestion":
            return "wrong_tool"
        if (
            isinstance(args, dict)
            and {"name", "suggestion_text"}.issubset(args)
            and norm(args.get("name")) == "jennifer smith"
            and all(
                token in suggestion.split()
                for token in ("state", "machine", "abstractions", "complex", "workflows")
            )
        ):
            return "correct_tool_and_arguments"
        return "correct_tool_wrong_or_missing_argument"
    text = str(row.get("assistant_text") or "")
    if FALSE_COMPLETION.search(text):
        return "no_tool_false_claim_of_completion"
    if QUESTION_OR_CONFIRM.search(text):
        return TURN12_REDUNDANT
    return "no_tool_other"


def classify_turn11(row: dict[str, Any]) -> str:
    if correct_turn11(row):
        return "correct_tool_and_arguments"
    calls = row.get("tool_calls") or []
    if calls:
        return "tool_wrong_or_missing_argument"
    text = str(row.get("assistant_text") or "")
    if FALSE_COMPLETION.search(text):
        return "no_tool_false_claim_of_completion"
    if QUESTION_OR_CONFIRM.search(text):
        return TURN12_REDUNDANT
    return "no_tool_other"


def redundant_subtype(text: str) -> str:
    lowered = " ".join(text.split()).lower()
    if re.search(
        r"i(?:'ve| have).{0,45}(?:sent|submitted|added|recorded)|"
        r"gone ahead and|successfully submitted",
        lowered,
    ) and re.search(r"anything else|\?", lowered):
        return "post_action_claim_then_followup"
    if re.search(
        r"under your name|use your name|also from you|coming from you|"
        r"for someone else|is this suggestion.*from you|is this.*also from you|"
        r"your name.*(?:again|still)|name.*jennifer",
        lowered,
    ):
        return "identity_or_ownership_reconfirmation"
    if re.search(
        r"you(?:'d| would) like (?:to suggest|a session|this suggestion)|"
        r"session on .*\b(?:correct|right)\b|suggestion.*\b(?:correct|right)\b|"
        r"is that (?:the|your) suggestion",
        lowered,
    ):
        return "content_or_intent_reconfirmation"
    if re.search(
        r"should i submit|shall i|want me to|would you like me to|go ahead|"
        r"would you like (?:for )?me to|may i submit|can i submit",
        lowered,
    ):
        return "authorization_reconfirmation"
    return "other_question"


def main() -> int:
    runs: dict[str, list[Path]] = defaultdict(list)
    # schedule.tsv is emitted by csv.writer and carries a CR before its final
    # field is extended by worker.sh. Split only on LF, then normalize CR, so
    # the campaign ledger remains readable as six tab-separated fields.
    with (CAMPAIGN / "included.tsv").open(newline="") as included_handle:
        included_lines = included_handle.read().split("\n")
    for raw_line in included_lines:
        if not raw_line.strip() or raw_line.startswith("ordinal\t"):
            continue
        fields = raw_line.replace("\r", "").split("\t")
        if len(fields) != 6:
            raise RuntimeError(f"malformed included.tsv row: {raw_line!r}")
        _ordinal, _block, _position, arm, _arm_index, run_dir = fields
        runs[arm].append(REPO / run_dir)
    if {arm: len(runs[arm]) for arm in ARMS} != {arm: 30 for arm in ARMS}:
        raise RuntimeError("collection is not complete at N=30 per arm")

    rng = random.Random(SEED)
    output: dict[str, Any] = {
        "schema_version": 1,
        "campaign": str(CAMPAIGN.relative_to(REPO)),
        "arms": {},
        "pairwise_strict_rate_differences": {},
        "pairwise_turn12_redundant_rate_differences": {},
        "pairwise_turn11_redundant_rate_differences": {},
        "pairwise_suggestion_turn_redundant_rate_differences": {},
        "bootstrap": {
            "unit": "conversation",
            "replicates": BOOTSTRAPS,
            "seed": SEED,
        },
    }
    conversation_scores: dict[str, list[int]] = {}
    turn12_redundant: dict[str, list[int]] = {}
    turn11_redundant: dict[str, list[int]] = {}
    suggestion_turn_redundant_scores: dict[str, list[int]] = {}

    for arm in ARMS:
        metric_counts = Counter()
        strict_by_turn = Counter()
        strict_scores: list[int] = []
        completion_tokens: list[int] = []
        raw_ttft: list[float] = []
        ttfat: list[float] = []
        response_latency: list[float] = []
        reasoning_delay: list[float] = []
        recovery_responses = 0
        turn12_categories: Counter[str] = Counter()
        turn12_subtypes: Counter[str] = Counter()
        turn11_categories: Counter[str] = Counter()
        turn11_subtypes: Counter[str] = Counter()
        turn11_eligible = 0
        turn12_by_turn11_success: dict[bool, Counter[str]] = {
            True: Counter(),
            False: Counter(),
        }
        arm_turn12_redundant: list[int] = []
        arm_turn11_redundant: list[int] = []
        arm_suggestion_turn_redundant_scores: list[int] = []

        for run_dir in runs[arm]:
            summary_path = run_dir / "claude_summary.json"
            judged_path = run_dir / "claude_judged.jsonl"
            transcript_path = run_dir / "transcript.jsonl"
            if not summary_path.exists() or not judged_path.exists():
                raise RuntimeError(f"missing judgment: {run_dir}")
            with summary_path.open() as handle:
                summary = json.load(handle)
            strict_scores.append(int(summary["turn_pass"]["count"]))
            recovery_responses += int(summary.get("recovery_turns_recorded") or 0)
            for metric in METRICS:
                metric_counts[metric] += int(summary["claude_passes"][metric])

            judged = load_jsonl(judged_path)
            if len(judged) != 30:
                raise RuntimeError(f"expected 30 judged turns: {run_dir}")
            for row in judged:
                if all(bool(row["scores"][metric]) for metric in METRICS):
                    strict_by_turn[int(row["turn"])] += 1

            transcript = load_jsonl(transcript_path)
            scripted = [
                row
                for row in transcript
                if not row.get("recovery_turn") and 0 <= int(row["turn"]) < 30
            ]
            if len(scripted) != 30:
                raise RuntimeError(f"expected 30 scripted turns: {run_dir}")
            by_turn = {int(row["turn"]): row for row in scripted}
            turn11_success = correct_turn11(by_turn.get(11))
            turn11_eligible += int(turn11_success)
            turn11_category = classify_turn11(by_turn[11])
            turn11_categories[turn11_category] += 1
            is_turn11_redundant = int(turn11_category == TURN12_REDUNDANT)
            arm_turn11_redundant.append(is_turn11_redundant)
            if is_turn11_redundant:
                turn11_subtypes[redundant_subtype(by_turn[11].get("assistant_text", ""))] += 1
            turn12_category = classify_turn12(by_turn[12])
            turn12_categories[turn12_category] += 1
            turn12_by_turn11_success[turn11_success][turn12_category] += 1
            is_redundant = int(turn12_category == TURN12_REDUNDANT)
            arm_turn12_redundant.append(is_redundant)
            arm_suggestion_turn_redundant_scores.append(
                is_turn11_redundant + is_redundant
            )
            if is_redundant:
                turn12_subtypes[redundant_subtype(by_turn[12].get("assistant_text", ""))] += 1
            for row in scripted:
                tokens = row.get("tokens") or {}
                if tokens.get("completion_tokens") is not None:
                    completion_tokens.append(int(tokens["completion_tokens"]))
                raw = row.get("raw_ttfb_ms")
                answer = row.get("ttfb_ms")
                latency = row.get("latency_ms")
                if raw is not None:
                    raw_ttft.append(float(raw))
                if answer is not None:
                    ttfat.append(float(answer))
                if latency is not None:
                    response_latency.append(float(latency))
                if raw is not None and answer is not None:
                    reasoning_delay.append(max(0.0, float(answer) - float(raw)))

        conversation_scores[arm] = strict_scores
        turn12_redundant[arm] = arm_turn12_redundant
        turn11_redundant[arm] = arm_turn11_redundant
        suggestion_turn_redundant_scores[arm] = arm_suggestion_turn_redundant_scores
        strict_total = sum(strict_scores)
        arm_result = {
            "conversations": len(strict_scores),
            "turns": len(strict_scores) * 30,
            "strict_pass": strict_total,
            "strict_rate": strict_total / (len(strict_scores) * 30),
            "strict_rate_cluster_bootstrap_95ci": bootstrap_rate(strict_scores, rng),
            "conversation_score": {
                "median": statistics.median(strict_scores),
                "min": min(strict_scores),
                "max": max(strict_scores),
            },
            "metric_passes": {metric: metric_counts[metric] for metric in METRICS},
            "recovery_responses": recovery_responses,
            "completion_tokens": {
                "total": sum(completion_tokens),
                "mean": statistics.fmean(completion_tokens),
                "median": statistics.median(completion_tokens),
                "p95": percentile(completion_tokens, 0.95),
            },
            "raw_ttft_ms": {
                "median": statistics.median(raw_ttft),
                "p95": percentile(raw_ttft, 0.95),
                "max": max(raw_ttft),
            },
            "ttfat_ms": {
                "median": statistics.median(ttfat),
                "p95": percentile(ttfat, 0.95),
                "max": max(ttfat),
            },
            "reasoning_delay_ms": {
                "median": statistics.median(reasoning_delay),
                "p95": percentile(reasoning_delay, 0.95),
                "max": max(reasoning_delay),
            },
            "response_latency_ms": {
                "median": statistics.median(response_latency),
                "p95": percentile(response_latency, 0.95),
                "max": max(response_latency),
            },
            "strict_pass_by_turn": {
                str(turn): strict_by_turn[turn] for turn in range(30)
            },
            "turn12": {
                "turn11_direct_tool_success": turn11_eligible,
                "categories": dict(turn12_categories),
                "redundant_rate": sum(arm_turn12_redundant) / len(arm_turn12_redundant),
                "redundant_rate_bootstrap_95ci": bootstrap_binary_rate(
                    arm_turn12_redundant, rng
                ),
                "redundant_subtypes": dict(turn12_subtypes),
                "categories_by_turn11_direct_tool_success": {
                    "true": dict(turn12_by_turn11_success[True]),
                    "false": dict(turn12_by_turn11_success[False]),
                },
            },
            "turn11": {
                "categories": dict(turn11_categories),
                "redundant_rate": sum(arm_turn11_redundant) / len(arm_turn11_redundant),
                "redundant_rate_bootstrap_95ci": bootstrap_binary_rate(
                    arm_turn11_redundant, rng
                ),
                "redundant_subtypes": dict(turn11_subtypes),
            },
            "suggestion_turns_11_and_12": {
                "opportunities": 60,
                "redundant": sum(arm_suggestion_turn_redundant_scores),
                "redundant_rate": sum(arm_suggestion_turn_redundant_scores) / 60,
                "conversation_cluster_bootstrap_95ci": bootstrap_score_rate(
                    arm_suggestion_turn_redundant_scores, 2, rng
                ),
            },
        }
        output["arms"][arm] = arm_result

    for left in ARMS:
        for right in ARMS:
            if ARMS.index(left) >= ARMS.index(right):
                continue
            left_rate = sum(conversation_scores[left]) / 900
            right_rate = sum(conversation_scores[right]) / 900
            key = f"{left}_minus_{right}"
            output["pairwise_strict_rate_differences"][key] = {
                "difference": left_rate - right_rate,
                "cluster_bootstrap_95ci": bootstrap_difference(
                    conversation_scores[left], conversation_scores[right], rng
                ),
            }
            left_redundant = turn12_redundant[left]
            right_redundant = turn12_redundant[right]
            output["pairwise_turn12_redundant_rate_differences"][key] = {
                "difference": (
                    sum(left_redundant) / len(left_redundant)
                    - sum(right_redundant) / len(right_redundant)
                ),
                "bootstrap_95ci": bootstrap_binary_difference(
                    left_redundant, right_redundant, rng
                ),
            }
            left_turn11 = turn11_redundant[left]
            right_turn11 = turn11_redundant[right]
            output["pairwise_turn11_redundant_rate_differences"][key] = {
                "difference": sum(left_turn11) / 30 - sum(right_turn11) / 30,
                "bootstrap_95ci": bootstrap_binary_difference(
                    left_turn11, right_turn11, rng
                ),
            }
            left_suggestion = suggestion_turn_redundant_scores[left]
            right_suggestion = suggestion_turn_redundant_scores[right]
            output["pairwise_suggestion_turn_redundant_rate_differences"][key] = {
                "difference": sum(left_suggestion) / 60 - sum(right_suggestion) / 60,
                "conversation_cluster_bootstrap_95ci": bootstrap_score_difference(
                    left_suggestion, right_suggestion, 2, rng
                ),
            }

    result_path = CAMPAIGN / "analysis.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    low_result = output["arms"]["low"]
    high_result = output["arms"]["high"]
    low_high = output["pairwise_strict_rate_differences"]["low_minus_high"]
    low_high_ci = low_high["cluster_bootstrap_95ci"]
    low_xhigh_redundant = output[
        "pairwise_suggestion_turn_redundant_rate_differences"
    ]["low_minus_xhigh"]
    low_xhigh_redundant_ci = low_xhigh_redundant[
        "conversation_cluster_bootstrap_95ci"
    ]
    token_reduction = 1 - (
        low_result["completion_tokens"]["mean"]
        / high_result["completion_tokens"]["mean"]
    )
    p95_reduction = 1 - (
        low_result["ttfat_ms"]["p95"] / high_result["ttfat_ms"]["p95"]
    )
    lines = [
        "# Muse Glimmer reasoning-strength sweep",
        "",
        "## Result",
        "",
        "`low` is the best operating point on this benchmark. It has the highest observed "
        "strict pass rate, but the accuracy differences are unresolved: low minus high is "
        f"{low_high['difference']:+.2%} (conversation-cluster bootstrap 95% CI "
        f"{low_high_ci[0]:+.2%} to {low_high_ci[1]:+.2%}). `low` nevertheless uses "
        f"{token_reduction:.1%} fewer mean completion tokens than `high` and cuts P95 "
        f"answer latency by {p95_reduction:.1%}.",
        "",
        "The exploratory redundant-confirmation result points in the same operational "
        "direction but is not conclusive. Across the two consecutive suggestion turns, "
        f"low minus xhigh is {low_xhigh_redundant['difference']:+.2%} (conversation-cluster "
        f"bootstrap 95% CI {low_xhigh_redundant_ci[0]:+.2%} to "
        f"{low_xhigh_redundant_ci[1]:+.2%}). Turn 12 alone remains highly failure-prone "
        "at every strength and shows no monotonic strength effect.",
        "",
        "## Accuracy and latency",
        "",
        "| Strength | Strict pass | 95% CI | Completion tok mean | Scripted TTFAT P50 / P95 | Reasoning delay P50 / P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        result = output["arms"][arm]
        ci = result["strict_rate_cluster_bootstrap_95ci"]
        lines.append(
            f"| {arm} | {result['strict_pass']}/900 ({result['strict_rate']:.1%}) "
            f"| {ci[0]:.1%}–{ci[1]:.1%} "
            f"| {result['completion_tokens']['mean']:.1f} "
            f"| {result['ttfat_ms']['median']:.0f}/{result['ttfat_ms']['p95']:.0f} ms "
            f"| {result['reasoning_delay_ms']['median']:.0f}/{result['reasoning_delay_ms']['p95']:.0f} ms |"
        )
    lines.extend(["", "## Pairwise strict-rate differences", ""])
    for key, result in output["pairwise_strict_rate_differences"].items():
        ci = result["cluster_bootstrap_95ci"]
        lines.append(
            f"- `{key}`: {result['difference']:+.2%} "
            f"(conversation-cluster bootstrap 95% CI {ci[0]:+.2%} to {ci[1]:+.2%})"
        )
    lines.extend(
        [
            "",
            "## Redundant confirmation on the two suggestion turns",
            "",
            "This exploratory composite counts redundant confirmations on scripted Turns 11 and 12. "
            "Its interval resamples whole conversations, preserving dependence between the two turns.",
            "",
            "| Strength | Turn 11 redundant | Turn 12 redundant | Combined | Combined 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        turn11 = output["arms"][arm]["turn11"]
        turn12 = output["arms"][arm]["turn12"]
        combined = output["arms"][arm]["suggestion_turns_11_and_12"]
        ci = combined["conversation_cluster_bootstrap_95ci"]
        lines.append(
            f"| {arm} | {turn11['categories'].get(TURN12_REDUNDANT, 0)}/30 "
            f"({turn11['redundant_rate']:.1%}) | "
            f"{turn12['categories'].get(TURN12_REDUNDANT, 0)}/30 "
            f"({turn12['redundant_rate']:.1%}) | "
            f"{combined['redundant']}/60 ({combined['redundant_rate']:.1%}) | "
            f"{ci[0]:.1%}–{ci[1]:.1%} |"
        )
    lines.extend(["", "Pairwise combined-rate differences:", ""])
    for key, result in output[
        "pairwise_suggestion_turn_redundant_rate_differences"
    ].items():
        ci = result["conversation_cluster_bootstrap_95ci"]
        lines.append(
            f"- `{key}`: {result['difference']:+.2%} "
            f"(conversation-cluster bootstrap 95% CI {ci[0]:+.2%} to {ci[1]:+.2%})"
        )
    lines.extend(
        [
            "",
            "## Turn 12 redundant confirmation",
            "",
            "These are on-policy outcomes: each strength generated its own preceding history. "
            "They measure the production configuration, not a same-prefix direct effect.",
            "",
            "| Strength | Turn 11 direct tool success | Correct call | Redundant confirmation | Other outcome |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        result = output["arms"][arm]["turn12"]
        categories = result["categories"]
        redundant = categories.get(TURN12_REDUNDANT, 0)
        correct = categories.get("correct_tool_and_arguments", 0)
        other = 30 - redundant - correct
        lines.append(
            f"| {arm} | {result['turn11_direct_tool_success']}/30 | {correct}/30 | "
            f"{redundant}/30 ({result['redundant_rate']:.1%}) | {other}/30 |"
        )
    lines.extend(
        [
            "",
            "Turn 12 stratified by whether the scripted Turn 11 response directly made its expected tool call:",
            "",
            "| Strength | Turn 11 direct call? | N | Turn 12 correct | Turn 12 redundant | Other |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        strata = output["arms"][arm]["turn12"]["categories_by_turn11_direct_tool_success"]
        for value in ("true", "false"):
            categories = strata[value]
            n = sum(categories.values())
            correct = categories.get("correct_tool_and_arguments", 0)
            redundant = categories.get(TURN12_REDUNDANT, 0)
            lines.append(
                f"| {arm} | {value} | {n} | {correct}/{n} | {redundant}/{n} | "
                f"{n - correct - redundant}/{n} |"
            )
    lines.extend(["", "Pairwise redundant-rate differences:", ""])
    for key, result in output["pairwise_turn12_redundant_rate_differences"].items():
        ci = result["bootstrap_95ci"]
        lines.append(
            f"- `{key}`: {result['difference']:+.2%} "
            f"(bootstrap 95% CI {ci[0]:+.2%} to {ci[1]:+.2%})"
        )
    lines.extend(["", "Redundant-confirmation subtypes:", ""])
    subtype_names = sorted(
        {
            name
            for arm in ARMS
            for name in output["arms"][arm]["turn12"]["redundant_subtypes"]
        }
    )
    lines.append("| Subtype | " + " | ".join(ARMS) + " |")
    lines.append("|---|" + "---:|" * len(ARMS))
    for subtype in subtype_names:
        lines.append(
            f"| {subtype} | "
            + " | ".join(
                str(output["arms"][arm]["turn12"]["redundant_subtypes"].get(subtype, 0))
                for arm in ARMS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Controls and interpretation",
            "",
            "- The official supported values are exactly `low`, `medium`, `high`, and "
            "`xhigh`; the model card says reasoning cannot be disabled.",
            "- The live embedded-template audit proves absent/default equals `high`, and "
            "that top-level `reasoning_effort=none` and `enable_thinking=false` are "
            "render no-ops. `none` and `minimal` merely render unsupported literal labels, "
            "so they were not promoted to experimental arms.",
            "- The benchmark system instruction is unchanged and appears exactly once in "
            "every audited render. The only intended arm difference is "
            "`chat_template_kwargs.reasoning_strength`.",
            "- These are independent on-policy trajectories, balanced and interleaved by "
            "arm. Pairwise intervals are descriptive fixed-sample comparisons with no "
            "multiplicity adjustment; a same-prefix replay would answer a different "
            "question.",
            "- Completion tokens include the model's hidden reasoning and answer output; "
            "the local backend does not expose a separate thinking-token count. TTFAT is "
            "time to the first answer/tool token, while raw TTFT is time to the first "
            "reasoning token.",
        ]
    )
    lines.extend(["", "## Per-turn strict passes", ""])
    lines.append("| Turn | " + " | ".join(ARMS) + " |")
    lines.append("|---:|" + "---:|" * len(ARMS))
    for turn in range(30):
        lines.append(
            f"| {turn} | "
            + " | ".join(
                f"{output['arms'][arm]['strict_pass_by_turn'][str(turn)]}/30"
                for arm in ARMS
            )
            + " |"
        )
    (CAMPAIGN / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
