#!/usr/bin/env python3
"""Descriptive post-analysis of turn-12 no-tool failure modes.

This script is intentionally downstream of the sealed confirmatory analysis.
It does not change the primary estimand, stopping rule, or confidence interval.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORRECT = "correct_tool_and_arguments"
REDUNDANT = "no_tool_redundant_confirmation_or_question"
FALSE_CLAIM = "no_tool_false_claim_of_completion"
OTHER = "no_tool_other"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def percent(count: int, denominator: int) -> float:
    return count / denominator * 100.0


def correlation(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            result[order[position]] = rank
        index = end + 1
    return result


def redundant_subtype(text: str) -> str:
    lowered = clean_text(text).lower()
    if re.search(
        r"i(?:'ve| have).{0,45}(?:sent|submitted|added|recorded)|"
        r"gone ahead and|successfully submitted",
        lowered,
    ) and re.search(r"anything else|\?", lowered):
        return "post_action_claim_then_followup"
    if re.search(
        r"under your name|use your name|also from you|coming from you|"
        r"for someone else|is this suggestion.*from you|is this.*also from you",
        lowered,
    ):
        return "identity_or_ownership_reconfirmation"
    if re.search(
        r"you(?:'d| would) like (?:to suggest|a session|this suggestion)|"
        r"session on .*\b(?:correct|right)\b|suggestion.*\b(?:correct|right)\b",
        lowered,
    ):
        return "content_or_intent_reconfirmation"
    if re.search(r"should i submit|shall i|want me to|would you like me to|go ahead", lowered):
        return "authorization_reconfirmation"
    return "other_question"


def other_subtype(text: str) -> str:
    normalized = clean_text(text)
    if re.search(
        r"(?i)\(action|action:|`submit_session_suggestion|"
        r"calling\s+`?submit_session|call\s+`?submit_session",
        normalized,
    ):
        return "textual_pseudo_tool_call"
    if re.search(
        r"(?i)\b(?:i(?:'ve| have)|we(?:'ve| have))\b.{0,35}"
        r"\b(?:sent|submitted|added|recorded|noted|filed)\b|"
        r"\bsuccessfully submitted\b",
        normalized,
    ):
        return "uncaught_false_completion"
    if re.search(
        r"(?i)\b(?:i(?:'ll| will)|let me)\b.{0,50}\b(?:submit|add|send|get)\b|"
        r"\bright now\b",
        normalized,
    ):
        return "future_action_promise"
    return "other_acknowledgement"


def canonical_recent_suffix(snapshot: dict[str, Any]) -> str:
    messages = copy.deepcopy(snapshot["request"]["messages"][-4:])
    for message in messages:
        message.pop("tool_call_id", None)
        for call in message.get("tool_calls") or []:
            call.pop("id", None)
    return json.dumps(messages, sort_keys=True, separators=(",", ":"))


def source_family(source: str) -> str:
    if source == "local_fp8":
        return "local_fp8_origin"
    if source in {"local_bf16", "baseten_bf16"}:
        return "bf16_based_origin"
    return source


def summarize_prefix_group(
    prefix_names: list[str],
    categories: dict[tuple[str, str], Counter[str]],
) -> dict[str, Any]:
    arm_counts: dict[str, Counter[str]] = {}
    for arm in ("bf16", "fp8"):
        counts: Counter[str] = Counter()
        for prefix in prefix_names:
            counts.update(categories[(prefix, arm)])
        arm_counts[arm] = counts
    denominator = len(prefix_names) * 512
    return {
        "prefixes": prefix_names,
        "rows_per_arm": denominator,
        "bf16_success_percent": percent(arm_counts["bf16"][CORRECT], denominator),
        "fp8_success_percent": percent(arm_counts["fp8"][CORRECT], denominator),
        "bf16_minus_fp8_success_points": percent(
            arm_counts["bf16"][CORRECT] - arm_counts["fp8"][CORRECT], denominator
        ),
        "bf16_redundant_percent": percent(arm_counts["bf16"][REDUNDANT], denominator),
        "fp8_redundant_percent": percent(arm_counts["fp8"][REDUNDANT], denominator),
        "fp8_minus_bf16_redundant_points": percent(
            arm_counts["fp8"][REDUNDANT] - arm_counts["bf16"][REDUNDANT], denominator
        ),
        "bf16_other_percent": percent(arm_counts["bf16"][OTHER], denominator),
        "fp8_other_percent": percent(arm_counts["fp8"][OTHER], denominator),
        "fp8_minus_bf16_other_points": percent(
            arm_counts["fp8"][OTHER] - arm_counts["bf16"][OTHER], denominator
        ),
    }


def analyze(study_dir: Path) -> dict[str, Any]:
    final_analysis = read_json(study_dir / "results/v2-primary-8192-analysis.json")
    rows: list[dict[str, Any]] = []
    for relative in final_analysis["source_files"]:
        with (study_dir / relative).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["turn"] == 12 and row["snapshot_kind"] == "real_prefix_bank":
                    rows.append(row)

    assert len(rows) == 12_288
    prefixes = sorted({row["snapshot_id"] for row in rows})
    assert len(prefixes) == 12

    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    categories: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    texts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["snapshot_id"], int(row["seed"]), row["arm"])
        assert key not in cells
        cells[key] = row
        category = row["score"]["category"]
        categories[(row["snapshot_id"], row["arm"])][category] += 1
        if category in {REDUNDANT, OTHER, FALSE_CLAIM}:
            texts[(row["arm"], category, row["snapshot_id"])][
                clean_text(row["score"].get("content"))
            ] += 1

    for prefix in prefixes:
        for arm in ("bf16", "fp8"):
            assert sum(categories[(prefix, arm)].values()) == 512
            assert {seed for p, seed, a in cells if p == prefix and a == arm} == set(range(512))

    aggregate: dict[str, Any] = {}
    for arm in ("bf16", "fp8"):
        counts = Counter()
        for prefix in prefixes:
            counts.update(categories[(prefix, arm)])
        assert sum(counts.values()) == 6_144
        aggregate[arm] = {
            "rows": 6_144,
            "counts": dict(counts),
            "percents": {key: percent(value, 6_144) for key, value in counts.items()},
        }

    confirmatory = final_analysis["paired_comparisons"]["warm_turn12_bank"]
    assert aggregate["bf16"]["counts"][CORRECT] == 3_943
    assert aggregate["fp8"]["counts"][CORRECT] == 2_967
    assert math.isclose(
        percent(3_943 - 2_967, 6_144),
        confirmatory["equal_prefix_weighted_difference_points"],
    )

    transition_counts: Counter[tuple[str, str]] = Counter()
    per_prefix: dict[str, Any] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        snapshot = read_json(study_dir / "snapshots" / f"{prefix}.json")
        snapshots[prefix] = snapshot
        bf16 = categories[(prefix, "bf16")]
        fp8 = categories[(prefix, "fp8")]
        prefix_transitions: Counter[tuple[str, str]] = Counter()
        for seed in range(512):
            before = cells[(prefix, seed, "bf16")]["score"]["category"]
            after = cells[(prefix, seed, "fp8")]["score"]["category"]
            transition_counts[(before, after)] += 1
            prefix_transitions[(before, after)] += 1
        metadata = snapshot["metadata"]
        name_acknowledgements = [
            clean_text(message.get("content"))
            for message in snapshot["request"]["messages"]
            if message.get("role") == "assistant"
            and "Jennifer" in clean_text(message.get("content"))
        ]
        assert len(name_acknowledgements) == 1
        name_acknowledgement = name_acknowledgements[0]
        per_prefix[prefix] = {
            "source": metadata["source"],
            "source_family": source_family(metadata["source"]),
            "campaign_slot": metadata["campaign_slot"],
            "prompt_tokens": cells[(prefix, 0, "bf16")]["completion"]["usage"]["prompt_tokens"],
            "name_acknowledgement": name_acknowledgement,
            "nice_to_meet_acknowledgement": name_acknowledgement.startswith(
                "Nice to meet you, Jennifer!"
            ),
            "bf16_counts": dict(bf16),
            "fp8_counts": dict(fp8),
            "bf16_success_percent": percent(bf16[CORRECT], 512),
            "fp8_success_percent": percent(fp8[CORRECT], 512),
            "bf16_minus_fp8_success_points": percent(bf16[CORRECT] - fp8[CORRECT], 512),
            "fp8_minus_bf16_redundant_points": percent(fp8[REDUNDANT] - bf16[REDUNDANT], 512),
            "fp8_minus_bf16_other_points": percent(fp8[OTHER] - bf16[OTHER], 512),
            "fp8_minus_bf16_false_claim_points": percent(fp8[FALSE_CLAIM] - bf16[FALSE_CLAIM], 512),
            "correct_to_redundant": prefix_transitions[(CORRECT, REDUNDANT)],
            "redundant_to_correct": prefix_transitions[(REDUNDANT, CORRECT)],
        }

    source_groups: dict[str, Any] = {}
    grouped_prefixes: dict[str, list[str]] = defaultdict(list)
    for prefix, item in per_prefix.items():
        grouped_prefixes[item["source"]].append(prefix)
    for source, source_prefixes in sorted(grouped_prefixes.items()):
        source_groups[source] = summarize_prefix_group(source_prefixes, categories)

    grouped_families: dict[str, list[str]] = defaultdict(list)
    for prefix, item in per_prefix.items():
        grouped_families[item["source_family"]].append(prefix)
    source_families = {
        family: summarize_prefix_group(sorted(family_prefixes), categories)
        for family, family_prefixes in sorted(grouped_families.items())
    }

    acknowledgement_groups = {
        "nice_to_meet": summarize_prefix_group(
            sorted(
                prefix
                for prefix, item in per_prefix.items()
                if item["nice_to_meet_acknowledgement"]
            ),
            categories,
        ),
        "other_acknowledgement": summarize_prefix_group(
            sorted(
                prefix
                for prefix, item in per_prefix.items()
                if not item["nice_to_meet_acknowledgement"]
            ),
            categories,
        ),
    }

    seed_quartiles: list[dict[str, Any]] = []
    for start in range(0, 512, 128):
        arm_counts: dict[str, Counter[str]] = {"bf16": Counter(), "fp8": Counter()}
        for prefix in prefixes:
            for seed in range(start, start + 128):
                for arm in ("bf16", "fp8"):
                    arm_counts[arm][cells[(prefix, seed, arm)]["score"]["category"]] += 1
        denominator = 128 * len(prefixes)
        seed_quartiles.append(
            {
                "seed_start": start,
                "seed_end": start + 127,
                "fp8_minus_bf16_redundant_points": percent(
                    arm_counts["fp8"][REDUNDANT] - arm_counts["bf16"][REDUNDANT],
                    denominator,
                ),
                "fp8_minus_bf16_other_points": percent(
                    arm_counts["fp8"][OTHER] - arm_counts["bf16"][OTHER],
                    denominator,
                ),
            }
        )

    normalized_suffixes = {canonical_recent_suffix(snapshot) for snapshot in snapshots.values()}
    assert len(normalized_suffixes) == 1

    redundant_subtypes: dict[str, Counter[str]] = defaultdict(Counter)
    other_subtypes: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {
        "redundant": {},
        "other": {},
    }
    for arm in ("bf16", "fp8"):
        redundant_texts = Counter()
        other_texts = Counter()
        for prefix in prefixes:
            redundant_texts.update(texts[(arm, REDUNDANT, prefix)])
            other_texts.update(texts[(arm, OTHER, prefix)])
        for text, count in redundant_texts.items():
            redundant_subtypes[arm][redundant_subtype(text)] += count
        for text, count in other_texts.items():
            other_subtypes[arm][other_subtype(text)] += count
        examples["redundant"][arm] = [
            {"count": count, "text": text}
            for text, count in redundant_texts.most_common(12)
        ]
        examples["other"][arm] = [
            {"count": count, "text": text}
            for text, count in other_texts.most_common(12)
        ]

    teacher: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("bf16", "fp8"):
        payload = read_json(study_dir / "results" / f"{arm}-teacher-forced-warm.json")
        teacher[arm] = {item["snapshot_id"]: item for item in payload["snapshots"]}

    behavioral_differences: list[float] = []
    redundant_differences: list[float] = []
    margin_differences: list[float] = []
    for prefix in prefixes:
        bf16_margin = teacher["bf16"][prefix]["first_expected_minus_alternative_logprob"]
        fp8_margin = teacher["fp8"][prefix]["first_expected_minus_alternative_logprob"]
        margin_difference = bf16_margin - fp8_margin
        per_prefix[prefix]["bf16_first_tool_margin"] = bf16_margin
        per_prefix[prefix]["fp8_first_tool_margin"] = fp8_margin
        per_prefix[prefix]["bf16_minus_fp8_first_tool_margin"] = margin_difference
        behavioral_differences.append(per_prefix[prefix]["bf16_minus_fp8_success_points"])
        redundant_differences.append(per_prefix[prefix]["fp8_minus_bf16_redundant_points"])
        margin_differences.append(margin_difference)

    transition_rows = [
        {"bf16_category": before, "fp8_category": after, "count": count}
        for (before, after), count in transition_counts.most_common()
    ]
    assert sum(item["count"] for item in transition_rows) == 6_144

    redundant_net_counts = sorted(
        (
            item["fp8_counts"].get(REDUNDANT, 0)
            - item["bf16_counts"].get(REDUNDANT, 0),
            prefix,
        )
        for prefix, item in per_prefix.items()
    )
    other_net_counts = sorted(
        (
            item["fp8_counts"].get(OTHER, 0) - item["bf16_counts"].get(OTHER, 0),
            prefix,
        )
        for prefix, item in per_prefix.items()
    )

    return {
        "schema_version": 1,
        "scope": "descriptive post-analysis of the sealed 8,192-look turn-12 real-prefix bank",
        "confirmatory_result_unchanged": True,
        "rows": len(rows),
        "prefixes": len(prefixes),
        "seeds_per_prefix_arm": 512,
        "aggregate": aggregate,
        "per_prefix": per_prefix,
        "source_groups": source_groups,
        "source_families": source_families,
        "acknowledgement_groups": acknowledgement_groups,
        "seed_quartiles": seed_quartiles,
        "concentration": {
            "redundant_net_total": sum(count for count, _ in redundant_net_counts),
            "redundant_top_four_net": sum(count for count, _ in redundant_net_counts[-4:]),
            "redundant_top_four_prefixes": [
                prefix for _, prefix in reversed(redundant_net_counts[-4:])
            ],
            "other_net_total": sum(count for count, _ in other_net_counts),
            "other_top_four_net": sum(count for count, _ in other_net_counts[-4:]),
            "other_top_four_prefixes": [
                prefix for _, prefix in reversed(other_net_counts[-4:])
            ],
        },
        "recent_suffix_check": {
            "messages_compared": 4,
            "generated_call_ids_removed": True,
            "unique_normalized_suffixes": len(normalized_suffixes),
        },
        "paired_category_transitions": transition_rows,
        "redundant_confirmation_subtypes": {
            arm: dict(counts) for arm, counts in redundant_subtypes.items()
        },
        "other_no_tool_subtypes": {
            arm: dict(counts) for arm, counts in other_subtypes.items()
        },
        "first_decision_margin_association": {
            "description": "post-hoc association across 12 prefixes; not a new confirmatory test",
            "behavioral_difference_vs_margin_difference_pearson": correlation(
                behavioral_differences, margin_differences
            ),
            "behavioral_difference_vs_margin_difference_spearman": correlation(
                ranks(behavioral_differences), ranks(margin_differences)
            ),
            "redundant_difference_vs_margin_difference_pearson": correlation(
                redundant_differences, margin_differences
            ),
        },
        "representative_exact_outputs": examples,
    }


def markdown(payload: dict[str, Any]) -> str:
    bf16 = payload["aggregate"]["bf16"]
    fp8 = payload["aggregate"]["fp8"]
    lines = [
        "# Turn 12 post-analysis: no-tool failure patterns",
        "",
        "This is a descriptive analysis performed after the sealed confirmatory result. It does not alter the primary effect estimate or interval.",
        "",
        "## Aggregate categories",
        "",
        "| Outcome | BF16 | FP8 | FP8 - BF16 |",
        "|---|---:|---:|---:|",
    ]
    for category in (CORRECT, REDUNDANT, FALSE_CLAIM, OTHER):
        bf_count = bf16["counts"].get(category, 0)
        fp_count = fp8["counts"].get(category, 0)
        lines.append(
            f"| `{category}` | {bf_count}/6144 ({percent(bf_count, 6144):.2f}%) | "
            f"{fp_count}/6144 ({percent(fp_count, 6144):.2f}%) | "
            f"{percent(fp_count - bf_count, 6144):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Prefix ranking",
            "",
            "| Prefix | Source | BF16 success | FP8 success | BF16-FP8 success | FP8-BF16 redundant | FP8-BF16 other |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        payload["per_prefix"].items(),
        key=lambda item: item[1]["fp8_minus_bf16_redundant_points"],
        reverse=True,
    )
    for prefix, item in ranked:
        lines.append(
            f"| `{prefix}` | {item['source']} | {item['bf16_success_percent']:.1f}% | "
            f"{item['fp8_success_percent']:.1f}% | {item['bf16_minus_fp8_success_points']:+.1f} pp | "
            f"{item['fp8_minus_bf16_redundant_points']:+.1f} pp | "
            f"{item['fp8_minus_bf16_other_points']:+.1f} pp |"
        )

    lines.extend(
        [
            "",
            "## Prefix provenance",
            "",
            "| Source | Prefixes | BF16 success | FP8 success | BF16-FP8 success | BF16 redundant | FP8 redundant |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, item in payload["source_groups"].items():
        lines.append(
            f"| {source} | {len(item['prefixes'])} | {item['bf16_success_percent']:.1f}% | "
            f"{item['fp8_success_percent']:.1f}% | {item['bf16_minus_fp8_success_points']:+.1f} pp | "
            f"{item['bf16_redundant_percent']:.1f}% | {item['fp8_redundant_percent']:.1f}% |"
        )

    bf16_origin = payload["source_families"]["bf16_based_origin"]
    fp8_origin = payload["source_families"]["local_fp8_origin"]
    nice = payload["acknowledgement_groups"]["nice_to_meet"]
    other_ack = payload["acknowledgement_groups"]["other_acknowledgement"]
    concentration = payload["concentration"]
    lines.extend(
        [
            "",
            "The provenance interaction is large and reverses sign. Across the eight BF16-based prefixes, BF16 success exceeds FP8 success by "
            f"{bf16_origin['bf16_minus_fp8_success_points']:.1f} points. Across the four local-FP8 prefixes, FP8 success exceeds BF16 success by "
            f"{-fp8_origin['bf16_minus_fp8_success_points']:.1f} points. The preregistered +15.9-point bank average is therefore conditional on a bank containing eight BF16-based and four FP8-origin histories; it is not an on-policy deployment estimate. The BaseTen histories also came from BF16 weights/KV plus MTP, not merely the local model with a different KV dtype.",
            "",
            "## Concentration and observable prefix qualities",
            "",
            f"Four prefixes contribute {concentration['redundant_top_four_net']} of the net {concentration['redundant_net_total']} additional redundant-question labels ({concentration['redundant_top_four_net'] / concentration['redundant_net_total'] * 100:.1f}%). Nevertheless, nine of twelve prefixes have a positive redundant-label shift, and the aggregate change is stable across the four 128-seed quartiles: "
            + ", ".join(
                f"{item['fp8_minus_bf16_redundant_points']:+.1f} pp"
                for item in payload["seed_quartiles"]
            )
            + ".",
            "",
            "All twelve prefixes have the same user messages, operational state, immediately preceding successful tool call, tool result, and target request. Their prompt lengths span only 13,956–14,051 tokens. The variable content is earlier assistant wording and generated call IDs.",
            "",
            f"One post-hoc wording association is visible: the four histories whose name acknowledgement begins `Nice to meet you, Jennifer!` shift from {nice['bf16_redundant_percent']:.1f}% BF16 redundant to {nice['fp8_redundant_percent']:.1f}% FP8 redundant ({nice['fp8_minus_bf16_redundant_points']:+.1f} pp), versus {other_ack['fp8_minus_bf16_redundant_points']:+.1f} pp for the other eight. This phrase does not isolate a cause: one non-`Nice to meet you` history also has a large positive shift, and the twelve histories differ at many earlier assistant turns.",
            "",
            "The source-origin reversal is the more consequential prefix property. It is compatible with an on-policy or history-manifold effect: each KV path can be more stable on histories it helped generate. With only four selected prefixes per source, source, wording, deployment, and selection are confounded, so this is a diagnosis and a reason to narrow the claim—not proof of that mechanism.",
        ]
    )

    transitions = {
        (item["bf16_category"], item["fp8_category"]): item["count"]
        for item in payload["paired_category_transitions"]
    }
    lines.extend(
        [
            "",
            "## Paired category transitions",
            "",
            f"The dominant paired movement is correct BF16 tool call to FP8 redundant question: {transitions[(CORRECT, REDUNDANT)]} cases, versus {transitions[(REDUNDANT, CORRECT)]} in the reverse direction (net +{transitions[(CORRECT, REDUNDANT)] - transitions[(REDUNDANT, CORRECT)]}). False-completion to redundant-question transitions add a net +{transitions[(FALSE_CLAIM, REDUNDANT)] - transitions[(REDUNDANT, FALSE_CLAIM)]}. Thus most of the redundant-question increase is a loss of correct calls, although some is relabeling among no-tool failure styles.",
        ]
    )

    lines.extend(
        [
            "",
            "## Redundant-confirmation subtypes",
            "",
            "| Descriptive subtype | BF16 | FP8 | FP8 - BF16 |",
            "|---|---:|---:|---:|",
        ]
    )
    redundant_names = [
        "identity_or_ownership_reconfirmation",
        "content_or_intent_reconfirmation",
        "authorization_reconfirmation",
        "post_action_claim_then_followup",
        "other_question",
    ]
    for name in redundant_names:
        bf_count = payload["redundant_confirmation_subtypes"]["bf16"].get(name, 0)
        fp_count = payload["redundant_confirmation_subtypes"]["fp8"].get(name, 0)
        lines.append(f"| `{name}` | {bf_count} | {fp_count} | {fp_count - bf_count:+d} |")

    lines.extend(
        [
            "",
            "## Other no-tool subtypes",
            "",
            "`no_tool_other` is a residual mechanical category, not one coherent behavior.",
            "",
            "| Descriptive subtype | BF16 | FP8 | FP8 - BF16 |",
            "|---|---:|---:|---:|",
        ]
    )
    other_names = [
        "future_action_promise",
        "textual_pseudo_tool_call",
        "uncaught_false_completion",
        "other_acknowledgement",
    ]
    for name in other_names:
        bf_count = payload["other_no_tool_subtypes"]["bf16"].get(name, 0)
        fp_count = payload["other_no_tool_subtypes"]["fp8"].get(name, 0)
        lines.append(f"| `{name}` | {bf_count} | {fp_count} | {fp_count - bf_count:+d} |")

    other_forward = transitions[(CORRECT, OTHER)]
    other_reverse = transitions[(OTHER, CORRECT)]
    lines.extend(
        [
            "",
            f"The residual category rises from {bf16['counts'][OTHER]} to {fp8['counts'][OTHER]} cases ({fp8['counts'][OTHER] / bf16['counts'][OTHER]:.2f}x). Future-action promises and textual pseudo-calls contribute 85 of the net 95 additional cases (89.5%). Four prefixes contribute {concentration['other_top_four_net']} of that net change. Correct-BF16 to FP8-other transitions number {other_forward}, versus {other_reverse} in reverse (net +{other_forward - other_reverse}).",
        ]
    )

    association = payload["first_decision_margin_association"]
    lines.extend(
        [
            "",
            "## Decision-boundary diagnostic",
            "",
            f"All twelve prefixes have one unique normalized four-message suffix after generated call IDs are removed. Across prefixes, the BF16-FP8 change in the teacher-forced first `<|tool_call>` margin against the best ordinary-assistant alternative has Pearson r={association['behavioral_difference_vs_margin_difference_pearson']:.3f} and Spearman rho={association['behavioral_difference_vs_margin_difference_spearman']:.3f} with the behavioral success difference. Its post-hoc Pearson association with the redundant-confirmation difference is r={association['redundant_difference_vs_margin_difference_pearson']:.3f}.",
            "",
            "The no-tool answers often preserve the correct name, topic, function name, and arguments. The likely failure boundary is therefore choosing structured tool-call syntax versus ordinary assistant prose, not simply forgetting required state. Once prose wins, the model falls into familiar confirmation, promise, narration, or completion templates.",
            "",
            "These associations are descriptive mechanism diagnostics, not additional confirmatory tests. The subtype rules are a post-hoc mechanical audit, not independently human-validated semantic labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/v2-turn12-error-pattern-analysis.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/v2-turn12-error-pattern-analysis.md"),
    )
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    payload = analyze(study_dir)
    json_output = args.json_output if args.json_output.is_absolute() else study_dir / args.json_output
    markdown_output = (
        args.markdown_output if args.markdown_output.is_absolute() else study_dir / args.markdown_output
    )
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_output.write_text(markdown(payload))


if __name__ == "__main__":
    main()
