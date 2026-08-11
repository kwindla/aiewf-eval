#!/usr/bin/env python3
"""Analyze whether earlier generated history mediates local target-turn errors."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from study import (
    HERE,
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    source_rows,
    target_row,
    transcript_by_turn,
)


SEED = 20260807
BOOTSTRAPS = 50_000


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def independent_difference_ci(left: list[int], right: list[int], salt: int) -> list[float]:
    rng = random.Random(SEED + salt)
    estimates = []
    for _ in range(BOOTSTRAPS):
        a = sum(left[rng.randrange(len(left))] for _ in left) / len(left)
        b = sum(right[rng.randrange(len(right))] for _ in right) / len(right)
        estimates.append((a - b) * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def words(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value))


def has_correct_tool(row: dict[str, Any], name: str) -> bool:
    return any(call.get("name") == name for call in row.get("tool_calls") or [])


def build_rows() -> list[dict[str, Any]]:
    result = []
    for arm in ("local_fp8", "local_bf16"):
        for manifest in source_rows(arm):
            run_dir = manifest["run_dir"]
            turns = transcript_by_turn(run_dir)
            judged12 = target_row(run_dir, 12, judged=True)
            judged15 = target_row(run_dir, 15, judged=True)
            t9_11 = " ".join(str(turns[index].get("assistant_text") or "") for index in (9, 10, 11))
            t13_14 = " ".join(str(turns[index].get("assistant_text") or "") for index in (13, 14))
            t14 = str(turns[14].get("assistant_text") or "").casefold()
            result.append(
                {
                    "arm": arm,
                    "run_dir": str(run_dir),
                    "turn12_failure": not bool(judged12["scores"]["tool_use_correct"]),
                    "turn15_failure": not bool(judged15["scores"]["tool_use_correct"]),
                    "turn9_11_words": words(t9_11),
                    "turn13_14_words": words(t13_14),
                    "turn11_tool_present": has_correct_tool(turns[11], "submit_session_suggestion"),
                    "turn11_text_with_tool": bool(turns[11].get("assistant_text")),
                    "turn10_mentions_jennifer": "jennifer" in str(turns[10].get("assistant_text") or "").casefold(),
                    "turn12_recovery_recorded": any(
                        row.get("recovery_for_turn") == 12
                        for row in read_jsonl(run_dir / "transcript.jsonl")
                    ),
                    "turn14_mentions_jennifer": "jennifer" in t14,
                    "turn14_mentions_vegan": "vegan" in t14,
                    "turn14_offers_request": "request" in t14 or "submit" in t14,
                    "turn14_asks_question": "?" in t14,
                    "turn14_tool_present": bool(turns[14].get("tool_calls")),
                }
            )
    if len(result) != 300:
        raise RuntimeError(f"loaded {len(result)} local conversations, expected 300")
    return result


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    return {
        "n": len(selected),
        "turn12_failure_percent": sum(row["turn12_failure"] for row in selected) / len(selected) * 100,
        "turn15_failure_percent": sum(row["turn15_failure"] for row in selected) / len(selected) * 100,
        "mean_turn9_11_words": sum(row["turn9_11_words"] for row in selected) / len(selected),
        "mean_turn13_14_words": sum(row["turn13_14_words"] for row in selected) / len(selected),
        "feature_percent": {
            key: sum(bool(row[key]) for row in selected) / len(selected) * 100
            for key in selected[0]
            if key.startswith("turn") and isinstance(selected[0][key], bool)
        },
    }


def predictor_table(
    rows: list[dict[str, Any]], predictor: str, outcome: str
) -> dict[str, Any]:
    positive = [int(row[outcome]) for row in rows if row[predictor]]
    negative = [int(row[outcome]) for row in rows if not row[predictor]]
    if not positive or not negative:
        return {"predictor": predictor, "outcome": outcome, "estimable": False}
    positive_rate = sum(positive) / len(positive) * 100
    negative_rate = sum(negative) / len(negative) * 100
    return {
        "predictor": predictor,
        "outcome": outcome,
        "estimable": True,
        "present_n": len(positive),
        "absent_n": len(negative),
        "failure_percent_when_present": positive_rate,
        "failure_percent_when_absent": negative_rate,
        "risk_difference_points": positive_rate - negative_rate,
        "bootstrap_95_percent": independent_difference_ci(
            positive, negative, salt=sum(map(ord, predictor + outcome))
        ),
    }


def main() -> int:
    rows = build_rows()
    predictors = [
        ("turn11_text_with_tool", "turn12_failure"),
        ("turn10_mentions_jennifer", "turn12_failure"),
        ("turn12_recovery_recorded", "turn15_failure"),
        ("turn14_mentions_jennifer", "turn15_failure"),
        ("turn14_offers_request", "turn15_failure"),
        ("turn14_asks_question", "turn15_failure"),
    ]
    payload = {
        "schema_version": 1,
        "n_conversations": len(rows),
        "scope": "existing 150 local FP8-KV and 150 local BF16-KV completed conversations",
        "arms": {arm: arm_summary(rows, arm) for arm in ("local_fp8", "local_bf16")},
        "predictors_pooled_across_arms": [
            predictor_table(rows, predictor, outcome) for predictor, outcome in predictors
        ],
        "notes": [
            "Associations are descriptive and post hoc; they are not causal mediation estimates.",
            "Prefix-bank replay isolates the direct KV effect conditional on frozen history.",
            "Arm differences in these prefix features quantify the upstream-history pathway that frozen replay removes.",
        ],
    }
    atomic_write_json(HERE / "historical-mediation.json", payload)

    lines = [
        "# Historical upstream-history mediation screen",
        "",
        "This post-hoc screen uses the 300 completed local conversations. It asks whether",
        "the KV arms produced different lead-in histories, and whether those features are",
        "associated with the target-turn failures. It is descriptive, not a formal causal",
        "mediation estimate.",
        "",
        "## Arm summaries",
        "",
        "| Arm | N | Turn 12 failure | Turn 15 failure | Turns 9–11 assistant words | Turns 13–14 assistant words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("local_fp8", "local_bf16"):
        row = payload["arms"][arm]
        lines.append(
            f"| {arm} | {row['n']} | {row['turn12_failure_percent']:.1f}% | "
            f"{row['turn15_failure_percent']:.1f}% | {row['mean_turn9_11_words']:.1f} | "
            f"{row['mean_turn13_14_words']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Feature associations pooled across arms",
            "",
            "| Feature | Outcome | Present N | Absent N | Failure if present | Failure if absent | Difference (95% bootstrap CI) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["predictors_pooled_across_arms"]:
        if not row["estimable"]:
            continue
        lo, hi = row["bootstrap_95_percent"]
        lines.append(
            f"| {row['predictor']} | {row['outcome']} | {row['present_n']} | {row['absent_n']} | "
            f"{row['failure_percent_when_present']:.1f}% | {row['failure_percent_when_absent']:.1f}% | "
            f"{row['risk_difference_points']:+.1f} pp ({lo:+.1f}, {hi:+.1f}) |"
        )
    lines.extend(["", "See `historical-mediation.json` for all arm feature rates.", ""])
    atomic_write_text(HERE / "historical-mediation.md", "\n".join(lines))
    print(json.dumps(payload["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
