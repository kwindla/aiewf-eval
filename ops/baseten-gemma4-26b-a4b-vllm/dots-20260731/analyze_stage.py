#!/usr/bin/env python3
"""Analyze a fully judged paired Gemma no-filler/+96-dot stage.

The default invocation validates and computes without writing. ``--execute``
atomically freezes stage aggregates, an included-run manifest, and a Markdown
report. At the initial stage it writes a collector-compatible promotion
decision only if a prespecified trigger fires and an explicit reviewer is
named. It never invokes Claude, BaseTen, or the collection driver.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import collect
import judge_stage


HERE = Path(__file__).resolve().parent
ROOT = collect.ROOT
ANALYSIS_DIR = HERE / "analysis"
LOCK_PATH = HERE / ".analysis.lock"
BOOTSTRAP_ITERATIONS = 100_000
METRICS = (
    "strict_pass",
    "any_error",
    "tool_error",
    "instruction_error",
    "kb_error",
)
ERROR_METRICS = (
    "any_error",
    "tool_error",
    "instruction_error",
    "kb_error",
)
INCLUDED_FIELDS = (
    "pair",
    "slot",
    "arm",
    "run_dir",
    "classification",
    "observed_turns",
    "strict_passes",
    "transcript_sha256",
    "judgment_sha256",
    "summary_sha256",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def percentile(values: list[float], probability: float) -> float:
    if not values:
        fail("cannot take a percentile of an empty sample")
    values.sort()
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def quantile(values: list[float], probability: float) -> float | None:
    return percentile(list(values), probability) if values else None


def write_tsv_atomic(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    rendered = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            fail(f"refusing to replace changed frozen analysis artifact: {path}")
        return rendered
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    return rendered


def stage_paths(stage: str) -> dict[str, Path]:
    return {
        "json": ANALYSIS_DIR / f"aggregates-{stage}.json",
        "markdown": ANALYSIS_DIR / f"REPORT-{stage}.md",
        "included": ANALYSIS_DIR / f"included-runs-{stage}.tsv",
        "decision": ANALYSIS_DIR / "promotion-decision-initial.json",
    }


def validate_judge_complete(
    stage: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_path = judge_stage.complete_path(stage)
    complete = read_json(complete_path)
    expected_runs = judge_stage.STAGE_SLOTS[stage]
    if complete.get("campaign") != collect.CAMPAIGN_ID:
        fail("judge completion marker campaign mismatch")
    if complete.get("stage") != stage:
        fail("judge completion marker stage mismatch")
    if complete.get("canonical_runs") != expected_runs:
        fail("judge completion marker run count mismatch")
    if complete.get("canonical_pairs") != expected_runs // 2:
        fail("judge completion marker pair count mismatch")
    if complete.get("judge_model") != judge_stage.EXPECTED_JUDGE_MODEL:
        fail("judge completion marker model mismatch")
    if complete.get("judge_version") != judge_stage.EXPECTED_JUDGE_VERSION:
        fail("judge completion marker version mismatch")
    inputs_path = judge_stage.input_path(stage)
    if complete.get("canonical_inputs_sha256") != judge_stage.sha256(inputs_path):
        fail("judge input manifest changed after completion")
    if complete.get("judge_source_sha256") != judge_stage.sha256(
        judge_stage.SOURCE_HASH_PATH
    ):
        fail("judge source manifest changed after completion")
    transcript_hashes = complete.get("transcript_sha256")
    expected_hashes = {
        entry["slot"]: entry["transcript_sha256"] for entry in entries
    }
    if transcript_hashes != expected_hashes:
        fail("judge completion transcript hashes do not match the stage")
    if judge_stage.SOURCE_HASH_PATH.read_text(
        encoding="utf-8"
    ) != judge_stage.source_hash_text():
        fail("judge or campaign source changed after judging completed")
    return complete


def validate_judge_inputs(
    stage: str,
    entries: list[dict[str, Any]],
) -> None:
    frozen = judge_stage.read_tsv(judge_stage.input_path(stage))
    expected = [
        {
            "slot": entry["slot"],
            "pair": str(entry["pair"]),
            "arm": entry["arm"],
            "run_dir": entry["run_dir_text"],
            "transcript_sha256": entry["transcript_sha256"],
            "scheduled_turns": str(len(entry["turns"])),
        }
        for entry in entries
    ]
    if frozen != expected:
        fail(f"frozen judge inputs do not match the {stage} stage")


def build_conversation(entry: dict[str, Any]) -> dict[str, Any]:
    valid, error = judge_stage.validate_outputs(entry)
    if not valid:
        fail(f"invalid judgment for {entry['slot']}: {error}")
    transcript_rows = judge_stage.read_jsonl(entry["transcript"])
    scheduled = [
        row for row in transcript_rows if row.get("recovery_turn") is not True
    ]
    judged_rows = judge_stage.read_jsonl(
        entry["run_dir"] / "claude_judged.jsonl"
    )
    judged_by_turn = {int(row["turn"]): row for row in judged_rows}
    metrics: dict[str, list[int]] = {metric: [] for metric in METRICS}
    turn_taking_errors: list[int] = []
    for turn in range(collect.N_TURNS):
        judged = judged_by_turn.get(turn)
        if judged is None:
            tool_ok = instruction_ok = kb_ok = turn_taking_ok = False
        else:
            scores = judged["scores"]
            tool_ok = bool(scores["tool_use_correct"])
            instruction_ok = bool(scores["instruction_following"])
            kb_ok = bool(scores["kb_grounding"])
            turn_taking_ok = bool(scores["turn_taking"])
        strict = bool(tool_ok and instruction_ok and kb_ok)
        metrics["strict_pass"].append(int(strict))
        metrics["any_error"].append(int(not strict))
        metrics["tool_error"].append(int(not tool_ok))
        metrics["instruction_error"].append(int(not instruction_ok))
        metrics["kb_error"].append(int(not kb_ok))
        turn_taking_errors.append(int(not turn_taking_ok))

    latencies = [
        float(row["ttfb_ms"])
        for row in scheduled
        if isinstance(row.get("ttfb_ms"), (int, float))
        and not isinstance(row.get("ttfb_ms"), bool)
        and math.isfinite(float(row["ttfb_ms"]))
        and float(row["ttfb_ms"]) >= 0
    ]
    run_dir = entry["run_dir"]
    return {
        "pair": entry["pair"],
        "slot": entry["slot"],
        "arm": entry["arm"],
        "run_dir": entry["run_dir_text"],
        "classification": entry["classification"],
        "complete": int(entry["classification"] == "strict_complete"),
        "observed_turns": len(entry["turns"]),
        "missing_turns": collect.N_TURNS - len(entry["turns"]),
        "strict_passes": sum(metrics["strict_pass"]),
        "metrics": metrics,
        "turn_taking_errors": turn_taking_errors,
        "latencies": latencies,
        "transcript_sha256": entry["transcript_sha256"],
        "judgment_sha256": judge_stage.sha256(run_dir / "claude_judged.jsonl"),
        "summary_sha256": judge_stage.sha256(run_dir / "claude_summary.json"),
    }


def load_conversations(
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entries = judge_stage.load_stage_entries(stage)
    validate_judge_inputs(stage, entries)
    complete = validate_judge_complete(stage, entries)
    conversations = [build_conversation(entry) for entry in entries]
    controls = [row for row in conversations if row["arm"] == "nofiller"]
    dots = [row for row in conversations if row["arm"] == "dots96"]
    expected_pairs = judge_stage.STAGE_SLOTS[stage] // 2
    if len(controls) != expected_pairs or len(dots) != expected_pairs:
        fail("analysis stage is not arm-balanced")
    control_pairs = {row["pair"] for row in controls}
    dot_pairs = {row["pair"] for row in dots}
    if control_pairs != dot_pairs or len(control_pairs) != expected_pairs:
        fail("analysis stage does not contain exact matched temporal pairs")
    return controls, dots, complete


def conversation_rate(conversation: dict[str, Any], metric: str) -> float:
    if metric == "completion":
        return float(conversation["complete"])
    return sum(conversation["metrics"][metric]) / collect.N_TURNS


def whole_conversation_ci(
    conversations: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rates = [conversation_rate(row, metric) for row in conversations]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        estimates.append(
            sum(rates[rng.randrange(len(rates))] for _ in rates)
            / len(rates)
            * 100
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_bootstrap_effect(
    controls: list[dict[str, Any]],
    dots: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    by_pair_control = {row["pair"]: row for row in controls}
    by_pair_dots = {row["pair"]: row for row in dots}
    if set(by_pair_control) != set(by_pair_dots):
        fail("cannot bootstrap unmatched pairs")
    pair_differences = [
        conversation_rate(by_pair_dots[pair], metric)
        - conversation_rate(by_pair_control[pair], metric)
        for pair in sorted(by_pair_control)
    ]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        estimates.append(
            sum(
                pair_differences[rng.randrange(len(pair_differences))]
                for _ in pair_differences
            )
            / len(pair_differences)
            * 100
        )
    return {
        "dots_minus_control_points": (
            sum(pair_differences) / len(pair_differences) * 100
        ),
        "paired_bootstrap_95_low": percentile(estimates, 0.025),
        "paired_bootstrap_95_high": percentile(estimates, 0.975),
    }


def summarize_arm(
    conversations: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    n = len(conversations)
    denominator = n * collect.N_TURNS
    counts = {
        metric: sum(sum(row["metrics"][metric]) for row in conversations)
        for metric in METRICS
    }
    metric_results: dict[str, Any] = {}
    for index, metric in enumerate(METRICS):
        low, high = whole_conversation_ci(
            conversations,
            metric,
            iterations=iterations,
            seed=seed + index,
        )
        metric_results[metric] = {
            "count": counts[metric],
            "total": denominator,
            "rate_percent": counts[metric] / denominator * 100,
            "whole_conversation_bootstrap_95": [low, high],
        }

    completion_count = sum(row["complete"] for row in conversations)
    completion_low, completion_high = whole_conversation_ci(
        conversations,
        "completion",
        iterations=iterations,
        seed=seed + 20,
    )
    latencies = [
        latency for row in conversations for latency in row["latencies"]
    ]
    per_turn: list[dict[str, Any]] = []
    for turn in range(collect.N_TURNS):
        result: dict[str, Any] = {"turn": turn}
        for metric in ERROR_METRICS:
            count = sum(row["metrics"][metric][turn] for row in conversations)
            result[f"{metric}_count"] = count
            result[f"{metric}_percent"] = count / n * 100
        per_turn.append(result)

    concentration: dict[str, Any] = {}
    for metric in ERROR_METRICS:
        ranked = sorted(
            per_turn,
            key=lambda row: (-row[f"{metric}_count"], row["turn"]),
        )
        total_errors = counts[metric]
        top_rows = [row for row in ranked if row[f"{metric}_count"] > 0]
        top_three = sum(row[f"{metric}_count"] for row in top_rows[:3])
        top_five = sum(row[f"{metric}_count"] for row in top_rows[:5])
        concentration[metric] = {
            "total_errors": total_errors,
            "turns_with_at_least_one_error": len(top_rows),
            "top_3_turn_error_share_percent": (
                top_three / total_errors * 100 if total_errors else 0.0
            ),
            "top_5_turn_error_share_percent": (
                top_five / total_errors * 100 if total_errors else 0.0
            ),
            "ranked_turns": [
                {
                    "turn": row["turn"],
                    "count": row[f"{metric}_count"],
                    "percent": row[f"{metric}_percent"],
                }
                for row in top_rows
            ],
        }

    return {
        "conversations": n,
        "fixed_turn_denominator": denominator,
        "observed_turns": sum(row["observed_turns"] for row in conversations),
        "missing_turns_counted_as_failures": sum(
            row["missing_turns"] for row in conversations
        ),
        "metrics": metric_results,
        "strict_completion": {
            "count": completion_count,
            "total": n,
            "rate_percent": completion_count / n * 100,
            "whole_conversation_bootstrap_95": [
                completion_low,
                completion_high,
            ],
        },
        "classifications": dict(
            sorted(Counter(row["classification"] for row in conversations).items())
        ),
        "ttfat_ms_observed_responses_only": {
            "count": len(latencies),
            "p50": median(latencies) if latencies else None,
            "p95": quantile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "turn_taking_error_count_fixed_denominator": sum(
            sum(row["turn_taking_errors"]) for row in conversations
        ),
        "per_turn": per_turn,
        "error_concentration": concentration,
    }


def aligned_recurring_turns(
    control_summary: dict[str, Any],
    dots_summary: dict[str, Any],
    strict_delta: float,
) -> list[dict[str, Any]]:
    if strict_delta == 0:
        return []
    dots_worse = strict_delta < 0
    result: list[dict[str, Any]] = []
    for control_row, dots_row in zip(
        control_summary["per_turn"], dots_summary["per_turn"]
    ):
        control_count = control_row["any_error_count"]
        dots_count = dots_row["any_error_count"]
        worse_count = dots_count if dots_worse else control_count
        aligned = dots_count > control_count if dots_worse else control_count > dots_count
        if aligned and worse_count >= 3:
            result.append(
                {
                    "turn": control_row["turn"],
                    "direction": (
                        "dots_more_errors" if dots_worse else "dots_fewer_errors"
                    ),
                    "control_error_count": control_count,
                    "dots_error_count": dots_count,
                    "recurrences_in_worse_arm": worse_count,
                }
            )
    return result


def evaluate_promotion(
    stage: str,
    effects: dict[str, dict[str, float]],
    control_summary: dict[str, Any],
    dots_summary: dict[str, Any],
) -> dict[str, Any]:
    if stage == "full":
        return {
            "evaluated": False,
            "terminal_stage": True,
            "triggered_rules": [],
            "promote_to_n30": False,
            "note": "The full 30-pair stage is terminal; no promotion rule applies.",
        }
    strict = effects["strict_pass"]
    delta = strict["dots_minus_control_points"]
    recurring = aligned_recurring_turns(
        control_summary, dots_summary, delta
    )
    rules = {
        "ci_excludes_zero": (
            strict["paired_bootstrap_95_low"] > 0
            or strict["paired_bootstrap_95_high"] < 0
        ),
        "absolute_effect_ge_3_and_aligned_same_turn_recurs_ge_3": (
            abs(delta) >= 3.0 and bool(recurring)
        ),
        "completion_differs": (
            control_summary["strict_completion"]["count"]
            != dots_summary["strict_completion"]["count"]
        ),
    }
    triggered = [name for name, fired in rules.items() if fired]
    return {
        "evaluated": True,
        "terminal_stage": False,
        "rules": rules,
        "triggered_rules": triggered,
        "promote_to_n30": bool(triggered),
        "aligned_recurring_turns": recurring,
        "collection_launched": False,
        "note": (
            "This analysis evaluates the prespecified sample-size rule only; "
            "it never launches collection."
        ),
    }


def analyze(
    stage: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls, dots, complete = load_conversations(stage)
    control_summary = summarize_arm(
        controls,
        iterations=iterations,
        seed=20260731 + (10 if stage == "initial" else 30) * 100,
    )
    dots_summary = summarize_arm(
        dots,
        iterations=iterations,
        seed=20260731 + (10 if stage == "initial" else 30) * 100 + 50,
    )
    effects = {
        metric: paired_bootstrap_effect(
            controls,
            dots,
            metric,
            iterations=iterations,
            seed=20260731 + (10 if stage == "initial" else 30) * 1000 + index,
        )
        for index, metric in enumerate((*METRICS, "completion"))
    }
    turn_comparison: list[dict[str, Any]] = []
    for control_row, dots_row in zip(
        control_summary["per_turn"], dots_summary["per_turn"]
    ):
        row: dict[str, Any] = {"turn": control_row["turn"]}
        for metric in ERROR_METRICS:
            row[f"control_{metric}_count"] = control_row[f"{metric}_count"]
            row[f"dots_{metric}_count"] = dots_row[f"{metric}_count"]
            row[f"dots_minus_control_{metric}_count"] = (
                dots_row[f"{metric}_count"] - control_row[f"{metric}_count"]
            )
        turn_comparison.append(row)
    promotion = evaluate_promotion(
        stage, effects, control_summary, dots_summary
    )

    conversations = sorted(controls + dots, key=lambda row: row["slot"])
    included = [
        {field: row[field] for field in INCLUDED_FIELDS}
        for row in conversations
    ]
    payload = {
        "schema_version": 1,
        "campaign_id": collect.CAMPAIGN_ID,
        "stage": stage,
        "generated_at": utc_now(),
        "model": collect.MODEL,
        "provider": "BaseTen",
        "configuration": {
            "control": "fresh contemporaneous nofiller",
            "treatment": "+96 space-separated suffix dots, request-only",
            "thinking_enabled": False,
            "fixed_turns_per_conversation": collect.N_TURNS,
            "temporal_pairing": True,
        },
        "judge": {
            "model": judge_stage.EXPECTED_JUDGE_MODEL,
            "version": judge_stage.EXPECTED_JUDGE_VERSION,
        },
        "method": {
            "fixed_denominator": True,
            "missing_future_turns_fail_all_displayed_accuracy_criteria": True,
            "arm_interval_unit": "whole conversation",
            "effect_interval_unit": "frozen temporal pair",
            "effect_bootstrap_design": "paired bootstrap",
            "bootstrap_iterations": iterations,
        },
        "input_hashes": {
            "configuration": judge_stage.sha256(collect.CONFIG_PATH),
            "frozen_order": judge_stage.sha256(collect.SCHEDULE_PATH),
            "canonical": judge_stage.sha256(collect.CANONICAL_PATH),
            "judge_inputs": judge_stage.sha256(judge_stage.input_path(stage)),
            "judge_complete": judge_stage.sha256(judge_stage.complete_path(stage)),
            "judge_source": judge_stage.sha256(judge_stage.SOURCE_HASH_PATH),
            "analysis_source": judge_stage.sha256(Path(__file__).resolve()),
        },
        "judging_complete": complete,
        "arms": {"nofiller": control_summary, "dots96": dots_summary},
        "effects": effects,
        "turn_error_comparison": turn_comparison,
        "promotion_evaluation": promotion,
    }
    return payload, included


def format_latency(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f}"


def render_markdown(result: dict[str, Any]) -> str:
    control = result["arms"]["nofiller"]
    dots = result["arms"]["dots96"]
    effect = result["effects"]["strict_pass"]
    promotion = result["promotion_evaluation"]
    lines = [
        f"# Gemma 4 26B A4B +96 dots — {result['stage']} stage",
        "",
        "Every conversation contributes 30 scheduled turns. Missing future turns are errors. "
        "Arm intervals resample whole conversations; effect intervals resample the frozen temporal pairs.",
        "",
        "| arm | conversations | strict pass | whole-conversation 95% CI | strict completion | observed / fixed turns | TTFAT P50 / P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm in (("no filler", control), ("+96 dots", dots)):
        strict = arm["metrics"]["strict_pass"]
        ci = strict["whole_conversation_bootstrap_95"]
        completion = arm["strict_completion"]
        latency = arm["ttfat_ms_observed_responses_only"]
        lines.append(
            f"| {label} | {arm['conversations']} | {strict['rate_percent']:.1f}% | "
            f"{ci[0]:.1f} to {ci[1]:.1f}% | {completion['count']}/{completion['total']} "
            f"({completion['rate_percent']:.1f}%) | {arm['observed_turns']} / "
            f"{arm['fixed_turn_denominator']} | {format_latency(latency['p50'])} / "
            f"{format_latency(latency['p95'])} ms |"
        )
    lines.extend(
        [
            "",
            f"Dots minus control strict-pass effect: **{effect['dots_minus_control_points']:+.1f} points** "
            f"(paired bootstrap 95% CI {effect['paired_bootstrap_95_low']:+.1f} "
            f"to {effect['paired_bootstrap_95_high']:+.1f}).",
            "",
            "## Error concentration",
            "",
            "| arm | total strict errors | turns affected | top-3 turn share | top-5 turn share |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, arm in (("no filler", control), ("+96 dots", dots)):
        concentration = arm["error_concentration"]["any_error"]
        lines.append(
            f"| {label} | {concentration['total_errors']} | "
            f"{concentration['turns_with_at_least_one_error']} | "
            f"{concentration['top_3_turn_error_share_percent']:.1f}% | "
            f"{concentration['top_5_turn_error_share_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "### Highest-error turns",
            "",
            "| arm | turn | errors | conversations |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, arm in (("no filler", control), ("+96 dots", dots)):
        ranked = arm["error_concentration"]["any_error"]["ranked_turns"][:10]
        if not ranked:
            lines.append(f"| {label} | — | 0 | 0.0% |")
        for row in ranked:
            lines.append(
                f"| {label} | {row['turn']} | {row['count']} | "
                f"{row['percent']:.1f}% |"
            )
    lines.extend(["", "## Sample-size decision", ""])
    if promotion["evaluated"]:
        recommendation = (
            "promote to 30 pairs"
            if promotion["promote_to_n30"]
            else "stop at 10 pairs"
        )
        lines.append(f"Recommendation: **{recommendation}**.")
        lines.append("")
        for name, fired in promotion["rules"].items():
            lines.append(f"- `{name}`: {'fired' if fired else 'did not fire'}")
        lines.extend(
            [
                "",
                "The analyzer does not launch collection. A promotion file is written only when a trigger fires and an explicit reviewer is supplied.",
            ]
        )
    else:
        lines.append("The full 30-pair stage is terminal; no promotion rule applies.")
    lines.append("")
    return "\n".join(lines)


def stable_result(result: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(result))
    stable.pop("generated_at", None)
    complete = stable.get("judging_complete")
    if isinstance(complete, dict):
        complete.pop("completed_at", None)
    return stable


def write_outputs(
    stage: str,
    result: dict[str, Any],
    included: list[dict[str, Any]],
) -> dict[str, Path]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(stage)
    write_tsv_atomic(paths["included"], INCLUDED_FIELDS, included)

    if paths["json"].exists():
        existing = read_json(paths["json"])
        if stable_result(existing) != stable_result(result):
            fail(f"refusing to replace changed frozen analysis: {paths['json']}")
    else:
        temporary = paths["json"].with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(paths["json"])

    rendered = render_markdown(result)
    if paths["markdown"].exists():
        if paths["markdown"].read_text(encoding="utf-8") != rendered:
            fail(
                f"refusing to replace changed frozen report: {paths['markdown']}"
            )
    else:
        temporary = paths["markdown"].with_suffix(".md.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(paths["markdown"])
    return paths


def write_reviewed_promotion(
    result: dict[str, Any],
    paths: dict[str, Path],
    reviewer: str,
) -> Path | None:
    promotion = result["promotion_evaluation"]
    if result["stage"] != "initial":
        fail("promotion decisions are valid only for the initial stage")
    if not promotion["evaluated"]:
        fail("the prespecified promotion rule was not evaluated")
    if not promotion["promote_to_n30"]:
        if paths["decision"].exists():
            fail("a stale promotion decision exists although no trigger fired")
        return None
    reviewer = reviewer.strip()
    if len(reviewer) < 2 or reviewer.upper().startswith(("TODO", "REPLACE")):
        fail("a real --reviewed-by value is required for promotion")

    payload = {
        "campaign_id": collect.CAMPAIGN_ID,
        "decision_after_n_per_arm": 10,
        "promote_to_n30": True,
        "triggered_rules": promotion["triggered_rules"],
        "aggregates_sha256": judge_stage.sha256(paths["json"]),
        "included_runs_sha256": judge_stage.sha256(paths["included"]),
        "decided_at": utc_now(),
        "reviewed_by": reviewer,
        "aggregates_path": str(paths["json"].relative_to(ROOT)),
        "included_runs_path": str(paths["included"].relative_to(ROOT)),
        "notes": (
            "Reviewed result of the prespecified initial-stage rule. This file "
            "authorizes only the frozen n=30 continuation; it does not launch it."
        ),
    }
    path = paths["decision"]
    if path.exists():
        existing = read_json(path)
        comparison = dict(payload)
        comparison["decided_at"] = existing.get("decided_at")
        if existing != comparison:
            fail("refusing to replace a changed reviewed promotion decision")
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(judge_stage.STAGE_SLOTS), required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="freeze analysis artifacts; default is read-only",
    )
    parser.add_argument(
        "--reviewed-by",
        help="required only when an initial-stage promotion trigger fires",
    )
    args = parser.parse_args()
    if args.stage == "full" and args.reviewed_by:
        fail("--reviewed-by is not accepted at the terminal full stage")

    result, included = analyze(args.stage)
    effect = result["effects"]["strict_pass"]
    promotion = result["promotion_evaluation"]
    print(
        f"Analysis preflight: stage={args.stage}, pairs={len(included) // 2}, "
        f"strict_delta={effect['dots_minus_control_points']:+.2f}, "
        f"paired_ci95=[{effect['paired_bootstrap_95_low']:+.2f}, "
        f"{effect['paired_bootstrap_95_high']:+.2f}], "
        f"promote={promotion['promote_to_n30']}"
    )
    if not args.execute:
        print(
            "Read-only analysis only. No file, Claude, BaseTen, or collection "
            "request was made."
        )
        return 0
    if (
        args.stage == "initial"
        and promotion["promote_to_n30"]
        and not args.reviewed_by
    ):
        fail("a fired promotion rule requires --reviewed-by before any write")

    with LOCK_PATH.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another paired-dots analysis owns .analysis.lock")
        result, included = analyze(args.stage)
        if (
            args.stage == "initial"
            and result["promotion_evaluation"]["promote_to_n30"]
            and not args.reviewed_by
        ):
            fail("a fired promotion rule requires --reviewed-by before any write")
        paths = write_outputs(args.stage, result, included)
        decision: Path | None = None
        if args.stage == "initial" and args.reviewed_by:
            decision = write_reviewed_promotion(
                result, paths, args.reviewed_by
            )

    print(
        f"Wrote {paths['json'].relative_to(ROOT)}, "
        f"{paths['included'].relative_to(ROOT)}, and "
        f"{paths['markdown'].relative_to(ROOT)}."
    )
    if decision is not None:
        print(
            f"Reviewed promotion decision: {decision.relative_to(ROOT)}. "
            "Collection was not launched."
        )
    else:
        print("No promotion decision was written; collection was not launched.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
