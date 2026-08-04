#!/usr/bin/env python3
"""Analyze one fully judged Inkling Small adaptive dots stage.

The default invocation is read-only. ``--execute`` atomically writes
``analysis/stage-N.json`` and a Markdown companion; it never executes the
extension gate or makes a provider request.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
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
LOCK = HERE / ".analysis.lock"
BOOTSTRAP_ITERATIONS = 100_000
METRICS = ("strict_pass", "any_error", "tool_error", "instruction_error", "kb_error")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise RuntimeError("cannot take percentile of empty values")
    values.sort()
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return percentile(list(values), probability)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def validate_judge_complete(path: Path, *, stage: int | None = None) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("judge_model") != judge_stage.EXPECTED_JUDGE_MODEL:
        raise RuntimeError(f"wrong judge model in {path}")
    if payload.get("judge_version") != judge_stage.EXPECTED_JUDGE_VERSION:
        raise RuntimeError(f"wrong judge version in {path}")
    if stage is None:
        total = payload.get("canonical_runs", payload.get("canonical_conversations"))
        if total != 60:
            raise RuntimeError(f"primary judge completion is not 60 runs: {path}")
    else:
        if payload.get("stage") != stage or payload.get("canonical_dots") != stage:
            raise RuntimeError(f"dots judge completion does not match stage {stage}")
    return payload


def output_is_valid(
    run_dir: Path,
    *,
    turns: list[int],
    transcript_sha256: str,
) -> list[dict[str, Any]]:
    transcript = run_dir / "transcript.jsonl"
    if judge_stage.sha256(transcript) != transcript_sha256:
        raise RuntimeError(f"frozen transcript changed: {run_dir}")
    judged = judge_stage.read_jsonl(run_dir / "claude_judged.jsonl")
    if [row.get("turn") for row in judged] != turns:
        raise RuntimeError(f"judged turns do not match frozen transcript: {run_dir}")
    summary = read_json(run_dir / "claude_summary.json")
    if summary.get("judge_model") != judge_stage.EXPECTED_JUDGE_MODEL:
        raise RuntimeError(f"wrong judge model: {run_dir}")
    if summary.get("judge_version") != judge_stage.EXPECTED_JUDGE_VERSION:
        raise RuntimeError(f"wrong judge version: {run_dir}")
    if summary.get("model_name") != collect.MODEL:
        raise RuntimeError(f"wrong judged model: {run_dir}")
    if summary.get("turns_scored") != len(turns):
        raise RuntimeError(f"wrong judged turn count: {run_dir}")
    for row in judged:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            raise RuntimeError(f"missing scores: {run_dir}")
        for key in (
            "turn_taking", "tool_use_correct", "instruction_following", "kb_grounding",
        ):
            if not isinstance(scores.get(key), bool):
                raise RuntimeError(f"missing boolean score {key}: {run_dir}")
    return judged


def build_conversation(
    *,
    slot: str,
    arm: str,
    run_dir: Path,
    transcript_sha256: str,
    scheduled_turns: int,
    classification: str,
) -> dict[str, Any]:
    transcript_rows = collect.read_transcript(run_dir)
    scheduled = [row for row in transcript_rows if row.get("recovery_turn") is not True]
    turns = [row.get("turn") for row in scheduled]
    if turns != list(range(scheduled_turns)):
        raise RuntimeError(f"frozen scheduled-turn count changed for {slot}")
    judged = output_is_valid(
        run_dir,
        turns=turns,
        transcript_sha256=transcript_sha256,
    )
    judged_by_turn = {int(row["turn"]): row for row in judged}
    metric_values = {metric: [] for metric in METRICS}
    turn_taking_errors: list[int] = []
    for turn in range(collect.N_TURNS):
        row = judged_by_turn.get(turn)
        if row is None:
            tool_ok = instruction_ok = kb_ok = False
            turn_taking_ok = False
        else:
            scores = row["scores"]
            tool_ok = scores["tool_use_correct"]
            instruction_ok = scores["instruction_following"]
            kb_ok = scores["kb_grounding"]
            turn_taking_ok = scores["turn_taking"]
        strict = bool(tool_ok and instruction_ok and kb_ok)
        metric_values["strict_pass"].append(int(strict))
        metric_values["any_error"].append(int(not strict))
        metric_values["tool_error"].append(int(not tool_ok))
        metric_values["instruction_error"].append(int(not instruction_ok))
        metric_values["kb_error"].append(int(not kb_ok))
        turn_taking_errors.append(int(not turn_taking_ok))

    latencies = [
        float(row["ttfb_ms"])
        for row in scheduled
        if isinstance(row.get("ttfb_ms"), (int, float))
        and not isinstance(row.get("ttfb_ms"), bool)
        and math.isfinite(float(row["ttfb_ms"]))
        and float(row["ttfb_ms"]) >= 0
    ]
    return {
        "slot": slot,
        "arm": arm,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "classification": classification,
        "complete": int(classification == "strict_complete"),
        "observed_turns": scheduled_turns,
        "missing_turns": collect.N_TURNS - scheduled_turns,
        "metrics": metric_values,
        "turn_taking_errors": turn_taking_errors,
        "latencies": latencies,
    }


def load_conversations(stage: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    collect.validate_configuration()
    collect.validate_source_hashes()
    controls_frozen = collect.read_tsv(collect.CONTROL_INPUTS)
    if len(controls_frozen) != 30:
        raise RuntimeError(f"control-inputs.tsv must freeze 30 controls; found {len(controls_frozen)}")
    if collect.validate_or_freeze_control(execute=False) != 30:
        raise RuntimeError("primary controls no longer match their frozen hashes")
    validate_judge_complete(
        collect.PRIMARY / "judging/COMPLETE.json",
        stage=None,
    )

    dots_entries = judge_stage.load_stage_entries(stage)
    dots_complete = judge_stage.complete_path(stage)
    validate_judge_complete(dots_complete, stage=stage)
    judge_inputs = judge_stage.read_tsv(judge_stage.INPUTS_PATH)
    if len(judge_inputs) < stage:
        raise RuntimeError(f"dots judge inputs contain only {len(judge_inputs)} rows")
    if [row["slot"] for row in judge_inputs[:stage]] != [entry["slot"] for entry in dots_entries]:
        raise RuntimeError("dots judge input order does not match the stage cohort")

    primary_canonical = {
        row["slot"]: row for row in collect.read_tsv(collect.PRIMARY / "canonical.tsv")
    }
    controls: list[dict[str, Any]] = []
    for frozen in controls_frozen:
        row = primary_canonical.get(frozen["slot"])
        if row is None or row.get("arm") != "none":
            raise RuntimeError(f"frozen control slot is absent from primary canonical: {frozen['slot']}")
        controls.append(
            build_conversation(
                slot=frozen["slot"],
                arm="none",
                run_dir=collect.resolve_repo_path(frozen["run_dir"]),
                transcript_sha256=frozen["transcript_sha256"],
                scheduled_turns=int(frozen["scheduled_turns"]),
                classification=row["classification"],
            )
        )

    dots_canonical = {row["slot"]: row for row in collect.read_tsv(collect.CANONICAL)}
    dots: list[dict[str, Any]] = []
    for entry, frozen in zip(dots_entries, judge_inputs[:stage]):
        row = dots_canonical[entry["slot"]]
        if frozen["transcript_sha256"] != entry["transcript_sha256"]:
            raise RuntimeError(f"dots transcript hash mismatch for {entry['slot']}")
        dots.append(
            build_conversation(
                slot=entry["slot"],
                arm="dots96",
                run_dir=entry["run_dir"],
                transcript_sha256=entry["transcript_sha256"],
                scheduled_turns=len(entry["turns"]),
                classification=row["classification"],
            )
        )
    return controls, dots


def summarize_arm(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(conversations)
    denominator = n * collect.N_TURNS
    counts = {
        metric: sum(sum(conv["metrics"][metric]) for conv in conversations)
        for metric in METRICS
    }
    latencies = [latency for conv in conversations for latency in conv["latencies"]]
    per_turn: list[dict[str, Any]] = []
    for turn in range(collect.N_TURNS):
        errors = {
            metric: sum(conv["metrics"][metric][turn] for conv in conversations)
            for metric in ("any_error", "tool_error", "instruction_error", "kb_error")
        }
        per_turn.append(
            {
                "turn": turn,
                **{f"{metric}_count": value for metric, value in errors.items()},
                **{f"{metric}_percent": value / n * 100 for metric, value in errors.items()},
            }
        )
    error_turn_counts = [row["any_error_count"] for row in per_turn]
    ranked = sorted(range(collect.N_TURNS), key=lambda turn: (-error_turn_counts[turn], turn))
    return {
        "conversations": n,
        "fixed_turn_denominator": denominator,
        "observed_turns": sum(conv["observed_turns"] for conv in conversations),
        "missing_turns": sum(conv["missing_turns"] for conv in conversations),
        "counts": counts,
        "rates_percent": {metric: count / denominator * 100 for metric, count in counts.items()},
        "strict_completion": {
            "count": sum(conv["complete"] for conv in conversations),
            "total": n,
            "percent": sum(conv["complete"] for conv in conversations) / n * 100,
        },
        "classifications": dict(sorted(Counter(conv["classification"] for conv in conversations).items())),
        "ttfat_ms_observed_responses_only": {
            "n": len(latencies),
            "p50": median(latencies) if latencies else None,
            "p95": quantile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "turn_taking_error_count_fixed_denominator": sum(
            sum(conv["turn_taking_errors"]) for conv in conversations
        ),
        "per_turn": per_turn,
        "top_any_error_turns": [
            {"turn": turn, "count": error_turn_counts[turn], "percent": error_turn_counts[turn] / n * 100}
            for turn in ranked if error_turn_counts[turn] > 0
        ],
    }


def conversation_rates(conversations: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "completion":
        return [float(conv["complete"]) for conv in conversations]
    return [sum(conv["metrics"][metric]) / collect.N_TURNS for conv in conversations]


def bootstrap_effect(
    controls: list[dict[str, Any]],
    dots: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    control = conversation_rates(controls, metric)
    treated = conversation_rates(dots, metric)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        c_mean = sum(control[rng.randrange(len(control))] for _ in control) / len(control)
        d_mean = sum(treated[rng.randrange(len(treated))] for _ in treated) / len(treated)
        samples.append((d_mean - c_mean) * 100)
    estimate = (sum(treated) / len(treated) - sum(control) / len(control)) * 100
    return {
        "dots_minus_control_points": estimate,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def aligned_recurring_turns(
    control_summary: dict[str, Any],
    dots_summary: dict[str, Any],
    strict_delta: float,
) -> list[dict[str, Any]]:
    if strict_delta == 0:
        return []
    result: list[dict[str, Any]] = []
    dots_worse = strict_delta < 0
    for control_row, dots_row in zip(control_summary["per_turn"], dots_summary["per_turn"]):
        control_rate = control_row["any_error_percent"]
        dots_rate = dots_row["any_error_percent"]
        worse_count = (
            dots_row["any_error_count"] if dots_worse else control_row["any_error_count"]
        )
        aligned = dots_rate > control_rate if dots_worse else dots_rate < control_rate
        if aligned and worse_count >= 3:
            result.append(
                {
                    "turn": control_row["turn"],
                    "direction": "dots_more_errors" if dots_worse else "dots_fewer_errors",
                    "control_error_percent": control_rate,
                    "dots_error_percent": dots_rate,
                    "recurrences_in_worse_arm": worse_count,
                }
            )
    return result


def adaptive_decision(
    stage: int,
    effects: dict[str, dict[str, float]],
    control_summary: dict[str, Any],
    dots_summary: dict[str, Any],
) -> dict[str, Any]:
    strict = effects["strict_pass"]
    delta = strict["dots_minus_control_points"]
    completion_differs = (
        control_summary["strict_completion"]["percent"]
        != dots_summary["strict_completion"]["percent"]
    )
    recurring = aligned_recurring_turns(control_summary, dots_summary, delta)
    ci_excludes_zero = strict["ci95_low"] > 0 or strict["ci95_high"] < 0
    if stage == 6:
        triggers = {
            "absolute_pass_delta_at_least_2_points": abs(delta) >= 2.0,
            "strict_completion_rates_differ": completion_differs,
        }
        fired = any(triggers.values())
        recommendation = "extend_to_10" if fired else "stop_at_6"
    elif stage == 10:
        triggers = {
            "bootstrap_ci_excludes_zero": ci_excludes_zero,
            "absolute_delta_at_least_3_with_recurring_aligned_turn": (
                abs(delta) >= 3.0 and bool(recurring)
            ),
            "strict_completion_rates_differ": completion_differs,
        }
        fired = any(triggers.values())
        recommendation = "extend_to_30" if fired else "stop_at_10"
    else:
        triggers = {}
        fired = False
        recommendation = "terminal_at_30"
    return {
        "evaluated_stage": stage,
        "triggers": triggers,
        "trigger_fired": fired,
        "recommendation": recommendation,
        "aligned_recurring_turns": recurring,
        "gate_executed": False,
        "note": "Analysis evaluates the prespecified rule but never writes stage-decisions.tsv.",
    }


def analyze(stage: int) -> dict[str, Any]:
    controls, dots = load_conversations(stage)
    control_summary = summarize_arm(controls)
    dots_summary = summarize_arm(dots)
    effects = {
        metric: bootstrap_effect(
            controls,
            dots,
            metric,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=20260731 + stage * 100 + index,
        )
        for index, metric in enumerate((*METRICS, "completion"))
    }
    turn_comparison = []
    for control_row, dots_row in zip(control_summary["per_turn"], dots_summary["per_turn"]):
        turn_comparison.append(
            {
                "turn": control_row["turn"],
                "control_any_error_count": control_row["any_error_count"],
                "control_any_error_percent": control_row["any_error_percent"],
                "dots_any_error_count": dots_row["any_error_count"],
                "dots_any_error_percent": dots_row["any_error_percent"],
                "dots_minus_control_error_points": (
                    dots_row["any_error_percent"] - control_row["any_error_percent"]
                ),
            }
        )
    decision = adaptive_decision(stage, effects, control_summary, dots_summary)
    return {
        "schema_version": 1,
        "campaign_id": json.loads((HERE / "configuration.json").read_text())["campaign_id"],
        "stage": stage,
        "generated_at": now(),
        "model": collect.MODEL,
        "provider": "BaseTen Model API",
        "configuration": {
            "control": "frozen primary none arm",
            "treatment": "+96 suffix dots",
            "reasoning_effort": "none",
            "fixed_turns_per_conversation": collect.N_TURNS,
        },
        "judge": {
            "model": judge_stage.EXPECTED_JUDGE_MODEL,
            "version": judge_stage.EXPECTED_JUDGE_VERSION,
        },
        "method": {
            "fixed_denominator": True,
            "missing_future_turns_fail_all_displayed_criteria": True,
            "bootstrap_unit": "whole conversation",
            "bootstrap_design": "independent-arm resampling",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "deployment_time_caveat": (
                "The reused primary controls precede dots collection; provider drift is not randomized away."
            ),
        },
        "inputs": {
            "control_inputs": str(collect.CONTROL_INPUTS.relative_to(ROOT)),
            "control_inputs_sha256": judge_stage.sha256(collect.CONTROL_INPUTS),
            "dots_judge_inputs": str(judge_stage.INPUTS_PATH.relative_to(ROOT)),
            "dots_judge_inputs_sha256": judge_stage.sha256(judge_stage.INPUTS_PATH),
            "dots_judge_complete": str(judge_stage.complete_path(stage).relative_to(ROOT)),
            "dots_judge_complete_sha256": judge_stage.sha256(judge_stage.complete_path(stage)),
        },
        "arms": {"control_none": control_summary, "dots96": dots_summary},
        "effects": effects,
        "turn_error_comparison": turn_comparison,
        "adaptive_decision": decision,
    }


def render_markdown(result: dict[str, Any]) -> str:
    control = result["arms"]["control_none"]
    dots = result["arms"]["dots96"]
    effect = result["effects"]["strict_pass"]
    decision = result["adaptive_decision"]
    lines = [
        f"# Inkling Small +96 dots — stage {result['stage']}",
        "",
        "| arm | conversations | strict pass | strict completion | observed / fixed turns | TTFAT P50 / P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in (("control none", control), ("+96 dots", dots)):
        latency = arm["ttfat_ms_observed_responses_only"]
        lines.append(
            f"| {name} | {arm['conversations']} | {arm['rates_percent']['strict_pass']:.1f}% | "
            f"{arm['strict_completion']['count']}/{arm['strict_completion']['total']} "
            f"({arm['strict_completion']['percent']:.1f}%) | {arm['observed_turns']} / "
            f"{arm['fixed_turn_denominator']} | {latency['p50']:.0f} / {latency['p95']:.0f} ms |"
        )
    lines.extend(
        [
            "",
            f"Dots minus control strict-pass effect: **{effect['dots_minus_control_points']:+.1f} points** "
            f"(whole-conversation bootstrap 95% CI "
            f"{effect['ci95_low']:+.1f} to {effect['ci95_high']:+.1f}).",
            "",
            f"Adaptive recommendation: **{decision['recommendation']}**. "
            "This analysis did not execute the stage gate.",
            "",
            "## Dot-arm error concentrations",
            "",
            "| turn | any-error count | rate |",
            "|---:|---:|---:|",
        ]
    )
    for row in dots["top_any_error_turns"][:10]:
        lines.append(f"| {row['turn']} | {row['count']} | {row['percent']:.1f}% |")
    if not dots["top_any_error_turns"]:
        lines.append("| — | 0 | 0.0% |")
    lines.extend(
        [
            "",
            "The denominator is 30 scripted turns per conversation. Missing future turns count as errors. "
            "Controls were collected earlier, so provider/deployment-time drift remains a limitation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(stage: int, result: dict[str, Any]) -> tuple[Path, Path]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ANALYSIS_DIR / f"stage-{stage}.json"
    md_path = ANALYSIS_DIR / f"stage-{stage}.md"
    json_temp = json_path.with_suffix(".json.tmp")
    md_temp = md_path.with_suffix(".md.tmp")
    json_temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_temp.write_text(render_markdown(result), encoding="utf-8")
    json_temp.replace(json_path)
    md_temp.replace(md_path)
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=collect.STAGES, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = analyze(args.stage)
    decision = result["adaptive_decision"]
    effect = result["effects"]["strict_pass"]
    print(
        f"analysis preflight: stage={args.stage} strict_delta="
        f"{effect['dots_minus_control_points']:+.2f} "
        f"ci95=[{effect['ci95_low']:+.2f}, {effect['ci95_high']:+.2f}] "
        f"recommendation={decision['recommendation']}"
    )
    if not args.execute:
        print("Read-only analysis only. No file, gate, Claude, or BaseTen request was made.")
        return 0
    with LOCK.open("a") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another dots analysis owns the lock") from exc
        # Recompute under lock so hashes and judgments cannot be swapped between
        # preflight and the atomic output write.
        result = analyze(args.stage)
        paths = write_outputs(args.stage, result)
    print(f"wrote {paths[0].relative_to(ROOT)} and {paths[1].relative_to(ROOT)}")
    print("The extension gate was not executed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
