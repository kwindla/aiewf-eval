#!/usr/bin/env python3
"""Aggregate the exact-boundary Async Gemini Live reliability cohort."""

from __future__ import annotations

import csv
import argparse
import json
import math
import random
import re
import statistics
import wave
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260805
RAW_EVENT = re.compile(r"\[GEMINI_RAW_EVENT\] (\{.*\})$")


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    margin /= denominator
    return [100 * (center - margin), 100 * (center + margin)]


def bootstrap_ci(rates: list[float]) -> list[float] | None:
    if not rates:
        return None
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAPS):
        means.append(100 * statistics.mean(rng.choice(rates) for _ in rates))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def load_raw_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        match = RAW_EVENT.search(line)
        if match:
            events.append(json.loads(match.group(1)))
    return events


def load_judgment(run_dir: Path) -> dict | None:
    summary_path = run_dir / "claude_summary.json"
    judged_path = run_dir / "claude_judged.jsonl"
    if not summary_path.is_file() or not judged_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("turns_scored") != N_TURNS:
        return None
    rows = [json.loads(line) for line in judged_path.read_text().splitlines() if line]
    by_turn = {row.get("turn"): row for row in rows if isinstance(row.get("turn"), int)}
    if set(by_turn) != set(range(N_TURNS)):
        return None
    dimensions = ("tool_use_correct", "instruction_following", "kb_grounding")
    passes = Counter()
    strict_failures = Counter()
    strict_count = 0
    for turn in range(N_TURNS):
        scores = by_turn[turn]["scores"]
        for dimension in (*dimensions, "turn_taking"):
            passes[dimension] += scores.get(dimension) is True
        strict = all(scores.get(dimension) is True for dimension in dimensions)
        strict_count += strict
        if not strict:
            strict_failures[turn] += 1
    return {
        "strict_passes": strict_count,
        "dimension_passes": dict(passes),
        "strict_failure_turns": dict(strict_failures),
        "judge_model": summary.get("judge_model"),
        "judge_version": summary.get("judge_version"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a Async Gemini Live exact-boundary campaign."
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=HERE,
        help="Campaign directory containing attempts.tsv (default: script directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()
    attempts_path = campaign_dir / "attempts.tsv"
    with attempts_path.open(newline="") as handle:
        attempts = list(csv.DictReader(handle, delimiter="\t"))

    classifications = Counter(row["classification"] for row in attempts)
    failure_turns = Counter(
        int(row["failure_turn"])
        for row in attempts
        if row.get("failure_turn", "").isdigit()
    )
    runs = []
    all_ttfb = []
    all_latency = []
    per_turn_ttfb: dict[int, list[float]] = {turn: [] for turn in range(N_TURNS)}
    per_turn_latency: dict[int, list[float]] = {turn: [] for turn in range(N_TURNS)}
    tool_calls = Counter()
    complete_tool_calls = Counter()
    status_counts = Counter()
    thought_tokens = []
    strict_rates = []
    judgment_passes = Counter()
    judgment_failure_turns = Counter()
    judge_models = set()
    judge_versions = set()
    judged_runs = 0

    for row in attempts:
        run_dir = ROOT / row["run_dir"]
        transcript = [
            json.loads(line)
            for line in (run_dir / "transcript.jsonl").read_text().splitlines()
            if line
        ]
        events = load_raw_events(run_dir / "run.log")
        for event in events:
            status = event.get("interaction_status")
            if status:
                status_counts[status] += 1
            usage = event.get("usage") or {}
            if isinstance(usage.get("thoughtsTokenCount"), int):
                thought_tokens.append(usage["thoughtsTokenCount"])

        for turn in transcript:
            ttfb = turn.get("ttfb_ms")
            latency = turn.get("latency_ms")
            if isinstance(ttfb, (int, float)):
                all_ttfb.append(float(ttfb))
                per_turn_ttfb[turn["turn"]].append(float(ttfb))
            if isinstance(latency, (int, float)):
                all_latency.append(float(latency))
                per_turn_latency[turn["turn"]].append(float(latency))
            for call in turn.get("tool_calls") or []:
                tool_name = call.get("name") or "unknown"
                tool_calls[tool_name] += 1
                if row["classification"] == "complete":
                    complete_tool_calls[tool_name] += 1

        judgment = load_judgment(run_dir)
        if judgment:
            judged_runs += 1
            strict_rates.append(judgment["strict_passes"] / N_TURNS)
            judgment_passes.update(judgment["dimension_passes"])
            judgment_failure_turns.update(
                {int(turn): count for turn, count in judgment["strict_failure_turns"].items()}
            )
            if judgment["judge_model"]:
                judge_models.add(judgment["judge_model"])
            if judgment["judge_version"]:
                judge_versions.add(judgment["judge_version"])

        runs.append(
            {
                "attempt": int(row["attempt"]),
                "run_dir": row["run_dir"],
                "classification": row["classification"],
                "turns": len(transcript),
                "failure_turn": int(row["failure_turn"]) if row["failure_turn"] else None,
                "raw_events": int(row["raw_events"]),
                "in_progress": int(row["in_progress"]),
                "terminal": int(row["terminal"]),
                "activity_starts": int(row["activity_starts"]),
                "activity_ends": int(row["activity_ends"]),
                "audio_bytes": int(row["audio_bytes"]),
                "audio_duration_seconds": wav_duration_seconds(run_dir / "conversation.wav"),
                "ttfb_ms": distribution(
                    [float(turn["ttfb_ms"]) for turn in transcript if turn.get("ttfb_ms") is not None]
                ),
                "completion_latency_ms": distribution(
                    [float(turn["latency_ms"]) for turn in transcript if turn.get("latency_ms") is not None]
                ),
                "judged": judgment is not None,
            }
        )

    complete = classifications["complete"]
    total = len(attempts)
    strict_passes = round(sum(strict_rates) * N_TURNS)
    aggregate = {
        "attempts": total,
        "classifications": dict(classifications),
        "completion_rate_pct": 100 * complete / total if total else None,
        "completion_wilson_ci95_pct": wilson(complete, total),
        "failure_turns": dict(sorted(failure_turns.items())),
        "boundary_validation": {
            "balanced_runs": sum(run["activity_starts"] == run["activity_ends"] for run in runs),
            "runs": total,
            "activity_starts": sum(run["activity_starts"] for run in runs),
            "activity_ends": sum(run["activity_ends"] for run in runs),
            "model_audio_during_input": sum(
                int(row.get("model_audio_during_input") or 0) for row in attempts
            ),
            "terminal_during_input": sum(
                int(row.get("terminal_during_input") or 0) for row in attempts
            ),
            "interruption_during_input": sum(
                int(row.get("interruption_during_input") or 0) for row in attempts
            ),
            "nonempty_transcript_missing_ttfb": sum(
                int(row.get("missing_ttfb") or 0) for row in attempts
            ),
        },
        "provider_status_counts": dict(status_counts),
        "runs_with_in_progress": sum(run["in_progress"] > 0 for run in runs),
        "thought_tokens_per_terminal_event": distribution([float(value) for value in thought_tokens]),
        "ttfb_ms": distribution(all_ttfb),
        "completion_latency_ms": distribution(all_latency),
        "per_turn_ttfb_ms": {
            str(turn): distribution(values) for turn, values in per_turn_ttfb.items()
        },
        "per_turn_completion_latency_ms": {
            str(turn): distribution(values) for turn, values in per_turn_latency.items()
        },
        "tool_calls": dict(tool_calls),
        "complete_run_tool_calls": dict(complete_tool_calls),
        "complete_run_expected_tool_calls": {
            "submit_session_suggestion": 2 * complete,
            "submit_dietary_request": complete,
            "request_tech_support": complete,
            "vote_for_session": complete,
            "end_session": complete,
        },
        "judging": {
            "judged_runs": judged_runs,
            "turns": judged_runs * N_TURNS,
            "strict_passes": strict_passes,
            "strict_pass_rate_pct": (
                100 * strict_passes / (judged_runs * N_TURNS) if judged_runs else None
            ),
            "strict_pass_bootstrap_ci95_pct": bootstrap_ci(strict_rates),
            "dimension_passes": dict(judgment_passes),
            "strict_failure_turns": dict(judgment_failure_turns.most_common()),
            "judge_models": sorted(judge_models),
            "judge_versions": sorted(judge_versions),
        },
        "runs": runs,
    }
    (campaign_dir / "cohort-aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )

    completion_ci = aggregate["completion_wilson_ci95_pct"]
    ttfb = aggregate["ttfb_ms"]
    latency = aggregate["completion_latency_ms"]
    lines = [
        "# Async Gemini Live exact-boundary cohort",
        "",
        "Date: 2026-08-05",
        "",
        "## Reliability",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Attempts | {total} |",
        f"| Complete 30-turn conversations | {complete}/{total} |",
        f"| Completion rate (Wilson 95% CI) | {aggregate['completion_rate_pct']:.1f}% ({completion_ci[0]:.1f}–{completion_ci[1]:.1f}%) |",
    ]
    for classification, count in sorted(classifications.items()):
        lines.append(f"| {classification.replace('_', ' ').title()} | {count} |")
    if failure_turns:
        lines += [
            "",
            "Failed zero-based turns: "
            + ", ".join(
                f"{turn} ({count})" for turn, count in sorted(failure_turns.items())
            )
            + ".",
        ]
    lines += [
        "",
        "## Timing",
        "",
        "| Metric | P50 | P90 | P95 | Max | N |",
        "|---|---:|---:|---:|---:|---:|",
        f"| First audio | {ttfb['p50']:.0f}ms | {ttfb['p90']:.0f}ms | {ttfb['p95']:.0f}ms | {ttfb['max']:.0f}ms | {ttfb['n']} |",
        f"| Response completion | {latency['p50']:.0f}ms | {latency['p90']:.0f}ms | {latency['p95']:.0f}ms | {latency['max']:.0f}ms | {latency['n']} |",
        "",
        "## Event-flow checks",
        "",
        f"- Balanced explicit activity boundaries: {aggregate['boundary_validation']['balanced_runs']}/{total} runs.",
        f"- Activity starts/ends: {aggregate['boundary_validation']['activity_starts']}/{aggregate['boundary_validation']['activity_ends']}.",
        f"- Model audio events during explicit input: {aggregate['boundary_validation']['model_audio_during_input']}.",
        f"- Harmless terminal/interruption control events during input: {aggregate['boundary_validation']['terminal_during_input']}/{aggregate['boundary_validation']['interruption_during_input']}.",
        f"- Non-empty transcript turns missing first-audio timing: {aggregate['boundary_validation']['nonempty_transcript_missing_ttfb']}.",
        f"- Provider statuses: `{json.dumps(dict(status_counts), sort_keys=True)}`.",
        f"- Runs containing at least one `IN_PROGRESS`: {aggregate['runs_with_in_progress']}/{total}.",
        f"- Tool calls: `{json.dumps(dict(tool_calls), sort_keys=True)}`.",
        "",
        "### Tool calls in complete conversations",
        "",
        "These are raw call counts; the content judge separately checks timing and arguments.",
        "",
        "| Tool | Observed | Expected |",
        "|---|---:|---:|",
    ]
    for tool_name, expected in aggregate["complete_run_expected_tool_calls"].items():
        lines.append(
            f"| `{tool_name}` | {complete_tool_calls[tool_name]} | {expected} |"
        )
    lines += [
        "",
        "## Judging",
        "",
    ]
    if judged_runs:
        strict_ci = aggregate["judging"]["strict_pass_bootstrap_ci95_pct"]
        lines += [
            f"Judged {judged_runs} complete runs ({judged_runs * N_TURNS} turns).",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Strict pass | {aggregate['judging']['strict_pass_rate_pct']:.1f}% ({strict_ci[0]:.1f}–{strict_ci[1]:.1f}%) |",
            f"| Tool use | {judgment_passes['tool_use_correct']}/{judged_runs * N_TURNS} |",
            f"| Instruction | {judgment_passes['instruction_following']}/{judged_runs * N_TURNS} |",
            f"| KB grounding | {judgment_passes['kb_grounding']}/{judged_runs * N_TURNS} |",
            f"| Legacy offline turn-tag match (diagnostic only) | {judgment_passes['turn_taking']}/{judged_runs * N_TURNS} |",
            "",
            "The legacy offline detector assumes one contiguous bot-audio segment per turn. "
            "Async Gemini Live often speaks in multiple asynchronous phases, so missing tag matches "
            "are not evidence of actual overlap; the raw boundary audit above is authoritative for "
            "whether model audio crossed explicit user input.",
        ]
    else:
        lines.append("No complete run has been judged yet.")
    (campaign_dir / "cohort-summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
