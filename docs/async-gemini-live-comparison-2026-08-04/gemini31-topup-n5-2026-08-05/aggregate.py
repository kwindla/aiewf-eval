#!/usr/bin/env python3
"""Combine the frozen Gemini 3.1 minimal cohort with its five-run top-up."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_MANIFEST = ROOT / "docs/ten-run-allowlists/gemini-3.1-live-minimal-2026-03-28.txt"
ATTEMPTS = HERE / "attempts.tsv"
AUGUST_CONTROL_DIRS = (
    "runs/aiwf_medium_context/"
    "20260804T141323_gemini-3.1-flash-live-preview_a9e08bb1",
    "runs/aiwf_medium_context/"
    "20260804T175455_gemini-3.1-flash-live-preview_f79bd81e",
)
DIMENSIONS = ("tool_use_correct", "instruction_following", "kb_grounding")
BOOTSTRAPS = 100_000
SEED = 20260805
NO_RESPONSE_RE = re.compile(r"\[(?:NO_RESPONSE|EMPTY_RESPONSE)\] turn=(\d+)")
RECONNECTION_MARKER = "Calling on_reconnecting callback"


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_run(run_dir_text: str) -> dict:
    run_dir = ROOT / run_dir_text
    summary = json.loads((run_dir / "claude_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "claude_judged.jsonl").read_text().splitlines()
        if line
    ]
    assert summary["turns_scored"] == 30, run_dir
    assert len(rows) == 30, run_dir
    assert {row["turn"] for row in rows} == set(range(30)), run_dir

    passes = Counter()
    strict = 0
    for row in rows:
        scores = row["scores"]
        for dimension in (*DIMENSIONS, "turn_taking"):
            passes[dimension] += scores.get(dimension) is True
        strict += all(scores.get(dimension) is True for dimension in DIMENSIONS)
    return {
        "run_dir": run_dir_text,
        "strict_passes": strict,
        "dimension_passes": dict(passes),
        "judge_model": summary.get("judge_model"),
        "judge_version": summary.get("judge_version"),
    }


def bootstrap_ci(strict_passes: list[int]) -> list[float]:
    rng = random.Random(SEED)
    rates = []
    for _ in range(BOOTSTRAPS):
        sampled = [rng.choice(strict_passes) for _ in strict_passes]
        rates.append(100 * sum(sampled) / (30 * len(sampled)))
    rates.sort()
    return [rates[int(0.025 * BOOTSTRAPS)], rates[int(0.975 * BOOTSTRAPS)]]


def wilson_ci(successes: int, attempts: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / attempts
    denominator = 1 + z**2 / attempts
    center = (rate + z**2 / (2 * attempts)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / attempts + z**2 / (4 * attempts**2))
        / denominator
    )
    return [100 * (center - half_width), 100 * (center + half_width)]


def summarize(runs: list[dict]) -> dict:
    passes = Counter()
    for run in runs:
        passes.update(run["dimension_passes"])
    turns = 30 * len(runs)
    strict_passes = sum(run["strict_passes"] for run in runs)
    return {
        "conversations": len(runs),
        "turns": turns,
        "strict_passes": strict_passes,
        "strict_pass_rate_pct": 100 * strict_passes / turns,
        "strict_pass_bootstrap_ci95_pct": bootstrap_ci(
            [run["strict_passes"] for run in runs]
        ),
        "dimension_passes": dict(passes),
        "dimension_pass_rates_pct": {
            dimension: 100 * passes[dimension] / turns
            for dimension in (*DIMENSIONS, "turn_taking")
        },
        "per_conversation_strict_passes": [run["strict_passes"] for run in runs],
        "per_conversation_strict_passes_median": statistics.median(
            run["strict_passes"] for run in runs
        ),
        "judge_models": sorted(
            {run["judge_model"] for run in runs if run["judge_model"]}
        ),
        "judge_versions": sorted(
            {run["judge_version"] for run in runs if run["judge_version"]}
        ),
        "run_dirs": [run["run_dir"] for run in runs],
    }


def current_reliability(attempts: list[dict]) -> dict:
    timeout_attempts = []
    timeout_turns = []
    timeout_events = 0
    reconnection_attempts = []
    for row in attempts:
        run_log = ROOT / row["run_dir"] / "run.log"
        text = run_log.read_text(errors="replace")
        turns = [int(turn) for turn in NO_RESPONSE_RE.findall(text)]
        timeout_events += len(turns)
        if turns:
            timeout_attempts.append(int(row["attempt"]))
            timeout_turns.append(turns[0])
        if RECONNECTION_MARKER in text:
            reconnection_attempts.append(int(row["attempt"]))

    completed = sum(
        row["classification"] == "complete"
        and int(row["attempt"]) not in timeout_attempts
        for row in attempts
    )
    return {
        "attempts": len(attempts),
        "run_dirs": [row["run_dir"] for row in attempts],
        "complete_without_no_response": completed,
        "no_response_attempts": timeout_attempts,
        "no_response_event_count": timeout_events,
        "first_no_response_turns": timeout_turns,
        "reconnection_attempts": reconnection_attempts,
    }


def historical_reliability(run_dirs: list[str]) -> dict:
    timeout_runs = []
    timeout_events = 0
    timeout_turns = []
    reconnection_runs = []
    complete = 0
    for run_dir_text in run_dirs:
        run_dir = ROOT / run_dir_text
        text = (run_dir / "run.log").read_text(errors="replace")
        failure_turns = [int(turn) for turn in NO_RESPONSE_RE.findall(text)]
        timeout_events += len(failure_turns)
        if failure_turns:
            timeout_runs.append(run_dir_text)
            timeout_turns.append(failure_turns[0])
        if RECONNECTION_MARKER in text:
            reconnection_runs.append(run_dir_text)
        transcript_turns = sum(
            1
            for line in (run_dir / "transcript.jsonl").read_text().splitlines()
            if line.strip()
        )
        complete += transcript_turns == 30 and not failure_turns
    return {
        "attempts": len(run_dirs),
        "run_dirs": run_dirs,
        "complete_without_no_response": complete,
        "no_response_runs": timeout_runs,
        "no_response_event_count": timeout_events,
        "first_no_response_turns": timeout_turns,
        "reconnection_runs": reconnection_runs,
    }


def main() -> None:
    old_dirs = read_manifest(OLD_MANIFEST)
    with ATTEMPTS.open(newline="") as handle:
        attempts = list(csv.DictReader(handle, delimiter="\t"))
    eligible_new_dirs = [
        row["run_dir"]
        for row in attempts
        if row["classification"] == "complete" and int(row["retry_events"]) == 0
    ]
    # Keep the content cohort fixed at the first five recovery-free completions.
    # Later attempts extend the reliability denominator only.
    new_dirs = eligible_new_dirs[:5]
    assert len(old_dirs) == 10, old_dirs
    assert len(new_dirs) == 5, new_dirs
    assert not set(old_dirs) & set(new_dirs)
    (HERE / "clean-complete-runs.txt").write_text("\n".join(new_dirs) + "\n")

    historical = historical_reliability(old_dirs + list(AUGUST_CONTROL_DIRS))
    current = current_reliability(attempts)

    old_runs = [load_run(run_dir) for run_dir in old_dirs]
    new_runs = [load_run(run_dir) for run_dir in new_dirs]
    aggregate = {
        "model": "gemini-3.1-flash-live-preview",
        "configuration": "minimal thinking",
        "old_frozen_cohort": summarize(old_runs),
        "new_topup_cohort": summarize(new_runs),
        "combined_cohort": summarize(old_runs + new_runs),
        "no_response_no_replay_reliability": {
            "historical_complete": historical["complete_without_no_response"],
            "historical_attempts": historical["attempts"],
            "historical_run_dirs": historical["run_dirs"],
            "historical_no_response_runs": historical["no_response_runs"],
            "historical_no_response_events": historical["no_response_event_count"],
            "historical_first_no_response_turns": historical["first_no_response_turns"],
            "historical_reconnection_runs": historical["reconnection_runs"],
            "current_complete": current["complete_without_no_response"],
            "current_attempts": current["attempts"],
            "current_run_dirs": current["run_dirs"],
            "current_no_response_attempts": current["no_response_attempts"],
            "current_no_response_events": current["no_response_event_count"],
            "current_first_no_response_turns": current["first_no_response_turns"],
            "current_reconnection_attempts": current["reconnection_attempts"],
            "current_completion_rate_pct": 100
            * current["complete_without_no_response"]
            / current["attempts"],
            "combined_complete": historical["complete_without_no_response"]
            + current["complete_without_no_response"],
            "combined_attempts": historical["attempts"] + current["attempts"],
            "combined_completion_rate_pct": 100
            * (
                historical["complete_without_no_response"]
                + current["complete_without_no_response"]
            )
            / (historical["attempts"] + current["attempts"]),
        },
    }
    (HERE / "combined-aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )

    combined = aggregate["combined_cohort"]
    new = aggregate["new_topup_cohort"]
    reliability = aggregate["no_response_no_replay_reliability"]
    combined_reliability_ci = wilson_ci(
        reliability["combined_complete"], reliability["combined_attempts"]
    )
    ci = combined["strict_pass_bootstrap_ci95_pct"]
    rates = combined["dimension_pass_rates_pct"]
    lines = [
        "# Gemini 3.1 Flash Live Preview minimal-thinking top-up",
        "",
        "Date: 2026-08-05",
        "",
        "The combined cohort contains the frozen March 10-run cohort and five new "
        "complete conversations.",
        "",
        "| Metric | New five | Combined fifteen |",
        "|---|---:|---:|",
        f"| Judged turns | {new['turns']} | {combined['turns']} |",
        f"| Strict pass | {new['strict_pass_rate_pct']:.1f}% | {combined['strict_pass_rate_pct']:.1f}% ({ci[0]:.1f}–{ci[1]:.1f}%) |",
        f"| Tool use | {new['dimension_pass_rates_pct']['tool_use_correct']:.1f}% | {rates['tool_use_correct']:.1f}% |",
        f"| Instruction following | {new['dimension_pass_rates_pct']['instruction_following']:.1f}% | {rates['instruction_following']:.1f}% |",
        f"| Knowledge grounding | {new['dimension_pass_rates_pct']['kb_grounding']:.1f}% | {rates['kb_grounding']:.1f}% |",
        "",
        "## No-response, no-replay reliability",
        "",
        "The same policy is applied to both models: a logged 15-second no-response "
        "event terminates the run and the utterance is not replayed. A replay-enabled "
        "August 4 Gemini control is therefore classified as failed at its first "
        "logged timeout, even though replay eventually completed the conversation. "
        "WebSocket reconnects without a no-response timeout are tracked separately.",
        "",
        "| Cohort | No-response-free completion | First failure turns |",
        "|---|---:|---:|",
        f"| Historical recent runs | {reliability['historical_complete']}/{reliability['historical_attempts']} ({100 * reliability['historical_complete'] / reliability['historical_attempts']:.1f}%) | {reliability['historical_first_no_response_turns']} |",
        f"| Current top-up attempts | {reliability['current_complete']}/{reliability['current_attempts']} ({reliability['current_completion_rate_pct']:.1f}%) | {reliability['current_first_no_response_turns']} |",
        f"| Combined | {reliability['combined_complete']}/{reliability['combined_attempts']} ({reliability['combined_completion_rate_pct']:.1f}%; Wilson 95% CI {combined_reliability_ci[0]:.1f}–{combined_reliability_ci[1]:.1f}%) | {reliability['historical_first_no_response_turns'] + reliability['current_first_no_response_turns']} |",
    ]
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
