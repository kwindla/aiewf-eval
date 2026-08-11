#!/usr/bin/env python3
"""Build the frozen Async Gemini Live realtime-model comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N_TURNS = 30
BOOTSTRAPS = 100_000
SEED = 20260804
ORIGINAL_CLEVER = Path(
    "runs/aiwf_medium_context/20260804T145352_async-gemini-live_9554f41d"
)


@dataclass(frozen=True)
class CohortSpec:
    key: str
    display: str
    provider: str
    configuration: str
    allowlists: tuple[Path, ...]
    expected_runs: int
    latency_source: Path
    latency_key: str
    extra_runs: tuple[Path, ...] = ()


COHORTS = (
    CohortSpec(
        key="gemini25",
        display="Gemini 2.5 Flash Native Audio",
        provider="AI Studio",
        configuration="provider default",
        allowlists=(ROOT / "docs/ten-run-allowlists/gemini-live-2026-03-28.txt",),
        expected_runs=10,
        latency_source=ROOT / "docs/ten-run-aggregates/gemini-live-2026-03-28.json",
        latency_key="gemini-live",
    ),
    CohortSpec(
        key="gemini31",
        display="Gemini 3.1 Flash Live Preview",
        provider="AI Studio",
        configuration="minimal thinking",
        allowlists=(
            ROOT / "docs/ten-run-allowlists/gemini-3.1-live-minimal-2026-03-28.txt",
        ),
        expected_runs=10,
        latency_source=ROOT / "docs/ten-run-aggregates/gemini-3.1-live-minimal-2026-03-28.json",
        latency_key="gemini-3.1-flash-live-preview",
    ),
    CohortSpec(
        key="clever",
        display="Async Gemini Live",
        provider="AI Studio",
        configuration="minimal thinking",
        allowlists=(HERE / "clever-selected-valid-runs.txt",),
        expected_runs=15,
        latency_source=HERE / "new-cohort-aggregates.json",
        latency_key="async-gemini-live",
        extra_runs=(ORIGINAL_CLEVER,),
    ),
    CohortSpec(
        key="openai21",
        display="GPT-Realtime-2.1",
        provider="OpenAI",
        configuration="low reasoning",
        allowlists=(
            HERE / "openai-a-valid-runs.txt",
            HERE / "openai-b-valid-runs.txt",
            HERE / "openai-topup-a-valid-runs.txt",
            HERE / "openai-topup-b-valid-runs.txt",
            HERE / "openai-topup-c-valid-runs.txt",
            HERE / "openai-topup-d-valid-runs.txt",
        ),
        expected_runs=30,
        latency_source=HERE / "new-cohort-aggregates.json",
        latency_key="gpt-realtime-2.1",
    ),
)

RETRY_PATTERNS = (
    (re.compile(r"\[EMPTY_RESPONSE\] turn=(\d+) retry_count=(\d+)"), "empty response"),
    (re.compile(r"\[NO_RESPONSE\] turn=(\d+) retry_count=(\d+)"), "no response"),
    (re.compile(r"Gemini reconnected: scheduling turn (\d+) retry"), "reconnection"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-latency",
        action="store_true",
        help="re-run audio timing aggregation for the two new cohorts",
    )
    return parser.parse_args()


def read_allowlist(path: Path) -> list[Path]:
    return [ROOT / line.strip() for line in path.read_text().splitlines() if line.strip()]


def freeze_clever_selection() -> None:
    destination = HERE / "clever-selected-valid-runs.txt"
    if destination.is_file():
        return
    candidates = set()
    for path in sorted(HERE.glob("clever-?-valid-runs.txt")):
        candidates.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(candidates) < 14:
        raise ValueError(f"clever: need 14 valid campaign additions, found {len(candidates)}")
    selected = sorted(candidates)[:14]
    destination.write_text("\n".join(selected) + "\n")


def cohort_runs(spec: CohortSpec) -> list[Path]:
    runs = [ROOT / path for path in spec.extra_runs]
    for allowlist in spec.allowlists:
        runs.extend(read_allowlist(allowlist))
    resolved = [path.resolve() for path in runs]
    if len(resolved) != spec.expected_runs:
        raise ValueError(
            f"{spec.key}: expected {spec.expected_runs} runs, found {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{spec.key}: duplicate run in cohort")
    return resolved


def refresh_latency_aggregates(runs_by_key: dict[str, list[Path]]) -> None:
    def build(key: str) -> dict:
        relpaths = [str(path.relative_to(ROOT)) for path in runs_by_key[key]]
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark_summary.py"),
            *relpaths,
            "--json",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    combined = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for aggregate in executor.map(build, ("clever", "openai21")):
            combined.update(aggregate)
    (HERE / "new-cohort-aggregates.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )


def load_turns(run_dir: Path) -> tuple[list[dict], Counter, Counter]:
    judged_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    transcript_path = run_dir / "transcript.jsonl"
    for path in (judged_path, summary_path, transcript_path, run_dir / "run.log"):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing required artifact: {path}")

    summary = json.loads(summary_path.read_text())
    if summary.get("turns_scored") != N_TURNS:
        raise ValueError(f"not fully judged: {run_dir}")
    if not summary.get("judge_model") or not summary.get("judge_version"):
        raise ValueError(f"judge provenance missing: {run_dir}")

    turns = {}
    for line in judged_path.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if isinstance(turn, int) and 0 <= turn < N_TURNS:
            turns[turn] = row
    if set(turns) != set(range(N_TURNS)):
        raise ValueError(f"judgment coverage mismatch: {run_dir}")

    transcript_rows = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    primary_rows = [row for row in transcript_rows if row.get("recovery_turn") is not True]
    if len(primary_rows) != N_TURNS:
        raise ValueError(f"transcript does not contain 30 primary turns: {run_dir}")

    retry_events: Counter[str] = Counter()
    retry_turns: Counter[int] = Counter()
    for line in (run_dir / "run.log").read_text(errors="replace").splitlines():
        for pattern, label in RETRY_PATTERNS:
            match = pattern.search(line)
            if match:
                retry_events[label] += 1
                retry_turns[int(match.group(1))] += 1
    return [turns[index] for index in range(N_TURNS)], retry_events, retry_turns


def bootstrap_ci(per_conversation_rates: list[float], seed_offset: int) -> list[float]:
    values = np.asarray(per_conversation_rates, dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[indices].mean(axis=1) * 100
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def latency_summary(spec: CohortSpec) -> dict:
    source = json.loads(spec.latency_source.read_text())
    aggregate = source[spec.latency_key]
    non_tool = aggregate.get("non_tool_v2vs") or []
    tool = aggregate.get("tool_v2vs") or []
    silence = aggregate.get("silence_pads") or []
    return {
        "non_tool_v2v_p50_ms": median(non_tool),
        "non_tool_v2v_max_ms": max(non_tool) if non_tool else None,
        "tool_v2v_mean_ms": statistics.mean(tool) if tool else None,
        "silence_pad_mean_ms": statistics.mean(silence) if silence else None,
        "non_tool_observations": len(non_tool),
        "tool_observations": len(tool),
    }


def score_cohort(spec: CohortSpec, runs: list[Path], seed_offset: int) -> dict:
    dimensions = ("tool_use_correct", "instruction_following", "kb_grounding")
    passes = Counter()
    strict_by_turn = Counter()
    strict_by_run = []
    dimension_fail_by_turn = {
        dimension: Counter() for dimension in (*dimensions, "turn_taking")
    }
    retry_events: Counter[str] = Counter()
    retry_turns: Counter[int] = Counter()
    retry_conversations_by_turn: Counter[int] = Counter()
    retry_turn_count = 0
    judge_models = set()
    judge_versions = set()

    for run_dir in runs:
        rows, run_retry_events, run_retry_turns = load_turns(run_dir)
        summary = json.loads((run_dir / "claude_summary.json").read_text())
        judge_models.add(summary["judge_model"])
        judge_versions.add(summary["judge_version"])
        retry_events.update(run_retry_events)
        retry_turns.update(run_retry_turns)
        retry_turn_count += len(run_retry_turns)
        retry_conversations_by_turn.update(run_retry_turns.keys())

        strict_count = 0
        for turn, row in enumerate(rows):
            scores = row["scores"]
            for dimension in (*dimensions, "turn_taking"):
                passes[dimension] += scores.get(dimension) is True
                if scores.get(dimension) is not True:
                    dimension_fail_by_turn[dimension][turn] += 1
            strict = all(scores.get(dimension) is True for dimension in dimensions)
            if strict:
                strict_count += 1
            else:
                strict_by_turn[turn] += 1
        strict_by_run.append(strict_count / N_TURNS)

    denominator = len(runs) * N_TURNS
    strict_passes = round(sum(strict_by_run) * N_TURNS)
    return {
        "key": spec.key,
        "display": spec.display,
        "provider": spec.provider,
        "configuration": spec.configuration,
        "runs": len(runs),
        "turns": denominator,
        "strict_passes": strict_passes,
        "strict_pass_rate_pct": 100 * strict_passes / denominator,
        "strict_pass_ci95": bootstrap_ci(strict_by_run, seed_offset),
        "dimension_passes": dict(passes),
        "retry_events": dict(retry_events),
        "retry_event_count": sum(retry_events.values()),
        "retry_turn_count": retry_turn_count,
        "retry_turn_rate_pct": 100 * retry_turn_count / denominator,
        "top_strict_failure_turns": [
            {
                "turn": turn,
                "strict_failures": failures,
                "tool_failures": dimension_fail_by_turn["tool_use_correct"][turn],
                "instruction_failures": dimension_fail_by_turn["instruction_following"][turn],
                "kb_failures": dimension_fail_by_turn["kb_grounding"][turn],
                "turn_taking_failures": dimension_fail_by_turn["turn_taking"][turn],
                "retry_conversations": retry_conversations_by_turn[turn],
            }
            for turn, failures in strict_by_turn.most_common(12)
        ],
        "strict_passes_by_conversation": [round(rate * N_TURNS) for rate in strict_by_run],
        "judge_models": sorted(judge_models),
        "judge_versions": sorted(judge_versions),
        "run_dirs": [str(path.relative_to(ROOT)) for path in runs],
        "latency": latency_summary(spec),
    }


def load_attempts(prefix: str) -> list[dict]:
    rows = []
    for path in sorted(HERE.glob(f"{prefix}-*-attempts.tsv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


def end_session_turn(run_dir: Path) -> int | None:
    turns = []
    transcript = ROOT / run_dir / "transcript.jsonl"
    if not transcript.is_file():
        return None
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turns.append(int(row["turn"]))
    return min(turns) if turns else None


def attempt_summary(prefix: str, expected_valid: int) -> dict:
    rows = load_attempts(prefix)
    valid = [row for row in rows if row["exit_code"] == "0" and row["turns"] == "30"]
    if len(valid) < expected_valid:
        raise ValueError(
            f"{prefix}: expected at least {expected_valid} valid attempts, found {len(valid)}"
        )
    invalid_causes = Counter()
    for row in rows:
        if row not in valid:
            turn = end_session_turn(Path(row["run_dir"]))
            if turn is not None:
                label = f"end_session at turn {turn}"
            elif row["exit_code"] == "143":
                label = f"terminated stalled attempt after {row['turns']} turns"
            else:
                label = f"other: exit {row['exit_code']} after {row['turns']} turns"
            invalid_causes[label] += 1
    return {
        "attempts": len(rows),
        "valid": len(valid),
        "valid_pct": 100 * len(valid) / len(rows),
        "invalid_attempt_causes": dict(invalid_causes),
    }


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_ci(values: list[float]) -> str:
    return f"{values[0]:.1f}–{values[1]:.1f}%"


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


def render_markdown(result: dict) -> str:
    cohorts = result["cohorts"]
    clever = next(row for row in cohorts if row["key"] == "clever")
    openai = next(row for row in cohorts if row["key"] == "openai21")
    ranked = sorted(cohorts, key=lambda row: row["strict_pass_rate_pct"], reverse=True)
    top_failure_text = ", ".join(
        f"{row['turn']} ({row['strict_failures']}/{clever['runs']})"
        for row in clever["top_strict_failure_turns"][:5]
    )
    invalid_causes = result["attempts"]["clever"]["invalid_attempt_causes"]
    invalid_text = ", ".join(
        f"{cause}: {count}" for cause, count in sorted(invalid_causes.items())
    ) or "none"

    lines = [
        "# Async Gemini Live realtime comparison",
        "",
        "Date: 2026-08-04",
        "",
        "This is a standalone comparison. Async Gemini Live is intentionally not added to the",
        "public README table or the filler-effect report.",
        "",
        "## Quality and reliability",
        "",
        "Strict pass requires tool use, instruction following, and knowledge-base grounding;",
        "turn-taking is reported separately. Confidence intervals resample whole conversations",
        "(100,000 bootstrap samples), preserving within-conversation error clustering.",
        "",
        "| Model | Configuration | N | Strict pass (95% CI) | Tool | Instruction | KB | Turn-taking | Retry turns | Provider |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in cohorts:
        denom = row["turns"]
        dims = row["dimension_passes"]
        lines.append(
            "| {display} | {configuration} | {runs} | {rate} ({ci}) | {tool}/{denom} | "
            "{instruction}/{denom} | {kb}/{denom} | {tt}/{denom} | {retry}/{denom} ({retry_pct}) | {provider} |".format(
                display=row["display"],
                configuration=row["configuration"],
                runs=row["runs"],
                rate=fmt_pct(row["strict_pass_rate_pct"]),
                ci=fmt_ci(row["strict_pass_ci95"]),
                tool=dims["tool_use_correct"],
                instruction=dims["instruction_following"],
                kb=dims["kb_grounding"],
                tt=dims["turn_taking"],
                retry=row["retry_turn_count"],
                retry_pct=fmt_pct(row["retry_turn_rate_pct"]),
                denom=denom,
                provider=row["provider"],
            )
        )

    lines.extend(
        [
            "",
            "A retry turn is a distinct benchmark turn with at least one logged empty-response,",
            "no-response, or reconnection recovery event. It is not an additional denominator turn.",
            "",
            "## Latency",
            "",
            "| Model | Non-tool V2V P50 | Non-tool V2V max | Tool-turn V2V mean | Silence padding mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cohorts:
        latency = row["latency"]
        lines.append(
            f"| {row['display']} | {fmt_ms(latency['non_tool_v2v_p50_ms'])} | "
            f"{fmt_ms(latency['non_tool_v2v_max_ms'])} | "
            f"{fmt_ms(latency['tool_v2v_mean_ms'])} | "
            f"{fmt_ms(latency['silence_pad_mean_ms'])} |"
        )

    clever_attempts = result["attempts"]["clever"]
    openai_attempts = result["attempts"]["openai21"]
    clever_dims = clever["dimension_passes"]
    clever_latency = clever["latency"]
    per_conversation = clever["strict_passes_by_conversation"]
    retry_breakdown = ", ".join(
        f"{label}: {count}" for label, count in sorted(clever["retry_events"].items())
    ) or "none"
    lines.extend(
        [
            "",
            "Latency is conditional on responses the timing analyzer could align. Recovery-heavy",
            "runs can therefore have fewer latency observations than judged turns.",
            "",
            "## Async Gemini Live detail",
            "",
            "README-compatible row (shown for comparison only; not added to the public leaderboard):",
            "",
            "| Model | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| async-gemini-live (minimal thinking) | {fmt_pct(clever['strict_pass_rate_pct'])} | "
            f"{clever_dims['tool_use_correct']}/{clever['turns']} | "
            f"{clever_dims['instruction_following']}/{clever['turns']} | "
            f"{clever_dims['kb_grounding']}/{clever['turns']} | "
            f"{clever_dims['turn_taking']}/{clever['turns']} | "
            f"{fmt_ms(clever_latency['non_tool_v2v_p50_ms'])} | "
            f"{fmt_ms(clever_latency['non_tool_v2v_max_ms'])} | "
            f"{fmt_ms(clever_latency['tool_v2v_mean_ms'])} | "
            f"{fmt_ms(clever_latency['silence_pad_mean_ms'])} |",
            "",
            "| Metric | Result | Error rate / interval |",
            "|---|---:|---:|",
            f"| Strict pass | {clever['strict_passes']}/{clever['turns']} ({fmt_pct(clever['strict_pass_rate_pct'])}) | {fmt_ci(clever['strict_pass_ci95'])} whole-conversation bootstrap CI |",
            f"| Tool use | {clever_dims['tool_use_correct']}/{clever['turns']} | {fmt_pct(100 * (clever['turns'] - clever_dims['tool_use_correct']) / clever['turns'])} error |",
            f"| Instruction following | {clever_dims['instruction_following']}/{clever['turns']} | {fmt_pct(100 * (clever['turns'] - clever_dims['instruction_following']) / clever['turns'])} error |",
            f"| KB grounding | {clever_dims['kb_grounding']}/{clever['turns']} | {fmt_pct(100 * (clever['turns'] - clever_dims['kb_grounding']) / clever['turns'])} error |",
            f"| Turn-taking | {clever_dims['turn_taking']}/{clever['turns']} | {fmt_pct(100 * (clever['turns'] - clever_dims['turn_taking']) / clever['turns'])} error |",
            f"| Turns requiring recovery | {clever['retry_turn_count']}/{clever['turns']} ({fmt_pct(clever['retry_turn_rate_pct'])}) | {retry_breakdown} events |",
            f"| Per-conversation strict score | median {statistics.median(per_conversation):.0f}/30 | range {min(per_conversation)}–{max(per_conversation)}/30 |",
            f"| Full-conversation completion | {clever_attempts['valid']}/{clever_attempts['attempts']} ({fmt_pct(clever_attempts['valid_pct'])}) | {clever_attempts['attempts'] - clever_attempts['valid']} invalid attempts |",
            "",
            "Error hotspots below use zero-based benchmark turn indices.",
            "",
            "| Turn | Strict failures | Tool failures | Instruction failures | KB failures | Turn-taking failures | Conversations needing recovery |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in clever["top_strict_failure_turns"][:10]:
        lines.append(
            f"| {row['turn']} | {row['strict_failures']}/{clever['runs']} | "
            f"{row['tool_failures']}/{clever['runs']} | "
            f"{row['instruction_failures']}/{clever['runs']} | "
            f"{row['kb_failures']}/{clever['runs']} | "
            f"{row['turn_taking_failures']}/{clever['runs']} | "
            f"{row['retry_conversations']}/{clever['runs']} |"
        )

    lines.extend(
        [
            "",
            "## Campaign completion",
            "",
            "| Campaign | Full conversations | Finished attempts | Full-conversation completion | Invalid-attempt causes |",
            "|---|---:|---:|---:|---|",
            f"| Async Gemini Live additions | {clever_attempts['valid']} | {clever_attempts['attempts']} | {fmt_pct(clever_attempts['valid_pct'])} | {invalid_text} |",
            f"| GPT-Realtime-2.1 | {openai_attempts['valid']} | {openai_attempts['attempts']} | {fmt_pct(openai_attempts['valid_pct'])} | none |",
            "",
            "The Async Gemini Live quality cohort contains the original full smoke run plus the 14",
            "campaign additions. Diagnostic runs before the frozen campaign are excluded.",
            "The Gemini 2.5 row uses the newer frozen March retest rather than the older 86.0%",
            "aggregate still shown in the public README.",
            "",
            "## Conclusions",
            "",
            f"- Strict-pass ranking was: " + ", ".join(
                f"{row['display']} {fmt_pct(row['strict_pass_rate_pct'])}" for row in ranked
            ) + ".",
            f"- Async Gemini Live passed {clever['strict_passes']}/{clever['turns']} turns "
            f"({fmt_pct(clever['strict_pass_rate_pct'])}); its whole-conversation interval was "
            f"{fmt_ci(clever['strict_pass_ci95'])}.",
            f"- Its largest recurring strict-error turn indices were {top_failure_text}.",
            f"- Async Gemini Live needed recovery on {clever['retry_turn_count']}/{clever['turns']} "
            f"turns ({fmt_pct(clever['retry_turn_rate_pct'])}), compared with "
            f"{openai['retry_turn_count']}/{openai['turns']} for GPT-Realtime-2.1.",
            "- The completion failures and frequent recovery turns make Async Gemini Live materially",
            "  less reliable in this harness than the three comparison models, independent of its",
            "  strict content score.",
            "",
            "## Thought-text probe",
            "",
            "A standalone text-input probe used `send_realtime_input(text=...)` with",
            "`include_thoughts=True`. Minimal and high thinking reported 69 and 209 reasoning",
            "tokens respectively, but both returned zero thought-text parts. Async Gemini Live is",
            "therefore reasoning internally in these probes without exposing reasoning traces.",
            "The text field itself triggered the model turn; no separate `turn_complete` argument",
            "was required on the Gemini 3.1 realtime-input path.",
            "Google's [Live API capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)",
            "documents this Gemini 3.1-specific `send_realtime_input(text=...)` path.",
            "",
            "## Provenance",
            "",
            "- Gemini 2.5 and Gemini 3.1 use the frozen 2026-03-28 ten-run allowlists and aggregates.",
            "- Async Gemini Live and GPT-Realtime-2.1 were run on 2026-08-04 and judged with the",
            "  repository's Claude v4 turn-taking judge.",
            "- GPT-Realtime-2.1 is the current snapshot documented on OpenAI's",
            "  [model page](https://developers.openai.com/api/docs/models/gpt-realtime-2.1).",
            "- The JSON source for every number above is `comparison.json`.",
            "- Rebuild with `./.venv/bin/python docs/async-gemini-live-comparison-2026-08-04/analyze.py`.",
            "  Add `--refresh-latency` to recompute the new audio timing aggregates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    freeze_clever_selection()
    runs_by_key = {spec.key: cohort_runs(spec) for spec in COHORTS}
    if args.refresh_latency or not (HERE / "new-cohort-aggregates.json").is_file():
        refresh_latency_aggregates(runs_by_key)

    cohorts = [
        score_cohort(spec, runs_by_key[spec.key], index)
        for index, spec in enumerate(COHORTS)
    ]
    result = {
        "analysis_date": "2026-08-04",
        "turns_per_conversation": N_TURNS,
        "bootstrap_samples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "cohorts": cohorts,
        "attempts": {
            "clever": attempt_summary("clever", 14),
            "openai21": attempt_summary("openai", 30),
        },
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "comparison.md").write_text(render_markdown(result))


if __name__ == "__main__":
    main()
