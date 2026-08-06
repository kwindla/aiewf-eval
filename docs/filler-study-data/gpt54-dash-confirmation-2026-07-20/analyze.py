#!/usr/bin/env python3
"""Frozen primary analysis for the GPT-5.4 dash confirmation.

The experimental unit is a complete conversation. Missing judged turns are failures
under the fixed 30-turn denominator. Allocation was randomized within 41 two-run time
blocks, so the primary randomization test flips the treatment label within each block.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import t


HERE = Path(__file__).resolve().parent
N_PAIRS = 41
N_TURNS = 30
N_RANDOMIZATIONS = 200_000
RANDOM_SEED = 20260720


@dataclass(frozen=True)
class Run:
    slot: int
    pair: int
    arm: str
    run_dir: Path
    score: float
    strict_complete: bool
    ttfat_median_ms: float | None


def strict_pass(row: dict) -> bool | None:
    values = [
        value
        for value in (row.get("scores") or {}).values()
        if isinstance(value, bool)
    ]
    return all(values) if values else None


def end_session_turn(transcript: Path) -> int:
    best = -1
    for line in transcript.read_text().splitlines():
        row = json.loads(line)
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            best = max(best, int(row.get("turn", -1)))
    return best


def load_schedule(path: Path) -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "slot\tpair\tpair_position\tarm":
        raise ValueError("unexpected schedule header")
    for line in lines[1:]:
        slot_s, pair_s, _position, arm = line.split("\t")
        rows[int(slot_s)] = (int(pair_s), arm)
    if len(rows) != 2 * N_PAIRS:
        raise ValueError(f"expected {2 * N_PAIRS} schedule rows, found {len(rows)}")
    if sum(arm == "dash96" for _, arm in rows.values()) != N_PAIRS:
        raise ValueError("schedule is not balanced")
    return rows


def load_counted(path: Path) -> dict[int, Path]:
    lines = path.read_text().splitlines()
    expected = (
        "slot\tpair\tarm\tattempt\tstart_utc\tend_utc\trun_rc\trun_dir"
        "\ttranscript_rows\tend_session_turn\tclassification\tjudge_rc"
    )
    if not lines or lines[0] != expected:
        raise ValueError("unexpected counted.tsv header")
    rows: dict[int, Path] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        slot = int(fields[0])
        if slot in rows:
            raise ValueError(f"duplicate counted slot {slot}")
        rows[slot] = Path(fields[7])
    return rows


def load_run(slot: int, pair: int, arm: str, run_dir: Path) -> Run:
    judged = run_dir / "claude_judged.jsonl"
    transcript = run_dir / "transcript.jsonl"
    if not judged.is_file():
        raise ValueError(f"slot {slot} lacks judgment: {judged}")
    if not transcript.is_file():
        raise ValueError(f"slot {slot} lacks transcript: {transcript}")

    by_turn: dict[int, bool] = {}
    ttfat: dict[int, float] = {}
    for line in judged.read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        passed = strict_pass(row)
        if isinstance(turn, int) and 0 <= turn < N_TURNS and passed is not None:
            by_turn[turn] = passed
            value = row.get("ttfb_ms")
            if isinstance(value, (int, float)):
                ttfat[turn] = float(value)

    # Unjudged and forfeited turns contribute zero by protocol.
    score = sum(by_turn.get(turn, False) for turn in range(N_TURNS)) / N_TURNS
    latency = statistics.median(ttfat.values()) if ttfat else None
    return Run(
        slot=slot,
        pair=pair,
        arm=arm,
        run_dir=run_dir,
        score=score,
        strict_complete=end_session_turn(transcript) == 29,
        ttfat_median_ms=latency,
    )


def studentized(values: list[float]) -> float:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    if sd == 0:
        return math.inf if mean > 0 else -math.inf if mean < 0 else 0.0
    return mean / (sd / math.sqrt(len(values)))


def randomization_p(differences: list[float]) -> tuple[float, float]:
    observed = studentized(differences)
    rng = random.Random(RANDOM_SEED)
    positive = 0
    two_sided = 0
    for _ in range(N_RANDOMIZATIONS):
        permuted = [value if rng.getrandbits(1) else -value for value in differences]
        statistic = studentized(permuted)
        if statistic >= observed - 1e-12:
            positive += 1
        if abs(statistic) >= abs(observed) - 1e-12:
            two_sided += 1
    denominator = N_RANDOMIZATIONS + 1
    return (positive + 1) / denominator, (two_sided + 1) / denominator


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=HERE / "schedule.tsv")
    parser.add_argument("--counted", type=Path, default=HERE / "counted.tsv")
    args = parser.parse_args()

    schedule = load_schedule(args.schedule)
    counted = load_counted(args.counted)
    if set(counted) != set(schedule):
        missing = sorted(set(schedule) - set(counted))
        extra = sorted(set(counted) - set(schedule))
        raise SystemExit(
            f"analysis locked until all 82 slots are counted; missing={missing}, extra={extra}"
        )

    runs = [
        load_run(slot, pair, arm, counted[slot])
        for slot, (pair, arm) in sorted(schedule.items())
    ]
    by_pair: dict[int, dict[str, Run]] = {}
    for run in runs:
        by_pair.setdefault(run.pair, {})[run.arm] = run
    if set(by_pair) != set(range(1, N_PAIRS + 1)):
        raise ValueError("pair identifiers are incomplete")
    if any(set(pair) != {"nofiller", "dash96"} for pair in by_pair.values()):
        raise ValueError("each pair must contain exactly one run per arm")

    control = [by_pair[pair]["nofiller"] for pair in range(1, N_PAIRS + 1)]
    dash = [by_pair[pair]["dash96"] for pair in range(1, N_PAIRS + 1)]
    differences = [treatment.score - baseline.score for baseline, treatment in zip(control, dash)]
    mean_difference = statistics.mean(differences)
    sd_difference = statistics.stdev(differences)
    se = sd_difference / math.sqrt(N_PAIRS)
    critical = float(t.ppf(0.975, N_PAIRS - 1))
    ci_low = mean_difference - critical * se
    ci_high = mean_difference + critical * se
    p_positive, p_two_sided = randomization_p(differences)

    print("GPT-5.4 96-dash prospective confirmation")
    print(f"conversations: {len(control)} control, {len(dash)} dash")
    print(f"mean strict pass: {percent(statistics.mean(r.score for r in control))} control")
    print(f"mean strict pass: {percent(statistics.mean(r.score for r in dash))} dash")
    print(f"paired mean difference: {100 * mean_difference:+.2f} percentage points")
    print(f"95% paired-t CI: [{100 * ci_low:+.2f}, {100 * ci_high:+.2f}] points")
    print(
        f"studentized block sign-flip p: one-sided={p_positive:.6f}, "
        f"two-sided={p_two_sided:.6f} ({N_RANDOMIZATIONS:,} draws)"
    )
    print(
        "strict completion: "
        f"{sum(r.strict_complete for r in control)}/{N_PAIRS} control, "
        f"{sum(r.strict_complete for r in dash)}/{N_PAIRS} dash"
    )
    print(
        "error-free conversations: "
        f"{sum(r.score == 1 for r in control)}/{N_PAIRS} control, "
        f"{sum(r.score == 1 for r in dash)}/{N_PAIRS} dash"
    )
    control_latency = [r.ttfat_median_ms for r in control if r.ttfat_median_ms is not None]
    dash_latency = [r.ttfat_median_ms for r in dash if r.ttfat_median_ms is not None]
    print(
        "median of conversation-median TTFAT: "
        f"{statistics.median(control_latency):.0f} ms control, "
        f"{statistics.median(dash_latency):.0f} ms dash"
    )


if __name__ == "__main__":
    main()
