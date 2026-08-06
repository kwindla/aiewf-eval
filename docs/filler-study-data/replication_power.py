#!/usr/bin/env python3
"""Plan fresh, fixed-size model-specific filler replications.

The experimental unit is one complete 30-turn conversation. The planning model
uses a directional two-sample t approximation on each conversation's strict pass
fraction. To limit pilot overfitting, it powers for 75% of the observed effect and
uses a common 6.2-point planning SD. That SD is approximately the largest one-sided
80% upper pilot bound and avoids treating ceiling-locked zero variance as known.

These calculations size new replications. Existing exploratory runs do not count
toward the confirmatory sample size.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import nct, t


ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.025
EFFECT_RETENTION = 0.75
PLANNING_SD = 0.062


@dataclass(frozen=True)
class Comparison:
    model: str
    manifest: str
    control: str
    treatment: str


COMPARISONS = (
    Comparison(
        "gpt-5.4",
        "gpt54_filler_manifest.tsv",
        "nofiller",
        "filler96",
    ),
    Comparison(
        "gpt-5.5",
        "gpt55_manifest.tsv",
        "gpt55_nofiller",
        "gpt55_dots96",
    ),
    Comparison(
        "gpt-5.6-sol",
        "expand_oai_ant_manifest.tsv",
        "sol_nofiller",
        "sol_filler96",
    ),
    Comparison(
        "glm-5.2",
        "broaden_baseten_manifest.tsv",
        "glm52_nofiller",
        "glm52_filler96",
    ),
)


def turn_pass(row: dict) -> bool | None:
    values = [
        value
        for value in (row.get("scores") or {}).values()
        if isinstance(value, bool)
    ]
    return all(values) if values else None


def load_scores(manifest: Path, label: str) -> list[float]:
    run_dirs = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        candidate, run_dir = line.split("\t", 1)
        if candidate == label:
            run_dirs.append(ROOT / run_dir.strip())

    scores = []
    for run_dir in run_dirs:
        judged = run_dir / "claude_judged.jsonl"
        if not judged.is_file():
            raise ValueError(f"missing judgment file: {judged}")
        by_turn = {}
        for line in judged.read_text().splitlines():
            row = json.loads(line)
            passed = turn_pass(row)
            turn = row.get("turn")
            if passed is not None and isinstance(turn, int):
                by_turn[turn] = passed
        if set(by_turn) != set(range(30)):
            raise ValueError(f"run is not a complete 30-turn cluster: {run_dir}")
        scores.append(sum(by_turn.values()) / 30)
    if len(scores) < 2:
        raise ValueError(f"need at least two runs for {label}")
    return scores


def two_sample_power(n_per_arm: int, standardized_effect: float) -> float:
    degrees_freedom = 2 * n_per_arm - 2
    critical = t.ppf(1 - ALPHA, degrees_freedom)
    noncentrality = standardized_effect * math.sqrt(n_per_arm / 2)
    return float(nct.sf(critical, degrees_freedom, noncentrality))


def required_n(standardized_effect: float, target_power: float) -> int:
    for n_per_arm in range(2, 1001):
        if two_sample_power(n_per_arm, standardized_effect) >= target_power:
            return n_per_arm
    raise ValueError("sample size exceeded search bound")


def main() -> None:
    print(
        "model\tpilot_n\tpilot_delta_pp\tplanning_delta_pp\tplanning_sd_pp"
        "\tn_per_arm_80\tn_per_arm_90"
    )
    for comparison in COMPARISONS:
        manifest = Path(__file__).resolve().parent / comparison.manifest
        control = load_scores(manifest, comparison.control)
        treatment = load_scores(manifest, comparison.treatment)
        observed_effect = statistics.mean(treatment) - statistics.mean(control)
        planning_effect = abs(observed_effect) * EFFECT_RETENTION
        planning_sd = PLANNING_SD
        standardized_effect = planning_effect / planning_sd
        print(
            f"{comparison.model}\t{len(control)}/{len(treatment)}"
            f"\t{100 * observed_effect:+.2f}\t{100 * planning_effect:.2f}"
            f"\t{100 * planning_sd:.2f}"
            f"\t{required_n(standardized_effect, 0.80)}"
            f"\t{required_n(standardized_effect, 0.90)}"
        )


if __name__ == "__main__":
    main()
