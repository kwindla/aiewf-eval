#!/usr/bin/env python3
"""Sensitivity replacing the counted bare-idle Lite control with its extra run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze


HERE = Path(__file__).resolve().parent
COUNTED_IDLE = (analyze.ROOT / "runs/aiwf_medium_context/20260721T172944_gemini-3.5-flash-lite_ab2a7e32").resolve()
EXTRA_REPLACEMENT = (analyze.ROOT / "runs/aiwf_medium_context/20260721T173037_gemini-3.5-flash-lite_9b1df5f3").resolve()


def public(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def main() -> None:
    cells = analyze.load_all()
    primary_runs = cells[("gemini35flashlite", "nofiller")]
    matches = [run for run in primary_runs if run.run_dir == COUNTED_IDLE]
    if len(matches) != 1:
        raise ValueError("counted idle-timeout control is not uniquely present")
    replacement = analyze.load_conversation("gemini35flashlite", "nofiller", EXTRA_REPLACEMENT)
    alternate_runs = [run for run in primary_runs if run.run_dir != COUNTED_IDLE] + [replacement]
    primary = analyze.arm_summary(primary_runs, np.random.default_rng(20260721 + 50_000))
    alternate = analyze.arm_summary(alternate_runs, np.random.default_rng(20260721 + 50_000))
    output = {
        "artifact_status": "SENSITIVITY_ONLY_NOT_PRIMARY",
        "question": "Effect of treating the bare harness idle timeout as replacement-eligible infrastructure",
        "primary_counted_run": str(COUNTED_IDLE.relative_to(analyze.ROOT)),
        "alternate_extra_run": str(EXTRA_REPLACEMENT.relative_to(analyze.ROOT)),
        "primary_attempt_based": public(primary),
        "replacement_sensitivity": public(alternate),
        "pass_rate_change_points": alternate["pass_rate_pct"] - primary["pass_rate_pct"],
        "strict_completion_change_points": alternate["strict_completion_pct"] - primary["strict_completion_pct"],
    }
    (HERE / "idle-timeout-sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
