#!/usr/bin/env python3
"""Apply the prospective dots sample-size rule to the current aggregate."""

from __future__ import annotations

import json
from pathlib import Path

import analyze


HERE = Path(__file__).resolve().parent


def main() -> None:
    payload = json.loads((HERE / "aggregates.json").read_text())
    cells = analyze.load_all()
    result = {
        "protocol_rule": {
            "stage1_topup": "abs(delta) >= 2.0 points or any strict-completion difference",
            "stage2_promotion": "95% bootstrap interval excludes zero, abs(delta) >= 3.0 points with a recurring same-turn direction in >=3 conversations, or strict-completion rates differ",
        },
        "models": {},
    }
    for model in analyze.MODELS:
        row = payload["models"][model]
        control = row["arms"]["nofiller"]
        dots = row["arms"]["dots96"]
        result["models"][model] = analyze.compute_adaptive_decision(
            cells[(model, "nofiller")], cells[(model, "dots96")], control, dots, row["effect"]
        )
    (HERE / "adaptive-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
