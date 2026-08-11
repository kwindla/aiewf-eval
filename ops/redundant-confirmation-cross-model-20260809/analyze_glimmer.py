#!/usr/bin/env python3
"""Classify Turn 12 in the selected Muse Glimmer N=30 campaign."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = (
    ROOT
    / "runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("census", HERE / "analyze.py")
    assert spec and spec.loader
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)

    categories: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    eligible = 0
    manifest = CAMPAIGN / "included-runs.txt"
    run_dirs = [ROOT / line for line in manifest.read_text().splitlines() if line]
    for run_dir in run_dirs:
        rows = census.read_jsonl(run_dir / "transcript.jsonl")
        by_turn = {
            row["turn"]: row for row in rows if isinstance(row.get("turn"), int)
        }
        category = census.classify_turn12(by_turn[12])
        categories[category] += 1
        eligible += int(census.correct_turn11(by_turn.get(11)))
        if category == "no_tool_redundant_confirmation_or_question":
            subtypes[census.redundant_subtype(by_turn[12].get("assistant_text", ""))] += 1

    assert len(run_dirs) == 30
    assert categories == {
        "no_tool_redundant_confirmation_or_question": 27,
        "correct_tool_and_arguments": 3,
    }
    payload = {
        "schema_version": 1,
        "model": "muse-glimmer-30b",
        "configuration": "thinking high, GGUF, Q8_0 KV, DFlash 15",
        "campaign": str(CAMPAIGN.relative_to(ROOT)),
        "runs": len(run_dirs),
        "turn11_eligible": eligible,
        "turn12_categories": dict(categories),
        "redundant_subtypes": dict(subtypes),
    }
    output = HERE / "results/glimmer-canonical.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
