#!/usr/bin/env python3
"""Update the three Gemini minimal rows from the campaign no-filler arms."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
AGGREGATES = HERE / "aggregates.json"
START = "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"


def row_for(model: dict) -> str:
    arm = model["arms"]["nofiller"]
    n = arm["n_attempts"]
    if n < 10 or arm["fixed_turn_denominator"] != 30 * n:
        raise ValueError(f"invalid README control pool for {model['display_name']}")
    values = (
        model["readme_label"],
        f"{arm['pass_rate_pct']:.1f}%",
        f"{arm['any_error_rate_pct']:.1f}%",
        f"{arm['tool_error_rate_pct']:.1f}%",
        f"{arm['instruction_error_rate_pct']:.1f}%",
        f"{arm['kb_error_rate_pct']:.1f}%",
        f"{round(arm['ttfat_p50_ms'])}ms",
        f"{round(arm['ttfat_p95_ms'])}ms",
        f"{round(arm['ttfat_max_ms'])}ms",
        "AI Studio",
    )
    return "| " + " | ".join(values) + " |"


def pass_rate(row: str) -> float:
    return float(row.split("|")[2].strip().replace("**", "").rstrip("%"))


def main() -> None:
    payload = json.loads(AGGREGATES.read_text())
    if payload.get("artifact_status") != "FINAL":
        raise ValueError("Gemini aggregate is not final")
    if payload.get("protocol", {}).get("thinking_mode") != "minimal":
        raise ValueError("Gemini aggregate is not the minimal-thinking campaign")
    order = payload["protocol"]["model_order"]
    models = payload["models"]
    if set(order) != {"gemini35flash", "gemini35flashlite", "gemini36flash"}:
        raise ValueError("unexpected Gemini model set")

    replacements = {models[key]["readme_label"]: row_for(models[key]) for key in order}
    text = README.read_text()
    start = text.index(START)
    end = text.index("\n\n", start)
    lines = text[start:end].splitlines()
    header = lines[:2]
    rows = lines[2:]
    seen: set[str] = set()
    updated: list[str] = []
    for row in rows:
        label = row.split("|")[1].strip().replace("**", "")
        if label in replacements:
            updated.append(replacements[label])
            seen.add(label)
        else:
            updated.append(row)
    for label in (models[key]["readme_label"] for key in order):
        if label not in seen:
            updated.append(replacements[label])
    updated.sort(key=pass_rate, reverse=True)
    new_table = "\n".join(header + updated)
    README.write_text(text[:start] + new_table + text[end:])

    check = README.read_text()
    for label, row in replacements.items():
        if check.count(f"| {label} |") != 1 or row not in check:
            raise ValueError(f"README update verification failed for {label}")
    print("README Gemini minimal rows updated and verified")


if __name__ == "__main__":
    main()
