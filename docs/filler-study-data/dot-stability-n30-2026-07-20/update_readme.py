#!/usr/bin/env python3
"""Refresh focused leaderboard rows from the frozen n=30 aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
AGGREGATES = HERE / "aggregates.json"

ROWS = {
    "gpt54_nofiller": ("gpt54", "nofiller", "gpt-5.4 (none)"),
    "gpt54_dots": ("gpt54", "dots96", "gpt-5.4 (none, +96 dots)"),
    "terra": ("terra", "nofiller", "gpt-5.6-terra (none)"),
    "gpt55": ("gpt55", "nofiller", "gpt-5.5 (none)"),
    "sol": ("sol", "nofiller", "gpt-5.6-sol (none)"),
    "gemma431": ("gemma431", "nofiller", "lilac/gemma-4-31b-it (thinking off)"),
    "inkling": ("inkling", "nofiller", "inkling (none)"),
    "qwen3_8b": ("qwen3_8b", "nofiller", "qwen3-8b (thinking off, BaseTen)"),
    "glm52": ("glm52", "nofiller", "glm-5.2 (none)"),
}

REMOVE_LABELS = {
    "gpt-5.4",
    "gpt-5.4 (none)",
    "gpt-5.4 (none, +96 dots)",
    "gpt-5.6-terra (none)",
    "gpt-5.5 (none)",
    "gpt-5.6-sol (none)",
    "google/gemma-4-31b-it",
    "lilac/gemma-4-31b-it (thinking off)",
    "inkling (none)",
    "qwen3-8b (thinking off)",
    "qwen3-8b (thinking off, BaseTen)",
    "glm-5.2 (none)",
}

PROVIDERS = {
    "gpt-5.4 (none)": "OpenAI",
    "gpt-5.4 (none, +96 dots)": "OpenAI",
    "gpt-5.6-terra (none)": "OpenAI",
    "gpt-5.5 (none)": "OpenAI",
    "gpt-5.6-sol (none)": "OpenAI",
    "lilac/gemma-4-31b-it (thinking off)": "Lilac",
    "inkling (none)": "BaseTen",
    "qwen3-8b (thinking off, BaseTen)": "BaseTen",
    "glm-5.2 (none)": "BaseTen",
}


def plain_label(value: str) -> str:
    return value.replace("**", "").strip()


def formatted_row(label: str, arm: dict) -> str:
    values = [
        f'{arm["pass_rate_pct"]:.1f}%',
        f'{arm["any_error_rate_pct"]:.1f}%',
        f'{arm["tool_error_rate_pct"]:.1f}%',
        f'{arm["instruction_error_rate_pct"]:.1f}%',
        f'{arm["kb_error_rate_pct"]:.1f}%',
        f'{round(arm["ttfat_p50_ms"])}ms',
        f'{round(arm["ttfat_p95_ms"])}ms',
        f'{round(arm["ttfat_max_ms"])}ms',
    ]
    return "| " + " | ".join([label, *values, PROVIDERS[label]]) + " |"


def pass_rate(row: str) -> float:
    cells = [cell.strip().replace("**", "") for cell in row.strip().strip("|").split("|")]
    return float(cells[1].removesuffix("%"))


def render() -> str:
    payload = json.loads(AGGREGATES.read_text())
    if payload.get("protocol", {}).get("target_per_arm") != 30:
        raise ValueError("README refresh requires target_per_arm=30")
    models = payload.get("models", {})
    expected = {model for model, _arm, _label in ROWS.values()}
    if set(models) != expected:
        raise ValueError(f"focused aggregate model mismatch: {sorted(models)}")
    qwen_source = payload.get("protocol", {}).get("primary_sources", {}).get("qwen3_8b", {})
    if models["qwen3_8b"].get("provider") != "BaseTen" or qwen_source != {
        "lane": "baseten-qwen",
        "provider": "BaseTen",
        "historical_attempts_included": 0,
        "openrouter_attempts_included": 0,
    }:
        raise ValueError("Qwen aggregate is not the BaseTen-only replacement cohort")

    text = README.read_text()
    header = "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"
    start = text.index(header)
    table_end = text.index("\n\n", start)
    block = text[start:table_end]
    lines = block.splitlines()
    kept = []
    for row in lines[2:]:
        label = plain_label(row.strip().strip("|").split("|", 1)[0])
        if label not in REMOVE_LABELS:
            kept.append(row)

    generated = []
    for model, arm, label in ROWS.values():
        result = models[model]["arms"][arm]
        if result.get("n_attempts") != 30 or result.get("fixed_turn_denominator") != 900:
            raise ValueError(f"unexpected denominator for {model}/{arm}")
        generated.append(formatted_row(label, result))

    rows = kept + generated
    rows.sort(key=pass_rate, reverse=True)
    new_block = "\n".join([lines[0], lines[1], *rows])
    return text[:start] + new_block + text[table_end:]


def check(text: str) -> None:
    payload = json.loads(AGGREGATES.read_text())
    for model, arm, label in ROWS.values():
        expected = formatted_row(label, payload["models"][model]["arms"][arm])
        if text.count(expected) != 1:
            raise ValueError(f"README row mismatch for {label}")
    for legacy in ("| **gpt-5.4** |", "| google/gemma-4-31b-it |"):
        if legacy in text:
            raise ValueError(f"stale README row remains: {legacy}")
    table = text[text.index("| Model | Pass Rate |"):].split("\n\n", 1)[0]
    if "| Runs |" in table:
        raise ValueError("README must not expose per-model run counts")
    rates = [pass_rate(row) for row in table.splitlines()[2:]]
    if rates != sorted(rates, reverse=True):
        raise ValueError("README text-model table is not sorted by pass rate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if README.read_text() != rendered:
            raise SystemExit("README focused rows are stale")
        check(rendered)
        print("README focused rows verified")
        return
    README.write_text(rendered)
    check(rendered)
    print(f"updated {README}")


if __name__ == "__main__":
    main()
