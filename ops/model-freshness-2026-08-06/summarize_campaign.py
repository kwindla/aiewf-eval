#!/usr/bin/env python3
"""Build the auditable usage table for the August 2026 freshness campaign."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEECH_WPM = 150.0


def load_manifest():
    path = HERE / "run_campaign.py"
    spec = importlib.util.spec_from_file_location("freshness_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ROWS


def read_results() -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in HERE.glob("results-*.json"):
        for item in json.loads(path.read_text()):
            by_key[item["key"]] = item
    return by_key


def read_readme_rows() -> tuple[list[str], dict[str, str]]:
    text = (ROOT / "README.md").read_text()
    section = text.split("Text mode models:", 1)[1].split("Speech-to-speech models:", 1)[0]
    labels: list[str] = []
    providers: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Model") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        label = re.sub(r"\*\*", "", cells[0])
        labels.append(label)
        providers[label] = cells[-1]
    return labels, providers


def main() -> int:
    rows = load_manifest()
    results = read_results()
    readme_labels, providers = read_readme_rows()
    by_label = {row.label: row for row in rows}
    output: list[dict[str, Any]] = []

    for label in readme_labels:
        row = by_label[label]
        result = results.get(row.key, {})
        attempts = result.get("attempts", [])
        accepted = next((attempt for attempt in reversed(attempts) if attempt.get("complete")), None)
        item: dict[str, Any] = {
            "key": row.key,
            "label": label,
            "provider": providers[label],
            "model": row.model,
            "service": row.service,
            "status": "complete" if accepted else ("attempted_incomplete" if attempts else "pending"),
            "attempts": len(attempts),
        }
        if accepted:
            words = int(accepted["user_words"]) + int(accepted["assistant_words"])
            minutes = words / SPEECH_WPM
            item.update(
                {
                    "run_dir": accepted["run_dir"],
                    "transcript_rows": accepted["transcript_rows"],
                    "prompt_tokens": accepted["prompt_tokens"],
                    "completion_tokens": accepted["completion_tokens"],
                    "cache_read_input_tokens": accepted["cache_read_input_tokens"],
                    "cache_creation_input_tokens": accepted["cache_creation_input_tokens"],
                    "thinking_tokens": accepted["thinking_tokens"],
                    "all_rows_have_tokens": accepted["all_rows_have_tokens"],
                    "user_words": accepted["user_words"],
                    "assistant_words": accepted["assistant_words"],
                    "estimated_speech_minutes_150wpm": round(minutes, 3),
                    "input_tokens_per_speech_minute": round(accepted["prompt_tokens"] / minutes),
                    "output_tokens_per_speech_minute": round(accepted["completion_tokens"] / minutes),
                    "benchmark_process_seconds": accepted["elapsed_seconds"],
                }
            )
        output.append(item)

    (HERE / "usage-results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Text-model freshness usage sample — 2026-08-06",
        "",
        f"Conversation minutes estimate actual user + assistant words at {SPEECH_WPM:.0f} spoken words/minute.",
        "Benchmark process time is not used as conversation time. Recovery calls are included when billed.",
        "",
        "| Model configuration | Provider | Status | Attempts | Rows | Input tokens | Cached input | Cache write | Output tokens | Est. speech min | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in output:
        if item["status"] == "complete":
            run = f"`{item['run_dir'].split('/')[-1]}`"
            lines.append(
                f"| {item['label']} | {item['provider']} | complete | {item['attempts']} | "
                f"{item['transcript_rows']} | {item['prompt_tokens']:,} | "
                f"{item['cache_read_input_tokens']:,} | "
                f"{item['cache_creation_input_tokens']:,} | {item['completion_tokens']:,} | "
                f"{item['estimated_speech_minutes_150wpm']:.2f} | {run} |"
            )
        else:
            lines.append(
                f"| {item['label']} | {item['provider']} | {item['status']} | "
                f"{item['attempts']} | — | — | — | — | — | — | — |"
            )
    (HERE / "usage-results.md").write_text("\n".join(lines) + "\n")

    complete = sum(item["status"] == "complete" for item in output)
    pending = len(output) - complete
    print(f"README rows={len(output)} complete={complete} pending_or_incomplete={pending}")
    return 0 if pending == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
