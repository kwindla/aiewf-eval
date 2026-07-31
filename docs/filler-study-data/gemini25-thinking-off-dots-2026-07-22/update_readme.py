#!/usr/bin/env python3
"""Replace the historical Gemini 2.5 Flash row with the thinking-off result."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
NO_PROVIDER_START = "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max |"
MIDDLE_PROVIDER_START = "| Model | Provider | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max |"
START = "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"
PROVIDERS = {
    "nemotron-3-ultra (128)": "Modal",
    "claude-sonnet-4-6": "Anthropic",
    "claude-fable-5 (low)": "Anthropic",
    "claude-fable-5 (default)": "Anthropic",
    "glm-5.2 (none)": "BaseTen",
    "qwen3.5-27b (thinking)": "Modal",
    "glm-5 (thinking)": "Modal",
    "nemotron-3-ultra (96)": "Modal",
    "kimi-k2.6 Cerebras (thinking)": "Cerebras",
    "claude-haiku-4-5": "Anthropic",
    "gpt-5.1": "OpenAI",
    "gpt-5.6-terra (medium)": "OpenAI",
    "lilac/gemma-4-31b-it (thinking off)": "Lilac",
    "gpt-5.5 (none)": "OpenAI",
    "gemini-3.6-flash (minimal)": "AI Studio",
    "gpt-5.4 (low)": "OpenAI",
    "nemotron-3-super-120b (512)": "Modal",
    "gemini-3.1-flash-lite-preview": "AI Studio",
    "gpt-5.6-sol (none)": "OpenAI",
    "gpt-4.1": "OpenAI",
    "zai-org/glm-5.1": "Lilac",
    "gpt-5.4 (none, +96 dots)": "OpenAI",
    "qwen3.5-4b (thinking)": "Modal",
    "inkling (none)": "BaseTen",
    "gpt-4o": "OpenAI",
    "qwen3.5-9b (thinking)": "Modal",
    "qwen3.5-27b": "Modal",
    "kimi-k2.6 Cerebras (instant)": "Cerebras",
    "gemini-3.5-flash (minimal)": "AI Studio",
    "claude-sonnet-5": "Anthropic",
    "gemini-2.5-flash (thinking off)": "AI Studio",
    "gpt-5.6-terra (none)": "OpenAI",
    "gpt-5.4-mini (medium)": "OpenAI",
    "nemotron-3-nano-30b (512)": "Modal",
    "nova-2-pro-preview": "AWS Bedrock",
    "gpt-5.4 (none)": "OpenAI",
    "qwen3.5-9b": "Modal",
    "gpt-5.2": "OpenAI",
    "qwen3.5-4b": "Modal",
    "gpt-5.6-luna (none)": "OpenAI",
    "gpt-oss-120b (groq)": "Groq",
    "poolside/laguna-s-2.1 (thinking off)": "OpenRouter",
    "gpt-4.1-mini": "OpenAI",
    "glm-4.7-flash": "Modal",
    "gpt-5-mini": "OpenAI",
    "gpt-5.4-mini (none)": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "qwen3-8b (thinking off, BaseTen)": "BaseTen",
    "gemini-3.5-flash-lite (minimal)": "AI Studio",
    "qwen3.6-27b (thinking off)": "BaseTen",
    "qwen3.6-35b-a3b (thinking off, FP8)": "BaseTen",
    "gemma-4-26b-a4b-it (thinking off)": "BaseTen",
}


def clean_label(row: str) -> str:
    return row.split("|")[1].strip().replace("**", "")


def pass_rate(row: str) -> float:
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    index = 1 if cells[1].replace("**", "").endswith("%") else 2
    return float(cells[index].strip().replace("**", "").rstrip("%"))


def with_provider(row: str) -> str:
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    label = cells[0].replace("**", "")
    if label not in PROVIDERS:
        raise ValueError(f"README provider is not mapped: {label}")
    if len(cells) == 9:
        pass
    elif len(cells) == 10:
        if cells[1] == PROVIDERS[label]:
            cells.pop(1)
        elif cells[-1] == PROVIDERS[label]:
            cells.pop()
        else:
            raise ValueError(f"unexpected README provider placement for {label}")
    else:
        raise ValueError(f"unexpected README row width for {label}: {len(cells)}")
    cells.append(PROVIDERS[label])
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    payload = json.loads((HERE / "aggregates.json").read_text())
    if payload.get("artifact_status") != "FINAL":
        raise ValueError("Gemini 2.5 aggregate is not final")
    protocol = payload.get("protocol", {})
    if (
        protocol.get("thinking_mode") != "disabled"
        or protocol.get("thinking_budget") != 0
        or protocol.get("full_thinking_off_guaranteed") is not True
    ):
        raise ValueError("Gemini 2.5 aggregate is not explicitly thinking-off")
    model = payload["models"]["gemini25flash"]
    arm = model["arms"]["nofiller"]
    if arm["n_attempts"] not in {10, 30} or arm["fixed_turn_denominator"] != 30 * arm["n_attempts"]:
        raise ValueError("invalid no-filler pool")
    values = (
        model["readme_label"], f'{arm["pass_rate_pct"]:.1f}%', f'{arm["any_error_rate_pct"]:.1f}%',
        f'{arm["tool_error_rate_pct"]:.1f}%', f'{arm["instruction_error_rate_pct"]:.1f}%',
        f'{arm["kb_error_rate_pct"]:.1f}%', f'{round(arm["ttfat_p50_ms"])}ms',
        f'{round(arm["ttfat_p95_ms"])}ms', f'{round(arm["ttfat_max_ms"])}ms', "AI Studio",
    )
    replacement = "| " + " | ".join(values) + " |"
    text = README.read_text()
    table_start = next(
        header for header in (START, MIDDLE_PROVIDER_START, NO_PROVIDER_START) if header in text
    )
    start = text.index(table_start)
    end = text.index("\n\n", start)
    lines = text[start:end].splitlines()
    rows = lines[2:]
    old_labels = {"gemini-2.5-flash", "gemini-2.5-flash (thinking off)"}
    kept = [row for row in rows if clean_label(row) not in old_labels]
    kept.append(replacement)
    kept = [with_provider(row) for row in kept]
    kept.sort(key=pass_rate, reverse=True)
    header = [START, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    README.write_text(text[:start] + "\n".join(header + kept) + text[end:])
    updated = README.read_text()[start:].split("\n\n", 1)[0]
    if updated.count(f'| {model["readme_label"]} |') != 1 or replacement not in updated:
        raise ValueError("README Gemini 2.5 update verification failed")
    if "| gemini-2.5-flash |" in updated:
        raise ValueError("historical provider-default Gemini 2.5 row remains")
    print("README Gemini 2.5 thinking-off row updated and verified")


if __name__ == "__main__":
    main()
