#!/usr/bin/env python3
"""Verify final README and report synchronization for the focused refresh."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
MARKDOWN = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML = ROOT / "docs/filler-token-latent-scratchpad-study.html"

FOCUSED_LABELS = (
    "gpt-5.4 (none)",
    "gpt-5.4 (none, +96 dots)",
    "gpt-5.6-terra (none)",
    "gpt-5.5 (none)",
    "gpt-5.6-sol (none)",
    "lilac/gemma-4-31b-it (thinking off)",
    "inkling (none)",
    "qwen3-8b (thinking off, BaseTen)",
    "glm-5.2 (none)",
)


def main() -> None:
    payload = json.loads((HERE / "aggregates.json").read_text())
    if payload.get("protocol", {}).get("target_per_arm") != 30:
        raise ValueError("aggregate target mismatch")
    if len(payload.get("models", {})) != 8:
        raise ValueError("aggregate model count mismatch")
    qwen = payload["models"].get("qwen3_8b", {})
    qwen_source = payload.get("protocol", {}).get("primary_sources", {}).get("qwen3_8b", {})
    if qwen.get("provider") != "BaseTen" or qwen_source != {
        "lane": "baseten-qwen",
        "provider": "BaseTen",
        "historical_attempts_included": 0,
        "openrouter_attempts_included": 0,
    }:
        raise ValueError("Qwen aggregate provenance is not BaseTen-only")

    readme = README.read_text()
    table = readme[readme.index("| Model | Pass Rate |"):].split("\n\n", 1)[0]
    if "| Runs |" in table:
        raise ValueError("README exposes a run-count column")
    if any(len(row.strip("|").split("|")) != 10 for row in table.splitlines()[2:]):
        raise ValueError("README row width mismatch")
    for label in FOCUSED_LABELS:
        if table.count(f"| {label} |") != 1:
            raise ValueError(f"README focused row missing or duplicated: {label}")

    markdown = MARKDOWN.read_text()
    primary = markdown.split("<!-- N30_PRIMARY_START -->", 1)[1].split("<!-- N30_PRIMARY_END -->", 1)[0]
    table_lines = [line for line in primary.splitlines() if line.startswith("|")]
    if len(table_lines) != 28:
        raise ValueError(f"Markdown primary table should have 26 data rows, found {len(table_lines) - 2}")
    if "nemotron-super" in "\n".join(table_lines).lower():
        raise ValueError("Markdown primary table includes the separate Nemotron configuration contrast")
    if "p-value" in primary.lower() or "cluster p" in primary.lower() or "bonferroni" in primary.lower():
        raise ValueError("Markdown primary screen contains prohibited inference language")
    qwen_markdown_rows = [line for line in table_lines if line.startswith("| qwen3-8b |")]
    if len(qwen_markdown_rows) != 1 or "| BaseTen |" not in qwen_markdown_rows[0]:
        raise ValueError("Markdown primary Qwen row is not labeled BaseTen")
    if "| qwen3-8b | OpenRouter |" in primary:
        raise ValueError("Markdown primary Qwen row still names OpenRouter")
    qwen_ttfat = round(qwen["arms"]["nofiller"]["ttfat_p50_ms"])
    if f"| {qwen_ttfat} |" not in qwen_markdown_rows[0]:
        raise ValueError("Markdown primary Qwen row does not use BaseTen no-filler TTFAT")
    for required in (
        "## Filler effects at two reasoning-effort settings",
        "| `none` | 90.2% | 95.2% | +5.0 [+3.0, +6.9] | 30 | 689 → 694 ms |",
        "| `low` | 96.2% | 99.6% | +3.3 [+1.2, +5.0] | 8 | 1,091 → 1,131 ms |",
        "not an\ninteraction estimate",
    ):
        if required not in markdown:
            raise ValueError(f"Markdown reasoning-effort section is missing: {required}")

    html = HTML.read_text()
    section = html.split('<h2><span class="no">3</span>', 1)[1].split('<h2><span class="no">4</span>', 1)[0]
    if section.count("<tr><td>") != 26:
        raise ValueError(f"HTML primary table should have 26 rows, found {section.count('<tr><td>')}")
    lowered = section.lower()
    if "p=" in lowered or "cluster-p" in lowered or "bonferroni" in lowered:
        raise ValueError("HTML Section 3 contains a p-value or Bonferroni reference")
    qwen_html = re.findall(r"<tr><td>qwen3-8b</td>.*?</tr>", section)
    if len(qwen_html) != 1 or '<td class="mut">BaseTen</td>' not in qwen_html[0]:
        raise ValueError("HTML primary Qwen row is not labeled BaseTen")
    if '<td class="mut">OpenRouter</td>' in qwen_html[0]:
        raise ValueError("HTML primary Qwen row still names OpenRouter")
    if f'<td class="r mut">{qwen_ttfat}</td>' not in qwen_html[0]:
        raise ValueError("HTML primary Qwen row does not use BaseTen no-filler TTFAT")
    for required in (
        "row-config P50 TTFAT",
        "95% CI, focused",
    ):
        if required not in section:
            raise ValueError(f"HTML Section 3 is missing: {required}")
    figure_and_table = section.split('<p class="measure"><b>Focused n=30 estimates.</b>', 1)[0]
    if "nemotron-super" in figure_and_table.lower():
        raise ValueError("HTML Figure 1 or its table includes the separate Nemotron configuration contrast")
    if "+2.93 points (95% CI +1.09 to +4.77)" not in html:
        raise ValueError("completed GPT-5.4 dash confirmation is missing")
    for required in (
        "Filler effects at two reasoning-effort settings",
        "fig 5 · gpt-5.4 · two parallel filler comparisons",
        "effort none · n=30/arm",
        "effort low · n=8/arm",
        "689 → 694 ms",
        "1091 → 1131 ms",
        "−1.7 is not an interaction estimate",
    ):
        if required not in html:
            raise ValueError(f"HTML reasoning-effort section is missing: {required}")
    section_numbers = re.findall(r'<h2><span class="no">(\d+)</span>', html)
    if section_numbers != [str(number) for number in range(1, 11)]:
        raise ValueError(f"HTML section numbering mismatch: {section_numbers}")
    print("README and report outputs verified")


if __name__ == "__main__":
    main()
