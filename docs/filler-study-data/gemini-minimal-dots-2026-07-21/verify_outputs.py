#!/usr/bin/env python3
"""Verify final Gemini aggregates, README rows, and report synchronization."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
MARKDOWN = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML = ROOT / "docs/filler-token-latent-scratchpad-study.html"


def main() -> None:
    payload = json.loads((HERE / "aggregates.json").read_text())
    if payload.get("artifact_status") != "FINAL":
        raise ValueError("Gemini aggregate is not final")
    protocol = payload.get("protocol", {})
    order = protocol.get("model_order")
    if protocol.get("thinking_mode") != "minimal" or protocol.get("full_thinking_off_guaranteed") is not False:
        raise ValueError("Gemini reasoning-setting provenance mismatch")
    if order != ["gemini35flash", "gemini35flashlite", "gemini36flash"]:
        raise ValueError("Gemini model order mismatch")

    readme = README.read_text()
    readme_table = readme[readme.index("| Model | Pass Rate |"):].split("\n\n", 1)[0]
    if "| Runs |" in readme_table:
        raise ValueError("README exposes a run-count column")
    markdown = MARKDOWN.read_text()
    primary = markdown.split("<!-- N30_PRIMARY_START -->", 1)[1].split("<!-- N30_PRIMARY_END -->", 1)[0]
    table_lines = [line for line in primary.splitlines() if line.startswith("|")]
    if len(table_lines) != 28:
        raise ValueError(f"Markdown primary table should have 26 rows, found {len(table_lines) - 2}")
    html = HTML.read_text()
    section = html.split('<h2><span class="no">3</span>', 1)[1].split('<h2><span class="no">4</span>', 1)[0]
    if section.count("<tr><td>") != 26:
        raise ValueError(f"HTML primary table should have 26 rows, found {section.count('<tr><td>')}")
    if "p=" in section.lower() or "bonferroni" in section.lower():
        raise ValueError("HTML primary screen contains prohibited p-value language")

    for key in order:
        model = payload["models"][key]
        control = model["arms"]["nofiller"]
        dots = model["arms"]["dots96"]
        decision = model["adaptive_decision"]
        if decision.get("decision_pending") is not False:
            raise ValueError(f"adaptive decision remains pending for {key}")
        if control["n_attempts"] not in {10, 30} or control["fixed_turn_denominator"] != 30 * control["n_attempts"]:
            raise ValueError(f"invalid control pool for {key}")
        if dots["n_attempts"] not in {6, 10, 30} or dots["fixed_turn_denominator"] != 30 * dots["n_attempts"]:
            raise ValueError(f"invalid dots pool for {key}")
        label = model["readme_label"]
        if readme_table.count(f"| {label} |") != 1:
            raise ValueError(f"README Gemini row missing or duplicated: {label}")
        expected_readme = (
            f"| {label} | {control['pass_rate_pct']:.1f}% | {control['any_error_rate_pct']:.1f}% | "
            f"{control['tool_error_rate_pct']:.1f}% | {control['instruction_error_rate_pct']:.1f}% | "
            f"{control['kb_error_rate_pct']:.1f}% | {round(control['ttfat_p50_ms'])}ms | "
            f"{round(control['ttfat_p95_ms'])}ms | {round(control['ttfat_max_ms'])}ms | AI Studio |"
        )
        if expected_readme not in readme_table:
            raise ValueError(f"README aggregate mismatch for {label}")
        name = model["display_name"]
        markdown_rows = [line for line in table_lines if line.startswith(f"| {name} |")]
        if len(markdown_rows) != 1 or "| Google |" not in markdown_rows[0]:
            raise ValueError(f"Markdown Gemini row mismatch for {name}")
        if f"| {round(control['ttfat_p50_ms'])} |" not in markdown_rows[0]:
            raise ValueError(f"Markdown Gemini TTFAT mismatch for {name}")
        html_rows = re.findall(fr"<tr><td>{re.escape(name)}</td>.*?</tr>", section)
        if len(html_rows) != 1 or '<td class="mut">Google</td>' not in html_rows[0]:
            raise ValueError(f"HTML Gemini row mismatch for {name}")

    for text, name in ((readme, "README"), (markdown, "Markdown"), (html, "HTML")):
        if "does not guarantee" not in text or "minimal" not in text:
            raise ValueError(f"{name} omits the Gemini minimal-thinking caveat")
    sensitivity = json.loads((HERE / "idle-timeout-sensitivity.json").read_text())
    if sensitivity.get("artifact_status") != "SENSITIVITY_ONLY_NOT_PRIMARY":
        raise ValueError("Flash Lite idle-timeout result is not marked sensitivity-only")
    primary_lite = payload["models"]["gemini35flashlite"]["arms"]["nofiller"]
    sensitivity_primary = sensitivity.get("primary_attempt_based", {})
    if (
        sensitivity_primary.get("pass_rate_pct") != primary_lite["pass_rate_pct"]
        or sensitivity_primary.get("strict_completion_pct") != primary_lite["strict_completion_pct"]
    ):
        raise ValueError("Flash Lite sensitivity does not preserve the published primary estimate")
    for text, name in ((markdown, "Markdown"), (html, "HTML")):
        if "Flash Lite attempt-policy sensitivity" not in text or "primary estimates remain" not in text.lower():
            raise ValueError(f"{name} omits the Flash Lite attempt-policy sensitivity")
    print("Gemini campaign, README, and 26-row report outputs verified")


if __name__ == "__main__":
    main()
