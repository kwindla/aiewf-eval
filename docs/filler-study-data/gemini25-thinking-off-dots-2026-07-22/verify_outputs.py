#!/usr/bin/env python3
"""Verify Gemini 2.5 campaign artifacts and report synchronization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from update_readme import PROVIDERS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
README = ROOT / "README.md"
MARKDOWN = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML = ROOT / "docs/filler-token-latent-scratchpad-study.html"


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = json.loads((HERE / "aggregates.json").read_text())
    protocol = payload.get("protocol", {})
    if payload.get("artifact_status") != "FINAL":
        raise ValueError("Gemini 2.5 aggregate is not final")
    if (
        protocol.get("model_order") != ["gemini25flash"]
        or protocol.get("thinking_mode") != "disabled"
        or protocol.get("thinking_budget") != 0
        or protocol.get("full_thinking_off_guaranteed") is not True
        or protocol.get("bootstrap_samples") != 100_000
    ):
        raise ValueError("Gemini 2.5 protocol mismatch")
    model = payload["models"]["gemini25flash"]
    control = model["arms"]["nofiller"]
    dots = model["arms"]["dots96"]
    effect = model["effect"]
    decision = model["adaptive_decision"]
    nc, nd = control["n_attempts"], dots["n_attempts"]
    if (
        nc not in {10, 30} or nd not in {6, 10, 30}
        or control["fixed_turn_denominator"] != 30 * nc
        or dots["fixed_turn_denominator"] != 30 * nd
        or decision.get("decision_pending") is not False
        or decision.get("action") not in {
            "stop_at_6", "stop_at_10", "focused_followup_complete",
            "control_precision_extension_complete",
        }
        or control.get("thought_tokens") != 0 or dots.get("thought_tokens") != 0
    ):
        raise ValueError("Gemini 2.5 final pool or decision mismatch")
    if model["report_tier"] == "focused" and (nc, nd) != (30, 30):
        raise ValueError("focused tier is not balanced n=30")
    if model["report_tier"] == "exploratory" and (nc, nd) == (30, 30):
        raise ValueError("balanced n=30 result is not marked focused")
    if decision.get("action") == "control_precision_extension_complete" and (
        (nc, nd) != (30, 6) or decision.get("stage1_action") != "stop_at_6"
    ):
        raise ValueError("control-only precision extension provenance mismatch")
    if not close(effect["pass_delta_points"], dots["pass_rate_pct"] - control["pass_rate_pct"]):
        raise ValueError("effect sign mismatch")

    with (HERE / "source-manifest.tsv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if len(manifest) != nc + nd or len({row["run_dir"] for row in manifest}) != nc + nd:
        raise ValueError("source manifest count or uniqueness mismatch")
    expected_dirs = set(control["run_dirs"] + dots["run_dirs"])
    if {row["run_dir"] for row in manifest} != expected_dirs:
        raise ValueError("source manifest differs from aggregate run pool")
    if any("20260722T122741_gemini-2.5-flash_2fab87f1" in row["run_dir"] for row in manifest):
        raise ValueError("excluded instrumentation smoke entered the final pool")
    file_columns = {
        "transcript.jsonl": "transcript_jsonl_sha256",
        "claude_judged.jsonl": "claude_judged_jsonl_sha256",
        "claude_summary.json": "claude_summary_json_sha256",
        "run.log": "run_log_sha256",
    }
    for row in manifest:
        run_dir = ROOT / row["run_dir"]
        for filename, column in file_columns.items():
            if digest(run_dir / filename) != row[column]:
                raise ValueError(f"frozen source changed: {run_dir / filename}")
        log = (run_dir / "run.log").read_text()
        if "thinking_budget=0 (disabled)" not in log:
            raise ValueError(f"thinking-off signature missing: {run_dir}")
        thoughts = sum(
            int((json.loads(line).get("tokens") or {}).get("thinking_tokens") or 0)
            for line in (run_dir / "transcript.jsonl").read_text().splitlines()
        )
        if thoughts:
            raise ValueError(f"thinking tokens present: {run_dir}")

    readme = README.read_text()
    readme_table = readme[readme.index("| Model | Pass Rate |"):].split("\n\n", 1)[0]
    label = model["readme_label"]
    expected_readme = (
        f"| {label} | {control['pass_rate_pct']:.1f}% | {control['any_error_rate_pct']:.1f}% | "
        f"{control['tool_error_rate_pct']:.1f}% | {control['instruction_error_rate_pct']:.1f}% | "
        f"{control['kb_error_rate_pct']:.1f}% | {round(control['ttfat_p50_ms'])}ms | "
        f"{round(control['ttfat_p95_ms'])}ms | {round(control['ttfat_max_ms'])}ms | AI Studio |"
    )
    if readme_table.count(f"| {label} |") != 1 or expected_readme not in readme_table:
        raise ValueError("README Gemini 2.5 row mismatch")
    if "| gemini-2.5-flash |" in readme_table or "| Runs |" in readme_table:
        raise ValueError("README retains provider-default row or exposes run counts")
    if "| TTFAT Max | Provider |" not in readme_table:
        raise ValueError("README provider column is missing")
    provider_rows = {}
    for row in readme_table.splitlines()[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) != 10:
            raise ValueError(f"README provider-table row width mismatch: {row}")
        provider_rows[cells[0].replace("**", "")] = cells[-1]
    if provider_rows != PROVIDERS:
        raise ValueError("README provider mapping differs from the audited 52-row map")

    markdown = MARKDOWN.read_text()
    primary = markdown.split("<!-- N30_PRIMARY_START -->", 1)[1].split("<!-- N30_PRIMARY_END -->", 1)[0]
    table_lines = [line for line in primary.splitlines() if line.startswith("|")]
    if len(table_lines) != 28:
        raise ValueError(f"Markdown primary table should have 26 model rows, found {len(table_lines)-2}")
    rows = [line for line in table_lines if line.startswith("| gemini-2.5-flash |")]
    if len(rows) != 1:
        raise ValueError("Markdown Gemini 2.5 row missing or duplicated")
    completion = f'{control["strict_completion_pct"]:.0f}% → {dots["strict_completion_pct"]:.0f}%'
    if (
        f"| {control['pass_rate_pct']:.1f} | {dots['pass_rate_pct']:.1f} |" not in rows[0]
        or completion not in rows[0]
        or f"| {round(control['ttfat_p50_ms'])} | {nc} / {nd} |" not in rows[0]
    ):
        raise ValueError("Markdown Gemini 2.5 values mismatch")
    ci_text = f"{signed(effect['pass_delta_points'])} [{signed(effect['pass_delta_ci95'][0])}, {signed(effect['pass_delta_ci95'][1])}]"
    if (model["report_tier"] == "focused") != (ci_text in rows[0]):
        raise ValueError("Markdown focused-interval display mismatch")

    html = HTML.read_text()
    if "a 26-Model Exploratory Study" not in html or "Twenty-six-model exploratory screen" not in html:
        raise ValueError("HTML report model count was not updated to 26")
    section = html.split('<section id="primary-screen">', 1)[1].split("</section>", 1)[0]
    if section.count("<tr><td>") != 26:
        raise ValueError("HTML primary table does not contain 26 rows")
    html_rows = re.findall(r"<tr><td>gemini-2\.5-flash</td>.*?</tr>", section)
    if len(html_rows) != 1 or '<td class="mut">Google</td>' not in html_rows[0] or completion not in html_rows[0]:
        raise ValueError("HTML Gemini 2.5 table row mismatch")
    figure = section.split("<figure>", 1)[1].split("</figure>", 1)[0]
    if not re.search(
        r'class="lbl">gemini-2\.5-flash</text>.*?<tspan class="provider"> · AI Studio</tspan>',
        figure,
    ):
        raise ValueError("HTML Gemini 2.5 chart provider label is missing")
    mechanism = html.split('<section id="mechanism">', 1)[1].split("</section>", 1)[0]
    if (
        mechanism.count('class="turn-cell') != 660
        or mechanism.count('class="family-contribution-cell') != 55
        or mechanism.count('class="family-contribution-total') != 11
    ):
        raise ValueError("frozen 11-model mechanism cohort changed")
    print("Gemini 2.5 campaign, README, and 26-row report verified")


if __name__ == "__main__":
    main()
