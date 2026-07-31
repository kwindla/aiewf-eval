#!/usr/bin/env python3
"""Verify the turn-family decomposition and its report integration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MARKDOWN = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML = ROOT / "docs/filler-token-latent-scratchpad-study.html"
N30 = HERE.parent / "dot-stability-n30-2026-07-20/aggregates.json"
GEMINI = HERE.parent / "gemini-minimal-dots-2026-07-21/aggregates.json"
SOURCE_HASHES = {
    N30: "573e53779774f61c8cc9641d553c02c2368c56a2785fddc87071cdb5c22a1d99",
    GEMINI: "41be324032aaecffd03b3e43ffa35242a3e9b19c82c404093138f5905a7ecff2",
}
FROZEN_INPUT_SHA256 = {
    "benchmarks/_shared/turns.py": "c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b",
    "benchmarks/aiwf_medium_context/prompts/system.py": "6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6",
    "docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py": "3d9094da5c9858554baf9760eec9bbba786e71f72de64e362cfde9ce814dfe70",
    "docs/filler-study-data/dot-stability-n30-2026-07-20/aggregates.json": "573e53779774f61c8cc9641d553c02c2368c56a2785fddc87071cdb5c22a1d99",
    "docs/filler-study-data/gemini-minimal-dots-2026-07-21/analyze.py": "aa7b2ed23cb5cb5ec612626f3aa788f85d6f8b5af286c8eed991ae165ab8d2ee",
    "docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json": "41be324032aaecffd03b3e43ffa35242a3e9b19c82c404093138f5905a7ecff2",
    "docs/filler-study-data/turn-family-secondary-2026-07-22/turn-families.json": "058ea3ada0d087ddd2afad5a4d02b9120e5a14c2a28e6ab952fdadc67f3e946e",
    "docs/filler-study-data/turn-family-secondary-2026-07-22/source-manifest.tsv": "5a0a222e8f0d4f5e4297ff618407d2dab2416feea5850c8cce4362e741b1fffb",
}
EXPECTED_ORDER = [
    "gpt54", "terra", "gpt55", "sol", "gemma431", "inkling", "qwen3_8b", "glm52",
    "gemini35flash", "gemini35flashlite", "gemini36flash",
]


class ReportDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.turn_wrappers: list[dict[str, str]] = []
        self.turn_cells: list[dict[str, str]] = []
        self.contribution_cells: list[dict[str, str]] = []
        self.contribution_totals: list[dict[str, str]] = []
        self.family_cells: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "g" and "turn-heatmap" in classes:
            self.turn_wrappers.append(values)
        elif tag == "rect" and "turn-cell" in classes:
            self.turn_cells.append(values)
        elif tag == "div" and "family-contribution-cell" in classes:
            self.contribution_cells.append(values)
        elif tag == "div" and "family-contribution-total" in classes:
            self.contribution_totals.append(values)
        elif tag == "div" and "family-cell" in classes:
            self.family_cells.append(values)


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)


def main() -> None:
    for path, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen parent aggregate changed: {path}")

    taxonomy = json.loads((HERE / "turn-families.json").read_text())
    aggregate = json.loads((HERE / "aggregates.json").read_text())
    protocol = aggregate.get("protocol", {})
    if aggregate.get("artifact_status") != "FINAL_EXPLORATORY_SECONDARY":
        raise ValueError("turn-family aggregate is not final and exploratory")
    if (
        aggregate.get("schema_version") != 2
        or
        protocol.get("model_order") != EXPECTED_ORDER
        or protocol.get("n_per_arm") != 30
        or protocol.get("bootstrap_unit") != "whole conversation"
        or protocol.get("bootstrap_samples") != 100_000
        or protocol.get("primary_estimand_unchanged") is not True
        or protocol.get("input_sha256") != FROZEN_INPUT_SHA256
    ):
        raise ValueError("turn-family protocol mismatch")
    if aggregate.get("taxonomy") != taxonomy:
        raise ValueError("embedded taxonomy differs from the frozen taxonomy file")
    families = taxonomy.get("families", [])
    family_keys = [row["key"] for row in families]
    family_by_turn = {
        turn: family["key"] for family in families for turn in family["turns"]
    }
    turns = [turn for row in families for turn in row["turns"]]
    if len(family_keys) != 5 or len(set(family_keys)) != 5 or sorted(turns) != list(range(30)):
        raise ValueError("taxonomy is not a five-family disjoint partition of turns 0-29")
    with (HERE / "source-manifest.tsv").open(newline="") as handle:
        source_manifest = list(csv.DictReader(handle, delimiter="\t"))
    if len(source_manifest) != 660 or len({row["run_dir"] for row in source_manifest}) != 660:
        raise ValueError("source manifest does not contain 660 unique conversations")
    for row in source_manifest:
        run_dir = ROOT / row["run_dir"]
        for filename, column in (("transcript.jsonl", "transcript_sha256"), ("claude_judged.jsonl", "judgment_sha256")):
            actual = hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()
            if actual != row[column]:
                raise ValueError(f"source content changed: {run_dir / filename}")

    n30 = json.loads(N30.read_text())
    gemini = json.loads(GEMINI.read_text())
    source_by_key = {
        **{key: row for key, row in n30["models"].items()},
        **{key: row for key, row in gemini["models"].items()},
    }
    models = aggregate.get("models", {})
    if list(models) != EXPECTED_ORDER or set(source_by_key) != set(EXPECTED_ORDER):
        raise ValueError("turn-family model set/order mismatch")

    for key in EXPECTED_ORDER:
        model = models[key]
        source = source_by_key[key]
        if model["display_name"] != source["display_name"]:
            raise ValueError(f"display-name mismatch for {key}")
        if model.get("families", {}).keys() != dict.fromkeys(family_keys).keys():
            raise ValueError(f"family order mismatch for {key}")
        for arm in ("nofiller", "dots96"):
            if not close(model["overall"][f"{arm}_pass_rate_pct"], source["arms"][arm]["pass_rate_pct"]):
                raise ValueError(f"overall arm mismatch for {key}/{arm}")
        if (
            not close(model["overall"]["delta_points"], source["effect"]["pass_delta_points"])
            or model["overall"]["published_delta_ci95"] != source["effect"]["pass_delta_ci95"]
        ):
            raise ValueError(f"published overall effect changed for {key}")

        turn_rows = model.get("turns", [])
        if len(turn_rows) != 30 or [row.get("turn") for row in turn_rows] != list(range(30)):
            raise ValueError(f"per-turn rows missing or out of order for {key}")
        turn_pass_sums = {"nofiller": 0, "dots96": 0}
        for turn_row in turn_rows:
            turn = turn_row["turn"]
            if (
                turn_row.get("family_key") != family_by_turn[turn]
                or turn_row.get("n_conversations_per_arm") != 30
                or turn_row.get("fixed_turn_denominator_per_arm") != 30
            ):
                raise ValueError(f"turn metadata mismatch for {key}/{turn}")
            for arm in ("nofiller", "dots96"):
                passed = turn_row[f"{arm}_pass_count"]
                missing = turn_row[f"{arm}_missing_turn_count"]
                observed_failure = turn_row[f"{arm}_observed_failure_count"]
                if (
                    not all(isinstance(value, int) for value in (passed, missing, observed_failure))
                    or passed + missing + observed_failure != 30
                ):
                    raise ValueError(f"turn count decomposition mismatch for {key}/{turn}/{arm}")
                for count_key, rate_key in (
                    (f"{arm}_pass_count", f"{arm}_pass_rate_pct"),
                    (f"{arm}_missing_turn_count", f"{arm}_missing_turn_rate_pct"),
                    (f"{arm}_observed_failure_count", f"{arm}_observed_failure_rate_pct"),
                ):
                    if not close(turn_row[rate_key], 100 * turn_row[count_key] / 30):
                        raise ValueError(f"turn count/rate mismatch for {key}/{turn}/{rate_key}")
                turn_pass_sums[arm] += passed
            pass_delta = turn_row["dots96_pass_rate_pct"] - turn_row["nofiller_pass_rate_pct"]
            missing_delta = (
                turn_row["dots96_missing_turn_rate_pct"]
                - turn_row["nofiller_missing_turn_rate_pct"]
            )
            observed_failure_delta = (
                turn_row["dots96_observed_failure_rate_pct"]
                - turn_row["nofiller_observed_failure_rate_pct"]
            )
            if (
                not close(pass_delta, turn_row["pass_delta_points"])
                or not close(missing_delta, turn_row["missing_turn_rate_delta_points"])
                or not close(-missing_delta, turn_row["aligned_missing_contribution_points"])
                or not close(observed_failure_delta, turn_row["observed_failure_rate_delta_points"])
                or not close(-observed_failure_delta, turn_row["aligned_observed_failure_contribution_points"])
                or not close(
                    pass_delta,
                    turn_row["aligned_missing_contribution_points"]
                    + turn_row["aligned_observed_failure_contribution_points"],
                )
                or turn_row["pass_delta_ci95"][0] > turn_row["pass_delta_ci95"][1]
            ):
                raise ValueError(f"turn effect decomposition mismatch for {key}/{turn}")
        for arm in ("nofiller", "dots96"):
            if turn_pass_sums[arm] != source["arms"][arm]["pass_count"]:
                raise ValueError(f"turn pass counts do not reconcile for {key}/{arm}")
        if not close(
            sum(row["pass_delta_points"] for row in turn_rows) / 30,
            source["effect"]["pass_delta_points"],
        ):
            raise ValueError(f"turn effects do not reconcile overall for {key}")

        pass_sums = {"nofiller": 0, "dots96": 0}
        denominator_sum = 0
        contribution_sum = 0.0
        for family in families:
            row = model["families"][family["key"]]
            n_turns = len(family["turns"])
            denominator = 30 * n_turns
            if (
                row["turns"] != family["turns"]
                or row["n_turns"] != n_turns
                or row["n_conversations_per_arm"] != 30
                or row["fixed_turn_denominator_per_arm"] != denominator
                or not close(row["turn_weight"], n_turns / 30)
            ):
                raise ValueError(f"family denominator mismatch for {key}/{family['key']}")
            denominator_sum += denominator
            for arm in ("nofiller", "dots96"):
                count = row[f"{arm}_pass_count"]
                rate = row[f"{arm}_pass_rate_pct"]
                pass_sums[arm] += count
                if not close(rate, 100 * count / denominator):
                    raise ValueError(f"family rate/count mismatch for {key}/{family['key']}/{arm}")
                missing = row[f"{arm}_missing_turn_rate_pct"]
                observed_failure = row[f"{arm}_observed_failure_rate_pct"]
                if not close(rate + missing + observed_failure, 100):
                    raise ValueError(f"failure decomposition mismatch for {key}/{family['key']}/{arm}")
                member_rows = [turn_rows[turn] for turn in family["turns"]]
                if (
                    count != sum(member[f"{arm}_pass_count"] for member in member_rows)
                    or not close(missing, sum(member[f"{arm}_missing_turn_count"] for member in member_rows) * 100 / denominator)
                    or not close(observed_failure, sum(member[f"{arm}_observed_failure_count"] for member in member_rows) * 100 / denominator)
                ):
                    raise ValueError(f"turn/family reconciliation mismatch for {key}/{family['key']}/{arm}")
            delta = row["dots96_pass_rate_pct"] - row["nofiller_pass_rate_pct"]
            if not close(delta, row["conditional_delta_points"]):
                raise ValueError(f"effect sign mismatch for {key}/{family['key']}")
            if not close(
                row["missing_turn_rate_delta_points"],
                row["dots96_missing_turn_rate_pct"] - row["nofiller_missing_turn_rate_pct"],
            ):
                raise ValueError(f"missing-turn effect sign mismatch for {key}/{family['key']}")
            expected_contribution = delta * n_turns / 30
            if not close(row["overall_contribution_points"], expected_contribution):
                raise ValueError(f"contribution mismatch for {key}/{family['key']}")
            if row["conditional_delta_ci95"][0] > row["conditional_delta_ci95"][1]:
                raise ValueError(f"reversed interval for {key}/{family['key']}")
            contribution_sum += row["overall_contribution_points"]
            turn_family_delta = sum(turn_rows[turn]["pass_delta_points"] for turn in family["turns"]) / n_turns
            if not close(turn_family_delta, delta):
                raise ValueError(f"turn effects do not reconcile with family for {key}/{family['key']}")
        if denominator_sum != 900:
            raise ValueError(f"family denominators do not sum to 900 for {key}")
        for arm in ("nofiller", "dots96"):
            if pass_sums[arm] != source["arms"][arm]["pass_count"]:
                raise ValueError(f"family pass counts do not reconcile for {key}/{arm}")
        if not close(contribution_sum, source["effect"]["pass_delta_points"]):
            raise ValueError(f"family contributions do not reconcile for {key}")

    markdown = MARKDOWN.read_text()
    if markdown.count("<!-- TURN_FAMILY_START -->") != 1 or markdown.count("<!-- TURN_FAMILY_END -->") != 1:
        raise ValueError("Markdown turn-family block markers are missing or duplicated")
    block = markdown.split("<!-- TURN_FAMILY_START -->", 1)[1].split("<!-- TURN_FAMILY_END -->", 1)[0]
    if markdown.count("<!-- TURN_FAMILY_INSERT -->") != 1:
        raise ValueError("Markdown family insertion marker is missing or duplicated")
    primary_markdown = markdown.split("<!-- N30_PRIMARY_START -->", 1)[1].split("<!-- N30_PRIMARY_END -->", 1)[0]
    if "TURN_FAMILY" in primary_markdown:
        raise ValueError("Markdown secondary analysis remains inside the primary screen")
    mechanism_heading = "## Where effects occur: exploratory turn and task-family analysis"
    mechanism_start = markdown.index(mechanism_heading)
    insertion = markdown.index("<!-- TURN_FAMILY_INSERT -->")
    block_start = markdown.index("<!-- TURN_FAMILY_START -->")
    block_end = markdown.index("<!-- TURN_FAMILY_END -->")
    next_heading = markdown.find("\n## ", block_end)
    if not (
        mechanism_start < insertion < block_start < block_end
        and (next_heading == -1 or block_end < next_heading)
    ):
        raise ValueError("Markdown secondary block is outside the mechanism section")
    table_lines = [line for line in block.splitlines() if line.startswith("|")]
    if len(table_lines) != 13:
        raise ValueError(f"Markdown family table should have 11 model rows, found {len(table_lines) - 2}")
    for key in EXPECTED_ORDER:
        model = models[key]
        rows = [line for line in table_lines if line.startswith(f'| {model["display_name"]} |')]
        if len(rows) != 1:
            raise ValueError(f"Markdown family row missing or duplicated for {key}")
        for family_key in family_keys:
            effect = model["families"][family_key]
            low, high = effect["conditional_delta_ci95"]
            expected = f"{signed(effect['conditional_delta_points'])} [{signed(low)}, {signed(high)}]"
            if expected not in rows[0]:
                raise ValueError(f"Markdown family effect mismatch for {key}/{family_key}")

    html = HTML.read_text()
    if html.count("<!-- TURN_FAMILY_HTML_START -->") != 1 or html.count("<!-- TURN_FAMILY_HTML_END -->") != 1:
        raise ValueError("HTML turn-family markers are missing or duplicated")
    primary_html = html.split('<section id="primary-screen">', 1)[1].split("</section>", 1)[0]
    if any(token in primary_html for token in ("turn-family-effects", "turn-heatmap", "family-contribution-cell")):
        raise ValueError("HTML secondary analysis remains inside the primary screen")
    mechanism_html = html.split('<section id="mechanism">', 1)[1].split("</section>", 1)[0]
    secondary = html.split("<!-- TURN_FAMILY_HTML_START -->", 1)[1].split("<!-- TURN_FAMILY_HTML_END -->", 1)[0]
    if secondary not in mechanism_html:
        raise ValueError("HTML secondary analysis is outside the mechanism section")
    parser = ReportDataParser()
    parser.feed(secondary)
    if len(parser.turn_wrappers) != 2 or {row.get("data-panel") for row in parser.turn_wrappers} != {"pass", "missing"}:
        raise ValueError("HTML must contain two distinct turn-heatmap panels")
    if any(row.get("data-color-cap") != "50.0" for row in parser.turn_wrappers):
        raise ValueError("HTML turn heatmaps do not share the fixed ±50-point scale")
    if len(parser.turn_cells) != 660:
        raise ValueError(f"HTML turn heatmaps must contain 660 cells, found {len(parser.turn_cells)}")
    turn_cell_by_key = {}
    for cell in parser.turn_cells:
        marker = (cell.get("data-panel"), cell.get("data-model-key"), int(cell.get("data-turn", "-1")))
        if marker in turn_cell_by_key:
            raise ValueError(f"duplicate HTML turn cell: {marker}")
        turn_cell_by_key[marker] = cell
    for key in EXPECTED_ORDER:
        model = models[key]
        for turn_row in model["turns"]:
            turn = turn_row["turn"]
            for panel, expected_value in (
                ("pass", turn_row["pass_delta_points"]),
                ("missing", turn_row["aligned_missing_contribution_points"]),
            ):
                marker = (panel, key, turn)
                cell = turn_cell_by_key.get(marker)
                if (
                    cell is None
                    or cell.get("data-family") != turn_row["family_key"]
                    or cell.get("data-color-cap") != "50.0"
                    or not close(float(cell["data-value"]), expected_value)
                ):
                    raise ValueError(f"HTML turn cell mismatch for {marker}")
            low, high = turn_row["pass_delta_ci95"]
            expected_pass_title = (
                f'pass: no filler {turn_row["nofiller_pass_rate_pct"]:.1f}%, '
                f'dots {turn_row["dots96_pass_rate_pct"]:.1f}%, '
                f'delta {turn_row["pass_delta_points"]:+.1f} points; '
                f'pointwise 95% CI [{low:+.1f}, {high:+.1f}]'
            )
            expected_missing_title = (
                f'missing: no filler {turn_row["nofiller_missing_turn_rate_pct"]:.1f}%, '
                f'dots {turn_row["dots96_missing_turn_rate_pct"]:.1f}%; '
                f'benefit-aligned contribution {turn_row["aligned_missing_contribution_points"]:+.1f} points'
            )
            if expected_pass_title not in secondary or expected_missing_title not in secondary:
                raise ValueError(f"HTML turn tooltip mismatch for {key}/{turn}")

    if len(parser.contribution_cells) != 55 or len(parser.contribution_totals) != 11:
        raise ValueError("HTML contribution matrix must contain 55 family cells and 11 totals")
    contribution_by_key = {
        (row.get("data-model-key"), row.get("data-family")): row
        for row in parser.contribution_cells
    }
    if len(contribution_by_key) != 55:
        raise ValueError("HTML contribution cells are duplicated")
    for key in EXPECTED_ORDER:
        for family_key in family_keys:
            row = contribution_by_key.get((key, family_key))
            expected_value = models[key]["families"][family_key]["overall_contribution_points"]
            if row is None or not close(float(row["data-value"]), expected_value):
                raise ValueError(f"HTML contribution mismatch for {key}/{family_key}")

    if len(parser.family_cells) != 55:
        raise ValueError("HTML family matrix must contain 55 cells")
    for key in EXPECTED_ORDER:
        name = models[key]["display_name"]
        for family_key in family_keys:
            marker = f'data-model="{name}" data-family="{family_key}"'
            if secondary.count(marker) != 1:
                raise ValueError(f"HTML family cell missing or duplicated for {key}/{family_key}")
            effect = models[key]["families"][family_key]
            low, high = effect["conditional_delta_ci95"]
            content = secondary.split(marker, 1)[1].split("</div>", 1)[0]
            expected = (
                f'<b>{signed(effect["conditional_delta_points"])}</b>'
                f'<span>[{signed(low)}, {signed(high)}]</span>'
            )
            if expected not in content:
                raise ValueError(f"HTML family values mismatch for {key}/{family_key}")
    required = (
        "retrospective and exploratory",
        "not treatment-by-turn or treatment-by-family interaction tests",
        "pointwise, unadjusted 95%",
        "55 intervals are not",
        "Fig 1 remains the primary inferential summary",
    )
    if any(text not in secondary for text in required):
        raise ValueError("HTML secondary-analysis caveat is incomplete")
    lowered = secondary.lower()
    if "cluster-p=" in lowered or "significant" in lowered or "bonferroni" in lowered:
        raise ValueError("HTML secondary analysis contains prohibited screening language")
    print("Turn-family aggregate and report integration verified")


if __name__ == "__main__":
    main()
