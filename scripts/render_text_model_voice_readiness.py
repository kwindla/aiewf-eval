#!/usr/bin/env python3
"""Render the README text-model accuracy and TTFAT range chart."""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = REPO_ROOT / "README.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs/text-model-voice-readiness.svg"


@dataclass(frozen=True)
class ModelRow:
    model: str
    pass_rate: float
    p50_ms: float
    p95_ms: float
    provider: str


def _plain_markdown(value: str) -> str:
    return value.strip().replace("**", "").replace("`", "")


def _number(value: str, suffix: str) -> float:
    cleaned = _plain_markdown(value)
    if not cleaned.endswith(suffix):
        raise ValueError(f"expected {value!r} to end in {suffix!r}")
    return float(cleaned[: -len(suffix)])


def load_rows(readme: Path) -> list[ModelRow]:
    """Read and rank the current README text-model table."""

    lines = readme.read_text(encoding="utf-8").splitlines()
    header = "| Model | Pass Rate | Any Error |"
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith(header))
    except StopIteration as exc:
        raise ValueError(f"text-model table not found in {readme}") from exc

    rows: list[ModelRow] = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 10:
            raise ValueError(f"expected 10 columns in README row: {line}")
        if re.fullmatch(r":?-+:?", cells[0]):
            continue
        rows.append(
            ModelRow(
                model=_plain_markdown(cells[0]),
                pass_rate=_number(cells[1], "%"),
                p50_ms=_number(cells[6], "ms"),
                p95_ms=_number(cells[7], "ms"),
                provider=_plain_markdown(cells[9]),
            )
        )

    if not rows:
        raise ValueError(f"text-model table is empty in {readme}")
    return sorted(rows, key=lambda row: (-row.pass_rate, row.p50_ms, row.model))


def pareto_efficient(rows: list[ModelRow]) -> set[ModelRow]:
    """Return rows not dominated on higher pass rate and lower P50."""

    return {
        row
        for row in rows
        if not any(
            other.p50_ms <= row.p50_ms
            and other.pass_rate >= row.pass_rate
            and (other.p50_ms < row.p50_ms or other.pass_rate > row.pass_rate)
            for other in rows
        )
    }


def _latency(value_ms: float) -> str:
    if value_ms < 1000:
        return f"{value_ms:.0f}ms"
    seconds = value_ms / 1000
    return f"{seconds:.1f}s" if seconds >= 3 else f"{seconds:.2f}s"


def render_svg(rows: list[ModelRow]) -> str:
    width = 1200
    left, right = 38, 34
    plot_left, plot_right = 445, width - right
    axis_top = 110
    first_row_y, row_height = 148, 41
    last_row_y = first_row_y + (len(rows) - 1) * row_height
    footnote_y = last_row_y + 56
    height = footnote_y + 42
    min_latency, max_latency = 80.0, 8000.0
    ticks = [100, 250, 500, 1000, 2000, 4000, 8000]
    voice_target = 700
    frontier = pareto_efficient(rows)

    def x(value: float) -> float:
        value = min(max(value, min_latency), max_latency)
        position = (math.log10(value) - math.log10(min_latency)) / (
            math.log10(max_latency) - math.log10(min_latency)
        )
        return plot_left + position * (plot_right - plot_left)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">Text models: TTFAT P50 and P95</title>',
        '  <desc id="desc">Text models ranked by descending pass rate and ascending median time to first answer token. A solid dot shows TTFAT P50 and an open dot shows P95 on a logarithmic latency axis. Dark dots identify speed and accuracy efficient rows.</desc>',
        "  <style>",
        "    text { fill: #292623; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-variant-numeric: tabular-nums; }",
        "    .title, .model { font-family: Georgia, 'Times New Roman', serif; }",
        "    .title { font-size: 27px; }",
        "    .subtitle { fill: #716c66; font-size: 13px; }",
        "    .axis-label { fill: #716c66; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }",
        "    .tick { fill: #88827c; font-size: 10.5px; }",
        "    .guide { stroke: #dedad5; stroke-width: .75; }",
        "    .target { stroke: #8c847c; stroke-width: 1; stroke-dasharray: 3 4; }",
        "    .target-label { fill: #736c66; font-size: 10.5px; }",
        "    .model { font-size: 14.5px; }",
        "    .meta { fill: #77716b; font-size: 10.25px; }",
        "    .range { stroke: #aaa49e; stroke-width: 1; }",
        "    .p50 { fill: #77716b; }",
        "    .p50.frontier { fill: #292623; }",
        "    .p95 { fill: #fff; stroke: #77716b; stroke-width: 1.1; }",
        "    .note { fill: #77716b; font-family: Georgia, 'Times New Roman', serif; font-size: 11.5px; font-style: italic; }",
        "  </style>",
        f'  <text class="title" x="{left}" y="38">Text models: TTFAT P50 and P95</text>',
        f'  <text class="subtitle" x="{left}" y="64">Rows ranked by pass rate, then P50 · solid dot P50 · open dot P95 · logarithmic latency scale</text>',
        f'  <text class="axis-label" x="{plot_right}" y="82" text-anchor="end">Time to first answer token</text>',
    ]

    for tick in ticks:
        px = x(tick)
        tick_label = f"{tick / 1000:.0f}s" if tick >= 1000 else _latency(tick)
        lines.extend(
            [
                f'  <line class="guide" x1="{px:.1f}" y1="{axis_top}" x2="{px:.1f}" y2="{last_row_y + 16}"/>',
                f'  <text class="tick" x="{px:.1f}" y="{axis_top - 7}" text-anchor="middle">{tick_label}</text>',
            ]
        )

    target_x = x(voice_target)
    lines.extend(
        [
            f'  <line class="target" x1="{target_x:.1f}" y1="{axis_top - 1}" x2="{target_x:.1f}" y2="{last_row_y + 16}"/>',
            f'  <text class="target-label" x="{target_x - 6:.1f}" y="{axis_top - 20}" text-anchor="end">~700ms voice guideline</text>',
        ]
    )

    for index, row in enumerate(rows):
        py = first_row_y + index * row_height
        p50_x, p95_x = x(row.p50_ms), x(row.p95_ms)
        model = html.escape(row.model)
        metadata = html.escape(
            f"{row.pass_rate:.1f}% pass · {_latency(row.p50_ms)} P50 / "
            f"{_latency(row.p95_ms)} P95 · {row.provider}"
        )
        details = html.escape(
            f"{row.model}: {row.pass_rate:.1f}% pass rate; "
            f"TTFAT {_latency(row.p50_ms)} P50 and {_latency(row.p95_ms)} P95; "
            f"{row.provider}"
        )
        point_class = "p50 frontier" if row in frontier else "p50"
        lines.extend(
            [
                f'  <text class="model" x="{left}" y="{py - 3}">{model}</text>',
                f'  <text class="meta" x="{left}" y="{py + 13}">{metadata}</text>',
                f'  <line class="range" x1="{p50_x:.1f}" y1="{py}" x2="{p95_x:.1f}" y2="{py}"/>',
                f'  <circle class="{point_class}" cx="{p50_x:.1f}" cy="{py}" r="4.3"><title>{details}</title></circle>',
                f'  <circle class="p95" cx="{p95_x:.1f}" cy="{py}" r="3.6"><title>{details}</title></circle>',
            ]
        )

    lines.extend(
        [
            f'  <text class="note" x="{left}" y="{footnote_y}">Dark P50 dots mark rows for which no model is both faster and at least as accurate.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing SVG differs from freshly rendered output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_svg(load_rows(args.readme.resolve()))
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{output} is stale; rerun {Path(__file__).name}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
