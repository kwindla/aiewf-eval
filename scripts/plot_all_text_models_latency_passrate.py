"""Plot every text-model row in README.md by pass rate and TTFAT P50.

The README table is the source of truth; model data is deliberately not duplicated
in this script. Lower latency and higher pass rate are better. Pareto-efficient
configurations are marked with an extra black ring.

Run:
    uv run --with matplotlib --with adjustText \
        python scripts/plot_all_text_models_latency_passrate.py

Output:
    docs/all-text-models-latency-vs-passrate.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

try:
    from adjustText import adjust_text
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete env
    raise SystemExit(
        "This plot needs adjustText. Run it with: "
        "uv run --with matplotlib --with adjustText python "
        "scripts/plot_all_text_models_latency_passrate.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = REPO_ROOT / "README.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "all-text-models-latency-vs-passrate.png"


@dataclass(frozen=True)
class ModelPoint:
    model: str
    pass_rate: float
    latency_ms: float
    provider: str


def _plain_markdown(value: str) -> str:
    """Remove the inline Markdown used for emphasized table cells."""

    return value.strip().replace("**", "").replace("`", "")


def _number(value: str, suffix: str) -> float:
    cleaned = _plain_markdown(value)
    if not cleaned.endswith(suffix):
        raise ValueError(f"Expected {value!r} to end in {suffix!r}")
    return float(cleaned[: -len(suffix)])


def parse_text_model_table(readme: Path) -> list[ModelPoint]:
    """Read the text-model results table from the current README."""

    lines = readme.read_text(encoding="utf-8").splitlines()
    header = "| Model | Pass Rate | Any Error |"
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(header))
    except StopIteration as exc:
        raise ValueError(f"Could not find the text-model table in {readme}") from exc

    points: list[ModelPoint] = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if points:
                break
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 10:
            raise ValueError(f"Expected 10 columns in README row: {line}")
        if re.fullmatch(r":?-+:?", cells[0]):
            continue

        points.append(
            ModelPoint(
                model=_plain_markdown(cells[0]),
                pass_rate=_number(cells[1], "%"),
                latency_ms=_number(cells[6], "ms"),
                provider=_plain_markdown(cells[9]),
            )
        )

    if not points:
        raise ValueError(f"The text-model table in {readme} has no data rows")
    return points


def pareto_efficient(points: list[ModelPoint]) -> list[ModelPoint]:
    """Return points not dominated on latency (lower) and pass rate (higher)."""

    return [
        point
        for point in points
        if not any(
            other.latency_ms <= point.latency_ms
            and other.pass_rate >= point.pass_rate
            and (
                other.latency_ms < point.latency_ms
                or other.pass_rate > point.pass_rate
            )
            for other in points
        )
    ]


def provider_colors(points: list[ModelPoint]) -> dict[str, tuple[float, ...]]:
    """Assign stable, well-separated colors to the providers in the table."""

    providers = sorted({point.provider for point in points})
    palette = plt.get_cmap("tab20").colors
    return {provider: palette[(i * 2) % len(palette)] for i, provider in enumerate(providers)}


def draw(points: list[ModelPoint], output: Path) -> list[ModelPoint]:
    colors = provider_colors(points)
    frontier = pareto_efficient(points)
    frontier_set = set(frontier)

    fig, ax = plt.subplots(figsize=(19, 14))
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    # Plot each provider separately so the legend maps directly to point color.
    for provider in sorted(colors):
        group = [point for point in points if point.provider == provider]
        ax.scatter(
            [point.latency_ms for point in group],
            [point.pass_rate for point in group],
            s=72,
            color=colors[provider],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.92,
            zorder=3,
            label=provider,
        )

    # Rings identify observed configurations on the Pareto frontier. There is no
    # connecting line because the space between tested configurations is not data.
    ax.scatter(
        [point.latency_ms for point in frontier],
        [point.pass_rate for point in frontier],
        s=132,
        facecolor="none",
        edgecolor="#161616",
        linewidth=1.7,
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xlim(75, 5000)
    ax.set_ylim(min(point.pass_rate for point in points) - 2.3, 101.8)
    ticks = [100, 150, 250, 400, 700, 1000, 1500, 2500, 4000]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.minorticks_off()
    ax.set_yticks(range(50, 101, 5))

    ax.axvline(700, color="#777777", linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)
    ax.text(
        700,
        50.2,
        "~700 ms voice-use guideline",
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    labels = []
    for point in points:
        labels.append(
            ax.text(
                point.latency_ms,
                point.pass_rate,
                point.model,
                fontsize=7.2,
                color="#242424",
                weight="semibold" if point in frontier_set else "normal",
                zorder=5,
            )
        )

    # Point labels are essential here, but the dense cluster around 500--1,000 ms
    # needs automatic repulsion. Leader lines preserve the point-label mapping.
    adjust_text(
        labels,
        x=[point.latency_ms for point in points],
        y=[point.pass_rate for point in points],
        ax=ax,
        expand=(1.06, 1.18),
        force_text=(0.35, 0.55),
        force_static=(0.18, 0.35),
        force_pull=(0.012, 0.018),
        max_move=(24, 28),
        ensure_inside_axes=True,
        prevent_crossings=True,
        time_lim=8,
        arrowprops={"arrowstyle": "-", "color": "#aaa7a1", "lw": 0.55},
    )

    ax.set_xlabel("TTFAT P50 (ms, logarithmic scale)  —  lower is better", fontsize=12)
    ax.set_ylabel("Turn pass rate (%)  —  higher is better", fontsize=12)
    ax.set_title(
        "Text-model accuracy and response latency",
        loc="left",
        fontsize=18,
        weight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.012,
        f"All {len(points)} configurations in the current README aiwf_medium_context table",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#555555",
        va="bottom",
    )

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.28, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#161616",
            markeredgewidth=1.7,
            markersize=9,
            label="Pareto-efficient row",
        )
    )
    legend_labels.append("Pareto-efficient row")
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=6,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
        columnspacing=1.25,
    )

    fig.text(
        0.125,
        0.012,
        "Source: README.md text-model table. Each point is one tested configuration; "
        "black rings mark rows for which no other row is both faster and at least as accurate.",
        ha="left",
        fontsize=8.5,
        color="#666666",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.15)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return sorted(frontier, key=lambda point: point.latency_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    points = parse_text_model_table(args.readme)
    frontier = draw(points, args.output)
    print(f"Wrote {args.output} from {len(points)} README rows")
    print("Pareto-efficient rows:")
    for point in frontier:
        print(f"  {point.latency_ms:,.0f}ms, {point.pass_rate:.1f}%  {point.model}")


if __name__ == "__main__":
    main()
