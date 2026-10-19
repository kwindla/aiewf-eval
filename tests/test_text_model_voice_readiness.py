import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_text_model_voice_readiness",
    REPO_ROOT / "scripts/render_text_model_voice_readiness.py",
)
assert SPEC and SPEC.loader
chart = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = chart
SPEC.loader.exec_module(chart)


def test_current_readme_rows_are_ranked_by_pass_rate_then_p50():
    rows = chart.load_rows(chart.DEFAULT_README)
    keys = [(float("inf") if row.pass_rate is None else -row.pass_rate, row.p50_ms, row.model) for row in rows]
    assert keys == sorted(keys)
    summary = json.loads((REPO_ROOT / 'docs/model-evaluations-2026-10/judge-recovery-context-20261019/summary.json').read_text())['data']
    ranked = [m for m in summary['text_models'] if m['complete'] > 0]
    assert len(rows) == len(ranked)
    assert {row.model for row in rows} == {m['label'] for m in ranked}
    assert not {
        "gpt-oss-120b (groq)",
        "grok-4.6 (low)",
        "claude-fable-5 (low)",
        "claude-fable-5 (default)",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "nemotron-3-ultra (128)",
        "nemotron-3-ultra (96)",
        "nemotron-3-ultra (thinking off)",
    } & {row.model for row in rows}
    table = chart.DEFAULT_README.read_text().split("| Model | Pass Rate | Any Error |")[1].split("\n\n", 1)[0]
    actual_order = [line.split("|")[1].strip() for line in table.splitlines() if line.startswith("| ")]
    assert actual_order == [row.model for row in rows]
    assert all(row.conversations and 0 < row.conversations <= 30 for row in rows)
    assert sum(row.conversations for row in rows) == sum(m['complete'] for m in ranked)
    nemotron_rows = [row for row in rows if row.model.startswith("nemotron-3-")]
    assert len(nemotron_rows) == 3
    assert {row.provider for row in nemotron_rows} == {"Baseten", "AWS B200"}
    assert all("NVFP4" in row.model for row in nemotron_rows if row.provider == "AWS B200")
    lightning_rows = [
        row for row in rows if row.model.startswith("nemotron-3.5-lightning")
    ]
    assert len(lightning_rows) == 2
    assert {row.provider for row in lightning_rows} == {"Local RTX 5090"}


def test_standalone_leaderboard_matches_readme_table():
    standalone = REPO_ROOT / "leaderboard-medium-context.md"
    assert chart.load_rows(standalone) == chart.load_rows(chart.DEFAULT_README)


def test_chart_has_one_p50_and_p95_mark_per_model():
    rows = chart.load_rows(chart.DEFAULT_README)
    svg = chart.render_svg(rows)
    assert svg.count('class="p50') == len(rows)
    assert svg.count('class="p95"') == len(rows)
    assert "~700ms voice guideline" in svg
    assert "All rows: Opus 5.5 medium" in svg
    assert "quality conditional on completion" in svg
    assert "legacy TTFT" not in svg
    for row in rows:
        assert f"N={row.conversations}" in svg
        assert ("quality unavailable" if row.pass_rate is None else f"{row.pass_rate:.1f}% pass") in svg
    assert "muse-glimmer-30b" in svg
    assert "nemotron-3.5-lightning (thinking on, NVFP4)" in svg
    assert "nemotron-3.5-lightning (thinking off, NVFP4)" in svg
    assert "qwen3.8-27b (thinking off, NVFP4)" in svg
    assert "qwen3-8b" not in svg


def test_speed_accuracy_frontier_handles_ties_and_tradeoffs():
    rows = [
        chart.ModelRow("Fast", 80, 100, 200, "test"),
        chart.ModelRow("Accurate", 99, 400, 500, "test"),
        chart.ModelRow("Dominated", 80, 200, 300, "test"),
        chart.ModelRow("Same speed", 79, 100, 200, "test"),
        chart.ModelRow("Equal", 80, 100, 200, "test"),
        chart.ModelRow("Unknown", None, 50, 70, "test"),
    ]
    assert {row.model for row in chart.pareto_efficient(rows)} == {"Fast", "Accurate", "Equal"}
