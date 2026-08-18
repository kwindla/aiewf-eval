import importlib.util
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
    keys = [(-row.pass_rate, row.p50_ms, row.model) for row in rows]
    assert keys == sorted(keys)
    assert len(rows) == 45
    assert [row.model for row in rows[:4]] == [
        "nemotron-3-ultra (128)",
        "claude-sonnet-4-6",
        "claude-fable-5 (low)",
        "claude-fable-5 (default)",
    ]
    historical_nemotron_rows = [
        row for row in rows if row.model.startswith("nemotron-3-")
    ]
    assert len(historical_nemotron_rows) == 4
    assert {row.provider for row in historical_nemotron_rows} == {"Baseten"}
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
    assert "muse-glimmer-30b" in svg
    assert "231ms P50 / 1.75s P95" in svg
    assert "nemotron-3.5-lightning (thinking on, NVFP4)" in svg
    assert "93.6% pass · 1.46s P50 / 5.8s P95 · Local RTX 5090" in svg
    assert "nemotron-3.5-lightning (thinking off, NVFP4)" in svg
    assert "50.9% pass · 62ms P50 / 70ms P95 · Local RTX 5090" in svg
    assert "qwen3.8-27b (thinking off, NVFP4)" in svg
    assert "97.8% pass · 101ms P50 / 318ms P95 · Local RTX 5090" in svg
    assert "qwen3-8b" not in svg


def test_current_speed_accuracy_frontier_is_stable():
    rows = chart.load_rows(chart.DEFAULT_README)
    assert {row.model for row in chart.pareto_efficient(rows)} == {
        "gpt-oss-120b (groq)",
        "nemotron-3.5-lightning (thinking off, NVFP4)",
        "qwen3.8-27b (thinking off, NVFP4)",
        "nemotron-3-ultra (96)",
        "nemotron-3-ultra (128)",
    }
