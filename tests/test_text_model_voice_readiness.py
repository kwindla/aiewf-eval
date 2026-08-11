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
    assert len(rows) == 39
    assert [row.model for row in rows[:4]] == [
        "nemotron-3-ultra (128)",
        "claude-sonnet-4-6",
        "claude-fable-5 (low)",
        "claude-fable-5 (default)",
    ]
    nemotron_rows = [row for row in rows if row.model.startswith("nemotron-3-")]
    assert len(nemotron_rows) == 4
    assert {row.provider for row in nemotron_rows} == {"Baseten"}


def test_chart_has_one_p50_and_p95_mark_per_model():
    rows = chart.load_rows(chart.DEFAULT_README)
    svg = chart.render_svg(rows)
    assert svg.count('class="p50') == len(rows)
    assert svg.count('class="p95"') == len(rows)
    assert "~700ms voice guideline" in svg
    assert "muse-glimmer-30b" in svg
    assert "232ms P50 / 2.80s P95" in svg
    assert "qwen3-8b" not in svg


def test_current_speed_accuracy_frontier_is_stable():
    rows = chart.load_rows(chart.DEFAULT_README)
    assert {row.model for row in chart.pareto_efficient(rows)} == {
        "gpt-oss-120b (groq)",
        "inkling (none)",
        "gemma-4-31b-it (thinking off)",
        "nemotron-3-ultra (96)",
        "nemotron-3-ultra (128)",
    }
