"""Lightweight local checks for the no-network adaptive dots bundle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_collector():
    spec = importlib.util.spec_from_file_location("inkling_small_dots_collect", HERE / "collect.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow_modules():
    collector = load_collector()
    sys.modules["collect"] = collector

    judge_spec = importlib.util.spec_from_file_location(
        "judge_stage", HERE / "judge_stage.py"
    )
    assert judge_spec and judge_spec.loader
    judge = importlib.util.module_from_spec(judge_spec)
    sys.modules["judge_stage"] = judge
    judge_spec.loader.exec_module(judge)

    analysis_spec = importlib.util.spec_from_file_location(
        "inkling_small_dots_analyze", HERE / "analyze_stage.py"
    )
    assert analysis_spec and analysis_spec.loader
    analysis = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis)
    return collector, judge, analysis


def test_frozen_configuration_and_dots_only_schedule():
    collector = load_collector()
    config = collector.validate_configuration()
    schedule = collector.validate_schedule()

    assert config["thinking_effort"] == "none"
    assert config["stage_caps"] == [6, 10, 30]
    assert len(schedule) == 30
    assert {row["arm"] for row in schedule} == {"dots96"}


def test_attempt_environment_is_exactly_none_plus_96_suffix_dots(monkeypatch):
    collector = load_collector()
    monkeypatch.setenv("MTE_BASETEN_ENABLE_THINKING", "true")
    monkeypatch.setenv("MTE_FILLER_DOTS", "7")
    monkeypatch.setenv("MTE_FILLER_TOKEN", "-")
    monkeypatch.setenv("MTE_FILLER_POSITION", "prefix")

    env = collector.build_attempt_environment("redacted")

    assert env["MTE_BASETEN_REASONING_EFFORT"] == "none"
    assert "MTE_BASETEN_ENABLE_THINKING" not in env
    assert env["MTE_FILLER_DOTS"] == "96"
    assert env["MTE_FILLER_TOKEN"] == "."
    assert env["MTE_FILLER_POSITION"] == "suffix"
    assert env["MTE_BASETEN_MAX_TOKENS"] == "16384"
    assert env["MTE_BASETEN_TEMPERATURE"] == "1.0"


def test_read_only_stage_six_preflight_does_not_freeze_control():
    collector = load_collector()
    before = collector.CONTROL_INPUTS.read_bytes()

    collector.preflight(execute=False, stage=6)

    assert collector.CONTROL_INPUTS.read_bytes() == before


def test_judge_child_environment_has_only_anthropic_provider_secret(monkeypatch):
    _, judge, _ = load_workflow_modules()
    monkeypatch.setenv("BASETEN_API_KEY", "must-not-leak")
    monkeypatch.setenv("BASETEN_BASE_URL", "must-not-leak")
    monkeypatch.setenv("MTE_BASETEN_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-leak")

    env = judge.judge_environment("anthropic-only")

    assert env["ANTHROPIC_API_KEY"] == "anthropic-only"
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert "BASETEN_API_KEY" not in env
    assert "BASETEN_BASE_URL" not in env
    assert not any(key.startswith("MTE_BASETEN_") for key in env)
    assert "OPENAI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert [key for key in env if key.endswith("_API_KEY")] == ["ANTHROPIC_API_KEY"]


def synthetic_conversation(*, failures: int, complete: bool = True):
    passes = [0] * failures + [1] * (30 - failures)
    errors = [1 - value for value in passes]
    return {
        "slot": "synthetic",
        "arm": "synthetic",
        "run_dir": "synthetic",
        "classification": "strict_complete" if complete else "model_abort",
        "complete": int(complete),
        "observed_turns": 30 - failures,
        "missing_turns": failures,
        "metrics": {
            "strict_pass": passes,
            "any_error": errors,
            "tool_error": errors,
            "instruction_error": errors,
            "kb_error": errors,
        },
        "turn_taking_errors": errors,
        "latencies": [500.0],
    }


def test_fixed_denominator_summary_and_bootstrap_are_deterministic():
    _, _, analysis = load_workflow_modules()
    controls = [synthetic_conversation(failures=0) for _ in range(30)]
    dots = [synthetic_conversation(failures=3) for _ in range(6)]

    summary = analysis.summarize_arm(dots)
    first = analysis.bootstrap_effect(
        controls, dots, "strict_pass", iterations=1_000, seed=123
    )
    second = analysis.bootstrap_effect(
        controls, dots, "strict_pass", iterations=1_000, seed=123
    )

    assert summary["fixed_turn_denominator"] == 180
    assert summary["counts"]["strict_pass"] == 162
    assert summary["counts"]["any_error"] == 18
    assert first == second
    assert round(first["dots_minus_control_points"], 10) == -10.0


def test_adaptive_rules_report_recommendation_without_executing_gate():
    _, _, analysis = load_workflow_modules()
    effects = {
        "strict_pass": {
            "dots_minus_control_points": -3.0,
            "ci95_low": -8.0,
            "ci95_high": 1.0,
        }
    }
    control = {
        "strict_completion": {"percent": 100.0},
        "per_turn": [
            {"turn": turn, "any_error_percent": 0.0, "any_error_count": 0}
            for turn in range(30)
        ],
    }
    dots = {
        "strict_completion": {"percent": 100.0},
        "per_turn": [
            {
                "turn": turn,
                "any_error_percent": 50.0 if turn == 12 else 0.0,
                "any_error_count": 3 if turn == 12 else 0,
            }
            for turn in range(30)
        ],
    }

    stage6 = analysis.adaptive_decision(6, effects, control, dots)
    stage10 = analysis.adaptive_decision(10, effects, control, dots)

    assert stage6["recommendation"] == "extend_to_10"
    assert stage10["recommendation"] == "extend_to_30"
    assert stage10["gate_executed"] is False
