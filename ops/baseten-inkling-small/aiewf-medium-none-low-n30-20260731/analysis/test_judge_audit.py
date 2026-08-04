"""Pure-function and fail-closed tests for the Inkling judge audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "inkling_small_judge_audit", HERE / "judge_audit.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_changes(module):
    base = {
        "tool_use_correct": True,
        "instruction_following": True,
        "kb_grounding": True,
    }
    rows = []
    for slot, turn in (("IS-18", 16), ("IS-18", 17), ("IS-47", 17)):
        after = dict(base)
        after["tool_use_correct"] = False
        rows.append(
            {
                "slot": slot,
                "turn": turn,
                "arm": "none",
                "official_scores": dict(base),
                "counterfactual_scores": after,
            }
        )
    mixed = dict(base)
    mixed["instruction_following"] = False
    mixed_after = dict(mixed)
    mixed_after["tool_use_correct"] = False
    rows.append(
        {
            "slot": "IS-47",
            "turn": 16,
            "arm": "none",
            "official_scores": mixed,
            "counterfactual_scores": mixed_after,
        }
    )
    return rows


def test_counterfactual_has_exact_fixed_denominator_deltas():
    module = load_module()
    official = {
        "none": {"strict_pass": 700, "any_error": 200, "tool_error": 80},
        "low": {"strict_pass": 400, "any_error": 500, "tool_error": 300},
    }
    result = module.apply_counterfactual(official, fixture_changes(module))

    none = result["none"]["metrics"]
    assert none["strict_pass"]["counterfactual_count"] == 697
    assert none["any_error"]["counterfactual_count"] == 203
    assert none["tool_error"]["counterfactual_count"] == 84
    assert none["strict_pass"]["delta_percentage_points"] == pytest.approx(-1 / 3)
    assert none["tool_error"]["delta_percentage_points"] == pytest.approx(4 / 9)
    assert result["low"]["metrics"]["strict_pass"]["delta_count"] == 0


def test_counterfactual_rejects_non_tool_changes():
    module = load_module()
    official = {
        "none": {"strict_pass": 700, "any_error": 200, "tool_error": 80},
        "low": {"strict_pass": 400, "any_error": 500, "tool_error": 300},
    }
    changes = fixture_changes(module)
    changes[0]["counterfactual_scores"]["instruction_following"] = False
    with pytest.raises(RuntimeError, match="non-tool"):
        module.apply_counterfactual(official, changes)


def test_final_gate_fails_closed(tmp_path):
    module = load_module()
    aggregates = tmp_path / "aggregates.json"
    complete = tmp_path / "COMPLETE.json"
    with pytest.raises(RuntimeError, match="final aggregates"):
        module.require_final_artifacts(aggregates, complete)
    aggregates.write_text("{}\n")
    with pytest.raises(RuntimeError, match="completion marker"):
        module.require_final_artifacts(aggregates, complete)


def test_stable_row_hash_detects_semantic_drift():
    module = load_module()
    row = {"turn": 16, "scores": {"tool_use_correct": True}, "text": "ok"}
    reordered = {"text": "ok", "scores": {"tool_use_correct": True}, "turn": 16}
    changed = {"turn": 16, "scores": {"tool_use_correct": False}, "text": "ok"}
    assert module.stable_row_hash(row) == module.stable_row_hash(reordered)
    assert module.stable_row_hash(row) != module.stable_row_hash(changed)
