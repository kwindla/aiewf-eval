from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = load("kimi_analysis", HERE / "analysis/analyze.py")
comparison = load("kimi_comparison", HERE / "analysis/compare_arms.py")
judge = load("kimi_judge", HERE / "judge_campaign.py")


def test_analysis_excludes_recovery_from_score_and_ttfat_but_bills_tokens(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    run_dir = root / "run"
    run_dir.mkdir()
    transcript = []
    judged = []
    for turn in range(30):
        transcript.append(
            {
                "turn": turn,
                "model_name": analysis.MODEL,
                "assistant_text": "ok",
                "tool_calls": [],
                "tokens": {"prompt_tokens": 10, "completion_tokens": 1},
                "ttfb_ms": turn + 1,
                "raw_ttfb_ms": turn + 0.5,
            }
        )
        judged.append(
            {
                "turn": turn,
                "scores": {
                    "tool_use_correct": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                },
            }
        )
    transcript.append(
        {
            "turn": 31,
            "model_name": analysis.MODEL,
            "assistant_text": "",
            "tool_calls": [{"name": "end_session", "args": {}}],
            "tokens": {"prompt_tokens": 20, "completion_tokens": 2},
            "ttfb_ms": 99999,
            "recovery_turn": True,
        }
    )
    (run_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in transcript), encoding="utf-8"
    )
    (run_dir / "claude_judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged), encoding="utf-8"
    )
    (run_dir / "claude_summary.json").write_text(
        json.dumps(
            {
                "judge_model": analysis.JUDGE_MODEL,
                "judge_version": analysis.JUDGE_VERSION,
                "model_name": analysis.MODEL,
                "turns_scored": 30,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis, "ROOT", root)
    conversation = analysis.load_conversation({"slot": "K26T-01", "run_dir": "run"})
    assert len(conversation.strict) == 30
    assert len(conversation.ttfat_ms) == 30
    assert max(conversation.ttfat_ms) == 30
    assert max(conversation.raw_ttfb_ms) == 29.5
    assert conversation.recovery_rows == 1
    assert conversation.token_totals["prompt_tokens"] == 320
    assert conversation.recovery_token_totals["prompt_tokens"] == 20


def test_analysis_uses_exact_thinking_request_signature() -> None:
    config = json.loads((HERE / "configuration.json").read_text(encoding="utf-8"))
    analysis.validate_config(config)
    assert analysis.request_signature(config) == {
        "endpoint": "https://inference.baseten.co/v1",
        "reasoning_effort": "omit",
        "chat_template_args": {"enable_thinking": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "filler": None,
    }
    changed = json.loads(json.dumps(config))
    changed["sampling"]["temperature"] = 0.6
    try:
        analysis.validate_config(changed)
    except RuntimeError as exc:
        assert "frozen configuration mismatch" in str(exc)
    else:
        raise AssertionError("off-arm sampling unexpectedly passed validation")


def test_comparison_cluster_difference_bootstrap_is_conversation_level() -> None:
    identical = [30] * 30
    assert comparison.independent_cluster_difference_ci(identical, identical) == [0.0, 0.0]


def test_comparison_distinguishes_transmitted_off_field_from_effective_control() -> None:
    assert comparison.OFF_TRANSMITTED_SIGNATURE["reasoning_effort"] == "none"
    assert "omitted" in comparison.OFF_CONTROL_INTERPRETATION
    assert "ignored" in comparison.OFF_CONTROL_INTERPRETATION
    assert "Zero thinking tokens" in comparison.OFF_CONTROL_INTERPRETATION


def test_judge_output_requires_exactly_scripted_turns_zero_through_29(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    transcript = run_dir / "transcript.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    judged = [
        {
            "turn": turn,
            "scores": {
                "tool_use_correct": True,
                "instruction_following": True,
                "kb_grounding": True,
            },
        }
        for turn in range(30)
    ]
    (run_dir / "claude_judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged), encoding="utf-8"
    )
    (run_dir / "claude_summary.json").write_text(
        json.dumps(
            {
                "judge_model": judge.JUDGE_MODEL,
                "judge_version": judge.JUDGE_VERSION,
                "model_name": judge.MODEL,
                "turns_scored": 30,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "claude_analysis.md").write_text("ok\n", encoding="utf-8")
    entry = {
        "run_dir": run_dir,
        "transcript": transcript,
        "transcript_sha256": judge.sha256(transcript),
    }
    assert judge.validate_output(entry) == (True, "")
    judged.append({**judged[-1], "turn": 30})
    (run_dir / "claude_judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged), encoding="utf-8"
    )
    valid, error = judge.validate_output(entry)
    assert not valid
    assert "scripted turns 0-29" in error
