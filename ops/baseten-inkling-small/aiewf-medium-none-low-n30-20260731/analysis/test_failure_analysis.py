"""Invariant tests for the raw Inkling Small failure analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "inkling_small_failure_analysis", HERE / "failure_analysis.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transcript(module, turns: int, *, scheduled_end: int | None = None):
    rows = []
    for turn in range(turns):
        calls = [{"name": "end_session", "args": {}}] if turn == scheduled_end else []
        rows.append(
            {
                "turn": turn,
                "model_name": module.MODEL,
                "assistant_text": "ok" if not calls else "",
                "tool_calls": calls,
            }
        )
    return rows


def membership_rows():
    canonical = []
    schedule = []
    attempts = []
    for index in range(1, 61):
        slot = f"IS-{index:02d}"
        pair = str((index + 1) // 2)
        arm = "low" if index % 2 else "none"
        run_dir = f"runs/fixture-{slot}"
        canonical_row = {
            "slot": slot,
            "pair": pair,
            "arm": arm,
            "attempt": "1",
            "run_dir": run_dir,
            "scheduled_rows": "30",
            "response_turns": "30",
            "end_session_turn": "29",
            "classification": "strict_complete",
        }
        canonical.append(canonical_row)
        schedule.append({"slot": slot, "pair": pair, "arm": arm})
        attempts.append(dict(canonical_row))
    return canonical, schedule, attempts


def test_classifies_each_prespecified_cause():
    module = load_module()
    strict = transcript(module, 30, scheduled_end=29)
    abort = transcript(module, 16, scheduled_end=15)
    recovery = transcript(module, 16)
    recovery.append(
        {
            "turn": 30,
            "model_name": module.MODEL,
            "recovery_turn": True,
            "assistant_text": "",
            "tool_calls": [{"name": "end_session", "args": {}}],
        }
    )
    serving = transcript(module, 12)

    assert module.classify_run(
        scheduled=module.scheduled_map(strict, slot="strict"),
        all_rows=strict,
        run_log_text="Using BaseTen",
    ) == "strict_complete"
    assert module.classify_run(
        scheduled=module.scheduled_map(abort, slot="abort"),
        all_rows=abort,
        run_log_text="Using BaseTen",
    ) == "model_abort"
    assert module.classify_run(
        scheduled=module.scheduled_map(recovery, slot="recovery"),
        all_rows=recovery,
        run_log_text="Using BaseTen",
    ) == "recovery_end_session"
    assert module.classify_run(
        scheduled=module.scheduled_map(serving, slot="429"),
        all_rows=serving,
        run_log_text=(
            "Using BaseTen\nqueue_turn: reason=next turn_idx=12\n"
            "Error code: 429 - {'error': 'Rate limit exceeded'}\n"
            "Idle timeout detected."
        ),
    ) == "baseten_429_idle"


def test_429_requires_baseten_rate_limit_and_idle_markers():
    module = load_module()
    rows = transcript(module, 12)
    scheduled = module.scheduled_map(rows, slot="short")

    for log in (
        "Using BaseTen\nError code: 429\nRate limit exceeded",
        "Using BaseTen\nIdle timeout detected.",
        "Error code: 429\nRate limit exceeded\nIdle timeout detected.",
    ):
        assert module.classify_run(
            scheduled=scheduled,
            all_rows=rows,
            run_log_text=log,
        ) == "unattributed_short"


def test_429_must_be_terminal_and_ordered_before_idle():
    module = load_module()
    rows = transcript(module, 12)
    scheduled = module.scheduled_map(rows, slot="short")
    terminal = (
        "Using BaseTen\nqueue_turn: reason=next turn_idx=12\n"
        "Error code: 429 - Rate limit exceeded\nIdle timeout detected."
    )
    stale = (
        "Using BaseTen\nError code: 429 - Rate limit exceeded\n"
        "queue_turn: reason=next turn_idx=12\nIdle timeout detected."
    )
    recovered = (
        "Using BaseTen\nqueue_turn: reason=next turn_idx=12\n"
        "Error code: 429 - Rate limit exceeded\nTTFB: 0.2s\n"
        "Recorded turn 12\nIdle timeout detected."
    )

    assert module.classify_run(
        scheduled=scheduled,
        all_rows=rows,
        run_log_text=terminal,
    ) == "baseten_429_idle"
    for log in (stale, recovered):
        assert module.classify_run(
            scheduled=scheduled,
            all_rows=rows,
            run_log_text=log,
        ) == "unattributed_short"


def test_canonical_arm_identity_is_bound_to_schedule_and_attempts():
    module = load_module()
    canonical, schedule, attempts = membership_rows()
    module.validate_canonical(
        canonical,
        schedule=schedule,
        attempts=attempts,
    )

    canonical[0]["arm"], canonical[1]["arm"] = (
        canonical[1]["arm"],
        canonical[0]["arm"],
    )
    with pytest.raises(RuntimeError, match="arm mismatch against frozen schedule"):
        module.validate_canonical(
            canonical,
            schedule=schedule,
            attempts=attempts,
        )


def test_scheduled_turns_must_be_a_unique_contiguous_prefix():
    module = load_module()
    duplicate = transcript(module, 2)
    duplicate.append(dict(duplicate[-1]))
    with pytest.raises(RuntimeError, match="duplicate scheduled turn"):
        module.scheduled_map(duplicate, slot="duplicate")

    gap = transcript(module, 3)
    gap[1]["turn"] = 5
    with pytest.raises(RuntimeError, match="contiguous prefix"):
        module.scheduled_map(gap, slot="gap")


def test_response_and_transition_patterns_are_exact():
    module = load_module()
    first = transcript(module, 18)
    first[14]["assistant_text"] = ""
    first[14]["tool_calls"] = [{"name": "submit_dietary_request", "args": {}}]
    first[15]["assistant_text"] = ""
    first[15]["tool_calls"] = [{"name": "end_session", "args": {}}]
    second = transcript(module, 15)
    runs = [
        {"scheduled": module.scheduled_map(first, slot="first")},
        {"scheduled": module.scheduled_map(second, slot="second")},
    ]

    turn = module.summarize_turn(runs, 15)
    assert turn["observed_responses"] == 1
    assert turn["missing_responses"] == 1
    assert turn["response_patterns"] == {
        "missing": 1,
        "tool_only:end_session": 1,
    }
    assert module.summarize_transition(runs, 14, 15) == {
        "t14=text_only -> t15=missing": 1,
        "t14=tool_only:submit_dietary_request -> t15=tool_only:end_session": 1,
    }
