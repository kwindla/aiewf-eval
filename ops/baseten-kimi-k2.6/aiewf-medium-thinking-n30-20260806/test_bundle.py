from __future__ import annotations

import importlib.util
import json
import py_compile
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("kimi26_collect", HERE / "collect.py")
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def signature_log() -> str:
    return "\n".join(
        (
            "Recovery nudges enabled=True",
            "Tool call dedupe enabled=True",
            "Tool result run_llm enabled=False",
            "Using BaseTen with base_url=https://inference.baseten.co/v1, "
            "model=moonshotai/Kimi-K2.6, reasoning_effort=omit, "
            "enable_thinking=true, max_tokens=8192, temperature=1.0, top_p=0.95",
            "Text pipeline idle_timeout_secs=45.0",
        )
    )


def write_run(path: Path, *, turns: int = 30, end_session: bool = True) -> None:
    path.mkdir()
    rows = []
    for turn in range(turns):
        rows.append(
            {
                "turn": turn,
                "model_name": collector.MODEL,
                "assistant_text": "ok",
                "tool_calls": [],
                "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "thinking_tokens": 1},
            }
        )
    if end_session:
        rows.append(
            {
                "turn": 30,
                "model_name": collector.MODEL,
                "assistant_text": "",
                "tool_calls": [{"name": "end_session", "args": {}}],
                "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "thinking_tokens": 1},
                "recovery_turn": True,
            }
        )
    (path / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "runtime.json").write_text(
        json.dumps(
            {
                "model_name": collector.MODEL,
                "turns": len(rows),
                "status": "completed",
                "valid": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "run.log").write_text(signature_log(), encoding="utf-8")


def test_frozen_configuration_schedule_and_seed_manifest() -> None:
    config = collector.validate_configuration(execute=True)
    schedule = collector.validate_schedule()
    attempts, canonical = collector.validate_manifests(schedule)
    assert config["sampling"] == {
        "reasoning_effort": "omit",
        "chat_template_args": {"enable_thinking": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    }
    assert config["runtime"]["provider_endpoint_concurrency"] == 1
    assert config["runtime"]["inter_attempt_cooldown_seconds"] == 30
    assert len(attempts) >= len(canonical) >= 1
    seed = next(row for row in canonical if row["slot"] == "K26T-01")
    assert seed["classification"] == "strict_complete"


def test_seed_smoke_matches_exact_signature_and_complete_policy() -> None:
    seed = collector.ROOT / (
        "runs/aiwf_medium_context/"
        "20260806T173519Z_moonshotai_Kimi-K2.6_thinking_SMOKE_attempt01"
    )
    assert collector.inspect_run(seed) == {
        "scheduled_rows": 30,
        "response_turns": 30,
        "tool_calls": 6,
        "token_rows": 30,
        "thinking_token_rows": 30,
        "thinking_tokens": 7854,
        "end_session_turn": 29,
        "classification": "strict_complete",
    }


def test_complete_only_policy_replaces_short_but_retains_missing_end(tmp_path: Path) -> None:
    short = tmp_path / "short"
    write_run(short, turns=29)
    assert collector.inspect_run(short)["classification"] == "incomplete_scheduled_turns"

    no_end = tmp_path / "no-end"
    write_run(no_end, end_session=False)
    result = collector.inspect_run(no_end)
    assert result["classification"] == "strict_complete"
    assert result["end_session_turn"] == -1


def test_child_environment_forces_no_filler_and_frozen_sampling(monkeypatch) -> None:
    monkeypatch.setenv("MTE_FILLER_DOTS", "96")
    monkeypatch.setenv("MTE_FILLER_POSITION", "before")
    monkeypatch.setenv("MTE_BASETEN_REASONING_EFFORT", "high")
    monkeypatch.setenv("MTE_BASETEN_ENABLE_THINKING", "true")
    env = collector.child_environment("secret")
    assert not any(name.startswith("MTE_FILLER_") for name in env)
    assert env["MTE_BASETEN_ENABLE_THINKING"] == "true"
    assert env["MTE_BASETEN_REASONING_EFFORT"] == "omit"
    assert env["MTE_BASETEN_MAX_TOKENS"] == "8192"
    assert env["MTE_BASETEN_TEMPERATURE"] == "1.0"
    assert env["MTE_BASETEN_TOP_P"] == "0.95"
    assert env["BASETEN_BASE_URL"] == collector.ENDPOINT


def test_source_integrity_manifest_is_deterministic() -> None:
    first = collector.source_hash_text()
    second = collector.source_hash_text()
    assert first == second
    assert "ops/aiewf-campaign-template/run_one.py" in first
    assert "src/multi_turn_eval/services/baseten_logged.py" in first


def test_all_campaign_python_files_compile() -> None:
    paths = sorted(HERE.glob("*.py")) + sorted((HERE / "analysis").glob("*.py"))
    assert paths
    for path in paths:
        py_compile.compile(str(path), doraise=True)
