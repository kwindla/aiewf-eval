"""Offline tests for the portable AIEWF campaign collector."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COLLECT_PATH = ROOT / "ops/aiewf-campaign-template/collect.py"


def load_collector():
    spec = importlib.util.spec_from_file_location("aiewf_campaign_template_collect", COLLECT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector():
    return load_collector()


def write_campaign(
    tmp_path: Path,
    *,
    target: int = 1,
    verified: bool = False,
) -> tuple[Path, Path, Path]:
    artifact_dir = tmp_path / "artifacts"
    run_root = tmp_path / "runs" / "aiwf_medium_context"
    schedule = tmp_path / "schedule.tsv"
    schedule.write_text(
        "slot\tarm\n" + "".join(f"{slot}\tnone\n" for slot in range(1, target + 1)),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "campaign_id": "offline-test-campaign",
        "benchmark": "aiwf_medium_context",
        "model": "test/model",
        "accepted_response_model_ids": ["test/model"],
        "service": "vllm-openai",
        "pipeline": "text",
        "endpoint": {
            "url": "https://offline.invalid/sync/v1",
            "request_env": "VLLM_BASE_URL",
        },
        "credential": {
            "request_env": "VLLM_API_KEY",
            "source_env": "TEST_CAMPAIGN_TOKEN",
            "source_file": "",
            "source_file_key": "TEST_CAMPAIGN_TOKEN",
        },
        "paths": {
            "campaign_artifact_dir": str(artifact_dir),
            "run_output_root": str(run_root),
        },
        "schedule_path": str(schedule),
        "source_integrity_paths": [],
        "target_eligible_runs": target,
        "fixed_scheduled_turns_per_conversation": 30,
        "serving": {"verified": verified},
        "eligibility": {
            "policy": "first_valid_response",
            "missing_future_turns": "retain_as_fixed_denominator_failures",
        },
        "collection": {
            "provider_endpoint_concurrency": 1,
            "max_attempts_per_slot_default": 3,
            "timeout_seconds_default": 60,
        },
        "unset_environment_prefixes": ["MTE_FILLER_"],
        "unset_environment": ["MTE_VLLM_THINKING_BUDGET"],
        "common_environment": {
            "MTE_VLLM_TEMPERATURE": "1.0",
            "MTE_VLLM_MAX_TOKENS": "8192",
        },
        "arms": {
            "none": {
                "environment": {"MTE_VLLM_THINKING": "0"},
                "provenance_log_needles": [
                    "thinking=False",
                    "T=1.0",
                    "max_tokens=8192",
                ],
                "forbidden_log_needles": ["MTE_FILLER_DOTS active"],
            }
        },
    }
    config_path = tmp_path / "configuration.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, artifact_dir, run_root


def write_fake_run(
    run_dir: Path,
    *,
    turns: int,
    valid_response: bool = True,
) -> None:
    run_dir.mkdir(parents=True)
    rows = []
    for turn in range(turns):
        rows.append(
            {
                "turn": turn,
                "model_name": "test/model",
                "assistant_text": "ok" if valid_response else "",
                "tool_calls": [],
            }
        )
    (run_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "run.log").write_text(
        "Using vllm-openai with "
        "base_url=https://offline.invalid/sync/v1, model=test/model, "
        "thinking=False, T=1.0, max_tokens=8192\n",
        encoding="utf-8",
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_dir_from_command(command: list[str]) -> Path:
    return Path(command[command.index("--run-dir") + 1])


def test_default_preflight_is_read_only(collector, tmp_path, capsys):
    config_path, artifact_dir, run_root = write_campaign(tmp_path)

    assert collector.main(["--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "Read-only preflight only" in output
    assert not artifact_dir.exists()
    assert not run_root.exists()


def test_execute_requires_verified_serving_before_writes(collector, tmp_path):
    config_path, artifact_dir, run_root = write_campaign(tmp_path, verified=False)

    with pytest.raises(ValueError, match="serving smoke gate is not complete"):
        collector.main(["--config", str(config_path), "--execute"])

    assert not artifact_dir.exists()
    assert not run_root.exists()


def test_execute_is_sequential_and_honors_configured_output_root(
    collector, tmp_path, monkeypatch
):
    config_path, artifact_dir, run_root = write_campaign(
        tmp_path, target=2, verified=True
    )
    monkeypatch.setenv("TEST_CAMPAIGN_TOKEN", "offline-secret")
    monkeypatch.setenv("MTE_FILLER_DOTS", "96")
    launched: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, **kwargs):
        run_dir = run_dir_from_command(command)
        assert run_dir.parent == run_root
        assert not run_dir.exists()
        assert kwargs["env"]["VLLM_BASE_URL"] == "https://offline.invalid/sync/v1"
        assert kwargs["env"]["VLLM_API_KEY"] == "offline-secret"
        assert kwargs["env"]["MTE_VLLM_THINKING"] == "0"
        assert "MTE_FILLER_DOTS" not in kwargs["env"]
        launched.append((command, kwargs["env"]))
        write_fake_run(run_dir, turns=30)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    assert collector.main(["--config", str(config_path), "--execute"]) == 0

    assert len(launched) == 2
    canonical = read_tsv(artifact_dir / "canonical.tsv")
    assert [row["slot"] for row in canonical] == ["1", "2"]
    assert all(Path(row["run_dir"]).parent == run_root for row in canonical)
    assert not (artifact_dir / "pending-attempt.json").exists()
    assert (artifact_dir / "source-sha256.txt").is_file()
    assert "RUN_COLLECTION_DONE total=2 none=2" in (
        artifact_dir / "campaign.log"
    ).read_text(encoding="utf-8")


def test_short_attempt_after_valid_response_is_canonical_without_replacement(
    collector, tmp_path, monkeypatch
):
    config_path, artifact_dir, _ = write_campaign(tmp_path, verified=True)
    monkeypatch.setenv("TEST_CAMPAIGN_TOKEN", "offline-secret")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        write_fake_run(run_dir_from_command(command), turns=1, valid_response=True)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    assert collector.main(["--config", str(config_path), "--execute"]) == 0

    assert calls == 1
    canonical = read_tsv(artifact_dir / "canonical.tsv")
    assert canonical[0]["classification"] == "fixed_denominator_short"
    assert canonical[0]["turns"] == "1"
    assert len(read_tsv(artifact_dir / "attempts.tsv")) == 1


def test_zero_valid_response_is_replaced_without_changing_schedule(
    collector, tmp_path, monkeypatch
):
    config_path, artifact_dir, _ = write_campaign(tmp_path, verified=True)
    monkeypatch.setenv("TEST_CAMPAIGN_TOKEN", "offline-secret")
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        write_fake_run(
            run_dir_from_command(command),
            turns=1 if calls == 1 else 30,
            valid_response=calls != 1,
        )
        return subprocess.CompletedProcess(command, 1 if calls == 1 else 0)

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    assert collector.main(["--config", str(config_path), "--execute"]) == 0

    attempts = read_tsv(artifact_dir / "attempts.tsv")
    canonical = read_tsv(artifact_dir / "canonical.tsv")
    assert [row["attempt"] for row in attempts] == ["1", "2"]
    assert attempts[0]["classification"] == "ineligible_no_valid_response_or_provenance"
    assert canonical[0]["slot"] == "1"
    assert canonical[0]["attempt"] == "2"


def test_restart_finalizes_pending_attempt_before_launching_new_work(
    collector, tmp_path, monkeypatch
):
    config_path, artifact_dir, run_root = write_campaign(tmp_path, verified=True)
    monkeypatch.setenv("TEST_CAMPAIGN_TOKEN", "offline-secret")
    config = collector.validate_configuration(
        config_path, require_serving_verified=True
    )
    collector.initialize_artifacts(config)
    run_dir = run_root / "recovered-run"
    write_fake_run(run_dir, turns=30)
    log_path = artifact_dir / "logs" / "slot001-none-attempt01.log"
    log_path.write_text("collector was interrupted after child exit\n", encoding="utf-8")
    collector.write_pending(
        config,
        {
            "slot": 1,
            "arm": "none",
            "attempt": 1,
            "started_at": "2026-07-31T00:00:00+00:00",
            "finished_at": "",
            "exit_code": "",
            "run_dir": str(run_dir),
            "log": str(log_path),
        },
    )
    collector.atomic_write(
        collector.artifact_paths(config)["source_hash"], collector.source_hashes(config)
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("recovered canonical attempt must prevent a new launch")

    monkeypatch.setattr(collector.subprocess, "run", unexpected_run)

    assert collector.main(["--config", str(config_path), "--execute"]) == 0

    assert not (artifact_dir / "pending-attempt.json").exists()
    canonical = read_tsv(artifact_dir / "canonical.tsv")
    assert canonical[0]["run_dir"] == str(run_dir)
    attempts = read_tsv(artifact_dir / "attempts.tsv")
    assert attempts[0]["exit_code"] == "unknown"
