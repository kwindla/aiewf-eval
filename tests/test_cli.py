import pytest
from click import UsageError
from click.testing import CliRunner

import multi_turn_eval.cli as cli_module
from multi_turn_eval.cli import cli, infer_pipeline, reject_openai_pro_model


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-07-01",
        "openai/gpt-5.5-pro",
    ],
)
def test_reject_openai_pro_model(model: str) -> None:
    with pytest.raises(UsageError, match="Pro models are excluded"):
        reject_openai_pro_model(model)


@pytest.mark.parametrize("model", ["o3-pro", "future-model-pro"])
def test_reject_pro_model_on_openai_service(model: str) -> None:
    with pytest.raises(UsageError, match="Pro models are excluded"):
        reject_openai_pro_model(model, service="openai")


def test_reject_openai_namespaced_pro_model_on_compatible_service() -> None:
    with pytest.raises(UsageError, match="Pro models are excluded"):
        reject_openai_pro_model("openai/o3-pro", service="openrouter")


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-mini",
        "claude-pro",
        "some-org/pro-model",
    ],
)
def test_allow_non_openai_pro_model(model: str) -> None:
    reject_openai_pro_model(model)


def test_gemini_live_service_routes_unknown_alias_to_realtime() -> None:
    assert (
        infer_pipeline("future-confidential-alias", service="gemini-live")
        == "realtime"
    )


def test_gemini_live_behavior_flags_reach_run_configuration(monkeypatch) -> None:
    received = {}

    async def fake_run(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli_module, "_run", fake_run)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "placeholder-benchmark",
            "--model",
            "opaque-live-model",
            "--service",
            "gemini-live",
            "--gemini-3-protocol",
            "--gemini-require-interaction-status",
            "--gemini-explicit-audio-activity",
            "--no-turn-replay",
            "--thinking",
            "minimal",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["model"] == "opaque-live-model"
    assert received["gemini_3_protocol"] is True
    assert received["gemini_require_interaction_status"] is True
    assert received["gemini_explicit_audio_activity"] is True
    assert received["no_turn_replay"] is True
