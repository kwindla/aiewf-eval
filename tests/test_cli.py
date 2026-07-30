import pytest
from click import UsageError

from multi_turn_eval.cli import reject_openai_pro_model


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
