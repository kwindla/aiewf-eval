"""Project-wide model eligibility rules."""

from __future__ import annotations


OPENAI_PRO_EXCLUSION_MESSAGE = (
    "OpenAI Pro models are excluded from this conversational-latency "
    "benchmark; choose a non-Pro model."
)


def is_openai_pro_model(model: str, service: str | None = None) -> bool:
    """Return whether a model is an OpenAI ``*-pro`` latency-ineligible variant."""
    normalized = model.strip().lower()
    model_id = normalized.rsplit("/", 1)[-1]
    is_pro_variant = model_id.endswith("-pro") or "-pro-" in model_id
    if not is_pro_variant:
        return False

    service_id = (service or "").strip().lower()
    explicitly_openai = service_id == "openai" or normalized.startswith("openai/")
    recognizable_openai_id = model_id.startswith(("gpt-", "chatgpt-", "codex-"))
    recognizable_openai_id = recognizable_openai_id or (
        len(model_id) >= 2 and model_id[0] == "o" and model_id[1].isdigit()
    )
    return explicitly_openai or recognizable_openai_id
