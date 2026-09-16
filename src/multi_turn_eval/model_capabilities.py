"""Central model capability classification for routing and provider behavior."""

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities that cannot always be inferred from a public model name."""

    realtime: bool = False
    gemini_live: bool = False
    gemini_3: bool = False
    thinking_levels: bool = False
    allow_turn_replay: bool = True
    default_thinking_mode: str = "default"
    explicit_audio_activity: bool = False
    requires_interaction_status: bool = False


_MODEL_CAPABILITY_OVERRIDES = {
    # Use the same no-replay reliability policy as asynchronous Live models so
    # full-conversation completion is comparable across the speech leaderboard.
    "gemini-3.1-flash-live-preview": ModelCapabilities(
        realtime=True,
        gemini_live=True,
        gemini_3=True,
        thinking_levels=True,
        allow_turn_replay=False,
        default_thinking_mode="minimal",
    ),
    "clever-chatter": ModelCapabilities(
        realtime=True,
        gemini_live=True,
        gemini_3=True,
        thinking_levels=True,
        allow_turn_replay=False,
        default_thinking_mode="minimal",
        explicit_audio_activity=True,
        requires_interaction_status=True,
    ),
    # This model has no asynchronous reasoning and never sends interactionStatus.
    "walkie-talkie": ModelCapabilities(
        realtime=True,
        gemini_live=True,
        gemini_3=True,
        thinking_levels=True,
        allow_turn_replay=False,
        default_thinking_mode="minimal",
        explicit_audio_activity=True,
    ),
}


# Local overrides file MTE_MODEL_CAPABILITIES_FILE > _MODEL_CAPABILITY_OVERRIDES
# Cached per (path, mtime) since lookups happen in frame hot paths.
_local_overrides_cache: tuple[str, float, dict] | None = None

_VALID_FIELDS = {f.name for f in fields(ModelCapabilities)}


def _get_local_overrides() -> dict[str, ModelCapabilities]:
    global _local_overrides_cache
    path = os.getenv("MTE_MODEL_CAPABILITIES_FILE")
    if not path:
        return {}
    mtime = Path(path).stat().st_mtime  # missing file raises FileNotFoundError
    if _local_overrides_cache and _local_overrides_cache[:2] == (path, mtime):
        return _local_overrides_cache[2]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = {}
    for model_id, caps in raw.items():
        unknown = set(caps) - _VALID_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown capability field(s) {sorted(unknown)} for "
                f"{model_id!r} in {path}; valid fields: {sorted(_VALID_FIELDS)}"
            )
        overrides[normalize_model_id(model_id)] = ModelCapabilities(**caps)
    _local_overrides_cache = (path, mtime, overrides)
    return overrides


def normalize_model_id(model: str) -> str:
    """Normalize provider resource prefixes used by Google model listings."""
    normalized = model.strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.removeprefix("models/")
    return normalized


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Return explicit or pattern-derived capabilities for ``model``."""
    normalized = normalize_model_id(model)
    override = _get_local_overrides().get(normalized) or _MODEL_CAPABILITY_OVERRIDES.get(
        normalized
    )
    if override is not None:
        return override

    gemini_live = normalized.startswith("gemini") and (
        "live" in normalized or "native-audio" in normalized
    )
    gemini_3 = "gemini-3" in normalized
    return ModelCapabilities(
        realtime=gemini_live,
        gemini_live=gemini_live,
        gemini_3=gemini_3,
        thinking_levels=gemini_3,
        default_thinking_mode="minimal" if gemini_3 else "default",
    )
