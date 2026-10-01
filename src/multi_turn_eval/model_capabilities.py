"""Central model capability classification for routing and provider behavior."""

from dataclasses import dataclass


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
    # GA successor to clever-chatter. The GA endpoint rejects minimal thinking;
    # low is the least expensive supported level and preserves the asynchronous
    # interaction-status protocol exercised by the EA benchmark.
    "gemini-3.8-live-extended-thinking": ModelCapabilities(
        realtime=True,
        gemini_live=True,
        gemini_3=True,
        thinking_levels=True,
        allow_turn_replay=False,
        default_thinking_mode="low",
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
    # The standard GA model rejects thinking levels and has the synchronous
    # turn-complete behavior previously observed on walkie-talkie.
    "gemini-3.8-live": ModelCapabilities(
        realtime=True,
        gemini_live=True,
        gemini_3=True,
        allow_turn_replay=False,
        default_thinking_mode="default",
        explicit_audio_activity=True,
    ),
}


def normalize_model_id(model: str) -> str:
    """Normalize provider resource prefixes used by Google model listings."""
    normalized = model.strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.removeprefix("models/")
    return normalized


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Return explicit or pattern-derived capabilities for ``model``."""
    normalized = normalize_model_id(model)
    override = _MODEL_CAPABILITY_OVERRIDES.get(normalized)
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
