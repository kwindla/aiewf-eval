"""Service wrappers used by the evaluation harness.

The legacy `NemotronLLMService` (text-mode OpenAI-compatible nemotron) was
written against pipecat 0.0.101 and has not yet been migrated to pipecat
1.1.0. Its import is deferred so the audio-in path (which uses the vendored
upstream service) keeps working while the migration waits for Phase 2.
"""

__all__: list[str] = []

try:
    from multi_turn_eval.services.nemotron import NemotronLLMService  # noqa: F401

    __all__.append("NemotronLLMService")
except ImportError:
    # Pipecat 1.1.0 migration pending for the text-mode nemotron service.
    pass
