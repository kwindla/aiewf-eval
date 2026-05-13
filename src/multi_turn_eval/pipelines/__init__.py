"""Pipeline implementations for different LLM service types.

Pipelines handle the full execution of multi-turn benchmarks including:
- Creating and configuring LLM services
- Managing turn flow (queuing turns, detecting end-of-turn)
- Recording transcripts and metrics
- Handling reconnection for long-running sessions

Available pipelines:
- TextPipeline: For text-based LLM services (OpenAI, Anthropic, Google, etc.)
- RealtimePipeline: For speech-to-speech services (OpenAI Realtime, Gemini Live)
- GrokRealtimePipeline: For xAI Grok Voice Agent API
- NovaSonicPipeline: For AWS Nova Sonic speech-to-speech service
- AudioInPipeline: For Nemotron audio-in/text-out (vendored upstream service)

NOTE: During the pipecat 0.0.101 -> 1.1.0 migration we deferred fixing the
realtime / grok-realtime / nova-sonic pipelines. Imports of those modules are
wrapped in try/except so the audio-in (and text) paths keep working while the
other pipelines wait for their migration in Phase 2. The CLI's
`get_pipeline_class()` does its own `importlib.import_module` per request and
will surface the breakage at use-time when those modules are needed.
"""

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.pipelines.text import TextPipeline

__all__ = ["BasePipeline", "TextPipeline"]

try:
    from multi_turn_eval.pipelines.audio_in import AudioInPipeline  # noqa: F401

    __all__.append("AudioInPipeline")
except ImportError as _exc:  # pragma: no cover
    pass

try:
    from multi_turn_eval.pipelines.realtime import (  # noqa: F401
        RealtimePipeline,
        GeminiLiveLLMServiceWithReconnection,
    )

    __all__.extend(["RealtimePipeline", "GeminiLiveLLMServiceWithReconnection"])
except ImportError:
    # Pipecat 1.1.0 migration pending for realtime pipeline. See Phase 2.
    pass

try:
    from multi_turn_eval.pipelines.grok_realtime import (  # noqa: F401
        GrokRealtimePipeline,
        XAIRealtimeLLMService,
    )

    __all__.extend(["GrokRealtimePipeline", "XAIRealtimeLLMService"])
except ImportError:
    # Pipecat 1.1.0 migration pending for grok-realtime pipeline. See Phase 2.
    pass

try:
    from multi_turn_eval.pipelines.nova_sonic import (  # noqa: F401
        NovaSonicPipeline,
        NovaSonicLLMServiceWithCompletionSignal,
        NovaSonicTurnGate,
    )

    __all__.extend(
        [
            "NovaSonicPipeline",
            "NovaSonicLLMServiceWithCompletionSignal",
            "NovaSonicTurnGate",
        ]
    )
except ImportError:
    # Pipecat 1.1.0 migration pending for nova-sonic pipeline. See Phase 2.
    pass
