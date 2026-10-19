"""Configuration for the medium context benchmark (~12K tokens)."""
from pathlib import Path

from benchmarks._shared import turns, ToolsSchemaForTest
from .prompts.system import system_instruction
from .judge import judge_system_prompt


class BenchmarkConfig:
    """Configuration for the medium context benchmark."""

    # Benchmark metadata
    name = "aiwf_medium_context"
    description = "Medium context benchmark with ~12K token knowledge base"

    # Shared data
    turns = turns
    tools_schema = ToolsSchemaForTest

    # Audio directory path
    audio_dir = Path(__file__).parent.parent / "_shared" / "audio"

    # System prompt
    system_instruction = system_instruction
    judge_system_prompt = judge_system_prompt

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn."""
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
