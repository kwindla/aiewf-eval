"""Separate speech prompt; identical tasks, knowledge base, audio, and tools."""
from benchmarks.aiwf_medium_context.config import BenchmarkConfig as LegacyBenchmark
from .prompts.system import system_instruction


class BenchmarkConfig(LegacyBenchmark):
    name = "aiwf_medium_context_speech_v2"
    description = "Medium context speech-to-speech benchmark, voice policy v2"
    allowed_pipelines = ("realtime", "grok-realtime", "nova-sonic")
    prompt_version = "speech-v2"
    system_instruction = system_instruction

    from .judge import judge_system_prompt
