"""Nova 2.5 Sonic prompt-optimized row; only the SYSTEM policy and tool descriptions differ.

Turns, audio, knowledge base bytes, tool names and parameters, and the speech-v2 judge are
inherited unchanged. Run with MTE_NOVA_KB_CONTEXT_ROLE=user, like the standard Nova Sonic row.
"""
from pathlib import Path
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as SpeechV2
from .tools import ToolsSchemaNovaSonicOptimized

HERE = Path(__file__).resolve().parent


class BenchmarkConfig(SpeechV2):
    name = "aiwf_medium_context_speech_nova_sonic_optimized"
    description = "Nova 2.5 Sonic prompt-optimized speech benchmark"
    allowed_pipelines = ("nova-sonic",)
    prompt_version = "nova-sonic-optimized-v1"
    system_policy = (HERE / "prompts/system.txt").read_text()
    # The unchanged speech-v2 knowledge base follows the policy, as in the standard row
    reference_context = SpeechV2.reference_context
    system_instruction = system_policy + reference_context
    tools_schema = ToolsSchemaNovaSonicOptimized