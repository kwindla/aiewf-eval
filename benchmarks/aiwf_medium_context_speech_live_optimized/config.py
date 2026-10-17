"""Separate prompt-optimized GPT-Live experiment; shared behavioral judge."""
from pathlib import Path
from benchmarks.aiwf_medium_context_speech_live.config import BenchmarkConfig as Standard
HERE = Path(__file__).resolve().parent
class BenchmarkConfig(Standard):
    name = 'aiwf_medium_context_speech_live_optimized'
    description = 'GPT-Live prompt-optimized; reconstructed-exchange experimental protocol'
    prompt_version = 'gpt-live-optimized-v1'
    frontend_system_instruction = (HERE/'prompts/frontend.txt').read_text()
    backend_system_instruction = (HERE/'prompts/backend.txt').read_text()
