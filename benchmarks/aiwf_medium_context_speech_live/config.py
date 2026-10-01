"""GPT-Live entry point; exact shared speech-v2 prompt, tools, tasks and judge."""
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as SpeechV2

class BenchmarkConfig(SpeechV2):
    name = 'aiwf_medium_context_speech_live'
    allowed_pipelines = ('openai-live',)
