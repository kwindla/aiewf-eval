"""Isolation of the canonical voice policy from existing text evaluations."""
import asyncio
import pytest
import click
from benchmarks.aiwf_medium_context.config import BenchmarkConfig as TextConfig
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as VoiceConfig
from multi_turn_eval.cli import _run
from multi_turn_eval.judging.claude_judge import JUDGE_SYSTEM_PROMPT


def test_voice_policy_shares_tasks_but_not_instructions():
    text, voice = TextConfig(), VoiceConfig()
    assert text.turns is voice.turns
    assert text.tools_schema is voice.tools_schema
    assert text.audio_dir == voice.audio_dir
    assert text.system_instruction != voice.system_instruction
    assert 'you must ask for their name and preference' in text.system_instruction
    assert 'you must ask for their name and preference' not in voice.system_instruction
    assert not hasattr(text, 'judge_system_prompt')
    assert 'Model\'s governing voice policy' in voice.judge_system_prompt
    assert 'Model\'s governing voice policy' not in JUDGE_SYSTEM_PROMPT


@pytest.mark.parametrize('pipeline', ['text', 'audio-in'])
def test_voice_benchmark_rejects_text_pipelines_before_startup(pipeline):
    with pytest.raises(click.UsageError, match='supports only'):
        asyncio.run(_run(
            benchmark_name='aiwf_medium_context_speech_v2', model='gpt-4o',
            service='openai', pipeline_type=pipeline, only_turns=None,
            thinking=None, verbose=False,
        ))
