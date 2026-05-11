import base64
import inspect

import pytest

from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from multi_turn_eval.pipelines.audio_in import AudioInPipeline, _audio_prompt


@pytest.fixture
def benchmark():
    return BenchmarkConfig()


@pytest.fixture
def pipeline(benchmark):
    return AudioInPipeline(benchmark)


def _audio_part(message):
    audio_parts = [
        part for part in message["content"] if part.get("type") == "input_audio"
    ]
    assert len(audio_parts) == 1
    return audio_parts[0]["input_audio"]


def _expected_audio_b64(benchmark, turn_index):
    return base64.b64encode(benchmark.get_audio_path(turn_index).read_bytes()).decode(
        "ascii"
    )


def test_first_turn_audio_message(pipeline, monkeypatch):
    monkeypatch.delenv("MTE_AUDIO_IN_PROMPT", raising=False)

    message = pipeline._build_audio_user_message(0)

    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert message["content"][1]["type"] == "input_audio"
    audio = message["content"][1]["input_audio"]
    assert audio["format"] == "wav"
    assert isinstance(audio["data"], str)
    assert audio["data"]


def test_next_turn_uses_actual_benchmark_index(pipeline, benchmark):
    pipeline._turn_indices = [5, 7]

    first_audio = _audio_part(pipeline._build_audio_user_message(0))
    second_audio = _audio_part(pipeline._build_audio_user_message(1))

    assert first_audio["data"] == _expected_audio_b64(benchmark, 5)
    assert second_audio["data"] == _expected_audio_b64(benchmark, 7)
    assert first_audio["data"] != _expected_audio_b64(benchmark, 0)


def test_only_turns_does_not_shift_audio_filenames(pipeline):
    pipeline._turn_indices = [5, 7]

    assert pipeline._get_actual_turn_index(0) == 5
    assert pipeline._get_actual_turn_index(1) == 7


def test_transcript_text_not_in_user_message(pipeline, benchmark, monkeypatch):
    monkeypatch.delenv("MTE_AUDIO_IN_PROMPT", raising=False)
    transcript_text = benchmark.turns[0]["input"]

    message = pipeline._build_audio_user_message(0)

    assert all(part.get("text") != transcript_text for part in message["content"])


def test_audio_prompt_default(monkeypatch):
    monkeypatch.delenv("MTE_AUDIO_IN_PROMPT", raising=False)
    assert _audio_prompt() == "Listen to the audio and respond to the spoken instruction."

    monkeypatch.setenv("MTE_AUDIO_IN_PROMPT", "")
    assert _audio_prompt() is None

    monkeypatch.setenv("MTE_AUDIO_IN_PROMPT", "false")
    assert _audio_prompt() is None

    monkeypatch.setenv("MTE_AUDIO_IN_PROMPT", "Use this custom prompt.")
    assert _audio_prompt() == "Use this custom prompt."


def test_recovery_message_is_text():
    assert "_queue_recovery_turn" in AudioInPipeline.__dict__
    source = inspect.getsource(AudioInPipeline._queue_recovery_turn)
    assert "Please go ahead." in source


def test_missing_audio_raises_file_not_found(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline._build_audio_user_message(999)
