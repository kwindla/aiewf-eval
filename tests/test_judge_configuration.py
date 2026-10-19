import asyncio
import hashlib
import json
from types import SimpleNamespace

from click.testing import CliRunner
import pytest

from multi_turn_eval.cli import cli
import multi_turn_eval.judging.claude_judge as judge


@pytest.fixture
def transcript(tmp_path):
    run = tmp_path / "aiwf_medium_context" / "blind-run"
    run.mkdir(parents=True)
    record = {"turn": 0, "model_name": "secret-model-identity",
              "user_text": "When are workshops?", "assistant_text": "June 3rd.", "tool_calls": []}
    (run / "transcript.jsonl").write_text(json.dumps(record) + "\n")
    return run, record


def install_query(monkeypatch, final=None):
    seen = {}
    if final is None:
        final = [{"turn": 0, "turn_taking": True, "tool_use_correct": True,
                  "instruction_following": True, "kb_grounding": True, "reasoning": "Correct."}]

    async def fake_query(*, prompt, options):
        seen.update(prompt=prompt, options=options)
        yield SimpleNamespace(model=options.model, content=[SimpleNamespace(text=json.dumps({
            "phase1_analysis": [], "realignment_notes": "", "function_call_tracking": {},
            "final_judgments": final,
        }))])

    monkeypatch.setattr(judge, "query", fake_query)
    return seen


def test_explicit_effort_blinding_and_actual_output_metadata(monkeypatch, transcript):
    run, record = transcript
    monkeypatch.setattr(judge, "JUDGE_MODEL", "claude-opus-5-5")
    seen = install_query(monkeypatch)
    monkeypatch.setenv("MTE_JUDGE_EFFORT", "high")
    result = asyncio.run(judge.judge_with_claude(
        run, expected_turns=[{"golden_text": "June 3rd."}], skip_turn_taking=True,
        judge_model="claude-opus-5", judge_effort="medium", judge_system_prompt="frozen rubric",
    ))
    options = seen["options"]
    assert options.model == "claude-opus-5"
    assert options.effort == "medium"
    assert options.thinking == {"type": "adaptive"}
    assert options.tools == [] and options.setting_sources == []
    assert "secret-model-identity" not in seen["prompt"]
    metadata = result["judge_metadata"]
    assert metadata["judge_response_models"] == ["claude-opus-5"]
    assert metadata["judge_prompt_sha256"] == hashlib.sha256(b"frozen rubric").hexdigest()
    assert metadata["judge_input_sha256"] == hashlib.sha256(seen["prompt"].encode()).hexdigest()
    judge.write_outputs(run, [record], result["judgments"], "", record["model_name"], judge_metadata=metadata)
    summary = json.loads((run / "claude_summary.json").read_text())
    assert summary["judge_model"] == "claude-opus-5"
    assert summary["judge_effort"] == "medium"
    assert "**Judge**: claude-opus-5" in (run / "claude_analysis.md").read_text()
    assert "**Judge effort**: medium" in (run / "claude_analysis.md").read_text()
    assert (run / "claude_response.txt").exists()


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_effort_environment(monkeypatch, transcript, effort):
    monkeypatch.setattr(judge, "JUDGE_MODEL", "claude-opus-5-5")
    seen = install_query(monkeypatch)
    monkeypatch.setenv("MTE_JUDGE_EFFORT", effort)
    asyncio.run(judge.judge_with_claude(transcript[0], expected_turns=[{}]))
    assert seen["options"].effort == effort


def test_invalid_effort_never_queries(monkeypatch, transcript):
    seen = install_query(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported judge effort"):
        asyncio.run(judge.judge_with_claude(transcript[0], expected_turns=[{}], judge_effort="minimal"))
    assert not seen


def test_legacy_model_override_keeps_its_own_thinking_defaults(monkeypatch, transcript):
    seen = install_query(monkeypatch)
    monkeypatch.delenv("MTE_JUDGE_EFFORT", raising=False)
    asyncio.run(judge.judge_with_claude(transcript[0], expected_turns=[{}], judge_model="claude-sonnet-4-5"))
    assert seen["options"].effort is None
    assert seen["options"].thinking is None


def test_verified_audio_scores_override_model_and_are_in_the_blind_prompt(monkeypatch, transcript):
    seen = install_query(monkeypatch)
    result = asyncio.run(judge.judge_with_claude(
        transcript[0], expected_turns=[{}],
        precomputed_turn_taking={0: {"turn_taking": False, "issues": ["no recorded speech"]}},
    ))
    assert result["judgments"][0]["scores"]["turn_taking"] is False
    assert "no recorded speech" in seen["prompt"]
    assert "secret-model-identity" not in seen["prompt"]


@pytest.mark.parametrize("data", [{}, {0: {"turn_taking": None}}])
def test_incomplete_or_unscored_audio_data_is_rejected_before_query(monkeypatch, transcript, data):
    seen = install_query(monkeypatch)
    with pytest.raises(ValueError, match="Precomputed turn-taking"):
        asyncio.run(judge.judge_with_claude(
            transcript[0], expected_turns=[{}], precomputed_turn_taking=data,
        ))
    assert not seen


@pytest.mark.parametrize("kind", ["duplicate", "extra", "missing", "nonboolean"])
def test_malformed_scores_rejected_and_raw_response_retained(monkeypatch, transcript, kind):
    valid = {"turn": 0, "tool_use_correct": True, "instruction_following": True, "kb_grounding": True}
    final = [valid]
    if kind == "duplicate":
        final.append(dict(valid))
    elif kind == "extra":
        final.append({**valid, "turn": 1})
    elif kind == "missing":
        final = []
    else:
        valid["kb_grounding"] = "true"
    install_query(monkeypatch, final)
    with pytest.raises(ValueError):
        asyncio.run(judge.judge_with_claude(transcript[0], expected_turns=[{}]))
    assert (transcript[0] / "claude_response.txt").exists()


def test_cli_passes_model_effort_and_benchmark_rubric(monkeypatch, transcript):
    from benchmarks.aiwf_medium_context.config import BenchmarkConfig

    monkeypatch.setattr(judge, "JUDGE_MODEL", "claude-opus-5-5")
    seen = install_query(monkeypatch)
    monkeypatch.setattr(BenchmarkConfig, "judge_system_prompt", "text rubric", raising=False)
    result = CliRunner().invoke(cli, ["judge", str(transcript[0]), "--judge-model", "claude-opus-5",
                                      "--judge-effort", "medium"])
    assert result.exit_code == 0, result.output
    assert seen["options"].model == "claude-opus-5"
    assert seen["options"].effort == "medium"
    assert seen["options"].system_prompt == "text rubric"


def test_speech_rubric_is_unchanged():
    from benchmarks.aiwf_medium_context_speech_v2.judge import judge_system_prompt

    assert hashlib.sha256(judge_system_prompt.encode()).hexdigest() == (
        "c5a8a3b3c3240172cc392529a1784b59d2913a32e2ec2ba843a456415b75ef27"
    )
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as Speech
    assert Speech.judge_system_prompt == judge_system_prompt


def test_text_benchmark_uses_the_frozen_calibrated_rubric():
    from benchmarks.aiwf_medium_context.config import BenchmarkConfig

    assert hashlib.sha256(BenchmarkConfig.judge_system_prompt.encode()).hexdigest() == (
        "70cdc7982a5f2485375e22ff43761ebf74acf21e38bde5d8518a9e95f6cfd3d2"
    )


def test_comprehensive_runner_respects_judge_configuration(monkeypatch, transcript):
    from benchmarks.aiwf_medium_context.config import BenchmarkConfig
    from scripts.run_comprehensive_eval import judge_run

    monkeypatch.setattr(BenchmarkConfig, "judge_system_prompt", "frozen text rubric", raising=False)
    monkeypatch.setattr(judge, "JUDGE_MODEL", "claude-opus-5-5")
    seen = install_query(monkeypatch)
    summary = asyncio.run(judge_run(transcript[0], "aiwf_medium_context", "claude-opus-5", "medium"))
    assert seen["options"].system_prompt == "frozen text rubric"
    assert seen["options"].model == "claude-opus-5"
    assert seen["options"].effort == "medium"
    assert summary["judge_model"] == "claude-opus-5"
    assert summary["judge_effort"] == "medium"


def test_task_clarification_is_additive_and_auditable(monkeypatch, transcript):
    original = (transcript[0] / 'transcript.jsonl').read_bytes()
    seen = install_query(monkeypatch)
    baseline = asyncio.run(judge.judge_with_claude(transcript[0], expected_turns=[{}], skip_turn_taking=True))
    ordinary_prompt = seen['prompt']
    note = 'Evaluate the recorded output and provide brief score justifications.'
    clarified = asyncio.run(judge.judge_with_claude(
        transcript[0], expected_turns=[{}], skip_turn_taking=True,
        task_clarification=note,
    ))
    assert seen['prompt'] == ordinary_prompt + '\n\n# Task clarification\n' + note
    assert clarified['judge_metadata']['judge_task_clarification'] == note
    assert clarified['judge_metadata']['judge_prompt_sha256'] == baseline['judge_metadata']['judge_prompt_sha256']
    assert clarified['judge_metadata']['judge_input_sha256'] != baseline['judge_metadata']['judge_input_sha256']
    assert (transcript[0] / 'transcript.jsonl').read_bytes() == original


@pytest.mark.parametrize(('source', 'answer'), [
    ('  Ordinary answer.\n', '  Ordinary answer.\n'),
    ('Private analysis.\n</think>\nJune 3rd.', 'June 3rd.'),
    ('<think>Private analysis.</think>June 3rd.', 'June 3rd.'),
    ('Before.<think>Private analysis.</think>After.', 'Before.After.'),
    ('<think>One.</think>A<think>Two.</think>B', 'AB'),
    ('Private analysis.</think>', ''),
    ('</think>Final answer.', 'Final answer.'),
])
def test_reasoning_cleanup_uses_boundaries_and_is_idempotent(source, answer):
    assert judge.strip_reasoning_sections(source) == answer
    assert judge.strip_reasoning_sections(answer) == answer


@pytest.mark.parametrize('source', [
    '<think>Unclosed analysis',
    '<think>Nested <think>analysis</think></think>',
    'Analysis</think>Answer</think>Other answer',
])
def test_ambiguous_reasoning_boundaries_are_rejected(source):
    with pytest.raises(ValueError, match='think'):
        judge.strip_reasoning_sections(source)


def test_judge_excludes_reasoning_preserves_tool_actions_and_raw_source(monkeypatch, transcript):
    run, record = transcript
    record['assistant_text'] = 'Private thought with an incorrect claim.</think>June 3rd.'
    record['tool_calls'] = [{'name': 'end_session', 'args': {'reason': 'finished'}}]
    (run/'transcript.jsonl').write_text(json.dumps(record)+'\n')
    original = (run/'transcript.jsonl').read_bytes()
    seen = install_query(monkeypatch)
    result = asyncio.run(judge.judge_with_claude(
        run, expected_turns=[{}], skip_turn_taking=True,
    ))
    assert 'Private thought' not in seen['prompt']
    assert '**Assistant**: June 3rd.' in seen['prompt']
    assert 'end_session' in seen['prompt'] and 'finished' in seen['prompt']
    assert (run/'transcript.jsonl').read_bytes() == original
    meta = result['judge_metadata']
    assert meta['judge_input_normalization'] == judge.REASONING_NORMALIZATION
    assert meta['judge_normalized_turns'] == [{
        'turn': 0,
        'source_text_sha256': hashlib.sha256(record['assistant_text'].encode()).hexdigest(),
        'judge_text_sha256': hashlib.sha256(b'June 3rd.').hexdigest(),
        'removed_characters': len(record['assistant_text'])-len('June 3rd.'),
    }]


def test_reasoning_only_tool_turn_keeps_the_tool_call():
    records = [{'turn': 0, 'assistant_text': 'Private analysis.</think>',
                'tool_calls': [{'name': 'request_tech_support', 'args': {'issue': 'app'}}]}]
    prepared, changes = judge.prepare_judge_records(records)
    assert prepared[0]['assistant_text'] == ''
    assert prepared[0]['tool_calls'] == records[0]['tool_calls']
    assert records[0]['assistant_text'] == 'Private analysis.</think>'
    assert len(changes) == 1
