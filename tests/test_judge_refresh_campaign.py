"""Operational safeguards for the paid, resumable refresh; no network requests."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "judge_refresh_campaign", ROOT / "ops/judge-refresh-opus55-20261017/campaign.py"
)
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    monkeypatch.setattr(refresh, "HERE", tmp_path / "artifacts")
    monkeypatch.setattr(refresh, "RAW", tmp_path / "raw")
    refresh.HERE.mkdir()
    source = tmp_path / "transcript.jsonl"
    records = [{"turn": n, "model_name": "hidden", "user_text": "Question", "assistant_text": "Answer", "tool_calls": []} for n in range(30)]
    data = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    source.write_bytes(data)
    spec = refresh.freeze({"key": "example", "label": "Example", "mode": "text", "target": 30,
                           "inputs": [{"capture_id": "original", "source": str(source),
                                       "transcript_sha256": hashlib.sha256(data).hexdigest()}]})
    return spec, spec["inputs"][0]


def result_for(spec):
    return {"judgments": {n: {"scores": {"tool_use_correct": True, "instruction_following": True,
                                          "kb_grounding": True, "turn_taking": True}, "reasoning": "Valid."} for n in range(30)},
            "summary": "", "model_name": "hidden", "realignment_notes": "", "function_tracking": {},
            "judge_metadata": {"judge_model": refresh.MODEL, "judge_effort": refresh.EFFORT,
                               "judge_response_models": [refresh.MODEL], "judge_prompt_sha256": spec["judge_prompt_sha256"]}}


def test_credit_failure_waits_ninety_seconds_and_keeps_both_attempts(monkeypatch, job):
    spec, item = job
    clock, calls = [0.0], []
    monkeypatch.setattr(refresh, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def sleep(seconds):
        clock[0] += seconds

    async def judge(*args, **kwargs):
        calls.append(clock[0])
        if len(calls) == 1:
            raise ValueError("Your credit balance is too low to access the Anthropic API")
        return result_for(spec)

    monkeypatch.setattr(refresh.asyncio, "sleep", sleep)
    monkeypatch.setattr(refresh, "judge_with_claude", judge)
    asyncio.run(refresh.evaluate(spec, item, refresh.CreditCooldown()))
    assert calls == [0, 90]
    first = refresh.RAW / "judging" / item["job_id"] / "attempt-1" / "error.json"
    assert json.loads(first.read_text())["credit_failure"] is True
    assert refresh.validate_existing(spec, item)
    assert json.loads(refresh.result_path(item).read_text())["attempt"] == 2


def test_reconstruction_reports_keep_unscored_audio_null(monkeypatch, job):
    spec, item = job
    spec["mode"] = "voice"
    spec["judge_prompt_sha256"] = hashlib.sha256(refresh.VOICE_RUBRIC.encode()).hexdigest()
    item["turn_taking_scored"] = False

    async def judge(*args, **kwargs):
        return result_for(spec)

    monkeypatch.setattr(refresh, "judge_with_claude", judge)
    asyncio.run(refresh.evaluate(spec, item, refresh.CreditCooldown()))
    result = json.loads(refresh.result_path(item).read_text())
    assert all(j["scores"]["turn_taking"] is None for j in result["judgments"].values())
    run = refresh.ROOT / result["raw_output"]
    summary = json.loads((run / "claude_summary.json").read_text())
    assert summary["claude_passes"]["turn_taking"] is None
    report = (run / "claude_analysis.md").read_text()
    assert "unscored (null)" in report
    assert "Failed Dimensions" not in report


def test_frozen_input_membership_cannot_change(job):
    spec, item = job
    changed = {**spec, "inputs": [{**item, "transcript_sha256": "changed"}]}
    with pytest.raises(AssertionError, match="Refusing to change frozen membership"):
        refresh.freeze(changed)


def test_resume_rejects_a_grade_from_an_obsolete_judge_input(monkeypatch, job):
    spec, item = job

    async def judge(*args, **kwargs):
        return result_for(spec)

    monkeypatch.setattr(refresh, 'judge_with_claude', judge)
    asyncio.run(refresh.evaluate(spec, item, refresh.CreditCooldown()))
    assert refresh.validate_existing(spec, item)
    normalization={'judge_input_normalization': 'think-sections-v1',
                   'judge_normalized_turns': [{'turn': 0, 'source_text_sha256': 'raw',
                                              'judge_text_sha256': 'answer', 'removed_characters': 10}]}
    monkeypatch.setattr(refresh, 'normalization_metadata', lambda _: normalization)
    with pytest.raises(AssertionError, match='normalization mismatch'):
        refresh.validate_existing(spec, item)
    data=json.loads(refresh.result_path(item).read_text())
    data['metadata'].update(normalization)
    refresh.dump(refresh.result_path(item), data)
    assert refresh.validate_existing(spec, item)


def test_provider_safeguard_is_preserved_unscored_without_repeated_rephrasing(monkeypatch, job):
    spec, item = job
    calls=[]

    async def judge(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError('Provider safeguards flagged this message: [reasoning_extraction]')

    monkeypatch.setattr(refresh, 'judge_with_claude', judge)
    asyncio.run(refresh.evaluate(spec, item, refresh.CreditCooldown()))
    assert len(calls)==1
    assert not refresh.result_path(item).exists()
    blocked=json.loads(refresh.blocked_path(item).read_text())
    assert blocked['quality_judgment'] is None
    assert blocked['transcript_sha256']==item['transcript_sha256']
    assert refresh.resolved(spec,item)


def test_incomplete_quality_never_becomes_a_complete_subset_score(monkeypatch, job):
    spec,item=job

    async def judge(*args, **kwargs):
        return result_for(spec)

    monkeypatch.setattr(refresh,'judge_with_claude',judge)
    asyncio.run(refresh.evaluate(spec,item,refresh.CreditCooldown()))
    blocked={**item,'job_id':item['job_id']+'-blocked'}
    spec={**spec,'actual_n':2,'inputs':[item,blocked]}
    refresh.preserve_provider_block(spec,blocked,1,'[reasoning_extraction]')
    monkeypatch.syspath_prepend(str(ROOT/'ops/judge-refresh-opus55-20261017'))
    module_spec=importlib.util.spec_from_file_location('refresh_analyzer',ROOT/'ops/judge-refresh-opus55-20261017/analyze.py')
    analyzer=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(analyzer)
    monkeypatch.setattr(analyzer,'campaign',refresh)
    result=analyzer.quality(spec)
    assert result['scores']['strict']==30
    assert result['graded_turns']==30 and result['unknown_turns']==30
    assert result['strict_percent'] is None and result['strict_ci95'] is None
    assert result['strict_bounds_percent']==[50,100]
    assert result['quality_complete'] is False


def test_resuming_completed_mode_preserves_the_original_phase_boundary(monkeypatch, job):
    spec,item=job

    async def judge(*args, **kwargs):
        return result_for(spec)

    monkeypatch.setattr(refresh,'judge_with_claude',judge)
    asyncio.run(refresh.evaluate(spec,item,refresh.CreditCooldown()))
    monkeypatch.setattr(refresh,'now',lambda:'2026-10-18T01:00:00+00:00')
    first=refresh.write_mode_completion('text',[spec])
    original=(refresh.HERE/'text-COMPLETE.json').read_bytes()
    monkeypatch.setattr(refresh,'now',lambda:'2026-10-19T01:00:00+00:00')
    assert refresh.write_mode_completion('text',[spec])==first
    assert (refresh.HERE/'text-COMPLETE.json').read_bytes()==original
