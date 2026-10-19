import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from multi_turn_eval.judging import claude_judge as judge
from multi_turn_eval.judging.context import outcome_text, validate_context

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT/'ops/judge-recovery-context-20261019'

def record(turn, **kwargs):
    return dict(turn=turn, user_text='Please submit it.', assistant_text='Done.', tool_calls=[], **kwargs)

def recovery(**kwargs):
    return record(30, recovery_turn=True, recovery_for_turn=0, **kwargs)

def test_context_is_chronological_unscored_and_lossless():
    rows = [record(0), recovery(tool_results=[{'name':'support', 'response':{'result':{'status':'success'},'is_duplicate':False}}]),
            record(31, recovery_turn=True), record(1)]
    original = deepcopy(rows)
    text = judge.format_turns_for_claude(rows, [{},{}])
    assert text.index('## Turn 0') < text.index('Recovery R1') < text.index('Recovery R2') < text.index('## Turn 1')
    assert '"status": "success"' in text and '"is_duplicate": false' in text
    assert 'Scripted scoring targets: [0, 1]' in text
    assert 'unnecessary deferral' in text and 'earlier recovery can satisfy' in text
    assert validate_context(rows,2)[2]['parent_inferred']
    assert rows == original

@pytest.mark.parametrize('response', [{'status':'failed'}, {'status':'duplicate_skipped'}, 'tool failed'])
def test_older_outcomes_are_preserved(response):
    assert json.dumps(response) in outcome_text({'tool_results':[{'name':'vote','response':response}]})
    assert 'unrecorded' in outcome_text({})

def test_full_context_survives_target_filter_and_private_text_is_removed():
    rows = [record(0), recovery(), record(1)]
    rows[1]['assistant_text'] = '<think>private analysis</think>Public response.'
    normalized, changes = judge.prepare_judge_records(rows)
    text = judge.format_turns_for_claude(normalized,[{},{}],{1})
    assert '**Scoring**: context only; do not judge' in text
    assert 'Recovery R1' in text and 'Public response.' in text and 'private analysis' not in text
    assert changes[0]['record_index']==1 and changes[0]['recovery_turn']
    assert 'private analysis' not in judge.format_turns_for_claude(rows,[{},{}])

@pytest.mark.parametrize('rows', [[recovery()], [record(0),record(30,recovery_turn=True,recovery_for_turn=1)],
                                [record(0),record(0)], [record(1),record(0)], [record(2)]])
def test_invalid_ancestry_and_scripted_ids_fail_closed(rows):
    with pytest.raises(ValueError):
        validate_context(rows,2)

def test_no_recovery_format_is_exactly_legacy():
    expected = (ROOT/"tests/fixtures/judge-no-recovery.txt").read_text()
    assert judge.format_turns_for_claude([record(0), record(1)], [{}, {}]) == expected

@pytest.mark.parametrize('invalid', [True,False,0.0,'R1'])
def test_response_ids_cannot_alias_scripted_integers(tmp_path,monkeypatch,invalid):
    (tmp_path/'transcript.jsonl').write_text(json.dumps(record(0))+'\n')
    async def query(**kw):
        yield SimpleNamespace(model=kw['options'].model,content=[SimpleNamespace(text=json.dumps({'phase1_analysis':[], 'realignment_notes':'', 'function_call_tracking':{}, 'final_judgments':[
            {'turn':invalid, 'tool_use_correct':True,'instruction_following':True,'kb_grounding':True}]}))])
    monkeypatch.setattr(judge,'query',query)
    with pytest.raises(ValueError,match='exactly one judgment'):
        asyncio.run(judge.judge_with_claude(tmp_path,expected_turns=[{}],skip_turn_taking=True))

def test_target_filter_changes_scope_but_keeps_all_records(tmp_path,monkeypatch):
    rows = [record(0), recovery(), record(1)]
    (tmp_path/'transcript.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    seen = {}
    async def query(**kw):
        seen.update(kw)
        yield SimpleNamespace(model=kw['options'].model,content=[SimpleNamespace(text=json.dumps({'phase1_analysis':[], 'realignment_notes':'', 'function_call_tracking':{}, 'final_judgments':[
            {'turn':1,'tool_use_correct':True,'instruction_following':True,'kb_grounding':True}]}))])
    monkeypatch.setattr(judge,'query',query)
    result = asyncio.run(judge.judge_with_claude(tmp_path,expected_turns=[{},{}],only_turns={1},skip_turn_taking=True))
    assert 'Recovery R1' in seen['prompt'] and '## Turn 0' in seen['prompt']
    assert 'only the explicitly declared scripted scoring target IDs' in seen['options'].system_prompt
    assert set(result['judgments'])=={1}
    assert result['judge_metadata']['judge_scored_targets']==[1]
