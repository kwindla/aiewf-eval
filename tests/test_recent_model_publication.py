"""New rows must reconcile to all registered grades, including early endings."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'ops/recent-models-20261018'
DIMS=('tool_use_correct','instruction_following','kb_grounding')

def test_recent_matrix_has_the_approved_counts_and_explicit_modes():
    configurations=[json.loads(p.read_text()) for p in HERE.glob('*/configuration.json')]
    assert len(configurations)==6
    assert sum(len(c['arms']) for c in configurations)==10
    assert sum(c['target_eligible_runs'] for c in configurations)==295
    assert {c['model'] for c in configurations}=={'gpt-6-luna','claude-sonnet-5-5','claude-haiku-5-5','meta/muse-spark-1.3','meta-models/muse-spark-open','grok-voice-3.0'}
    for c in configurations:
        lines=(ROOT/c['schedule_path']).read_text().splitlines()
        assert len(lines)-1==c['target_eligible_runs']
        assert all(a['environment'] for a in c['arms'].values())
        assert c['common_environment']['MTE_ENABLE_RECOVERY']==('0' if c['model']=='grok-voice-3.0' else '1')

@pytest.mark.local_evidence
def test_recent_grades_reconcile_completed_and_fixed_denominators():
    if not (HERE/'aggregate.json').exists():
        pytest.skip('Campaign has not completed judging')
    data=json.loads((HERE/'aggregate.json').read_text())
    inputs=json.loads((HERE/'judging/inputs.json').read_text())
    cases={c['job_id']:c for c in inputs['cases']}
    assert len(cases)==295
    assert data['all_quality_complete']
    cost=0
    seen=[]
    for model in data['text_models']+data['voice_models']:
        strict=0;counts={d:0 for d in DIMS}
        for job in model['cases']:
            c=cases[job];result=json.loads((HERE/'judging/results'/(job+'.json')).read_text())
            assert (c['name'],c['arm'])==(model['name'],model['arm'])
            seen.append(job)
            meta=result['metadata']
            assert result['transcript_sha256']==c['transcript_sha256']
            assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
            assert meta['judge_response_models']==['claude-opus-5-5']
            assert meta['judge_tools']==[] and meta['judge_setting_sources']==[]
            cost+=meta.get('judge_cost_usd',0) or 0
            rubric=ROOT/('benchmarks/aiwf_medium_context_speech_v2/judge.py' if c['name']=='grok-voice' else 'benchmarks/aiwf_medium_context/judge.py')
            assert set(result['judgments'])==set(map(str,c['observed_turns']))
            for j in result['judgments'].values():
                assert all(type(j['scores'][d]) is bool for d in DIMS)
                for d in DIMS:counts[d]+=j['scores'][d]
                strict+=all(j['scores'][d] for d in DIMS)
        assert model['all_assigned']['strict_passes']==strict
        assert model['all_assigned']['dimension_passes']==counts
        assert model['all_assigned']['denominator']==30*model['assigned']
        assert abs(model['all_assigned']['strict_percent']-100*strict/(30*model['assigned']))<1e-10
        assert model['complete_only']['denominator']==30*model['complete']
        assert model['complete']+model['incomplete']==model['assigned']
        if model['complete']==0:
            assert model['strict_percent'] is None
            assert model['latency']['positive']['n']==0
            assert model['latency']['positive']['p50_ms'] is None
    assert len(seen)==len(set(seen))==len(cases)
    assert set(seen)==set(cases)
    assert abs(cost-data['successful_judge_cost_usd'])<1e-8
    if (HERE/'PUBLICATION.json').exists():
        voice=data['voice_models'][0]
        text=(ROOT/'README.md').read_text().split('Speech-to-speech models:\n',1)[1]
        row=next(line for line in text.splitlines() if line.startswith('| '+voice['label']+' |'))
        cells=[c.strip() for c in row.strip('|').split('|')]
        den=30*voice['complete']
        assert cells[1]==str(voice['complete'])
        assert cells[2]==f"{voice['failed_conversations']}/{voice['eligible_attempts']} ({voice['failure_rate_percent']:.1f}%)"
        assert cells[3]==f"{voice['strict_percent']:.1f}%"
        assert cells[4:7]==[f"{voice['complete_only']['dimension_passes'][d]}/{den}" for d in DIMS]
        assert cells[7]==(f"{voice['turn_taking_passes']}/{den}" if voice['turn_taking_scored'] else '—')
        fmt=lambda v:'—' if v is None else f'{v:.0f}ms'
        assert cells[8:]==[fmt(voice['non_tool']['median_ms']),fmt(voice['non_tool']['max_ms']),fmt(voice['tool']['mean_ms']),fmt(voice['silence']['mean_ms'])]
