"""High/none comparison must preserve the baseline and reconcile to raw captures."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'ops/grok-voice-high-20261018'
OLD=ROOT/'ops/recent-models-20261018'
DIMS=('tool_use_correct','instruction_following','kb_grounding')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_high_uses_the_same_voice_tasks_and_an_explicit_high_setting():
    high=json.loads((HERE/'grok-voice/configuration.json').read_text())
    none=json.loads((OLD/'grok-voice/configuration.json').read_text())
    for field in ['model','benchmark','service','pipeline','endpoint','common_environment','unset_environment_prefixes']:
        assert high[field]==none[field]
    assert high['target_eligible_runs']==25
    assert set(high['arms'])=={'high'}
    assert high['arms']['high']['environment']=={'MTE_XAI_VOICE_REASONING_EFFORT':'high'}
    assert high['arms']['high']['provenance_log_needles']==['session reasoning.effort=high']
    schedule=(ROOT/high['schedule_path']).read_text().splitlines()
    assert schedule==['slot\tarm']+[f'{i}\thigh' for i in range(1,26)]
    captured=dict(line.split(maxsplit=1)[::-1] for line in (OLD/'grok-voice/artifacts/source-sha256.txt').read_text().splitlines())
    for name in ['src/multi_turn_eval/pipelines/grok_realtime.py','src/multi_turn_eval/pipelines/realtime.py','src/multi_turn_eval/cli.py','src/multi_turn_eval/model_capabilities.py']:
        if name=='src/multi_turn_eval/cli.py':
            archive=ROOT/'ops/judge-recovery-context-20261019'
            snapshot=archive/'legacy'/name
            assert json.loads((archive/'legacy-sha256.json').read_text())['legacy/'+name]==captured[name]
        else:
            snapshot=OLD/'runtime-snapshot'/name
            assert json.loads((OLD/'runtime-sha256.json').read_text())[name]==captured[name]
        assert sha(snapshot)==captured[name]

def test_completed_none_study_is_immutable():
    for name,digest in json.loads((HERE/'baseline-sha256.json').read_text()).items():
        assert sha(ROOT/name)==digest,name

def test_high_results_reconcile_to_all_frozen_judgments_and_audio():
    if not (HERE/'PUBLICATION.json').exists():
        pytest.skip('High cohort has not finished publication')
    data=json.loads((HERE/'aggregate.json').read_text())
    comparison=json.loads((HERE/'comparison.json').read_text())
    inputs=json.loads((HERE/'judging/inputs.json').read_text())
    m=data['voice_models'][0]
    assert data['all_quality_complete'] and data['judged_conversations']==25
    assert data['text_models']==[] and m['arm']=='high' and m['complete']==25
    assert comparison['none']==json.loads((OLD/'aggregate.json').read_text())['voice_models'][0]
    assert comparison['high']==m
    assert comparison['strict_delta_percentage_points']==m['strict_percent']-comparison['none']['strict_percent']
    assert len(inputs['cases'])==25 and len(set(c['job_id'] for c in inputs['cases']))==25
    counts={d:0 for d in DIMS};strict=0;cost=0;turn_taking=0
    for c in inputs['cases']:
        assert c['name']=='grok-voice' and c['arm']=='high'
        assert c['observed_turns']==list(range(30))
        assert (ROOT/c['source']).read_bytes()==(ROOT/c['input']).read_bytes()
        assert sha(ROOT/c['source'])==c['transcript_sha256']
        r=json.loads((HERE/'judging/results'/(c['job_id']+'.json')).read_text());meta=r['metadata']
        assert r['transcript_sha256']==c['transcript_sha256']
        assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
        assert meta['judge_response_models']==['claude-opus-5-5'] and meta['judge_tools']==[] and meta['judge_setting_sources']==[]
        assert meta['judge_prompt_sha256']==inputs['rubric_sha256']['voice']
        assert set(r['judgments'])==set(map(str,range(30)))
        assert r['turn_taking_analysis'] and not r['turn_taking_analysis'].get('error')
        cost+=meta['judge_cost_usd']
        for j in r['judgments'].values():
            assert all(type(j['scores'][d]) is bool for d in DIMS+('turn_taking',))
            strict+=all(j['scores'][d] for d in DIMS)
            turn_taking+=j['scores']['turn_taking']
            for d in DIMS:counts[d]+=j['scores'][d]
        folder=HERE/'grok-voice/audio';proof=json.loads((folder/(c['job_id']+'-source.json')).read_text())
        assert proof['identity']['run_dir']==str(Path(c['source']).parent)
        for name,digest in proof['identity']['raw_sha256'].items():assert sha(ROOT/Path(c['source']).parent/name)==digest
        assert sha(folder/(c['job_id']+'.json'))==proof['metrics_sha256']
        wave=folder/(c['job_id']+'-input-waveform.json');assert sha(wave)==proof['input_waveform_sha256']
        assert not json.loads(wave.read_text())['issues']
    assert counts==m['complete_only']['dimension_passes']
    assert strict==m['complete_only']['strict_passes']
    assert m['complete_only']['denominator']==750 and m['strict_percent']==100*strict/750
    assert turn_taking==m['turn_taking_passes']
    assert abs(cost-data['successful_judge_cost_usd'])<1e-9
    assert m['first_speech_positive']['n']+m['nonpositive_n']+m['missing_latency_n']==750
    attempts=json.loads((HERE/'grok-voice/attempts.json').read_text())
    assert m['eligible_attempts']==len(attempts) and m['failed_conversations']==sum(not a['complete'] for a in attempts)
    row=next(line for line in (ROOT/'README.md').read_text().splitlines() if line.startswith('| '+m['label']+' |'))
    cells=[v.strip() for v in row.strip('|').split('|')]
    assert cells[1:4]==['25',f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",f"{m['strict_percent']:.1f}%"]
    assert cells[4:7]==[f'{counts[d]}/750' for d in DIMS]
    assert cells[7]==f'{turn_taking}/750'
    fmt=lambda v:'—' if v is None else f'{v:.0f}ms'
    assert cells[8:]==[fmt(m['non_tool']['median_ms']),fmt(m['non_tool']['max_ms']),fmt(m['tool']['mean_ms']),fmt(m['silence']['mean_ms'])]
    for name,digest in json.loads((HERE/'judging/driver-sha256.json').read_text()).items():assert sha(ROOT/name)==digest

# These reconcile immutable full captures, not just the public summary.
pytestmark = pytest.mark.local_evidence
