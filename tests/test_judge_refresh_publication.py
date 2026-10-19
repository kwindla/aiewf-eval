"""Current tables must trace to the complete, uniformly judged frozen refresh."""

import pytest
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'ops/judge-refresh-opus55-20261017'
DIMS = ('tool_use_correct','instruction_following','kb_grounding')


def read(name):
    return json.loads((DATA/name).read_text())


def pct(value):
    return '—' if value is None else f'{value:.1f}%'


def ms(value):
    return '—' if value is None else f'{value:.0f}ms'


def test_all_selected_inputs_have_uniform_grades_or_explicit_provider_blocks():
    aggregate=read('aggregate.json')
    assert aggregate['processing_complete'] is True
    assert {k:aggregate['counts'][k] for k in ('text_configurations','text_conversations','voice_configurations','voice_conversations','scripted_turns')}=={'text_configurations':43,'text_conversations':1091,
                                'voice_configurations':9,'voice_conversations':225,'scripted_turns':39480}
    jobs=set()
    total_graded=0
    total_blocked=0
    clarifications=read('task-clarifications.json')
    cleanup_path=DATA/'reasoning-cleanup/manifest.json'
    cleaned={i['job_id']:i for i in json.loads(cleanup_path.read_text())['jobs']} if cleanup_path.exists() else {}
    for m in aggregate['text_models']+aggregate['voice_models']:
        c=read(m['manifest'])
        assert c['actual_n']==m['complete']
        assert c['judge_model']=='claude-opus-5-5' and c['judge_effort']=='medium'
        assert c['judge_prompt_sha256']==hashlib.sha256((DATA/f"rubric-{c['mode']}.txt").read_bytes()).hexdigest()
        counts={dim:0 for dim in DIMS}; strict=0; cost=0; unknown_cost=0; graded=0; blocked=0
        for i in c['inputs']:
            assert i['job_id'] not in jobs
            jobs.add(i['job_id'])
            block_path=DATA/'provider-blocked'/(i['job_id']+'.json')
            if block_path.exists():
                block=json.loads(block_path.read_text())
                assert block['status']=='provider_safeguard_rejection'
                assert block['transcript_sha256']==i['transcript_sha256']
                assert block['judge_model']=='claude-opus-5-5' and block['judge_effort']=='medium'
                assert block['judge_prompt_sha256']==c['judge_prompt_sha256']
                assert block['quality_judgment'] is None
                assert not (DATA/'results'/(i['job_id']+'.json')).exists()
                blocked+=1
                continue
            graded+=1
            result=read('results/'+i['job_id']+'.json')
            assert result['transcript_sha256']==i['transcript_sha256']
            meta=result['metadata']
            assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
            assert meta['judge_response_models']==['claude-opus-5-5']
            assert meta['judge_thinking']=='adaptive'
            assert meta['judge_tools']==[] and meta['judge_setting_sources']==[]
            if i['job_id'] in cleaned:
                transform=cleaned[i['job_id']]
                assert meta['judge_input_normalization']==transform['judge_input_normalization']
                assert meta['judge_normalized_turns']==transform['judge_normalized_turns']
                assert transform['transcript_sha256']==i['transcript_sha256']
                assert 'judge_task_clarification' not in meta
            elif i['job_id'] in clarifications:
                assert meta['judge_task_clarification']==clarifications[i['job_id']]['text']
            else:
                assert 'judge_task_clarification' not in meta
            assert meta['judge_prompt_sha256']==c['judge_prompt_sha256']
            assert set(result['judgments'])=={str(t) for t in range(30)}
            value=meta.get('judge_cost_usd')
            if type(value) in (int,float) and math.isfinite(value): cost+=value
            else: unknown_cost+=1
            for t,j in result['judgments'].items():
                scores=j['scores']
                assert all(type(scores[d]) is bool for d in DIMS)
                for d in DIMS:counts[d]+=scores[d]
                strict+=all(scores[d] for d in DIMS)
                if c['mode']=='voice':
                    if i['turn_taking_scored']:
                        assert scores['turn_taking'] is i['turn_taking_data'][t]['turn_taking']
                    else:assert scores['turn_taking'] is None
        assert counts=={d:m['scores'][d] for d in DIMS}
        assert strict==m['scores']['strict']
        assert m['accuracy_denominator']==30*m['complete']
        assert m['graded_conversations']==graded
        assert m['provider_blocked_conversations']==blocked
        assert m['graded_turns']==30*graded and m['unknown_turns']==30*blocked
        assert m['quality_complete'] is (blocked==0)
        if blocked:
            assert m['strict_percent'] is None and m['strict_ci95'] is None
            assert all(v is None for v in m['dimension_error_percent'].values())
        else:
            assert abs(m['strict_percent']-100*strict/m['accuracy_denominator'])<1e-10
        total_graded+=graded;total_blocked+=blocked
        assert abs(cost-m['successful_judge_cost_usd'])<1e-8
        assert unknown_cost==m['unknown_cost_jobs']
    assert len(jobs)==1316
    assert total_graded+total_blocked==1316
    assert total_graded==aggregate['counts']['graded_conversations']
    assert total_blocked==aggregate['counts']['provider_blocked_conversations']
    assert aggregate['all_quality_complete'] is (total_blocked==0)
    assert aggregate['complete'] is (total_blocked==0)
    if 'cost_accounting' in aggregate:
        archived=[json.loads(p.read_text()) for p in (DATA/'reasoning-cleanup/prior-results').glob('*.json')]
        old_cost=sum(r['metadata'].get('judge_cost_usd',0) or 0 for r in archived)
        current_cost=sum(m['successful_judge_cost_usd'] for m in aggregate['text_models']+aggregate['voice_models'])
        accounting=aggregate['cost_accounting']
        assert accounting['superseded_successful_requests']==len(archived)
        assert abs(accounting['total_successful_request_cost_usd']-current_cost-old_cost)<1e-8
        events=[json.loads(line) for line in (DATA/'attempts.jsonl').read_text().splitlines()]
        billed=sum(e.get('cost_usd',0) or 0 for e in events if e['event']=='done')
        assert abs(accounting['total_successful_request_cost_usd']-billed)<1e-8


def test_current_text_cells_and_sample_sizes_match_recalculated_aggregate():
    data=read('aggregate.json')
    study=ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018'
    publication=json.loads((study/'PUBLICATION.json').read_text())
    fresh=json.loads((study/'aggregate.json').read_text())['arms']['on']
    models=[m for m in data['text_models'] if m['label'] not in publication['retired_labels']]
    models.append({'label':publication['label'],'provider':'Baseten','complete':30,'shortfall':0,
        'strict_percent':fresh['strict_percent'],'dimension_error_percent':fresh['observed_dimension_error_percent'],
        'latency':{'positive':fresh['ttfat'],'missing_n':fresh['missing_ttfat'],
            'nonpositive_n':fresh['nonpositive_ttfat'],'eligible_response_n':fresh['ttfat']['n']+fresh['missing_ttfat']+fresh['nonpositive_ttfat']}})
    recent_path=ROOT/'ops/recent-models-20261018/PUBLICATION.json'
    recent=json.loads(recent_path.read_text()) if recent_path.exists() else {}
    if recent:
        addition=json.loads((recent_path.parent/'aggregate.json').read_text())
        models += [m for m in addition['text_models'] if m['label'] in recent['ranked_labels']]
    repair_path=ROOT/'ops/responses-repair-20261018/PUBLICATION.json'
    repair=json.loads(repair_path.read_text()) if repair_path.exists() else {}
    if repair:
        addition=json.loads((repair_path.parent/'aggregate.json').read_text())
        models=[m for m in models if m['label'] not in repair['replaced_labels']]
        models += [m for m in addition['text_models'] if m['label'] in repair['ranked_labels']]
    migration=ROOT/'ops/judge-recovery-context-20261019'
    if (migration/'PUBLICATION.json').exists():
        corrected={m['label']:m for m in json.loads((migration/'aggregate.json').read_text())['text_models']}
        for m in models:
            latest=corrected[m['label']]
            assert latest['complete']==m['complete']
            m.update(strict_percent=latest['strict_percent'],dimension_error_percent=latest['dimension_error_percent'])
    text=(ROOT/'README.md').read_text().split('Text mode models:\n')[1].split('Speech-to-speech models:\n')[0]
    text=text[text.index('| Model | Pass Rate |'):].split('\n\n',1)[0]
    rows={line.split('|')[1].strip():[c.strip() for c in line.strip('|').split('|')]
          for line in text.splitlines() if line.startswith('| ') and not line.startswith('| Model')}
    assert set(rows)=={m['label'] for m in models}
    assert sum(m['shortfall']>0 for m in models)==(repair['current_shortfalls'] if repair else 13+recent.get('text_shortfalls_added',0))
    for m in models:
        expected=[m['label'],pct(m['strict_percent']),pct(None if m['strict_percent'] is None else 100-m['strict_percent'])]
        expected += [pct(m['dimension_error_percent'][d]) for d in DIMS]
        expected += [ms(m['latency']['positive'][k]) for k in ('p50_ms','p95_ms','max_ms')]
        expected += [m['provider'],str(m['complete'])]
        assert rows[m['label']]==expected
        assert m['latency']['positive']['n']+m['latency']['nonpositive_n']+m['latency']['missing_n']==m['latency']['eligible_response_n']


def test_published_ultra_matches_frozen_thinking_on_grades():
    study=ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018'
    publication=json.loads((study/'PUBLICATION.json').read_text())
    data=json.loads((study/'aggregate.json').read_text())
    assert publication['selected_arm']=='on' and publication['further_budget_experiments'] is False
    assert publication['source_aggregate_sha256']==hashlib.sha256((study/'aggregate.json').read_bytes()).hexdigest()
    cases=[c for c in data['conversations'] if c['arm']=='on'];assert len(cases)==30
    counts={d:0 for d in DIMS};strict=0
    for case in cases:
        result=json.loads((study/'judging/results'/f"{case['job_id']}.json").read_text())
        assert result['transcript_sha256']==case['transcript_sha256']
        meta=result['metadata'];assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
        assert set(result['judgments'])=={str(t) for t in range(30)}
        for judgment in result['judgments'].values():
            for d in DIMS:counts[d]+=judgment['scores'][d]
            strict+=all(judgment['scores'][d] for d in DIMS)
    assert strict==data['arms']['on']['known_strict_passes']
    assert counts==data['arms']['on']['dimension_passes']
    for name in ['README.md','leaderboard-medium-context.md']:
        table=(ROOT/name).read_text().split('| Model | Pass Rate |',1)[1].split('\n\n',1)[0]
        ultra=[line for line in table.splitlines() if '| nemotron-3-ultra' in line]
        assert len(ultra)==1 and publication['label'] in ultra[0]


def test_current_voice_tables_preserve_nulls_and_audio_reliability_evidence():
    data=read('aggregate.json')
    baseline=json.loads((ROOT/'docs/speech-results-2026-09/aggregate.json').read_text())
    old={m['key']:m for m in baseline['models']}
    section=(ROOT/'README.md').read_text().split('Speech-to-speech models:\n')[1].split('### Sample Recordings')[0]
    found={line.split('|')[1].strip():[c.strip() for c in line.strip('|').split('|')]
           for line in section.splitlines() if line.startswith('| ') and not line.startswith('| Model')}
    recent_path=ROOT/'ops/recent-models-20261018/PUBLICATION.json'
    high_path=ROOT/'ops/grok-voice-high-20261018/PUBLICATION.json'
    nova_path=ROOT/'ops/nova-2-5-sonic-20261019/user-context-v2/PUBLICATION.json'
    assert len(found)==9+int(recent_path.exists())+int(high_path.exists())+int(nova_path.exists())
    for m in data['voice_models']:
        row=found[m['label']]
        assert row[:4]==[m['label'],'25',f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",pct(m['strict_percent'])]
        if m['additional_experiment']:
            assert row[4:]==[ms(m['first_speech_positive']['mean_ms']),ms(m['first_speech_positive']['median_ms']),f"{m['first_speech_positive']['n']}/750"]
        else:
            assert row[4:7]==[(f"{m['scores'][d]}/750" if m["quality_complete"] else "—") for d in DIMS]
            assert row[7]==(f"{m['scores']['turn_taking']}/750" if m['turn_taking_scored'] else '—')
            assert row[8:]==[ms(m['non_tool']['median_ms']),ms(m['non_tool']['max_ms']),ms(m['tool']['mean_ms']),ms(m['silence']['mean_ms'])]
        assert m['first_speech_positive']['n']+m['nonpositive_n']+m['missing_latency_n']==750
        assert m['tool']['n']+m['non_tool']['n']==m['first_speech_positive']['n']
        for k in ('eligible_attempts','failed_conversations','failure_rate_percent','harness_excluded','failure_details','turn_taking_scored','nonpositive_n','missing_latency_n'):
            assert m[k]==old[m['key']][k]
        for group in ('first_speech_positive','first_speech_signed','tool','non_tool','silence'):
            for field in ('n','mean_ms','median_ms','max_ms'):
                before,after=old[m['key']][group][field],m[group][field]
                assert before is after if before is None else abs(before-after)<1e-7


def test_campaign_freezes_before_requests_and_finishes_text_before_voice():
    from collections import Counter
    from datetime import datetime

    events=[json.loads(line) for line in (DATA/'attempts.jsonl').read_text().splitlines() if line.strip()]
    cohorts=[json.loads(p.read_text()) for p in (DATA/'cohorts').glob('*.json')]
    frozen={i['job_id']:datetime.fromisoformat(c['frozen_at']) for c in cohorts for i in c['inputs']}
    text_complete=datetime.fromisoformat(read('text-COMPLETE.json')['at'])
    done=Counter(e['job'] for e in events if e['event'] in ('done','provider_blocked'))
    assert set(done)==set(frozen)
    retries=Counter()
    for path in sorted((DATA/'retries').glob('*/retry.json')):
        retry=json.loads(path.read_text())
        original_marker=json.loads((path.parent/'text-COMPLETE.json').read_text())
        text_complete=min(text_complete,datetime.fromisoformat(original_marker['at']))
        start=datetime.fromisoformat(retry['started_at'])
        finish=datetime.fromisoformat(retry['finished_at'])
        assert set(retry['jobs'])=={o['job_id'] for o in retry['outcomes']}
        for outcome in retry['outcomes']:
            job=outcome['job_id']
            prior=json.loads((path.parent/(job+'.json')).read_text())
            assert prior['status']=='provider_safeguard_rejection'
            terminal=[e for e in events if e.get('job')==job
                      and e['event'] in ('done','provider_blocked')
                      and start<=datetime.fromisoformat(e['at'])<=finish]
            assert len(terminal)==1
            assert terminal[0]['attempt']==outcome['attempt']
            assert terminal[0]['event']==('done' if outcome['status']=='graded' else 'provider_blocked')
            retries[job]+=1
    revisions=Counter()
    superseded=Counter()
    cleanup_path=DATA/'reasoning-cleanup/manifest.json'
    if cleanup_path.exists():
        cleanup=json.loads(cleanup_path.read_text())
        original_marker=read('reasoning-cleanup/text-COMPLETE.json')
        text_complete=min(text_complete,datetime.fromisoformat(original_marker['at']))
        start=datetime.fromisoformat(cleanup['started_at'])
        finish=datetime.fromisoformat(cleanup['finished_at'])
        assert {o['job_id'] for o in cleanup['outcomes']}=={i['job_id'] for i in cleanup['jobs']}
        for outcome in cleanup['outcomes']:
            job=outcome['job_id']
            prior_grade=DATA/'reasoning-cleanup/prior-results'/(job+'.json')
            prior_block=DATA/'reasoning-cleanup/prior-blocked'/(job+'.json')
            assert prior_grade.exists() != prior_block.exists()
            if prior_grade.exists():
                prior=json.loads(prior_grade.read_text())
                assert 'judge_input_normalization' not in prior['metadata']
                superseded[job]+=1
            terminal=[e for e in events if e.get('job')==job
                      and e['event'] in ('done','provider_blocked')
                      and start<=datetime.fromisoformat(e['at'])<=finish]
            assert len(terminal)==1 and terminal[0]['attempt']==outcome['attempt']
            revisions[job]+=1
    assert done==Counter({job:1+retries[job]+revisions[job] for job in frozen})
    # A grade can be superseded only by an explicitly audited changed input.
    successes=Counter(e['job'] for e in events if e['event']=='done')
    assert all(count<=1+superseded[job] for job,count in successes.items())
    for e in events:
        if e['event']=='start':
            at=datetime.fromisoformat(e['at'])
            assert at >= frozen[e['job']]
            if e['job'].startswith('voice-'):
                assert at >= text_complete
        elif e['event']=='concurrency_change':
            assert e['in_flight_at_stop']==0

# These reconcile immutable full captures, not just the public summary.
pytestmark = pytest.mark.local_evidence
