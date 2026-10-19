"""Reconcile corrected and proven-equivalent grades without altering old studies."""
from collections import Counter
from copy import deepcopy
import numpy as np
from common import *
from driver import valid_result

def interval(counts):
    if not counts:
        return None
    rng=np.random.default_rng(20261019)
    draws=rng.choice(np.asarray(counts),size=(20000,len(counts)),replace=True).mean(axis=1)/30*100
    return list(np.percentile(draws,[2.5,97.5]))

def summarize(cases,grades):
    counts={d:0 for d in DIMS}
    strict=[]
    observed=0
    for c in cases:
        r=grades[c['job_id']]
        score=0
        for j in r['judgments'].values():
            observed+=1
            score+=all(j['scores'][d] for d in DIMS)
            for d in DIMS:
                counts[d]+=j['scores'][d]
        strict.append(score)
    denominator=30*len(cases)
    return {'strict_passes':sum(strict),'dimension_passes':counts,'denominator':denominator,
            'strict_percent':100*sum(strict)/denominator if denominator else None,
            'strict_ci95':interval(strict),'conversation_strict_passes':strict,
            'dimension_error_percent':{d:100*(denominator-counts[d])/denominator if denominator else None for d in DIMS},
            'observed_scripted_turns':observed,'missing_future_turns':denominator-observed}

def costs():
    events=[]
    for path in (HERE/'judging').glob('*/*/attempt-*/sdk-events.jsonl'):
        messages=[json.loads(line) for line in path.read_text().splitlines()]
        result=[m for m in messages if m['type']=='ResultMessage']
        if result:
            assert len(result)==1
            events.append({'attempt':str(path.parent.relative_to(HERE)),'phase':path.parents[2].name,
                           'cost_usd':result[0].get('total_cost_usd'),'is_error':result[0].get('is_error')})
        else:
            events.append({'attempt':str(path.parent.relative_to(HERE)),'phase':path.parents[2].name,'cost_usd':None})
    return {'opus_known_cost_usd':sum(e.get('cost_usd') or 0 for e in events),
            'opus_missing_cost_attempts':[e for e in events if e['cost_usd'] is None],
            'opus_cost_by_phase':{phase:sum(e.get('cost_usd') or 0 for e in events if e['phase']==phase) for phase in sorted({e['phase'] for e in events})},
            'fable_review_and_reference_cost_usd':read(HERE/'FABLE-PLAN-REVIEW.json')['total_cost_usd']+sum(read(p).get('cost_usd') or 0 for p in (HERE/'references/fable').glob('batch-*.json')),
            'astra_reference_usage':[json.loads(line) for p in (HERE/'references/astra').glob('batch-*-attempt-1.stdout') for line in p.read_text().splitlines() if json.loads(line).get('type')=='turn.completed'],
            'astra_cost_note':'Codex CLI uses explicit gpt-6-astra; token usage retained but USD billing is not exposed by this CLI.',
            'attempts':events}

def main():
    from acceptance import verify_acceptance
    verify_acceptance()
    assert read(HERE/'VALIDATION.json')['passed']
    verify_prior()
    inventory=read(HERE/'inventory.json')
    cases=inventory['cases']
    grades={}
    ledger=[]
    changes=[]
    lookup={c['job_id']:c for c in cases}
    for c in cases:
        if c['disposition']=='reuse':
            assert c['historical_request']==c['candidate_request']
            path=rootpath(c['old_grade'])
            grade=old_grade(c)
        else:
            path=HERE/'judging/canonical'/c['job_id']/'result.json'
            assert valid_result(path,c)
            grade=read(path)
            old=old_grade(c)
            for t,j in grade['judgments'].items():
                for d in DIMS:
                    before=old['judgments'][t]['scores'][d]
                    after=j['scores'][d]
                    if before!=after:
                        changes.append({'job_id':c['job_id'],'label':c['label'],'complete':c['complete'],'turn':int(t),'dimension':d,
                                        'before':before,'after':after,'old_reasoning':old['judgments'][t]['reasoning'],
                                        'new_reasoning':j['reasoning']})
        grades[c['job_id']]=grade
        ledger.append({'job_id':c['job_id'],'disposition':c['disposition'],'grade_path':str(path.relative_to(ROOT)),
                       'grade_sha256':sha(path),'source_sha256':c['transcript_sha256'],'complete':c['complete']})
    models=[]
    for original in inventory['models']:
        m=deepcopy(original)
        # Old generation evidence stays current; old judge accounting and
        # redundant quality summaries belong to their immutable original study.
        for field in ['strict_bounds_percent','conversation_strict_passes','successful_judge_cost_usd',
                      'unknown_cost_jobs','judge_duration_ms','judge_usage']:
            m.pop(field,None)
        if m['campaign_layer']=='refresh' and 'manifest' in m:
            m['source_cohort_manifest']=str(Path('ops/judge-refresh-opus55-20261017')/m.pop('manifest'))
        cohort=[lookup[job] for job in m['cases']]
        complete=[c for c in cohort if c['complete']]
        m['assigned']=len(cohort)
        m['complete']=len(complete)
        m['incomplete']=len(cohort)-len(complete)
        m['shortfall']=max(0,(25 if m['mode']=='voice' else 30)-len(complete))
        m['all_assigned']=summarize(cohort,grades)
        m['complete_only']=summarize(complete,grades)
        m['strict_percent']=m['complete_only']['strict_percent']
        m['strict_ci95']=m['complete_only']['strict_ci95']
        m['dimension_error_percent']=m['complete_only']['dimension_error_percent']
        m['scores']={**m['complete_only']['dimension_passes'],'strict':m['complete_only']['strict_passes']}
        if m['mode']=='voice':
            m['scores']['turn_taking']=(sum(j['scores']['turn_taking'] for c in complete for j in grades[c['job_id']]['judgments'].values())
                                       if m.get('turn_taking_scored') else None)
        m['accuracy_denominator']=30*len(complete)
        m['quality_complete']=True
        m['graded_conversations']=len(cohort)
        m['graded_turns']=m['all_assigned']['observed_scripted_turns']
        m['unknown_turns']=0
        m['provider_blocked_conversations']=0
        m['provider_blocked_jobs']=[]
        m['old_strict_percent']=(original.get('strict_percent') if m['campaign_layer']!='ultra' else original['original_metrics']['strict_percent'])
        m['delta_percentage_points']=m['strict_percent']-m['old_strict_percent'] if m['strict_percent'] is not None and m['old_strict_percent'] is not None else None
        m['grade_dispositions']=dict(Counter(c['disposition'] for c in cohort))
        models.append(m)
    aggregate={'schema_version':1,'judge_model':MODEL,'judge_effort':'medium','validation_sha256':sha(HERE/'VALIDATION.json'),
               'inventory_sha256':sha(HERE/'inventory.json'),'all_quality_complete':True,
               'counts':{'assignments':len(cases),'complete':sum(c['complete'] for c in cases),'incomplete':sum(not c['complete'] for c in cases),
                         'reused':sum(c['disposition']=='reuse' for c in cases),'rejudged':sum(c['disposition']=='rejudge' for c in cases),
                         'changed_dimensions':len(changes),'false_to_true':sum(not e['before'] for e in changes),'true_to_false':sum(e['before'] for e in changes)},
               'text_models':[m for m in models if m['mode']=='text'],'voice_models':[m for m in models if m['mode']=='voice'],
               'ledger':ledger,'costs':costs(),'limits':'Corrected context plus judge variation; changed labels are not a pure causal estimate. Historical cohorts and latency evidence unchanged.'}
    dump(HERE/'aggregate.json',aggregate)
    dump(HERE/'changed-labels.json',changes)
    print(json.dumps(aggregate['counts']),flush=True)

if __name__=='__main__':main()
