"""Enumerate all active assignments and prove historical request identity."""
import asyncio
from collections import Counter
from copy import deepcopy
import json
from common import *
from multi_turn_eval.judging import claude_judge as candidate
from multi_turn_eval.judging.context import validate_context

def inventory():
    cases, models = [], []
    repair = ROOT/'ops/responses-repair-20261018'
    replaced = set(read(repair/'PUBLICATION.json')['replaced_labels'])
    ultra = ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018'
    replaced.update(read(ultra/'PUBLICATION.json')['retired_labels'])
    base = ROOT/'ops/judge-refresh-opus55-20261017'
    original = read(base/'aggregate.json')
    def add(model, entries, folder, layer):
        m = deepcopy(model)
        m['campaign_layer'] = layer
        m['mode'] = model['mode']
        ids = []
        for entry in entries:
            c = deepcopy(entry)
            c['old_job_id'] = c['job_id']
            c['job_id'] = layer + '--' + c['job_id']
            c.update(label=m['label'], mode=m['mode'], provider=m.get('provider'), layer=layer)
            grade = folder/(c['old_job_id']+'.json')
            c['old_grade'] = str(grade.relative_to(ROOT))
            c['old_grade_sha256'] = sha(grade)
            grade_data = read(grade)
            if m['mode'] == 'voice' and 'turn_taking_data' not in c:
                analysis = grade_data.get('turn_taking_analysis')
                c['turn_taking_data'] = analysis['per_turn'] if analysis else None
            c['observed_turns'] = [r['turn'] for r in records(c) if not r.get('recovery_turn')]
            c['complete'] = c['observed_turns'] == list(range(30))
            ids.append(c['job_id'])
            cases.append(c)
        m['cases'] = ids
        models.append(m)
    for mode in ['text', 'voice']:
        for m in original[mode+'_models']:
            if m['label'] not in replaced:
                add({**m, 'mode': mode}, read(base/m['manifest'])['inputs'], base/'results', 'refresh')
    d = read(ultra/'aggregate.json')
    arm = d['arms']['on']
    add({'label': 'nemotron-3-ultra (thinking on)', 'mode': 'text', 'provider': 'Baseten',
         'complete': 30, 'assigned': 30, 'latency': {'positive': arm['complete_only_ttfat']},
         'original_metrics': arm}, [c for c in d['conversations'] if c['arm']=='on'], ultra/'judging/results', 'ultra')
    for layer, folder in [('recent', 'recent-models-20261018'), ('responses', 'responses-repair-20261018'), ('high', 'grok-voice-high-20261018')]:
        base = ROOT/'ops'/folder
        d = read(base/'aggregate.json')
        lookup = {c['job_id']: c for c in d['cases']}
        for mode in ['text', 'voice']:
            for m in d[mode+'_models']:
                if layer == 'recent' and m['name']=='luna':
                    continue
                add({**m, 'mode': mode}, [lookup[job] for job in m['cases']], base/'judging/results', layer)
    assert len(models)==62 and len(cases)==1686
    assert len({c['job_id'] for c in cases})==len(cases)
    assert sum(c['complete'] for c in cases)==1619
    return models, cases

async def main():
    verify_prior()
    legacy = load_legacy()
    assert sha(HERE/'legacy/src/multi_turn_eval/judging/claude_judge.py') == '7adda4fc26b4670a52b2b6dc41c65767a340bd590a05508e22dc591a99f41df1'
    models, cases = inventory()
    holds = []
    for index, c in enumerate(cases):
        try:
            raw = records(c)
            ctx = validate_context(raw, len(turns(c)))
            restored, applied = restored_records(c)
            candidate.prepare_judge_records(restored)
            c['recovery_records'] = sum(r.get('recovery_turn', False) for r in raw)
            c['inferred_parents'] = [r for r in ctx if r.get('parent_inferred')]
            grade = old_grade(c)
            meta = grade['metadata']
            assert grade['transcript_sha256'] == c['transcript_sha256']
            assert meta['judge_model'] == MODEL and meta['judge_effort']=='medium'
            assert meta['judge_response_models']==[MODEL] and meta['judge_thinking']=='adaptive'
            assert meta['judge_tools']==[] and meta['judge_setting_sources']==[]
            assert meta['judge_prompt_sha256']==digest(rubric(c))
            assert set(grade['judgments'])==set(map(str,c['observed_turns']))
            assert all(type(j['scores'][d]) is bool for j in grade['judgments'].values() for d in DIMS)
            old_request, _ = await request_identity(legacy, c)
            assert old_request['user_sha256']==meta['judge_input_sha256'], ('Historical prompt mismatch', old_request['user_sha256'], meta['judge_input_sha256'])
            new_request, _ = await request_identity(candidate, c)
            c['historical_request'] = old_request
            c['candidate_request'] = new_request
            c['restorations_applied'] = applied
            c['disposition'] = 'reuse' if old_request==new_request else 'rejudge'
            assert not c['recovery_records'] or c['disposition']=='rejudge'
        except Exception as error:
            holds.append({'job_id': c['job_id'], 'error': str(error)})
        if index % 100==0:
            print('Preflight', index, 'holds', len(holds), flush=True)
    summary = dict(Counter(c.get('disposition','hold') for c in cases))
    data = {'models': models, 'cases': cases, 'counts': summary, 'holds': holds,
            'policy_sha256': sha(HERE/'PLAN.md'), 'judge_source_sha256': sha(ROOT/'src/multi_turn_eval/judging/claude_judge.py'),
            'context_source_sha256': sha(ROOT/'src/multi_turn_eval/judging/context.py')}
    dump(HERE/'inventory.json', data)
    print(json.dumps({'counts':summary,'holds':holds[:12]}), flush=True)
    assert not holds, f'{len(holds)} active provenance/input holds'

if __name__=='__main__':
    asyncio.run(main())
