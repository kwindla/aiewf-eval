"""Preregister complete, disjoint diagnostic samples and independent raw views."""
from collections import Counter
from common import *
from multi_turn_eval.judging.claude_judge import strip_reasoning_sections

SEED = 'recovery-context-validation-20261019-v1'

def excluded(c):
    # Conservative whole-family exclusions cover manually inspected cohorts.
    label = c['label'].lower()
    return ('luna' in label or 'ultra' in label or 'gemini-3.5' in label
            or 'gemini-3.8' in label or 'gemini-3.7' in label)

def category(c):
    return 'voice' if c['mode']=='voice' else ('recovery' if c['recovery_records'] else 'clean')

def choose_targets(c):
    grade = old_grade(c)['judgments']
    failures = [int(t) for t,j in grade.items() if not all(j['scores'][d] for d in DIMS)]
    if not failures:
        return None
    # Old-label stratification deliberately overrepresents errors; candidate
    # judgments and rankings are unavailable at selection time.
    first = min(failures, key=lambda t:digest(SEED+c['transcript_sha256']+str(t)))
    expected = turns(c)
    alternatives = [int(t) for t,j in grade.items() if int(t)!=first and all(j['scores'][d] for d in DIMS)]
    if not alternatives:
        return None
    second = min(alternatives, key=lambda t:(not bool(expected[t].get('required_function_call')),digest(SEED+str(t)+c['transcript_sha256'])))
    return sorted([first,second])

def select(pool, used):
    chosen = []
    for kind, count in [('recovery',8),('clean',4),('voice',12)]:
        eligible = [c for c in pool if category(c)==kind and c['transcript_sha256'] not in used and choose_targets(c)]
        counts = Counter(c['label'] for c in chosen)
        providers = Counter(c.get('provider') or c['label'].split()[0] for c in chosen)
        for _ in range(count):
            c = min(eligible,key=lambda c:(counts[c['label']],providers[c.get('provider') or c['label'].split()[0]],digest(SEED+c['transcript_sha256'])))
            chosen.append(c)
            used.add(c['transcript_sha256'])
            counts[c['label']]+=1
            providers[c.get('provider') or c['label'].split()[0]]+=1
            eligible = [x for x in eligible if x['transcript_sha256'] not in used]
    return chosen

def independent_packet(c, alias, targets):
    # This view intentionally does not call the candidate formatter or ancestry
    # inference code. Source positions and explicit flags are represented directly.
    raw, applied = restored_records(c)
    view = []
    for position, r in enumerate(raw):
        view.append({'source_position':position, 'scripted_id':None if r.get('recovery_turn') else r['turn'],
                     'is_extra_recovery':bool(r.get('recovery_turn')), 'explicit_recovery_parent':r.get('recovery_for_turn'),
                     'user':r['user_text'], 'public_assistant':strip_reasoning_sections(r['assistant_text']),
                     'tool_calls':r.get('tool_calls',[]),'recorded_tool_results':r.get('tool_results',[])})
    return {'case':alias,'mode':c['mode'],'target_scripted_ids':targets,'raw_chronological_records':view,
            'expected_scripted_examples':turns(c),'frozen_audio_annotations':audio(c)}

def main():
    data = read(HERE/'inventory.json')
    assert not data['holds']
    pool = [c for c in data['cases'] if c['complete'] and not excluded(c)]
    used = set()
    primary, reserve = select(pool,used), select(pool,used)
    # The confirmed support regression and all conservative exclusions cannot leak.
    assert len(used)==48
    frozen = {'seed':SEED,'selection':'complete30 only; old-label failure/pass stratification; greedy configuration/provider diversity; SHA tie-break',
              'exclusions':[c['transcript_sha256'] for c in data['cases'] if excluded(c)],
              'inventory_sha256':sha(HERE/'inventory.json'),'primary':[],'reserve':[]}
    for name, cohort in [('primary',primary),('reserve',reserve)]:
        for i,c in enumerate(cohort):
            frozen[name].append({'case':f'H{i+1:02d}' if name=='primary' else f'R{i+1:02d}',
                                  'job_id':c['job_id'],'source_sha256':c['transcript_sha256'],
                                  'targets':choose_targets(c),'category':category(c),'mode':c['mode']})
    path = HERE/'validation-selection.json'
    if path.exists():
        assert read(path)==frozen
    else:
        dump(path,frozen)
    lookup = {c['job_id']:c for c in data['cases']}
    primary_view = [independent_packet(lookup[e['job_id']],e['case'],e['targets']) for e in frozen['primary']]
    for batch in range(6):
        packet = {'instructions':'Independently label ONLY the specified scripted targets. Extra recovery records are context, never targets. Evaluate original scripted responses using information available at that time; later successful recovery does not erase an unnecessary earlier deferral. Necessary missing-argument clarification remains valid. Actual earlier recovery actions affect later state and can fulfill a later expected action. Absent recorded outcomes are unknown. Do not infer success from prose. Use the governing policy and KB, not literal golden wording. For each target return all three scores, per-dimension uncertain flags, and concise reasons. Flag policy ambiguity rather than inventing requirements. Output JSON only: {"labels":[{"case":"H01","turn":0,"scores":{"tool_use_correct":true,"instruction_following":true,"kb_grounding":true},"uncertain":{"tool_use_correct":false,"instruction_following":false,"kb_grounding":false},"reasoning":"..."}]}.',
                  'reference_rubrics':{'text':(HERE/'legacy/text-rubric.txt').read_text(),'voice':(HERE/'legacy/voice-rubric.txt').read_text()},
                  'cases':primary_view[batch*4:batch*4+4]}
        dump(HERE/f'references/packet-{batch+1}.json',packet)
    dump(HERE/'validation-freeze.json',{'selection_sha256':sha(path),'packets':{f'packet-{i}.json':sha(HERE/f'references/packet-{i}.json') for i in range(1,7)},
                                      'primary_conversations':24,'primary_targets':48,'primary_dimensions':144,'reserve_conversations':24,
                                      'candidate_diagnostics_started':False,'references':'independent model labels, not human truth'})
    print('Frozen 24 primary + 24 disjoint reserve complete conversations',flush=True)

if __name__=='__main__':main()
