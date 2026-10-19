"""Locked, first-valid diagnostic and canonical judging with shared credit waits."""
import asyncio
from contextvars import ContextVar
import fcntl
import json
import sys
import time
from common import *
from multi_turn_eval.judging import claude_judge as judge

CURRENT=ContextVar('attempt_folder')
ORIGINAL_QUERY=judge.query

def log(event,**kwargs):
    row={'at':time.time(),'event':event,**kwargs}
    with (HERE/'attempts.jsonl').open('a') as out:
        out.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

async def audited_query(**kwargs):
    folder=CURRENT.get()
    async for message in ORIGINAL_QUERY(**kwargs):
        data={'type':type(message).__name__}
        for field in ['id','model','session_id','total_cost_usd','usage','duration_ms','is_error','result']:
            if hasattr(message,field):
                data[field]=getattr(message,field)
        with (folder/'sdk-events.jsonl').open('a') as out:
            out.write(json.dumps(data,default=str)+'\n')
        yield message

judge.query=audited_query

class Cooldown:
    def __init__(self):
        self.path=HERE/'credit-cooldown.json'
    async def wait(self):
        while self.path.exists() and (delay:=read(self.path)['until']-time.time())>0:
            await asyncio.sleep(min(delay,30))
    def suspend(self):
        dump(self.path,{'until':time.time()+90,'seconds':90})

def frozen_identity():
    names=['PLAN.md','inventory.json','preflight.py','common.py','driver.py','holdout.py','validation-selection.json',
           'RECOVERY-ENDING-AUDIT.json','RECOVERY-PUBLIC-TEXT-RESTORATIONS-v1.json',
           '../../src/multi_turn_eval/judging/claude_judge.py','../../src/multi_turn_eval/judging/context.py']
    return {name:sha(HERE/name) for name in names}

def valid_result(path,c):
    if not path.exists():
        return False
    r=read(path)
    m=r['metadata']
    assert r['transcript_sha256']==c['transcript_sha256']
    assert m['judge_model']==MODEL and m['judge_response_models']==[MODEL] and m['judge_effort']=='medium'
    assert m['judge_thinking']=='adaptive' and m['judge_tools']==[] and m['judge_setting_sources']==[]
    assert m['judge_input_sha256']==c['candidate_request']['user_sha256']
    assert m['judge_prompt_sha256']==c['candidate_request']['system_sha256']
    assert set(r['judgments'])==set(map(str,c['observed_turns']))
    assert all(type(j['scores'][d]) is bool for j in r['judgments'].values() for d in DIMS)
    return True

async def grade(c,phase,cooldown,semaphore):
    folder=HERE/'judging'/phase/c['job_id']
    target=folder/'result.json'
    if valid_result(target,c):
        return
    async with semaphore:
        failures=0
        attempt=max([int(p.name.split('-')[1]) for p in folder.glob('attempt-*')],default=0)
        while failures<4:
            await cooldown.wait()
            attempt+=1
            work=folder/f'attempt-{attempt}'
            work.mkdir(parents=True)
            source,applied=restored_records(c)
            (work/'transcript.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in source))
            dump(work/'input-audit.json',{'source_sha256':c['transcript_sha256'],'derived_transcript_sha256':sha(work/'transcript.jsonl'),
                 'restored_records':applied,'phase':phase,'canonical':phase=='canonical'})
            token=CURRENT.set(work)
            log('start',phase=phase,job_id=c['job_id'],attempt=attempt)
            try:
                result=await asyncio.wait_for(judge.judge_with_claude(
                    work,expected_turns=turns(c),skip_turn_taking=audio(c) is None,precomputed_turn_taking=audio(c),
                    judge_model=MODEL,judge_effort='medium',judge_system_prompt=rubric(c),
                    task_clarification=old_grade(c)['metadata'].get('judge_task_clarification')),timeout=900)
                payload={'job_id':c['job_id'],'transcript_sha256':c['transcript_sha256'],'raw_output':str(work.relative_to(ROOT)),
                         'judgments':result['judgments'],'metadata':result['judge_metadata'],
                         'realignment_notes':result['realignment_notes'],'function_tracking':result['function_tracking'],
                         'turn_taking_analysis':old_grade(c).get('turn_taking_analysis'),'phase':phase,
                         'restorations_applied':applied}
                # Validate before atomic first-valid publication.
                pending=work/'candidate-result.json'
                dump(pending,payload)
                valid_result(pending,c)
                assert not target.exists()
                dump(target,payload)
                log('done',phase=phase,job_id=c['job_id'],attempt=attempt,cost=result['judge_metadata'].get('judge_cost_usd'))
                return
            except Exception as error:
                text=str(error)
                credit=any(s in text.lower() for s in ['credit balance','insufficient credit','balance is too low'])
                dump(work/'error.json',{'error':text,'credit_failure':credit})
                log('error',phase=phase,job_id=c['job_id'],attempt=attempt,error=text,credit_failure=credit)
                if credit:
                    cooldown.suspend()
                else:
                    failures+=1
                    await asyncio.sleep(min(15*2**(failures-1),60))
            finally:
                CURRENT.reset(token)
        raise RuntimeError('Four operational failures: '+c['job_id'])

def development_cases(data):
    ids=['refresh--gemini-3.5-flash-023-68432ef89f55','refresh--deepseek-v4-flash-0731-high-023-c4c7afd555bf']
    selected=read(HERE/'validation-selection.json')
    assert not set(ids)&{e['job_id'] for k in ['primary','reserve'] for e in selected[k]}
    return [c for c in data['cases'] if c['job_id'] in ids]

async def main(phase):
    if phase=='canonical' and not globals().get('_CANONICAL_WRAPPER_ACTIVE'):
        raise RuntimeError('Canonical requests must use the registered canonical.py wrapper')
    credentials()
    verify_prior()
    data=read(HERE/'inventory.json')
    assert not data['holds']
    identity=HERE/('canonical-driver-sha256.json' if phase=='canonical' else 'driver-sha256.json')
    if identity.exists():
        assert read(identity)==frozen_identity(),'Frozen judging source changed'
    else:
        dump(identity,frozen_identity())
    if phase.startswith('development'):
        cases=development_cases(data)
        expectations={'support_case':{'turn':19,'dimension':'instruction_following','expected':True,
                       'rationale':'Actual successful prior recovery ticket exists; the response acknowledges it and refuses unsupported navigation.'},
                      'delayed_actions':{'turns':[15,24],'dimensions':['tool_use_correct','instruction_following'],'expected':False,
                       'rationale':'Name and preference/session were known, unnecessary reconfirmations cannot receive recovery hindsight credit.'}}
        prereg=HERE/'development-expectations.json'
        if prereg.exists():assert read(prereg)==expectations
        else:dump(prereg,expectations)
    elif phase in ['validation-A','validation-B']:
        refs=read(HERE/'references/FROZEN.json')
        assert refs['minimum_gate_passed']
        for path,expected in refs['files'].items():assert sha(HERE/path)==expected
        ids={e['job_id'] for e in read(HERE/'validation-selection.json')['primary']}
        cases=[c for c in data['cases'] if c['job_id'] in ids]
    elif phase=='canonical':
        assert read(HERE/'VALIDATION.json')['passed'],'Validation gate holds bulk'
        cases=[c for c in data['cases'] if c['disposition']=='rejudge']
    else:
        raise ValueError(phase)
    cooldown=Cooldown()
    semaphore=asyncio.Semaphore(8)
    await asyncio.gather(*(grade(c,phase,cooldown,semaphore) for c in cases))
    assert read(identity)==frozen_identity()
    dump(HERE/f'{phase}-COMPLETE.json',{'phase':phase,'conversations':len(cases),'graded':sum(valid_result(HERE/'judging'/phase/c['job_id']/'result.json',c) for c in cases)})

if __name__=='__main__':
    with (HERE/'.judge.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main(sys.argv[1]))
