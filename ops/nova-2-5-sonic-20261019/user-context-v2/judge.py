"""Freeze and judge the complete Nova cohort with Opus 5.5 Medium."""
import asyncio,fcntl,hashlib,importlib.util,json,os
from pathlib import Path
from dotenv import dotenv_values
from common import HERE,ROOT,RAW,sha,dump,now,freeze as generation_freeze
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from benchmarks.aiwf_medium_context_speech_v2.judge import judge_system_prompt as RUBRIC
from multi_turn_eval.judging.claude_judge import judge_with_claude,load_transcript,prepare_judge_records,write_outputs

MODEL='claude-opus-5-5';EFFORT='medium'
DIMS=('tool_use_correct','instruction_following','kb_grounding')
RETRY=ROOT/'ops/judge-refresh-opus55-20261017/campaign.py'
spec=importlib.util.spec_from_file_location('nova_credit_retry',RETRY)
retry=importlib.util.module_from_spec(spec);spec.loader.exec_module(retry)

def log(event,**fields):
    record={'at':now(),'event':event,**fields}
    with (HERE/'judging/attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

def freeze():
    generation_freeze()
    assert (HERE/'COLLECTION-COMPLETE.json').exists()
    cases=[]
    for row in json.loads((HERE/'COLLECTED.json').read_text())['cases']:
        run=ROOT/row['run_dir'];source=run/'transcript.jsonl';records=load_transcript(run)
        assert len(records)==30 and [r['turn'] for r in records]==list(range(30))
        _,changes=prepare_judge_records(records)
        job=f"nova-standard-{row['slot']:03d}";snapshot=RAW/'judging/inputs'/job/'transcript.jsonl'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists():assert snapshot.read_bytes()==source.read_bytes()
        else:snapshot.write_bytes(source.read_bytes())
        cases.append({'job_id':job,'slot':row['slot'],'source':str(source.relative_to(ROOT)),
            'input':str(snapshot.relative_to(ROOT)),'transcript_sha256':sha(source),'normalization_changes':changes})
    assert len(cases)==25
    sources=[HERE/'judge.py',HERE/'common.py',HERE/'PROTOCOL.md',RETRY,
        ROOT/'src/multi_turn_eval/judging/claude_judge.py',ROOT/'src/multi_turn_eval/judging/context.py',
        ROOT/'src/multi_turn_eval/judging/turn_taking.py',ROOT/'benchmarks/aiwf_medium_context_speech_v2/judge.py']
    identity={'cases':cases,'judge_model':MODEL,'judge_effort':EFFORT,
        'rubric_sha256':hashlib.sha256(RUBRIC.encode()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'generation_manifest_sha256':sha(HERE/'generation-sha256.json')}
    path=HERE/'judging/inputs.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Judge identity changed'
    else:dump(path,identity)
    return cases

def result_path(case):return HERE/'judging/results'/(case['job_id']+'.json')
def valid(case):
    path=result_path(case)
    if not path.exists():return False
    r=json.loads(path.read_text());m=r['metadata']
    assert r['transcript_sha256']==case['transcript_sha256']
    assert m['judge_model']==MODEL and m['judge_effort']==EFFORT
    assert m['judge_response_models']==[MODEL] and m['judge_tools']==[] and m['judge_setting_sources']==[]
    assert m['judge_prompt_sha256']==hashlib.sha256(RUBRIC.encode()).hexdigest()
    assert m.get('judge_normalized_turns',[])==case['normalization_changes']
    assert set(r['judgments'])==set(map(str,range(30)))
    assert all(type(j['scores'][d]) is bool for j in r['judgments'].values() for d in DIMS+('turn_taking',))
    assert r['turn_taking_analysis'] and not r['turn_taking_analysis'].get('error')
    return True

async def grade(case,cooldown):
    if valid(case):return
    source=ROOT/case['source'];assert sha(source)==case['transcript_sha256']
    assert source.read_bytes()==(ROOT/case['input']).read_bytes()
    folder=RAW/'judging'/case['job_id'];folder.mkdir(parents=True,exist_ok=True)
    number=max([int(p.name.split('-')[-1]) for p in folder.glob('attempt-*')],default=0);failures=0
    while failures<4:
        await cooldown.wait();number+=1;run=folder/f'attempt-{number}';run.mkdir()
        (run/'transcript.jsonl').write_bytes(source.read_bytes())
        for name in ['conversation.wav','run.log']:(run/name).symlink_to((source.parent/name).resolve())
        log('start',job=case['job_id'],attempt=number)
        try:
            result=await asyncio.wait_for(judge_with_claude(run,expected_turns=BenchmarkConfig.turns,
                judge_model=MODEL,judge_effort=EFFORT,judge_system_prompt=RUBRIC),600)
            meta=result['judge_metadata']
            write_outputs(run,load_transcript(run),result['judgments'],result['summary'],result['model_name'],
                result['realignment_notes'],result['function_tracking'],result.get('turn_taking_analysis'),judge_metadata=meta)
            dump(result_path(case),{'job_id':case['job_id'],'transcript_sha256':case['transcript_sha256'],
                'raw_output':str(run.relative_to(ROOT)),'judgments':result['judgments'],'metadata':meta,
                'function_tracking':result['function_tracking'],'realignment_notes':result['realignment_notes'],
                'turn_taking_analysis':result.get('turn_taking_analysis')})
            assert valid(case);log('done',job=case['job_id'],attempt=number,cost_usd=meta.get('judge_cost_usd'));return
        except Exception as exc:
            error=retry.safe_error(exc);credit=retry.credit_failure(error)
            dump(run/'error.json',{'error':error,'credit_failure':credit})
            log('error',job=case['job_id'],attempt=number,error=error,credit_failure=credit)
            if credit:cooldown.suspend();log('credit_cooldown',seconds=90)
            else:failures+=1;await asyncio.sleep(15*2**(failures-1))
    raise RuntimeError('Four operational failures: '+case['job_id'])

async def main():
    for path in ['/home/khkramer/src/aiewf-eval/.env','/home/khkramer/src/gb-benchmarks/.env']:
        key=dotenv_values(path).get('ANTHROPIC_API_KEY')
        if key:os.environ.setdefault('ANTHROPIC_API_KEY',key)
    cases=freeze();cooldown=retry.CreditCooldown()
    for offset in range(0,25,4):await asyncio.gather(*(grade(c,cooldown) for c in cases[offset:offset+4]))
    assert all(valid(c) for c in cases)
    dump(HERE/'judging/COMPLETE.json',{'at':now(),'graded':25,'judge_model':MODEL,'judge_effort':EFFORT})

if __name__=='__main__':
    (HERE/'judging').mkdir(exist_ok=True)
    with (HERE/'judging/.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
