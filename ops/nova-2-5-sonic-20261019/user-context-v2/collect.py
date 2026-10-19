"""Excluded smoke then first 25 complete conversations; preserve every attempt."""
import asyncio,fcntl,json,sys
from common import HERE,ROOT,RAW,MODEL,env,dump,now,freeze,config

RUN_ONE=ROOT/'ops/aiewf-campaign-template/run_one.py'

def validate(run):
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
    path=run/'transcript.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    assert [r['turn'] for r in rows]==list(range(len(rows)))
    assert not any(r.get('recovery_turn') or r.get('reconnection_count',0) for r in rows)
    assert all(r['model_name']==MODEL for r in rows)
    assert all(r['user_text']==BenchmarkConfig.turns[r['turn']]['input'] for r in rows)
    logs=(run/'run.log').read_text()
    assert 'Nova reference layout: policy SYSTEM (4310 chars), knowledge base USER (36851 chars); full prompt preserved' in logs
    assert 'dropping older history' not in logs and 'minimal system prompt for reconnection' not in logs
    for row in rows:
        for result in row.get('tool_results',[]):
            call_id=result['response']['tool_call_id']
            assert f'Nova native tool result delivered: tool_call_id={call_id}' in logs,call_id
    responded=sum(bool(r.get('assistant_text','').strip() or r.get('tool_calls')) and '[NO RESPONSE' not in r.get('assistant_text','') and '[audio response - no text' not in r.get('assistant_text','') for r in rows)
    return {'turns':len(rows),'response_turns':responded,'tool_calls':sum(len(r.get('tool_calls',[])) for r in rows)}

async def attempt(number,smoke=False):
    name=config().get('excluded_smoke_name','smoke') if smoke else f'attempt-{number:03d}'
    record=HERE/(f'smoke/{name}.json' if smoke else f'attempts/{name}.json')
    if record.exists():return json.loads(record.read_text())
    run=RAW/name;log=HERE/(f'smoke/{name}.log' if smoke else f'attempts/{name}.log');log.parent.mkdir(parents=True,exist_ok=True)
    code=None
    if not run.exists():
        with log.open('w') as out:
            p=await asyncio.create_subprocess_exec(sys.executable,str(RUN_ONE),'--config',str(HERE/'configuration.json'),'--arm','standard','--run-dir',str(run),cwd=ROOT,env=env(),stdout=out,stderr=asyncio.subprocess.STDOUT)
            try:code=await asyncio.wait_for(p.wait(),1800)
            except asyncio.TimeoutError:p.terminate();await p.wait();code=124
    stats={};error=None
    try:stats=validate(run)
    except Exception as exc:error=str(exc)
    complete=code in (None,0) and stats.get('turns')==stats.get('response_turns')==30 and error is None
    row={'attempt':number,'run_dir':str(run.relative_to(ROOT)),'at':now(),'exit_code':code,'complete':complete,'validation_error':error,**stats}
    dump(record,row);print(json.dumps(row),flush=True)
    return row

async def main(smoke_only=False):
    import direct
    await direct.main()
    smoke=await attempt(0,True)
    assert smoke['complete'],('Hold smoke for review',smoke)
    from audio import measure
    measure(config().get('excluded_smoke_name','smoke'),ROOT/smoke['run_dir'])
    if smoke_only:return
    freeze();attempts=[];number=1
    dump(HERE/'SESSION.json',{'at':now(),'status':'collecting','complete':0,'target':25})
    while sum(a['complete'] for a in attempts)<25:
        freeze();count=min(3,25-sum(a['complete'] for a in attempts))
        attempts.extend(await asyncio.gather(*(attempt(n) for n in range(number,number+count))))
        number+=count;dump(HERE/'attempts.json',attempts)
        # Hold completed timing defects immediately; never replace them by score.
        for slot,row in enumerate([a for a in attempts if a['complete']],1):
            measure(f'nova-standard-{slot:03d}',ROOT/row['run_dir'])
        dump(HERE/'SESSION.json',{'at':now(),'status':'collecting','complete':sum(a['complete'] for a in attempts),'attempts':len(attempts),'target':25})
        if len(attempts)>=50 and sum(a['complete'] for a in attempts)<25:raise RuntimeError('Review attempt cap')
    cases=[{**a,'slot':i} for i,a in enumerate([a for a in attempts if a['complete']],1)]
    assert len(cases)==25
    dump(HERE/'COLLECTED.json',{'at':now(),'cases':cases})
    dump(HERE/'COLLECTION-COMPLETE.json',{'at':now(),'complete':25,'attempts':len(attempts)})

if __name__=='__main__':
    with (HERE/'.collection.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main('--smoke-only' in sys.argv))
