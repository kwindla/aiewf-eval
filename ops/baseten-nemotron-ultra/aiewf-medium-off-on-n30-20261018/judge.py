"""Freeze and judge the registered Baseten Ultra conversations with Opus 5.5 medium."""

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from dotenv import dotenv_values

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=ROOT/'runs/baseten-ultra-20261018-paced/judging'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
from preflight import collect
from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from benchmarks.aiwf_medium_context.judge import judge_system_prompt as RUBRIC
from multi_turn_eval.judging.claude_judge import judge_with_claude, load_transcript, prepare_judge_records, write_outputs

spec=importlib.util.spec_from_file_location('refresh_retry_policy',ROOT/'ops/judge-refresh-opus55-20261017/campaign.py')
retry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)
MODEL='claude-opus-5-5'
EFFORT='medium'
DIMS=('tool_use_correct','instruction_following','kb_grounding')


def now():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def log(event,**fields):
    record={'at':now(),'event':event,**fields}
    with (HERE/'judging/attempts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def freeze():
    config=collect.validate_configuration(HERE/'configuration.json',require_serving_verified=True)
    schedule=collect.validate_schedule(config)
    _,rows=collect.validate_manifests(config,schedule)
    assert len(rows)==60
    collect.verify_source_hashes(config)
    cases=[]
    for row in rows:
        run=collect.path_from_manifest(row['run_dir'])
        source=run/'transcript.jsonl';raw=source.read_bytes()
        records=load_transcript(run)
        turns=[r['turn'] for r in records if not r.get('recovery_turn')]
        assert turns==list(range(len(turns))) and turns
        _,changes=prepare_judge_records(records)
        job=f"slot-{int(row['slot']):03d}-{row['arm']}"
        snapshot=RAW/'inputs'/job/'transcript.jsonl'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists():assert snapshot.read_bytes()==raw
        else:snapshot.write_bytes(raw)
        cases.append({'job_id':job,'slot':int(row['slot']),'arm':row['arm'],
                      'source':str(source.relative_to(ROOT)),'input':str(snapshot.relative_to(ROOT)),
                      'transcript_sha256':sha(raw),'observed_turns':turns,'normalization_changes':changes,
                      'classification':row['classification']})
    path=HERE/'judging/inputs.json'
    identity={'judge_model':MODEL,'judge_effort':EFFORT,'rubric_sha256':sha(RUBRIC.encode()),
              'cases':cases,'source_sha256':{
                  p:sha((ROOT/p).read_bytes()) for p in [str(Path(__file__).relative_to(ROOT)),
                      'src/multi_turn_eval/judging/claude_judge.py',
                      'ops/judge-refresh-opus55-20261017/campaign.py',
                      'benchmarks/aiwf_medium_context/judge.py']}}
    if path.exists():
        prior=json.loads(path.read_text())
        assert all(prior[k]==v for k,v in identity.items()),'Frozen judge inputs or implementation changed'
        return prior
    data={'frozen_at':now(),**identity}
    dump(path,data)
    (HERE/'judging/rubric.txt').write_text(RUBRIC)
    return data


def result_path(case):return HERE/'judging/results'/f"{case['job_id']}.json"
def blocked_path(case):return HERE/'judging/provider-blocked'/f"{case['job_id']}.json"


def valid(case):
    path=result_path(case)
    if not path.exists():return False
    result=json.loads(path.read_text());meta=result['metadata']
    assert result['transcript_sha256']==case['transcript_sha256']
    assert meta['judge_model']==MODEL and meta['judge_response_models']==[MODEL]
    assert meta['judge_effort']==EFFORT and meta['judge_prompt_sha256']==sha(RUBRIC.encode())
    assert meta['judge_tools']==[] and meta['judge_setting_sources']==[]
    assert meta.get('judge_normalized_turns',[])==case['normalization_changes']
    assert set(result['judgments'])=={str(t) for t in case['observed_turns']}
    assert all(type(j['scores'][d]) is bool for j in result['judgments'].values() for d in DIMS)
    return True


async def grade(case,cooldown):
    source=ROOT/case['source'];data=(ROOT/case['input']).read_bytes()
    assert source.read_bytes()==data and sha(data)==case['transcript_sha256']
    folder=RAW/case['job_id']
    number=max([int(p.name.split('-')[-1]) for p in folder.glob('attempt-*')],default=0)
    failures=0
    while failures<4:
        await cooldown.wait();number+=1
        run=folder/f'attempt-{number}';run.mkdir(parents=True)
        (run/'transcript.jsonl').write_bytes(data)
        log('start',job=case['job_id'],attempt=number)
        try:
            result=await asyncio.wait_for(judge_with_claude(
                run,expected_turns=BenchmarkConfig.turns,skip_turn_taking=True,
                judge_model=MODEL,judge_effort=EFFORT,judge_system_prompt=RUBRIC,
            ),timeout=600)
            meta=result['judge_metadata']
            write_outputs(run,load_transcript(run),result['judgments'],result['summary'],
                          result['model_name'],result['realignment_notes'],result['function_tracking'],
                          judge_metadata=meta)
            dump(result_path(case),{'job_id':case['job_id'],'attempt':number,
                 'transcript_sha256':case['transcript_sha256'],'raw_output':str(run.relative_to(ROOT)),
                 'judgments':result['judgments'],'metadata':meta,
                 'function_tracking':result['function_tracking'],'realignment_notes':result['realignment_notes']})
            assert valid(case)
            log('done',job=case['job_id'],attempt=number,cost_usd=meta.get('judge_cost_usd'))
            return
        except Exception as exc:
            error=retry.safe_error(exc);credit=retry.credit_failure(error)
            dump(run/'error.json',{'at':now(),'error':error,'credit_failure':credit})
            log('error',job=case['job_id'],attempt=number,error=error,credit_failure=credit)
            if 'reasoning_extraction' in error:
                dump(blocked_path(case),{'job_id':case['job_id'],'attempt':number,
                     'transcript_sha256':case['transcript_sha256'],'error':error,'quality_judgment':None})
                return
            if credit:
                cooldown.suspend();log('credit_cooldown',seconds=90)
            else:
                failures+=1
                if failures<4:await asyncio.sleep(15*2**(failures-1))
    raise RuntimeError(f"Four operational failures for {case['job_id']}")


async def main(args):
    collection=HERE/'artifacts/campaign.log'
    while not collection.exists() or 'CAMPAIGN_COLLECTION_DONE' not in collection.read_text():
        if not args.watch:raise RuntimeError('Collection has not completed')
        await asyncio.sleep(15)
    for path in [Path('/home/khkramer/src/aiewf-eval/.env'),Path('/home/khkramer/src/gb-benchmarks/.env')]:
        key=dotenv_values(path).get('ANTHROPIC_API_KEY')
        if key:os.environ.setdefault('ANTHROPIC_API_KEY',key)
    if not os.getenv('ANTHROPIC_API_KEY'):raise RuntimeError('Anthropic credential unavailable')
    manifest=freeze();cooldown=retry.CreditCooldown()
    pending=[c for c in manifest['cases'] if not valid(c) and not blocked_path(c).exists()]
    while pending:
        await asyncio.gather(*(grade(c,cooldown) for c in pending[:args.workers]))
        pending=[c for c in manifest['cases'] if not valid(c) and not blocked_path(c).exists()]
    marker=HERE/'judging/COMPLETE.json'
    if not marker.exists():
        dump(marker,{'at':now(),'conversations':60,'graded':sum(valid(c) for c in manifest['cases']),
                     'provider_blocked':sum(blocked_path(c).exists() for c in manifest['cases'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('workers must be 1..8')
    (HERE/'judging').mkdir(exist_ok=True)
    with (HERE/'judging/.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main(args))
