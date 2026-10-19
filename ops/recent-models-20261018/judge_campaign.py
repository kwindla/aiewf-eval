"""Freeze collected transcripts and rejudge with Opus 5.5 Medium."""
import asyncio, fcntl, hashlib, importlib.util, json, os, sys
from pathlib import Path
from dotenv import dotenv_values
from preflight import HERE,ROOT,collect,dump,cfg
from collect_campaign import NAMES
from benchmarks.aiwf_medium_context.config import BenchmarkConfig
from benchmarks.aiwf_medium_context.judge import judge_system_prompt as TEXT_RUBRIC
from benchmarks.aiwf_medium_context_speech_v2.judge import judge_system_prompt as VOICE_RUBRIC
from multi_turn_eval.judging.claude_judge import judge_with_claude,load_transcript,prepare_judge_records,write_outputs
spec=importlib.util.spec_from_file_location('retry_policy',ROOT/'ops/judge-refresh-opus55-20261017/campaign.py');retry=importlib.util.module_from_spec(spec);spec.loader.exec_module(retry)
RAW=ROOT/'runs/recent-models-20261018/judging';MODEL='claude-opus-5-5';EFFORT='medium';DIMS=('tool_use_correct','instruction_following','kb_grounding')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rubric(case):return VOICE_RUBRIC if case['name']=='grok-voice' else TEXT_RUBRIC
def log(event,**kw):
 row={'at':collect.utc_now(),'event':event,**kw}
 with (HERE/'judging/attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def freeze():
 cases=[]
 for name in NAMES:
  c=cfg(name)
  if name=='grok-voice':assert (HERE/name/'artifacts/source-sha256.txt').read_text()==collect.source_hashes(c)
  else:collect.verify_source_hashes(c)
  for row in json.loads((HERE/name/'COLLECTED.json').read_text())['cases']:
   run=ROOT/row['run_dir'];source=run/'transcript.jsonl';raw=source.read_bytes();records=load_transcript(run)
   job=f"{name}-{row['arm']}-{int(row['slot']):03d}";snapshot=RAW/'inputs'/job/'transcript.jsonl'
   snapshot.parent.mkdir(parents=True,exist_ok=True)
   if snapshot.exists():assert snapshot.read_bytes()==raw
   else:snapshot.write_bytes(raw)
   _,changes=prepare_judge_records(records)
   cases.append({'job_id':job,'name':name,'arm':row['arm'],'slot':int(row['slot']),'source':str(source.relative_to(ROOT)),'input':str(snapshot.relative_to(ROOT)),'transcript_sha256':sha(raw),'observed_turns':[r['turn'] for r in records if not r.get('recovery_turn')],'normalization_changes':changes})
 identity={'judge_model':MODEL,'judge_effort':EFFORT,'cases':cases,'rubric_sha256':{'text':sha(TEXT_RUBRIC.encode()),'voice':sha(VOICE_RUBRIC.encode())},'source_sha256':{p:sha((ROOT/p).read_bytes()) for p in ['ops/recent-models-20261018/judge_campaign.py','src/multi_turn_eval/judging/claude_judge.py','ops/judge-refresh-opus55-20261017/campaign.py','benchmarks/aiwf_medium_context/judge.py','benchmarks/aiwf_medium_context_speech_v2/judge.py']}}
 path=HERE/'judging/inputs.json'
 if path.exists():assert json.loads(path.read_text())==identity,'Frozen judging inputs changed'
 else:dump(path,identity)
 return cases
def result_path(c):return HERE/'judging/results'/(c['job_id']+'.json')
def blocked_path(c):return HERE/'judging/provider-blocked'/(c['job_id']+'.json')
def valid(c):
 p=result_path(c)
 if not p.exists():return False
 r=json.loads(p.read_text());m=r['metadata'];assert r['transcript_sha256']==c['transcript_sha256']
 assert m['judge_model']==MODEL and m['judge_effort']==EFFORT and m['judge_response_models']==[MODEL]
 assert m['judge_prompt_sha256']==sha(rubric(c).encode()) and m['judge_tools']==[] and m['judge_setting_sources']==[]
 assert m.get('judge_normalized_turns',[])==c['normalization_changes']
 assert set(r['judgments'])==set(map(str,c['observed_turns']))
 assert all(type(j['scores'][d]) is bool for j in r['judgments'].values() for d in DIMS)
 return True
async def grade(c,cooldown):
 raw=(ROOT/c['source']).read_bytes();assert sha(raw)==c['transcript_sha256'] and raw==(ROOT/c['input']).read_bytes()
 folder=RAW/c['job_id'];number=max([int(p.name.split('-')[-1]) for p in folder.glob('attempt-*')],default=0);failures=0
 while failures<4:
  await cooldown.wait();number+=1;run=folder/f'attempt-{number}';run.mkdir(parents=True);(run/'transcript.jsonl').write_bytes(raw)
  # Audio analysis uses the actual captured waveform and original timing logs.
  if c['name']=='grok-voice':
   for name in ['conversation.wav','run.log']:
    source=(ROOT/c['source']).parent/name
    if source.exists():(run/name).symlink_to(source.resolve())
  log('start',job=c['job_id'],attempt=number)
  try:
   result=await asyncio.wait_for(judge_with_claude(run,expected_turns=BenchmarkConfig.turns,skip_turn_taking=c['name']!='grok-voice',judge_model=MODEL,judge_effort=EFFORT,judge_system_prompt=rubric(c)),timeout=600)
   meta=result['judge_metadata']
   write_outputs(run,load_transcript(run),result['judgments'],result['summary'],result['model_name'],result['realignment_notes'],result['function_tracking'],result.get('turn_taking_analysis'),judge_metadata=meta)
   dump(result_path(c),{'job_id':c['job_id'],'transcript_sha256':c['transcript_sha256'],'raw_output':str(run.relative_to(ROOT)),'judgments':result['judgments'],'metadata':meta,'function_tracking':result['function_tracking'],'realignment_notes':result['realignment_notes'],'turn_taking_analysis':result.get('turn_taking_analysis')})
   assert valid(c);log('done',job=c['job_id'],attempt=number,cost_usd=meta.get('judge_cost_usd'));return
  except Exception as e:
   error=retry.safe_error(e);credit=retry.credit_failure(error);dump(run/'error.json',{'error':error,'credit_failure':credit});log('error',job=c['job_id'],attempt=number,error=error,credit_failure=credit)
   if 'reasoning_extraction' in error:dump(blocked_path(c),{'job_id':c['job_id'],'error':error,'quality_judgment':None});return
   if credit:cooldown.suspend();log('credit_cooldown',seconds=90)
   else:failures+=1;await asyncio.sleep(15*2**(failures-1))
 raise RuntimeError('Four operational failures: '+c['job_id'])
async def main():
 while not (HERE/'COLLECTION-COMPLETE.json').exists():await asyncio.sleep(10)
 for path in ['/home/khkramer/src/aiewf-eval/.env','/home/khkramer/src/gb-benchmarks/.env']:
  key=dotenv_values(path).get('ANTHROPIC_API_KEY')
  if key:os.environ.setdefault('ANTHROPIC_API_KEY',key)
 cases=freeze();cooldown=retry.CreditCooldown()
 for offset in range(0,len(cases),4):
  await asyncio.gather(*(grade(c,cooldown) for c in cases[offset:offset+4] if not valid(c) and not blocked_path(c).exists()))
 dump(HERE/'judging/COMPLETE.json',{'at':collect.utc_now(),'conversations':len(cases),'graded':sum(valid(c) for c in cases),'provider_blocked':sum(blocked_path(c).exists() for c in cases)})
if __name__=='__main__':
 (HERE/'judging').mkdir(exist_ok=True)
 with (HERE/'judging/.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
