"""Collect frozen text slots and first-25-complete Grok voice captures."""
import asyncio, fcntl, hashlib, json, sys
from pathlib import Path
from preflight import HERE, ROOT, cfg, collect, dump
NAMES=['luna','sonnet','haiku','spark13','spark-open','grok-voice']

def load(p):return json.loads(p.read_text())
def log(event,**kw):
 row={'at':collect.utc_now(),'event':event,**kw}
 with (HERE/'collection-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)

async def text_lane(names):
 for name in names:
  while not cfg(name)['serving']['verified']:await asyncio.sleep(10)
  c=cfg(name);assert c['serving']['verified']
  path=HERE/name/'collection.log'
  with path.open('a') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'ops/aiewf-campaign-template/collect.py'),'--config',str(c['_config_path']),'--execute',cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT)
   rc=await p.wait()
  if rc:raise RuntimeError(f'{name} collection exited {rc}; see {path}')
  _,rows=collect.validate_manifests(c,collect.validate_schedule(c));collect.verify_source_hashes(c)
  dump(HERE/name/'COLLECTED.json',{'at':collect.utc_now(),'name':name,'cases':rows})
  log('text_collected',name=name,conversations=len(rows))

async def voice_attempt(c,number):
 run=ROOT/c['paths']['run_output_root']/f'attempt-{number:03d}';path=HERE/'grok-voice/artifacts'/f'attempt-{number:03d}.json'
 if path.exists():return load(path)
 log_path=path.with_suffix('.log');log_path.parent.mkdir(parents=True,exist_ok=True)
 # A prior interrupted attempt is retained and validated rather than overwritten.
 if not run.exists():
  env=collect.child_environment(c,'none',collect.extract_credential(c))
  with log_path.open('w') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,str(collect.RUN_ONE),'--config',str(c['_config_path']),'--arm','none','--run-dir',str(run),cwd=ROOT,env=env,stdout=f,stderr=asyncio.subprocess.STDOUT)
   try:rc=await asyncio.wait_for(p.wait(),1800)
   except asyncio.TimeoutError:p.terminate();await p.wait();rc=124
 else:rc=None
 stats={};error=None
 try:
  stats=collect.validate_transcript(c,run);collect.validate_run_provenance(c,'none',run)
 except Exception as e:error=str(e)
 complete=stats.get('turns')==30 and stats.get('response_turns')==30 and error is None
 row={'attempt':number,'arm':'none','run_dir':str(run.relative_to(ROOT)),'exit_code':rc,'complete':complete,'validation_error':error,**stats}
 dump(path,row);log('voice_attempt',**row);return row

async def voice_lane():
 while not cfg('grok-voice')['serving']['verified']:await asyncio.sleep(10)
 c=cfg('grok-voice');assert c['serving']['verified'];art=HERE/'grok-voice/artifacts';art.mkdir(exist_ok=True)
 frozen=art/'source-sha256.txt';current=collect.source_hashes(c)
 if frozen.exists():assert frozen.read_text()==current,'Voice capture sources changed'
 else:frozen.write_text(current)
 attempts=[];next_number=1
 while sum(a['complete'] for a in attempts)<25:
  remaining=25-sum(a['complete'] for a in attempts)
  results=await asyncio.gather(*(voice_attempt(c,n) for n in range(next_number,next_number+min(3,remaining))))
  attempts.extend(results);next_number+=len(results)
  dump(HERE/'grok-voice/attempts.json',attempts)
  if len(attempts)>=50 and sum(a['complete'] for a in attempts)<25:raise RuntimeError('Voice attempt review cap reached')
 cases=[{**a,'slot':i} for i,a in enumerate([a for a in attempts if a['complete']],1)]
 dump(HERE/'grok-voice/COLLECTED.json',{'at':collect.utc_now(),'name':'grok-voice','cases':cases})

async def main():
 result=await asyncio.gather(text_lane(['luna']),text_lane(['sonnet','haiku']),text_lane(['spark13']),text_lane(['spark-open']),voice_lane(),return_exceptions=True)
 failures=[str(r) for r in result if isinstance(r,BaseException)]
 if failures:dump(HERE/'COLLECTION-ERRORS.json',failures);raise RuntimeError(failures)
 dump(HERE/'COLLECTION-COMPLETE.json',{'at':collect.utc_now(),'campaigns':NAMES})
if __name__=='__main__':
 with (HERE/'.collection.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
