"""Collect the first 25 complete high-mode voice conversations, in batches of up to 3."""
import asyncio,fcntl,json,sys
from common import HERE,ROOT,RAW,collect,cfg,dump,sha,verify_baseline
import smoke

def log(event,**kw):
 row={'at':collect.utc_now(),'event':event,**kw}
 with (HERE/'collection-events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def freeze():
 c=cfg();assert c['serving']['verified'];verify_baseline()
 text=collect.source_hashes(c);path=HERE/'grok-voice/artifacts/source-sha256.txt'
 if path.exists():assert path.read_text()==text
 else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 hashes={}
 for line in text.splitlines():
  digest,name=line.split(maxsplit=1);source=ROOT/name;target=HERE/'runtime-snapshot'/name
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.read_bytes()==source.read_bytes()
  else:target.write_bytes(source.read_bytes())
  hashes[name]=digest
 snapshot=HERE/'runtime-sha256.json'
 if snapshot.exists():assert json.loads(snapshot.read_text())==hashes
 else:dump(snapshot,hashes)
 return c
async def attempt(c,number):
 path=HERE/'grok-voice/artifacts'/f'attempt-{number:03d}.json';run=RAW/f'attempt-{number:03d}'
 if path.exists():return json.loads(path.read_text())
 assert (HERE/'grok-voice/artifacts/source-sha256.txt').read_text()==collect.source_hashes(c)
 if not run.exists():
  env=collect.child_environment(c,'high',collect.extract_credential(c))
  with path.with_suffix('.log').open('w') as out:
   proc=await asyncio.create_subprocess_exec(sys.executable,str(collect.RUN_ONE),'--config',str(c['_config_path']),'--arm','high','--run-dir',str(run),cwd=ROOT,env=env,stdout=out,stderr=asyncio.subprocess.STDOUT)
   try:code=await asyncio.wait_for(proc.wait(),1800)
   except asyncio.TimeoutError:proc.terminate();await proc.wait();code=124
 else:code=None
 stats={};error=None
 try:stats=collect.validate_transcript(c,run);collect.validate_run_provenance(c,'high',run)
 except Exception as exc:error=str(exc)
 complete=stats.get('turns')==30 and stats.get('response_turns')==30 and error is None
 row={'attempt':number,'arm':'high','run_dir':str(run.relative_to(ROOT)),'exit_code':code,'complete':complete,'validation_error':error,**stats}
 dump(path,row);log('voice_attempt',**row);return row
async def main():
 await smoke.main();c=freeze();attempts=[];number=1
 while sum(a['complete'] for a in attempts)<25:
  count=min(3,25-sum(a['complete'] for a in attempts))
  attempts.extend(await asyncio.gather(*(attempt(c,n) for n in range(number,number+count))))
  number+=count;dump(HERE/'grok-voice/attempts.json',attempts)
  if len(attempts)>=50 and sum(a['complete'] for a in attempts)<25:raise RuntimeError('Voice attempt review cap reached')
 cases=[{**a,'slot':i} for i,a in enumerate([a for a in attempts if a['complete']],1)]
 assert len(cases)==25
 dump(HERE/'grok-voice/COLLECTED.json',{'at':collect.utc_now(),'name':'grok-voice','cases':cases})
 dump(HERE/'COLLECTION-COMPLETE.json',{'at':collect.utc_now(),'complete':25,'attempts':len(attempts)})
if __name__=='__main__':
 with (HERE/'.collection.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
