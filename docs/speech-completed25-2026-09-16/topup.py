"""Authorized top-ups: keep every attempt, take first validated completions."""
import asyncio,json,os,re,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SPECS=[('gemini31',1,[HERE/'capture_legacy_gemini.py','--model','gemini-3.1-flash-live-preview','--thinking','minimal'],'legacy'),('gemini38low',1,[HERE/'capture_legacy_gemini.py','--model','gemini-3.8-live-extended-thinking','--thinking','low'],'legacy'),('gemini38medium_new',2,[ROOT/'docs/gemini38-extended-exchanges-2026-09-16/capture.py','--label','completed25-topup'],'gemini'),('live_optimized',1,[ROOT/'docs/gpt-live-optimized-quiet5-2026-09-16/capture.py','--label','completed25-topup'],'live')]
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def append(p,r):
 with p.open('a') as f:f.write(json.dumps(r)+'\n')
async def command(args,log,timeout=1300):
 with log.open('w') as f:
  p=await asyncio.create_subprocess_exec(sys.executable,*map(str,args),cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
  try:return await asyncio.wait_for(p.wait(),timeout)
  except asyncio.TimeoutError:
   os.killpg(p.pid,signal.SIGTERM)
   try:await asyncio.wait_for(p.wait(),20)
   except asyncio.TimeoutError:os.killpg(p.pid,signal.SIGKILL);await p.wait()
   return 124
async def main():
 async def arm(key,target,args,kind):
  existing=[r for r in rows(HERE/'attempts.jsonl') if r['arm']==key];success=sum(r.get('accepted',False) for r in existing)
  for n in range(len(existing)+1,len(existing)+7):
   if success>=target:return
   log=HERE/f'{key}-{n:02d}.log'
   if log.exists():raise RuntimeError(f'Orphaned capture: {log}')
   append(HERE/'started.jsonl',dict(arm=key,index=n,time=time.time()));print('START',key,n,flush=True)
   code=await command(args,log);match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M);run=ROOT/match[1] if match else None
   ts=rows(run/'transcript.jsonl') if run else [];runtime=json.loads((run/'runtime.json').read_text()) if run and (run/'runtime.json').exists() else {}
   full=code==0 and runtime.get('valid') is not False and {t['turn'] for t in ts}==set(range(30))
   r=dict(arm=key,index=n,run_dir=str(run.relative_to(ROOT)) if run else None,exit_code=code,turns=len(ts),full=full,failure=runtime.get('failure'),accepted=False)
   if run and ts:
    if kind=='legacy':process=[HERE/'process_legacy.py',run]
    elif kind=='gemini':process=[ROOT/'docs/gemini38-extended-exchanges-2026-09-16/process_run.py',run]
    else:process=[ROOT/'docs/gpt-live-optimized-quiet5-2026-09-16/process_run.py',run]
    r['processing_exit']=await command(process,HERE/f'{key}-{n:02d}-process.log')
    vp=run/({'legacy':'topup-validation','gemini':'exchange-validation','live':'experimental-validation'}[kind])/'complete.json'
    validation=json.loads(vp.read_text()) if vp.exists() else {}
    r['validation']=validation;r['accepted']=full and validation.get('approved',False)
   append(HERE/'attempts.jsonl',r);print('FINISHED',key,n,'full',full,'accepted',r['accepted'],flush=True)
   if full and not r['accepted']:raise RuntimeError(f'{key}: completed capture needs validation review; no selective replacement')
   success+=r['accepted']
  raise RuntimeError(f'{key}: six additional attempts without target; review failures')
 await asyncio.gather(*(arm(*s) for s in SPECS))
 (HERE/'topups-complete.json').write_text(json.dumps(dict(time=time.time()))+'\n')
if __name__=='__main__':asyncio.run(main())
