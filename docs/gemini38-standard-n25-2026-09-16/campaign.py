"""First 25 validated complete conversations; one validation pilot then N=5."""
import asyncio, importlib.util, json, re, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'docs/speech-completed25-2026-09-16/topup.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
async def attempt(n):
 log=HERE/f'capture-{n:02d}.log'
 if log.exists():raise RuntimeError(f'Orphaned attempt {n}; review before resuming')
 h.append(HERE/'started.jsonl',dict(index=n,time=time.time()));print('START',n,flush=True)
 code=await h.command([HERE/'capture.py'],log)
 match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M);run=ROOT/match[1] if match else None
 ts=h.rows(run/'transcript.jsonl') if run else [];runtime=json.loads((run/'runtime.json').read_text()) if run and (run/'runtime.json').exists() else {}
 full=code==0 and runtime.get('valid') is not False and not runtime.get('failure') and {t['turn'] for t in ts}==set(range(30))
 r=dict(index=n,run_dir=str(run.relative_to(ROOT)) if run else None,exit_code=code,turns=len(ts),full=full,failure=runtime.get('failure'),accepted=False)
 if run and ts:
  r['processing_exit']=await h.command([ROOT/'docs/speech-completed25-2026-09-16/process_legacy.py',run],HERE/f'process-{n:02d}.log')
  vp=run/'topup-validation/complete.json';r['validation']=json.loads(vp.read_text()) if vp.exists() else {};r['accepted']=full and r['validation'].get('approved',False)
 h.append(HERE/'attempts.jsonl',r);print('FINISHED',n,'turns',len(ts),'accepted',r['accepted'],flush=True)
 return r
async def main():
 while True:
  rows=h.rows(HERE/'attempts.jsonl');success=sum(r['accepted'] for r in rows)
  if any(r['full'] and not r['accepted'] for r in rows):raise RuntimeError('Full capture needs validation review; do not selectively replace')
  if any(not r['run_dir'] or (not r['full'] and not r['failure']) for r in rows):raise RuntimeError('Unclassified failure needs review')
  if success==25:
   (HERE/'complete.json').write_text(json.dumps(dict(completed=success,attempts=len(rows),time=time.time()),indent=2)+'\n');return
  if len(rows)>=50:raise RuntimeError('Attempt cap reached; review failures')
  count=1 if len(rows) in (0,35) else min(5,25-success,50-len(rows))
  await asyncio.gather(*(attempt(n) for n in range(len(rows)+1,len(rows)+count+1)))
if __name__=='__main__':asyncio.run(main())
