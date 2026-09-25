"""Audit original low and new medium WAVs without changing scores or captures."""
import asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'audio-audit'
def records(p):return [json.loads(l) for l in p.read_text().splitlines() if l] if p.exists() else []
async def main():
 OUT.mkdir(exist_ok=True);sem=asyncio.Semaphore(1);jobs={}
 async def audit(key,r):
  async with sem:
   out=OUT/f'{key}.json';alignment=OUT/f'{key}.alignment.json';done=OUT/f'{key}.done.json'
   if done.exists():return
   run=ROOT/r['run_dir'] if r.get('run_dir') else None
   if not run or not (run/'conversation.wav').exists():
    done.write_text(json.dumps(dict(run_dir=r.get('run_dir'),audit=False,reason='No recorded WAV')));return
   codes=[]
   commands=[[sys.executable,str(HERE/'audit_gemini_audio.py'),str(run),str(out)],[sys.executable,str(HERE/'audit_gemini_alignment.py'),str(run),str(out),str(alignment)]]
   for i,cmd in enumerate(commands):
    with (OUT/f'{key}-{i}.log').open('w') as f:
     p=await asyncio.create_subprocess_exec(*cmd,stdout=f,stderr=asyncio.subprocess.STDOUT,cwd=ROOT)
     codes.append(await p.wait())
    if codes[-1]:break
   result=dict(run_dir=r['run_dir'],audit=all(c==0 for c in codes) and len(codes)==2,codes=codes,metrics=str(out),alignment=str(alignment))
   done.write_text(json.dumps(result,indent=2)+'\n');print(key,result['audit'],codes,flush=True)
 old=records(ROOT/'docs/speech-v2-canonical-2026-09-15/capture/attempts.jsonl')
 for r in old:
  if r['arm']=='gemini38':
   key=f"gemini38low-{r['index']:02d}";jobs[key]=asyncio.create_task(audit(key,r))
 while True:
  new=[r for r in records(HERE/'capture/attempts.jsonl') if r['arm']=='gemini38medium']
  for r in new:
   key=f"gemini38medium-{r['index']:02d}"
   if key not in jobs:jobs[key]=asyncio.create_task(audit(key,r))
  if len(new)==25:break
  await asyncio.sleep(5)
 await asyncio.gather(*jobs.values());(OUT/'complete.json').write_text(json.dumps(dict(conversations=50))+'\n');print('COMPLETE',flush=True)
asyncio.run(main())
