"""Judge and analyze completed captures while the frozen campaign runs."""
import asyncio, json, sys
from pathlib import Path
ROOT=Path.cwd();HERE=ROOT/'docs/speech-v2-canonical-2026-09-15'
async def main():
 sem=asyncio.Semaphore(5);tasks={}
 async def process(record):
  key=f"{record['arm']}-{record['index']:02d}"
  out=HERE/'processed'/key;out.mkdir(parents=True,exist_ok=True)
  if (out/'done.json').exists():return
  rd=record.get('run_dir')
  if not rd or record['turns']==0:
   (out/'done.json').write_text(json.dumps({'record':record,'judged':False,'audio':False})+'\n');return
  async with sem:
   run=ROOT/rd
   if not (run/'claude_summary.json').exists():
    for attempt in range(1,4):
     with (out/f'judge-{attempt}.log').open('w') as log:
      p=await asyncio.create_subprocess_exec(str(ROOT/'.venv/bin/multi-turn-eval'),'judge',str(run),'--judge-model','claude-opus-5',stdout=log,stderr=asyncio.subprocess.STDOUT)
      code=await p.wait()
     if code==0:break
    else:raise RuntimeError(f'Judge failed: {key}')
   with (out/'metrics.json').open('w') as data,(out/'metrics.log').open('w') as log:
    p=await asyncio.create_subprocess_exec(sys.executable,'scripts/analyze_turn_metrics.py',str(run),'--json',stdout=data,stderr=log)
    code=await p.wait()
   if code:raise RuntimeError(f'Audio analysis failed: {key}')
   metrics=json.loads((out/'metrics.json').read_text())
   summary=json.loads((run/'claude_summary.json').read_text())
   frozen=json.loads((HERE/'protocol.json').read_text())
   assert summary['judge_model']==frozen['judge_model']
   assert summary['judge_prompt_sha256']==frozen['judge_prompt_sha256']
   (out/'done.json').write_text(json.dumps({'record':record,'judged':True,'audio':True,'summary':summary})+'\n')
   print('processed',key,summary['turn_pass'],flush=True)
 while True:
  manifest=HERE/'capture/attempts.jsonl'
  records=[json.loads(l) for l in manifest.read_text().splitlines() if l] if manifest.exists() else []
  for r in records:
   key=(r['arm'],r['index'])
   if key not in tasks:tasks[key]=asyncio.create_task(process(r))
  for task in tasks.values():
   if task.done() and task.exception():raise task.exception()
  if len(records)==125:break
  await asyncio.sleep(5)
 await asyncio.gather(*tasks.values())
 print('processed all 125 attempts',flush=True)
if __name__=='__main__':asyncio.run(main())
