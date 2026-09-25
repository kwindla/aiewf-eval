"""Frozen attempt-based campaign. No replacements for model/service failures."""
import asyncio, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path.cwd(); HERE=ROOT/'docs/speech-v2-canonical-2026-09-15'
BENCHMARK='aiwf_medium_context_speech_v2'
ARMS=[
 dict(key='openai21',model='gpt-realtime-2.1',service='openai-realtime',provider='openai',reasoning='low'),
 dict(key='grok',model='grok-voice-think-fast-1.0',pipeline='grok-realtime',provider='xai'),
 dict(key='openai15',model='gpt-realtime-1.5',service='openai-realtime',provider='openai'),
 dict(key='gemini31',model='gemini-3.1-flash-live-preview',service='gemini-live',provider='google',thinking='minimal'),
 dict(key='gemini38',model='gemini-3.8-live-extended-thinking',service='gemini-live',provider='google',thinking='low'),
]
def now():return datetime.now(timezone.utc).isoformat()
def command(arm,smoke):
 cmd=[str(ROOT/'.venv/bin/multi-turn-eval'),'run',BENCHMARK,'--model',arm['model']]
 for key in ['service','pipeline','thinking']:
  if key in arm:cmd+=['--'+key,arm[key]]
 if smoke:cmd+=['--only-turns','0,1']
 return cmd
async def main():
 smoke='--smoke' in sys.argv
 count=1 if smoke else 25
 out=HERE/('smoke' if smoke else 'capture');out.mkdir(exist_ok=True)
 manifest=out/'attempts.jsonl'
 existing=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
 completed={(x['arm'],x['index']) for x in existing}
 sems={p:asyncio.Semaphore(5) for p in ['openai','google','xai']}
 from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
 config=BenchmarkConfig()
 signature={'benchmark':BENCHMARK,'prompt_sha256':hashlib.sha256(config.system_instruction.encode()).hexdigest(),'judge_prompt_sha256':hashlib.sha256(config.judge_system_prompt.encode()).hexdigest(),'judge_model':'claude-opus-5','attempts_per_arm':25,'arms':ARMS,'recovery_nudges':False,'replace_incomplete':False}
 frozen=HERE/'protocol.json'
 if frozen.exists():assert json.loads(frozen.read_text())==signature,'Protocol changed since freeze'
 else:
  frozen.write_text(json.dumps(signature,indent=2)+'\n')
  (HERE/'system-prompt.txt').write_text(config.system_instruction)
  (HERE/'judge-prompt.txt').write_text(config.judge_system_prompt)
 async def run(arm,index):
  if (arm['key'],index) in completed:return
  async with sems[arm['provider']]:
   logpath=out/f"{arm['key']}-{index:02d}.log"
   if logpath.exists():raise RuntimeError(f'Orphaned attempt needs audit: {logpath}')
   env=os.environ.copy();env.pop('MTE_OPENAI_REALTIME_REASONING_EFFORT',None);env.pop('MTE_GOOGLE_THINKING_MODE',None)
   env['MTE_ENABLE_RECOVERY']='0'
   if arm.get('reasoning'):env['MTE_OPENAI_REALTIME_REASONING_EFFORT']=arm['reasoning']
   started=now();print('start',arm['key'],index,flush=True)
   with logpath.open('w') as log:
    p=await asyncio.create_subprocess_exec(*command(arm,smoke),stdout=log,stderr=asyncio.subprocess.STDOUT,env=env,start_new_session=True)
    try:code=await asyncio.wait_for(p.wait(),timeout=900)
    except asyncio.TimeoutError:
     import signal
     os.killpg(p.pid,signal.SIGTERM);await p.wait();code=124
   import re
   text=logpath.read_text();m=re.search(r'^Output directory: (.+)$',text,re.M);run_dir=m.group(1) if m else None
   rows=[]
   if run_dir and (ROOT/run_dir/'transcript.jsonl').exists():rows=[json.loads(x) for x in (ROOT/run_dir/'transcript.jsonl').read_text().splitlines() if x]
   scripted=[r for r in rows if not r.get('recovery_turn')]
   record={'arm':arm['key'],'model':arm['model'],'index':index,'run_dir':run_dir,'exit_code':code,'turns':len(scripted),'full':code==0 and {r['turn'] for r in scripted}==set(range(2 if smoke else 30)),'started':started,'finished':now(),'log':str(logpath.relative_to(ROOT))}
   with manifest.open('a') as f:f.write(json.dumps(record)+'\n')
   print(json.dumps(record),flush=True)
 await asyncio.gather(*(run(arm,index) for index in range(1,count+1) for arm in ARMS))
if __name__=='__main__':asyncio.run(main())
