"""20 GPT-Live additions + 25 Gemini medium attempts; no capture replacements."""
import asyncio,hashlib,importlib.metadata,json,os,re,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
LIVE=ROOT/'docs/gpt-live-readiness-2026-09-15'
CANONICAL=ROOT/'docs/speech-v2-canonical-2026-09-15'
REVISION=CANONICAL/'rejudge-policy-compliance-1'
sys.path.insert(0,str(ROOT))
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from benchmarks.aiwf_medium_context_speech_live.config import BenchmarkConfig as LiveConfig

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p):return [json.loads(l) for l in p.read_text().splitlines() if l] if p.exists() else []
def append(p,r):
 with p.open('a') as f:f.write(json.dumps(r)+'\n')

def prepare():
 b=BenchmarkConfig();live=LiveConfig();frozen=json.loads((CANONICAL/'protocol.json').read_text());rev=json.loads((REVISION/'protocol.json').read_text())
 assert b.system_instruction==live.system_instruction
 assert b.judge_system_prompt==live.judge_system_prompt
 assert hashlib.sha256(b.system_instruction.encode()).hexdigest()==frozen['prompt_sha256']
 assert hashlib.sha256(b.judge_system_prompt.encode()).hexdigest()==rev['judge_prompt_sha256']
 pilots=sorted(read_jsonl(LIVE/'concurrency-pilot/attempts.jsonl'),key=lambda r:r['index'])
 assert len(pilots)==5 and {r['index'] for r in pilots}==set(range(1,6))
 for p in pilots:
  meta=json.loads((ROOT/p['run_dir']/'benchmark-provenance.json').read_text())
  assert meta['implementation_sha256']==sha(ROOT/'src/multi_turn_eval/pipelines/openai_live.py')
  assert meta['frontend-prompt_sha256']==frozen['prompt_sha256']
  for package,version in meta['versions'].items():assert importlib.metadata.version(package)==version
 signature=dict(prompt_sha256=frozen['prompt_sha256'],judge_prompt_sha256=rev['judge_prompt_sha256'],judge_model='claude-opus-5',judge_revision=rev['revision'],attempts_per_arm=25,planned_turns_per_arm=750,arms=[dict(key='live',model='gpt-live-1',backend='gpt-5.6-terra',reasoning='medium',voice='marin',imported_attempts=5,new_attempts=20),dict(key='gemini38medium',model='gemini-3.8-live-extended-thinking',thinking='medium',new_attempts=25)],capture_concurrency_per_provider=5,recovery=False,replace_failed_attempts=False,excluded_live_single_validation='20260915T150622_gpt-live-1_ba419085',versions={p:importlib.metadata.version(p) for p in ['openai','pipecat-ai','google-genai']},source_hashes={p:sha(ROOT/p) for p in ['src/multi_turn_eval/pipelines/openai_live.py','src/multi_turn_eval/pipelines/realtime.py','benchmarks/aiwf_medium_context_speech_v2/judge.py']})
 target=HERE/'protocol.json'
 if target.exists():assert json.loads(target.read_text())==signature
 else:
  target.write_text(json.dumps(signature,indent=2)+'\n')
  (HERE/'system-prompt.txt').write_text(b.system_instruction)
  (HERE/'judge-prompt.txt').write_text(b.judge_system_prompt)
  for p in ['pyproject.toml','uv.lock']:(HERE/p).write_bytes((ROOT/p).read_bytes())
 (HERE/'capture').mkdir(exist_ok=True);(HERE/'processed').mkdir(exist_ok=True)
 existing={(r['arm'],r['index']) for r in read_jsonl(HERE/'capture/attempts.jsonl')}
 for p in pilots:
  key=f"live-{p['index']:02d}";out=HERE/'processed'/key;out.mkdir(exist_ok=True)
  record=dict(arm='live',index=p['index'],model='gpt-live-1',run_dir=p['run_dir'],exit_code=p['exit_code'],turns=p['runtime']['turns'],full=p['runtime']['valid'] and p['runtime']['turns']==30,start_unix=p['start_unix'],end_unix=p['end_unix'],log=p['log'],imported=True)
  if ('live',p['index']) not in existing:append(HERE/'capture/attempts.jsonl',record)
  score=REVISION/key
  assert json.loads((score/'claude_summary.json').read_text())['judge_prompt_sha256']==signature['judge_prompt_sha256']
  done=dict(record=record,judge=True,audio=True,alignment=True,score_dir=str(score),audio_path=str(ROOT/p['run_dir']/'live-audio-metrics.json'),alignment_path=str(ROOT/p['run_dir']/'audio-alignment-audit.json'),imported=True)
  (out/'done.json').write_text(json.dumps(done,indent=2)+'\n')
 return signature

async def main():
 protocol=prepare();completed={(r['arm'],r['index']) for r in read_jsonl(HERE/'capture/attempts.jsonl')}
 sems={k:asyncio.Semaphore(5) for k in ['live','gemini38medium']};judges=asyncio.Semaphore(5);audio=asyncio.Semaphore(2);processing=[]
 env=os.environ.copy();env['MTE_ENABLE_RECOVERY']='0';env.pop('MTE_GOOGLE_THINKING_MODE',None);env.pop('MTE_OPENAI_REALTIME_REASONING_EFFORT',None)
 async def command(args,log,timeout=900,output=None):
  with log.open('w') as f:
   stream=output.open('w') if output else f
   try:
    p=await asyncio.create_subprocess_exec(*args,stdout=stream,stderr=f if output else asyncio.subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
    try:return await asyncio.wait_for(p.wait(),timeout)
    except asyncio.TimeoutError:
     os.killpg(p.pid,signal.SIGTERM)
     try:await asyncio.wait_for(p.wait(),20)
     except asyncio.TimeoutError:os.killpg(p.pid,signal.SIGKILL);await p.wait()
     return 124
   finally:
    if output:stream.close()
 async def process(r):
  key=f"{r['arm']}-{r['index']:02d}";out=HERE/'processed'/key;out.mkdir(exist_ok=True)
  if (out/'done.json').exists():return
  run=ROOT/r['run_dir'] if r.get('run_dir') else None
  result=dict(record=r,judge=False,audio=False,alignment=None,score_dir=str(run) if run else None)
  if run and r['turns']:
   async def judge():
    async with judges:
     if (run/'claude_summary.json').exists():
      assert json.loads((run/'claude_summary.json').read_text())['judge_prompt_sha256']==protocol['judge_prompt_sha256'];return True
     for attempt in range(1,4):
      code=await command([str(ROOT/'.venv/bin/multi-turn-eval'),'judge',str(run),'--judge-model','claude-opus-5'],out/f'judge-{attempt}.log')
      if code==0:
       assert json.loads((run/'claude_summary.json').read_text())['judge_prompt_sha256']==protocol['judge_prompt_sha256'];return True
     return False
   async def analyze():
    async with audio:
     if not (run/'conversation.wav').exists():return False
     if r['arm']=='live':
      code=await command([sys.executable,str(LIVE/'analyze_live_audio.py'),str(run)],out/'audio.log',timeout=300)
      result['audio_path']=str(run/'live-audio-metrics.json')
      align=await command([sys.executable,str(LIVE/'audit_audio_alignment.py'),str(run)],out/'alignment.log',timeout=300)
      result['alignment']=align==0;result['alignment_path']=str(run/'audio-alignment-audit.json');result['audio_exit']=code
      # Keep partial metrics for incomplete attempts; report any audit flags.
      return (run/'live-audio-metrics.json').exists()
     code=await command([sys.executable,'scripts/analyze_turn_metrics.py',str(run),'--json'],out/'audio.log',timeout=300,output=out/'metrics.json')
     result['audio_path']=str(out/'metrics.json');result['audio_exit']=code
     return code==0
   result['judge'],result['audio']=await asyncio.gather(judge(),analyze())
  (out/'done.json').write_text(json.dumps(result,indent=2)+'\n')
  print('PROCESSED',key,'judge',result['judge'],'audio',result['audio'],flush=True)
 async def capture(arm,index):
  if (arm,index) in completed:return
  async with sems[arm]:
   key=f'{arm}-{index:02d}';log=HERE/'capture'/f'{key}.log'
   if log.exists():raise RuntimeError(f'Orphaned capture needs audit: {key}')
   start=time.time();append(HERE/'capture/started.jsonl',dict(arm=arm,index=index,start_unix=start));print('START',key,flush=True)
   args=[sys.executable,str(LIVE/'validate.py'),'--label',f'canonical-n25-{index}'] if arm=='live' else [str(ROOT/'.venv/bin/multi-turn-eval'),'run','aiwf_medium_context_speech_v2','--model','gemini-3.8-live-extended-thinking','--service','gemini-live','--thinking','medium']
   code=await command(args,log,timeout=930 if arm=='live' else 900)
   match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M);rd=match.group(1) if match else None
   run=ROOT/rd if rd else None;rows=read_jsonl(run/'transcript.jsonl') if run else [];rows=[r for r in rows if not r.get('recovery_turn')]
   runtime=json.loads((run/'runtime.json').read_text()) if run and (run/'runtime.json').exists() else {}
   r=dict(arm=arm,index=index,model='gpt-live-1' if arm=='live' else 'gemini-3.8-live-extended-thinking',run_dir=rd,exit_code=code,turns=len(rows),full=code==0 and {r['turn'] for r in rows}==set(range(30)) and runtime.get('valid') is not False,start_unix=start,end_unix=time.time(),log=str(log.relative_to(ROOT)),imported=False)
   append(HERE/'capture/attempts.jsonl',r);print('CAPTURED',json.dumps(r),flush=True)
   processing.append(asyncio.create_task(process(r)))
 for r in read_jsonl(HERE/'capture/attempts.jsonl'):
  if not r.get('imported'):processing.append(asyncio.create_task(process(r)))
 await asyncio.gather(*(capture(arm,i) for i in range(1,26) for arm in ['live','gemini38medium']))
 await asyncio.gather(*processing)
 rows=read_jsonl(HERE/'capture/attempts.jsonl');assert len(rows)==50 and len({(r['arm'],r['index']) for r in rows})==50
 (HERE/'complete.json').write_text(json.dumps(dict(attempts=50,new_captures=45,finished_unix=time.time()))+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':asyncio.run(main())
