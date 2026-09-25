"""25 fresh experimental attempts, concurrency five, no score-based replacement."""
import asyncio,hashlib,importlib.metadata,json,os,re,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.aiwf_medium_context_speech_live_optimized.config import BenchmarkConfig
OUT=HERE/'cohort';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row)+'\n')
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
async def main():
 b=BenchmarkConfig();files=['src/multi_turn_eval/pipelines/openai_live.py',*[str((HERE/x).relative_to(ROOT)) for x in ['capture.py','coordinator.py','disk_audio.py','reconstruct.py','judge_reconstructed.py','process_run.py']], 'benchmarks/aiwf_medium_context_speech_live_optimized/config.py','benchmarks/aiwf_medium_context_speech_live_optimized/prompts/frontend.txt','benchmarks/aiwf_medium_context_speech_live_optimized/prompts/backend.txt']
 protocol=dict(label='GPT-Live prompt-optimized',method='live-exchange-v1',attempts=25,planned_turns=750,concurrency=5,replace_failures=False,model='gpt-live-1',backend='gpt-5.6-terra',reasoning='medium',voice='marin',coordinator='gpt-5.6-luna low',source_hashes={p:sha(ROOT/p) for p in files},frontend_sha256=hashlib.sha256(b.frontend_system_instruction.encode()).hexdigest(),backend_sha256=hashlib.sha256(b.backend_system_instruction.encode()).hexdigest(),judge_sha256=hashlib.sha256(b.judge_system_prompt.encode()).hexdigest(),versions={p:importlib.metadata.version(p) for p in ['openai','pipecat-ai']},substantive_latency='Not validated; first-speech metrics include acknowledgments. Report limitation.')
 path=OUT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==protocol
 else:path.write_text(json.dumps(protocol,indent=2)+'\n')
 sem=asyncio.Semaphore(5);processing=asyncio.Semaphore(2);tasks=[]
 async def command(args,log,timeout):
  with log.open('w') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,*map(str,args),cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
   try:return await asyncio.wait_for(p.wait(),timeout)
   except asyncio.TimeoutError:
    os.killpg(p.pid,signal.SIGTERM)
    try:await asyncio.wait_for(p.wait(),20)
    except asyncio.TimeoutError:os.killpg(p.pid,signal.SIGKILL);await p.wait()
    return 124
 async def process(r):
  async with processing:
   if not r['run_dir']:return
   code=await command([HERE/'process_run.py',ROOT/r['run_dir']],OUT/f"process-{r['index']:02d}.log",1200)
   await command([HERE/'summarize_run.py',ROOT/r['run_dir']],OUT/f"summary-{r['index']:02d}.log",60)
   append(OUT/'processed.jsonl',dict(index=r['index'],run_dir=r['run_dir'],exit_code=code));print('PROCESSED',r['index'],code,flush=True)
 completed={r['index'] for r in rows(OUT/'attempts.jsonl')}
 async def capture(i):
  if i in completed:return
  async with sem:
   log=OUT/f'capture-{i:02d}.log'
   if log.exists():raise RuntimeError('Orphaned attempt requires audit; do not replace')
   assert all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())
   append(OUT/'started.jsonl',dict(index=i,time=time.time()));print('START',i,flush=True)
   code=await command([HERE/'capture.py','--label',f'experimental-n25-{i}'],log,1230)
   match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M);rd=match.group(1) if match else None
   transcript=rows(ROOT/rd/'transcript.jsonl') if rd else [];runtime=json.loads((ROOT/rd/'runtime.json').read_text()) if rd and (ROOT/rd/'runtime.json').exists() else {}
   record=dict(index=i,run_dir=rd,exit_code=code,turns=len(transcript),full=code==0 and {r['turn'] for r in transcript}==set(range(30)) and runtime.get('valid') is not False)
   append(OUT/'attempts.jsonl',record);print('CAPTURED',json.dumps(record),flush=True);tasks.append(asyncio.create_task(process(record)))
 already={r['index'] for r in rows(OUT/'processed.jsonl')}
 for r in rows(OUT/'attempts.jsonl'):
  if r['index'] not in already:tasks.append(asyncio.create_task(process(r)))
 await asyncio.gather(*(capture(i) for i in range(1,26)));await asyncio.gather(*tasks)
 assert len(rows(OUT/'attempts.jsonl'))==25
 (OUT/'complete.json').write_text(json.dumps(dict(attempts=25,finished_unix=time.time()))+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':asyncio.run(main())
