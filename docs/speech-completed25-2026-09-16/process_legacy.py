import asyncio,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];AUDIT=ROOT/'docs/speech-v2-live-medium-2026-09-15'
async def main():
 run=Path(sys.argv[1]);out=run/'topup-validation';out.mkdir(exist_ok=True)
 async def command(name,args):
  with (out/(name+'.log')).open('w') as f:
   p=await asyncio.create_subprocess_exec(*map(str,args),cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT);return await p.wait()
 async def judge():
  for n in range(3):
   c=await command(f'judge-{n}',[ROOT/'.venv/bin/multi-turn-eval','judge',run,'--judge-model','claude-opus-5'])
   if c==0:return c
  return c
 async def audio():
  c=await command('audio',[sys.executable,AUDIT/'audit_gemini_audio.py',run,out/'audio.json'])
  a=await command('alignment',[sys.executable,AUDIT/'audit_gemini_alignment.py',run,out/'audio.json',out/'alignment.json']) if c==0 else None
  return c,a
 j,(a,align)=await asyncio.gather(judge(),audio());ts=[json.loads(l) for l in (run/'transcript.jsonl').read_text().splitlines()];inputs=[int(m[1]) for m in re.finditer(r'\[USER_AUDIO_QUEUED\] turn=(\d+)',(run/'run.log').read_text())]
 metrics=json.loads((out/'audio.json').read_text()) if (out/'audio.json').exists() else {};tmap={t['turn']:t for t in ts};issues=[]
 for issue in metrics.get('issues',[]):
  t=tmap.get(issue['turn'],{})
  if issue['reason']=='missing bot speech' and not t.get('assistant_text','').strip() and any(c['name']=='end_session' for c in t.get('tool_calls',[])):continue
  issues.append(issue)
 meta=json.loads((run/'topup-provenance.json').read_text());summary=json.loads((run/'claude_summary.json').read_text()) if j==0 else {};verified=summary.get('judge_model')=='claude-opus-5' and summary.get('judge_prompt_sha256')==meta['judge_prompt_sha256']
 result=dict(judge_exit=j,audio_exit=a,alignment_exit=align,input_sequence_valid=inputs==list(range(30)),judge_verified=verified,issues=issues,approved=j==a==align==0 and verified and inputs==list(range(30)) and not issues)
 (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':asyncio.run(main())
