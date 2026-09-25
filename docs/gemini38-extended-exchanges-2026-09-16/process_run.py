"""Validate exchanges, then run shared judge and waveform metrics independently."""
import asyncio,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];AUDIT=ROOT/'docs/speech-v2-live-medium-2026-09-15'
async def main():
 run=Path(sys.argv[1]);out=run/'exchange-validation';out.mkdir(exist_ok=True)
 async def command(name,args):
  with (out/(name+'.log')).open('w') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,*map(str,args),cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT)
   return await p.wait()
 reconstruction=await command('reconstruction',[HERE/'reconstruct.py',run])
 async def judge():
  if reconstruction:return None
  for i in range(1,4):
   code=await command(f'judge-{i}',[HERE/'judge_reconstructed.py',run])
   if code==0:return code
  return code
 async def audio():
  code=await command('audio',[AUDIT/'audit_gemini_audio.py',run,out/'audio.json'])
  align=await command('alignment',[AUDIT/'audit_gemini_alignment.py',run,out/'audio.json',out/'alignment.json']) if code==0 else None
  return code,align
 j,(a,align)=await asyncio.gather(judge(),audio())
 if j==0:
  summary=json.loads((run/'reconstructed/claude_summary.json').read_text());meta=json.loads((run/'benchmark-provenance.json').read_text())
  assert summary['judge_model']=='claude-opus-5' and summary['judge_prompt_sha256']==meta['judge_prompt_sha256']
 transcript=[json.loads(l) for l in (run/'transcript.jsonl').read_text().splitlines()];tmap={t['turn']:t for t in transcript};metrics=json.loads((out/'audio.json').read_text()) if (out/'audio.json').exists() else {}
 silent_closures=[];issues=[]
 for issue in metrics.get('issues',[]):
  t=tmap.get(issue['turn'],{})
  if issue['reason']=='missing bot speech' and not t.get('assistant_text','').strip() and any(c['name']=='end_session' for c in t.get('tool_calls',[])):silent_closures.append(issue['turn'])
  else:issues.append(issue)
 runtime=json.loads((run/'runtime.json').read_text())
 full={t['turn'] for t in transcript}==set(range(30)) and runtime.get('valid') is not False and not runtime.get('failure')
 result=dict(run=str(run),turns=len(transcript),full=full,runtime_failure=runtime.get('failure'),reconstruction_exit=reconstruction,judge_exit=j,audio_exit=a,alignment_exit=align,silent_closing_turns=silent_closures,audio_issues=issues,approved=full and reconstruction==j==a==align==0 and not issues)
 (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':asyncio.run(main())
