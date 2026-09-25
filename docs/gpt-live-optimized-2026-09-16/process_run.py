"""Reconstruct, then judge and quantify one completed experimental recording."""
import asyncio,hashlib,json,sys
import numpy as np
import soundfile as sf
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];LIVE=ROOT/'docs/gpt-live-readiness-2026-09-15'
async def main():
 run=Path(sys.argv[1]);out=run/'experimental-validation';out.mkdir(exist_ok=True)
 async def command(name,args):
  with (out/(name+'.log')).open('w') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,*map(str,args),cwd=ROOT,stdout=f,stderr=asyncio.subprocess.STDOUT)
   return await p.wait()
 reconstruction=await command('reconstruction',[HERE/'reconstruct.py',run])
 async def judge():
  if reconstruction:return None
  if (run/'reconstructed/claude_summary.json').exists():return 0
  for n in range(1,4):
   c=await command(f'judge-{n}',[HERE/'judge_reconstructed.py',run])
   if c==0:return c
  return c
 j,a,align=await asyncio.gather(judge(),command('audio',[LIVE/'analyze_live_audio.py',run]),command('alignment',[LIVE/'audit_audio_alignment.py',run]))
 events=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()];ends={e['turn']:e['t'] for e in events if e['type']=='input_end'};gate=[e['turn'] for e in events if e['type']=='turn_complete' and (e['turn'] not in ends or e['t']-ends[e['turn']]<2.95)]
 provenance=json.loads((run/'benchmark-provenance.json').read_text())
 starts=[e['event']['session'] for e in events if e.get('event',{}).get('type')=='session.started']
 prompt_echo=bool(starts) and all(hashlib.sha256(s['instructions'].encode()).hexdigest()==provenance['frontend-prompt_sha256'] and hashlib.sha256(s['delegation']['responses']['instructions'].encode()).hexdigest()==provenance['backend-prompt_sha256'] for s in starts)
 incremental=None
 if (run/'incremental-conversation.wav').exists():
  original,sr=sf.read(run/'conversation.wav',dtype='int16');copy,cr=sf.read(run/'incremental-conversation.wav',dtype='int16')
  incremental=sr==cr and np.array_equal(original,copy)
 input_indices=[e['turn'] for e in events if e['type']=='input_start']
 sequence_valid=input_indices==provenance['turns']
 result=dict(run=str(run),reconstruction_exit=reconstruction,judge_exit=j,audio_exit=a,alignment_exit=align,premature_commits=gate,prompt_echo_verified=prompt_echo,input_sequence_valid=sequence_valid,incremental_recording_matches=incremental,approved=reconstruction==j==a==align==0 and not gate and prompt_echo and sequence_valid and incremental is not False)
 (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':asyncio.run(main())
