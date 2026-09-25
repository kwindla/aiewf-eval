"""Five full conversations launched together; all attempts retained, no retries."""
import asyncio,json,re,sys,time
from pathlib import Path
HERE=Path(__file__).parent
OUT=HERE/'concurrency-pilot'
async def main():
    OUT.mkdir(exist_ok=False)
    # Preserve background provider load when interpreting this concurrency test.
    canonical=Path('docs/speech-v2-canonical-2026-09-15/capture/attempts.jsonl')
    captured=[json.loads(l) for l in canonical.read_text().splitlines()] if canonical.exists() else []
    (OUT/'background-load.json').write_text(json.dumps({'canonical_attempts_finished_at_pilot_start':len(captured),'canonical_attempts_planned':125,'note':'The earlier canonical campaign may still have up to five concurrent OpenAI Realtime sessions. This pilot adds five GPT-Live sessions; these are separate captures and models.'},indent=2))
    async def one(index):
        start=time.time();log=OUT/f'capture-{index}.log'
        with log.open('w') as f:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(HERE/'validate.py'),'--label',f'concurrency-pilot-{index}',stdout=f,stderr=asyncio.subprocess.STDOUT)
            code=await proc.wait()
        match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M)
        run=match.group(1) if match else None
        record=dict(index=index,start_unix=start,end_unix=time.time(),exit_code=code,run_dir=run,log=str(log))
        if run and (Path(run)/'runtime.json').exists():record['runtime']=json.loads((Path(run)/'runtime.json').read_text())
        with (OUT/'attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if run and (Path(run)/'transcript.jsonl').stat().st_size:
            async def process(name,args):
                with (OUT/f'{name}-{index}.log').open('w') as f:
                    p=await asyncio.create_subprocess_exec(*args,stdout=f,stderr=asyncio.subprocess.STDOUT)
                    return await p.wait()
            results=await asyncio.gather(
                process('judge',[str(Path(sys.executable).parent/'multi-turn-eval'),'judge',run,'--judge-model','claude-opus-5']),
                process('audio',[sys.executable,str(HERE/'analyze_live_audio.py'),run]),
                process('alignment',[sys.executable,str(HERE/'audit_audio_alignment.py'),run]))
            (OUT/f'processed-{index}.json').write_text(json.dumps(dict(run_dir=run,judge_exit=results[0],audio_exit=results[1],alignment_exit=results[2]),indent=2))
    await asyncio.gather(*(one(i) for i in range(1,6)))
asyncio.run(main())
