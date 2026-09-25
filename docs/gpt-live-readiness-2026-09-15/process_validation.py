"""Wait for a validation capture, then run Opus 5 and WAV analysis concurrently."""
import asyncio,json,re,sys,time
from pathlib import Path
HERE=Path(__file__).parent
async def main():
    log=HERE/sys.argv[1];deadline=time.monotonic()+1200
    while True:
        match=re.search(r'^Output directory: (.+)$',log.read_text(),re.M) if log.exists() else None
        run=Path(match.group(1)) if match else None
        if run and (run/'runtime.json').exists():break
        if time.monotonic()>deadline:raise TimeoutError('Capture did not finalize')
        await asyncio.sleep(2)
    async def launch(label,command):
        with (run/f'{label}.log').open('w') as f:
            p=await asyncio.create_subprocess_exec(*command,stdout=f,stderr=asyncio.subprocess.STDOUT)
            return await p.wait()
    codes=await asyncio.gather(
        launch('judge',[str(Path(sys.executable).parent/'multi-turn-eval'),'judge',str(run),'--judge-model','claude-opus-5']),
        launch('audio',[sys.executable,str(HERE/'analyze_live_audio.py'),str(run)]),
        launch('alignment',[sys.executable,str(HERE/'audit_audio_alignment.py'),str(run)]))
    result=dict(run_dir=str(run),judge_exit=codes[0],audio_exit=codes[1],alignment_exit=codes[2])
    (run/'validation-processing.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
asyncio.run(main())
