"""Capture an auditable GPT-Live validation conversation, excluded from canonical runs."""
import argparse, asyncio, hashlib, json, importlib.metadata
from pathlib import Path
from dotenv import load_dotenv
from multi_turn_eval.cli import create_run_directory, setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from multi_turn_eval.pipelines.openai_live import OpenAILivePipeline
from benchmarks.aiwf_medium_context_speech_live.config import BenchmarkConfig

async def main():
    p=argparse.ArgumentParser();p.add_argument('--turns');p.add_argument('--label',default='validation');args=p.parse_args()
    load_dotenv()
    b=BenchmarkConfig();out=create_run_directory(b.name,'gpt-live-1');setup_logging(out)
    indices=[int(i) for i in args.turns.split(',')] if args.turns else None
    metadata=dict(benchmark=b.name,prompt_version=b.prompt_version,model='gpt-live-1',backend='gpt-5.6-terra',reasoning='medium',voice='marin',pipeline='openai-live',quiet_seconds=3,turn_timeout_seconds=90,recovery=False,replace_failed_attempts=False,label=args.label,turns=indices or list(range(30)),versions={p:importlib.metadata.version(p) for p in ['openai','pipecat-ai']})
    for name,text in [('system-prompt',b.system_instruction),('frontend-prompt',b.system_instruction),('judge-prompt',b.judge_system_prompt)]:
        (out/(name+'.txt')).write_text(text);metadata[name+'_sha256']=hashlib.sha256(text.encode()).hexdigest()
    (out/'openai_live_snapshot.py').write_bytes(Path('src/multi_turn_eval/pipelines/openai_live.py').read_bytes())
    metadata['implementation_sha256']=hashlib.sha256(Path('src/multi_turn_eval/pipelines/openai_live.py').read_bytes()).hexdigest()
    (out/'benchmark-provenance.json').write_text(json.dumps(metadata,indent=2))
    print('Output directory:',out,flush=True)
    recorder=TranscriptRecorder(out,'gpt-live-1 + gpt-5.6-terra (medium)')
    try:await asyncio.wait_for(OpenAILivePipeline(b,recorder).run(indices),900)
    finally:recorder.close()
if __name__=='__main__':asyncio.run(main())
