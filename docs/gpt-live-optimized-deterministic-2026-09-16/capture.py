"""Capture experimental Live exchanges; no judging or prompt changes during a run."""
import argparse,asyncio,hashlib,importlib.metadata,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from multi_turn_eval.cli import create_run_directory,setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from multi_turn_eval.pipelines.openai_live import OpenAILivePipeline
from benchmarks.aiwf_medium_context_speech_live_optimized.config import BenchmarkConfig
from disk_audio import DiskAudio
async def main():
 p=argparse.ArgumentParser();p.add_argument('--turns');p.add_argument('--label',required=True);a=p.parse_args();load_dotenv()
 b=BenchmarkConfig();indices=[int(x) for x in a.turns.split(',')] if a.turns else list(range(30))
 out=create_run_directory(b.name,'gpt-live-1');setup_logging(out)
 meta=dict(label=a.label,benchmark=b.name,model='gpt-live-1',backend='gpt-5.6-terra',reasoning='medium',voice='marin',prompt_version=b.prompt_version,protocol='live-playback-exchanges-v1',turns=indices,judge_model='claude-opus-5',versions={p:importlib.metadata.version(p) for p in ['openai','pipecat-ai']},completion_gate='backend lifecycle + playback + 3s quiet; no text classifier')
 for name,txt in [('frontend-prompt',b.frontend_system_instruction),('backend-prompt',b.backend_system_instruction),('judge-prompt',b.judge_system_prompt)]:
  (out/(name+'.txt')).write_text(txt);meta[name+'_sha256']=hashlib.sha256(txt.encode()).hexdigest()
 for name,path in [('adapter',ROOT/'src/multi_turn_eval/pipelines/openai_live.py'),('disk_audio',HERE/'disk_audio.py')]:
  data=path.read_bytes();(out/(name+'-snapshot.py')).write_bytes(data);meta[name+'_sha256']=hashlib.sha256(data).hexdigest()
 (out/'benchmark-provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print('Output directory:',out,flush=True)
 recorder=TranscriptRecorder(out,'GPT-Live prompt-optimized');disk=DiskAudio(out)
 try:await asyncio.wait_for(OpenAILivePipeline(b,recorder,recording_observer=disk).run(indices),1200)
 finally:recorder.close();disk.close_tracks()
if __name__=='__main__':asyncio.run(main())
