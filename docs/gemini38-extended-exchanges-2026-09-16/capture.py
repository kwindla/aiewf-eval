"""Capture Gemini extended thinking with raw exchange provenance."""
import argparse,asyncio,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from multi_turn_eval.cli import create_run_directory,setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from adapter import AuditedGeminiPipeline,AuditedGeminiService
MODEL='gemini-3.8-live-extended-thinking'
async def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();load_dotenv();os.environ['MTE_ENABLE_RECOVERY']='0';b=BenchmarkConfig();out=create_run_directory(b.name,MODEL);setup_logging(out)
 meta=dict(label=a.label,model=MODEL,thinking='medium',benchmark=b.name,prompt_sha256=hashlib.sha256(b.system_instruction.encode()).hexdigest(),judge_prompt_sha256=hashlib.sha256(b.judge_system_prompt.encode()).hexdigest(),method='gemini-idle-exchanges-v1',recovery=False,versions={p:importlib.metadata.version(p) for p in ['google-genai','pipecat-ai']},source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'adapter.py',HERE/'capture.py',ROOT/'src/multi_turn_eval/pipelines/realtime.py']})
 (out/'benchmark-provenance.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'system-prompt.txt').write_text(b.system_instruction);(out/'judge-prompt.txt').write_text(b.judge_system_prompt);(out/'exchange-adapter-snapshot.py').write_bytes((HERE/'adapter.py').read_bytes())
 print('Output directory:',out,flush=True);recorder=TranscriptRecorder(out,MODEL);pipeline=AuditedGeminiPipeline(b,out/'gemini-events.jsonl',require_interaction_status=True,explicit_audio_activity=True,allow_turn_replay=False)
 try:await asyncio.wait_for(pipeline.run(recorder=recorder,model=MODEL,service_class=AuditedGeminiService,service_name='gemini-live',thinking='medium'),900)
 finally:recorder.close();pipeline.event_file.close()
if __name__=='__main__':asyncio.run(main())
