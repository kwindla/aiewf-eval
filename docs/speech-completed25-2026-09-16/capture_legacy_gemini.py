"""Unchanged standard configuration, with startup output excluded from turn zero."""
import argparse,asyncio,hashlib,json,os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from multi_turn_eval.cli import create_run_directory,setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from multi_turn_eval.pipelines.realtime import RealtimePipeline,GeminiLiveLLMServiceWithReconnection
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
class StartupGuard(RealtimePipeline):
 async def _on_turn_end(self,text):
  if self._first_user_end_time is None:
   logger.info('[STARTUP_GUARD] Ignoring transcript before scripted input: {}',text)
   return
  await super()._on_turn_end(text)
async def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--thinking',required=True);a=p.parse_args();load_dotenv();os.environ['MTE_ENABLE_RECOVERY']='0'
 b=BenchmarkConfig();out=create_run_directory(b.name,a.model);setup_logging(out);print('Output directory:',out,flush=True)
 (out/'topup-provenance.json').write_text(json.dumps(dict(model=a.model,thinking=a.thinking,prompt_sha256=hashlib.sha256(b.system_instruction.encode()).hexdigest(),judge_prompt_sha256=hashlib.sha256(b.judge_system_prompt.encode()).hexdigest(),startup_guard=True),indent=2)+'\n')
 (out/'startup-guard-snapshot.py').write_bytes(Path(__file__).read_bytes())
 r=TranscriptRecorder(out,a.model)
 try:await asyncio.wait_for(StartupGuard(b,allow_turn_replay=False).run(recorder=r,model=a.model,service_class=GeminiLiveLLMServiceWithReconnection,service_name='gemini-live',thinking=a.thinking),900)
 finally:r.close()
if __name__=='__main__':asyncio.run(main())
