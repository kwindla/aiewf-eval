"""Regular Gemini 3.8: standard prompt, no thinking config, synchronous completion."""
import asyncio, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/gemini38-extended-exchanges-2026-09-16'))
from adapter import AuditedGeminiPipeline, AuditedGeminiService
from multi_turn_eval.pipelines.realtime import GeminiLiveLLMServiceWithReconnection, GeminiVADParams
from multi_turn_eval.cli import create_run_directory, setup_logging
from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from dotenv import load_dotenv
class RegularService(AuditedGeminiService):
 async def _connection_task_handler(self, config):
  assert config.thinking_config is None, 'Regular model must omit thinking config'
  self.audit('setup',config=config.model_dump(mode='json',exclude_none=True))
  await GeminiLiveLLMServiceWithReconnection._connection_task_handler(self,config)
 async def _handle_msg_turn_complete(self,message):
  await super()._handle_msg_turn_complete(message)
  if not self._explicit_input_activity_open and not self.is_interaction_in_progress():
   self.audit('synchronous_turn_complete')
class RegularPipeline(AuditedGeminiPipeline):
 def _create_llm(self,service_class,model):
  return RegularService(audit=self.audit,api_key=os.environ['GOOGLE_API_KEY'],tools=self.benchmark.tools_schema,inference_on_context_initialization=True,on_reconnecting=self._on_gemini_reconnecting,on_reconnected=self._on_gemini_reconnected,gemini_3_protocol=True,require_interaction_status=False,settings=RegularService.Settings(model=model,system_instruction=self.benchmark.system_instruction,vad=GeminiVADParams(disabled=True)))
 async def _function_catchall(self,params):
  await super()._function_catchall(params)
  if params.tool_call_id in self._duplicate_tool_call_ids:
   # ToolCallRecorder preserves duplicate evidence but drops its result frame.
   # Gemini still needs an answer for this distinct provider call ID. Return
   # the same dedupe result directly, without executing the action again.
   await self.llm._tool_result(params.tool_call_id,params.function_name,{'status':'duplicate_skipped'})
   self.audit('duplicate_result_returned',tool_call_id=params.tool_call_id,function_name=params.function_name)
 def audit(self,kind,**data):
  super().audit(kind,**data)
  if kind=='synchronous_turn_complete' and self._end_session_requested and not self.done:
   if self._closing_task is None or self._closing_task.done():self._closing_task=asyncio.create_task(self._finish_silent_closure())
 async def _on_turn_end(self,text):
  if self._first_user_end_time is None:
   self.audit('startup_ignored',text=text);return
  await super()._on_turn_end(text)
async def main():
 load_dotenv();os.environ['MTE_ENABLE_RECOVERY']='0';b=BenchmarkConfig();model='gemini-3.8-live';out=create_run_directory(b.name,model);setup_logging(out)
 print('Output directory:',out,flush=True)
 meta=dict(model=model,thinking=None,prompt_sha256=hashlib.sha256(b.system_instruction.encode()).hexdigest(),judge_prompt_sha256=hashlib.sha256(b.judge_system_prompt.encode()).hexdigest(),startup_guard=True,completion='synchronous turnComplete plus playback; deferred end_session')
 (out/'topup-provenance.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'capture-snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'adapter-snapshot.py').write_bytes((ROOT/'docs/gemini38-extended-exchanges-2026-09-16/adapter.py').read_bytes());(out/'system-prompt.txt').write_text(b.system_instruction)
 recorder=TranscriptRecorder(out,model);pipeline=RegularPipeline(b,out/'gemini-events.jsonl',require_interaction_status=False,explicit_audio_activity=True,allow_turn_replay=False)
 try:await asyncio.wait_for(pipeline.run(recorder=recorder,model=model,service_class=RegularService,service_name='gemini-live',thinking='default'),900)
 finally:recorder.close();pipeline.event_file.close()
if __name__=='__main__':asyncio.run(main())
