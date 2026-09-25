import asyncio,importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from google.genai import types
from multi_turn_eval.pipelines.realtime import GeminiLiveLLMServiceWithReconnection
P=Path(__file__).resolve().parents[1]/'docs/gemini38-extended-exchanges-2026-09-16/adapter.py'
spec=importlib.util.spec_from_file_location('gemini_exchange_adapter',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_extended_tools_are_explicitly_nonblocking(monkeypatch):
 async def run():
  handler=AsyncMock();monkeypatch.setattr(GeminiLiveLLMServiceWithReconnection,'_connection_task_handler',handler)
  s=m.AuditedGeminiService.__new__(m.AuditedGeminiService);s.audit=lambda *a,**k:None
  config=types.LiveConnectConfig(tools=[types.Tool(function_declarations=[types.FunctionDeclaration(name='lookup',description='Lookup')])])
  await s._connection_task_handler(config)
  assert str(config.tools[0].function_declarations[0].behavior)=='NON_BLOCKING'
  assert handler.await_count==1
 asyncio.run(run())

def test_status_only_idle_is_processed_and_raw_messages_preserved(monkeypatch):
 async def run():
  monkeypatch.setattr(GeminiLiveLLMServiceWithReconnection,'_handle_session_ready',AsyncMock())
  s=m.AuditedGeminiService.__new__(m.AuditedGeminiService);events=[];s.audit=lambda kind,**data:events.append((kind,data));s._handle_msg_turn_complete=AsyncMock()
  first=types.LiveServerMessage(server_content=types.LiveServerContent(interaction_status='IN_PROGRESS',turn_complete=True))
  final=types.LiveServerMessage(server_content=types.LiveServerContent(interaction_status='IDLE'))
  async def receive():
   yield first
   yield final
  session=SimpleNamespace(receive=receive)
  await s._handle_session_ready(session)
  assert [v async for v in session.receive()]==[first,final]
  s._handle_msg_turn_complete.assert_awaited_once_with(final)
  assert len(events)==2
 asyncio.run(run())

def test_reconstruction_keeps_interim_and_final_speech_with_rounded_input_time(tmp_path):
 import hashlib,json
 spec=importlib.util.spec_from_file_location('gemini_reconstruct',P.with_name('reconstruct.py'));r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 (tmp_path/'benchmark-provenance.json').write_text(json.dumps(dict(prompt_sha256=hashlib.sha256(b'standard').hexdigest())))
 (tmp_path/'run.log').write_text('[USER_AUDIO_QUEUED] turn=0 predicted_end=2.000 duration=1.000\n[USER_AUDIO_QUEUED] turn=1 predicted_end=10.000 duration=1.000\n')
 row=dict(turn=0,assistant_text='Checking. The answer.',tool_calls=[])
 (tmp_path/'transcript.jsonl').write_text(json.dumps(row)+'\n')
 ev=[dict(monotonic=0,turn=0,type='setup',config=dict(system_instruction='standard',tools=[],thinking_config={'thinking_level':'MEDIUM'})),dict(monotonic=3,turn=0,type='server',message=dict(server_content=dict(output_transcription=dict(text='Checking. ')))),dict(monotonic=4,turn=0,type='phase_boundary',status='IN_PROGRESS',input_open=False),dict(monotonic=6,turn=0,type='server',message=dict(server_content=dict(output_transcription=dict(text='The answer.')))),dict(monotonic=8,turn=0,type='phase_boundary',status='IDLE',input_open=False),dict(monotonic=8.01,turn=0,type='interaction_idle'),dict(monotonic=9.0001,turn=0,type='exchange_commit',text='Checking. The answer.')]
 (tmp_path/'gemini-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in ev))
 assert r.reconstruct(tmp_path)
 assert json.loads((tmp_path/'reconstructed/transcript.jsonl').read_text())==row
 ev[4]['status']='IN_PROGRESS'
 (tmp_path/'gemini-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in ev))
 assert not r.reconstruct(tmp_path)

def test_end_session_returns_tool_result_without_canceling_spoken_farewell():
 async def run():
  p=m.AuditedGeminiPipeline.__new__(m.AuditedGeminiPipeline)
  p._enable_tool_call_dedupe=True;p._seen_tool_calls=set();p._duplicate_tool_call_ids=set();p._end_session_requested=False;p.done=False
  p.audit=lambda *a,**k:None;p._emit_tool_result_callback=AsyncMock();p.task=SimpleNamespace(cancel=AsyncMock())
  params=SimpleNamespace(function_name='end_session',arguments={},tool_call_id='call_1')
  await p._function_catchall(params)
  assert p._end_session_requested and not p.done
  p._emit_tool_result_callback.assert_awaited_once_with(params,{'status':'success'})
  p.task.cancel.assert_not_awaited()
  await p._function_catchall(params)
  assert 'call_1' in p._duplicate_tool_call_ids
  p.task.cancel.assert_not_awaited()
 asyncio.run(run())
