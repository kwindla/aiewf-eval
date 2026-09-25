import asyncio,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[1]/'docs/gpt-live-optimized-2026-09-16/coordinator.py'
spec=importlib.util.spec_from_file_location('live_experiment_coordinator',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_optimized_prompts_keep_full_kb_only_in_backend_and_shared_judge():
 from benchmarks.aiwf_medium_context_speech_live_optimized.config import BenchmarkConfig as Opt
 from benchmarks.aiwf_medium_context_speech_live.config import BenchmarkConfig as Standard
 kb=(P.parents[2]/'benchmarks/aiwf_medium_context/data/knowledge_base.txt').read_text()
 assert Opt.backend_system_instruction.endswith(kb)
 assert kb not in Opt.frontend_system_instruction
 assert Opt.system_instruction==Standard.system_instruction
 assert Opt.judge_system_prompt==Standard.judge_system_prompt
 assert Opt.turns is Standard.turns and Opt.tools_schema is Standard.tools_schema

def make_coordinator(tmp_path,create):
 c=m.ExchangeCoordinator.__new__(m.ExchangeCoordinator);c.client=SimpleNamespace(responses=SimpleNamespace(create=create));c.cache={};c.errors=0
 owner=SimpleNamespace(turn=0,state=SimpleNamespace(transcript='Let me check.'),recorder=SimpleNamespace(run_dir=tmp_path),benchmark=SimpleNamespace(turns=[{'input':'When is the event?','golden_text':'DO NOT SEND GOLDEN'}]),log=lambda *a,**k:None)
 return c,owner

def test_progress_does_not_advance_and_does_not_receive_goldens(tmp_path):
 async def run():
  async def create(**kwargs):
   assert 'DO NOT SEND GOLDEN' not in json.dumps(kwargs)
   return SimpleNamespace(output_text=json.dumps(dict(ready=False,kind='progress_only',reason='Checking only')),usage=None)
  c,o=make_coordinator(tmp_path,create)
  assert not await c(o)
  assert not await c(o)
 asyncio.run(run())

def test_response_changing_during_check_cannot_advance(tmp_path):
 async def run():
  async def create(**kwargs):
   o.state.transcript+=' More speech arrived.'
   return SimpleNamespace(output_text=json.dumps(dict(ready=True,kind='answer',reason='Answer')),usage=None)
  c,o=make_coordinator(tmp_path,create)
  assert not await c(o)
 asyncio.run(run())

def test_incorrect_substantive_answer_still_can_advance(tmp_path):
 async def run():
  async def create(**kwargs):return SimpleNamespace(output_text=json.dumps(dict(ready=True,kind='answer',reason='An answer regardless of factual accuracy')),usage=None)
  c,o=make_coordinator(tmp_path,create);o.state.transcript='The event is on February 30.'
  assert await c(o)
 asyncio.run(run())

def test_reconstruction_keeps_interim_speech_and_never_substitutes_backend(tmp_path):
 spec=importlib.util.spec_from_file_location('live_experiment_reconstruct',P.with_name('reconstruct.py'));r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 row=dict(turn=0,user_text='When?',assistant_text='Checking. Wrong date.',tool_calls=[])
 (tmp_path/'transcript.jsonl').write_text(json.dumps(row)+'\n')
 events=[dict(t=1,turn=0,type='server',event=dict(type='session.output_transcript.delta',event_id='a',delta='Checking.',start_ms=100,end_ms=200)),dict(t=2,turn=0,type='server',event=dict(type='session.output_transcript.delta',event_id='b',delta=' Wrong date.',start_ms=300,end_ms=400)),dict(t=3,turn=0,type='server',event=dict(type='response.event',delegation_id='d',event=dict(type='response.output_item.done',item=dict(type='message',content=[dict(type='output_text',text='Correct backend date.')]))))]
 (tmp_path/'live-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
 assert r.reconstruct(tmp_path)
 assert json.loads((tmp_path/'reconstructed/transcript.jsonl').read_text())==row
 # A late fragment must trigger review, not silently improve the committed answer.
 events.append(dict(t=4,turn=0,type='server',event=dict(type='session.output_transcript.delta',event_id='c',delta=' Late answer.',start_ms=150,end_ms=200)))
 (tmp_path/'live-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
 assert not r.reconstruct(tmp_path)
 assert not json.loads((tmp_path/'reconstructed/mapping.json').read_text())['approved_for_judging']
