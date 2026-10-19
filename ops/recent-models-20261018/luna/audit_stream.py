"""Excluded diagnostic: compare actual streaming tool names with final provider items."""
import asyncio,json,os,sys
from contextlib import asynccontextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
from preflight import cfg,collect,dump
c=cfg('luna');os.environ.update(collect.child_environment(c,'none',collect.extract_credential(c)))
from multi_turn_eval.services.openai_responses import OpenAIResponsesLLMService
original_init=OpenAIResponsesLLMService.__init__;records=[]
class Stream:
 def __init__(self,upstream,record):self.upstream=upstream;self.record=record
 def __aiter__(self):return self
 async def __anext__(self):
  event=await self.upstream.__anext__()
  if event.type in ['response.output_item.added','response.function_call_arguments.done','response.output_item.done','response.completed']:
   self.record['events'].append(event.model_dump(mode='json'))
  return event

def init(self,*args,**kwargs):
 original_init(self,*args,**kwargs);original=self._client.responses.stream
 @asynccontextmanager
 async def wrap(**params):
  record={'request':params,'events':[]};records.append(record)
  async with original(**params) as stream:yield Stream(stream.__aiter__(),record)
 self._client.responses.stream=wrap
OpenAIResponsesLLMService.__init__=init
async def main():
 import importlib.util
 spec=importlib.util.spec_from_file_location('run_one',collect.RUN_ONE);run_one=importlib.util.module_from_spec(spec);spec.loader.exec_module(run_one)
 raw_config=json.loads(c['_config_path'].read_text());run=ROOT/c['paths']['run_output_root']/'diagnostic-stream-audit-none'
 await run_one.run_conversation(raw_config,run)
 dump(run/'provider-events.json',records)
 mismatches=[];calls=[]
 for index,record in enumerate(records):
  final=next(e['response'] for e in record['events'] if e['type']=='response.completed')
  added={e['item']['id']:e['item'] for e in record['events'] if e['type']=='response.output_item.added' and e['item']['type']=='function_call'}
  for item in final['output']:
   if item['type']!='function_call':continue
   initial=added[item['id']]
   calls.append({'request_index':index,'initial_name':initial['name'],'completed_name':item['name'],'arguments':item['arguments']})
   if initial['name']!=item['name']:mismatches.append(calls[-1])
 dump(HERE/'stream-audit-summary.json',{'excluded_run':str(run.relative_to(ROOT)),'requests':len(records),'calls':calls,'tool_name_mismatches':mismatches,'diagnostic_only':True})
 print(json.dumps({'requests':len(records),'calls':calls,'mismatches':mismatches}),flush=True)
asyncio.run(main())
