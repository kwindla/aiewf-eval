"""Excluded 12-call archive replay; never imports or mutates benchmark runtime."""
import asyncio, copy, hashlib, importlib.util, json, time
from pathlib import Path
from openai import AsyncOpenAI
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARCHIVE=ROOT/'runs/recent-models-20261018/luna/diagnostic-stream-audit-none/provider-events.json'
OUT=HERE/'excluded-replay';OUT.mkdir(exist_ok=True)
D=json.loads(ARCHIVE.read_text())
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,default=str)+'\n')
def completed(d):return next(e['response'] for e in d['events'] if e['type']=='response.completed')
def replay_item(item):
 # Remove only SDK parse conveniences / None fields absent from API input schema.
 if item['type']=='message':
  r={k:copy.deepcopy(v) for k,v in item.items() if k in {'id','type','role','status','phase'} and v is not None}
  r['content']=[{k:copy.deepcopy(v) for k,v in c.items() if k in {'type','text','annotations','logprobs'} and v is not None} for c in item['content']]
  return r
 if item['type']=='function_call':return {k:copy.deepcopy(v) for k,v in item.items() if k in {'id','type','name','arguments','call_id','caller','namespace','status'} and v is not None}
 raise AssertionError(item['type'])
def restored(index):
 history=copy.deepcopy(D[0]['request']['input'])
 for j in range(index):
  output=completed(D[j])['output'];history.extend(replay_item(x) for x in output)
  # Newly appended items after this original request: tool outputs and next user.
  nextinput=D[j+1]['request']['input'];prior=D[j]['request']['input']
  assert nextinput[:len(prior)]==prior
  additions=nextinput[len(prior):]
  history.extend(copy.deepcopy(x) for x in additions if x.get('role')=='user' or x.get('type')=='function_call_output')
 original=D[index]['request']['input']
 # Exactly the same user and tool-result sequence; all differences are assistant output history.
 signature=lambda items:[x for x in items if x.get('role') in {'system','user'} or x.get('type')=='function_call_output']
 assert signature(history)==signature(original)
 origcalls=[(x['call_id'],x['name'],json.loads(x['arguments'])) for x in original if x['type']=='function_call']
 newcalls=[(x['call_id'],x['name'],json.loads(x['arguments'])) for x in history if x['type']=='function_call']
 assert origcalls==newcalls
 assert not any(x['type']=='reasoning' for x in history)
 return history
async def main():
 spec=importlib.util.spec_from_file_location('credential_reader',ROOT/'ops/aiewf-campaign-template/collect.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 cfg=json.loads((ROOT/'ops/recent-models-20261018/luna/configuration.json').read_text());key=mod.extract_credential(cfg)
 manifest={'excluded':True,'purpose':'Test full output-item replay versus existing flattened history; not phase-only ablation.','archive':str(ARCHIVE.relative_to(ROOT)),'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),'calls_planned':12,'model':'gpt-6-luna','effort':'none','normalizations':'Remove SDK parsed/parsed_arguments/async_ convenience keys and None fields only; preserve all assistant message phases, boundaries, empty finals and API output fields. Original function output order restored before original tool-result order.','reasoning_items_in_source':sum(x['type']=='reasoning' for d in D for x in completed(d)['output'])}
 dump(OUT/'manifest.json',manifest)
 records=[]
 async with AsyncOpenAI(api_key=key,timeout=180,max_retries=0) as client:
  for index in (7,8):
   for repeat in range(1,4):
    for mode in ('flattened','original-items'):
     name=f'request-{index}-repeat-{repeat}-{mode}';path=OUT/(name+'.json')
     if path.exists():records.append(json.loads(path.read_text())['summary']);continue
     request=copy.deepcopy(D[index]['request'])
     if mode=='original-items':request['input']=restored(index)
     start=time.monotonic();record={'excluded':True,'request':request,'archive_request_index':index,'repeat':repeat,'history_mode':mode};events=[]
     try:
      async with client.responses.stream(**request) as stream:
       async for event in stream:events.append(event.model_dump(mode='json'))
       response=await stream.get_final_response()
      record['response']=response.model_dump(mode='json');record['provider_request_id']=getattr(response,'_request_id',None)
      summary={'name':name,'valid_response':True,'response_id':response.id,'status':response.status,'output_types':[x.type for x in response.output],'phases':[getattr(x,'phase',None) for x in response.output if x.type=='message'],'tools':[x.name for x in response.output if x.type=='function_call'],'text':response.output_text,'usage':response.usage.model_dump(mode='json') if response.usage else None,'seconds':time.monotonic()-start}
     except Exception as exc:
      summary={'name':name,'valid_response':False,'error_type':type(exc).__name__,'error':str(exc),'seconds':time.monotonic()-start}
     record['events']=events;record['summary']=summary;dump(path,record);records.append(summary);dump(OUT/'summary.json',records)
     print(json.dumps({k:v for k,v in summary.items() if k not in {'text','usage'} }),flush=True)
 dump(OUT/'COMPLETE.json',{'calls':len(records),'valid':sum(r['valid_response'] for r in records),'archive_sha256':manifest['archive_sha256']})
if __name__=='__main__':asyncio.run(main())
