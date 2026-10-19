"""Exclude simple tool probes; freeze provider model aliases before generation."""
import asyncio,json,time
from openai import AsyncOpenAI
from common import HERE,NAMES,cfg,collect,dump
async def probe(name,semaphore):
 async with semaphore:
  c=cfg(name);path=HERE/name/'DIRECT.json'
  if path.exists():return
  env={**c['common_environment'],**c['arms']['standard']['environment']}
  params={'model':c['model'],'store':True,'service_tier':env['MTE_OPENAI_SERVICE_TIER'],
          'input':[{'role':'user','content':'Call probe with value 323. After the tool result, say OK.'}],
          'tools':[{'type':'function','name':'probe','description':'Return the supplied number.','parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}}]}
  effort=env.get('MTE_OPENAI_RESPONSES_REASONING_EFFORT')
  if effort:params['reasoning']={'effort':effort}
  start=time.monotonic()
  async with AsyncOpenAI(api_key=collect.extract_credential(c),base_url=c['endpoint']['url']) as client:
   first=await client.responses.create(**params)
   calls=[i for i in first.output if i.type=='function_call'];assert calls,(name,first)
   assert all(i.name=='probe' and json.loads(i.arguments)=={'value':323} for i in calls)
   continuation={k:v for k,v in params.items() if k!='input'}
   continuation.update(previous_response_id=first.id,input=[{'type':'function_call_output','call_id':i.call_id,'output':'Success: 323'} for i in calls])
   second=await client.responses.create(**continuation)
   assert 'OK' in second.output_text.upper(),(name,second)
  dump(path,{'request':params,'response':first.model_dump(mode='json'),'continuation':second.model_dump(mode='json'),'seconds':time.monotonic()-start})
  raw=json.loads(c['_config_path'].read_text());raw['accepted_response_model_ids']=sorted({*raw['accepted_response_model_ids'],first.model,second.model});dump(c['_config_path'],raw)
  print(name,'tool continuation verified',first.model,second.service_tier,flush=True)
async def main():await asyncio.gather(*(probe(name,sem) for name in NAMES))
if __name__=='__main__':
 sem=asyncio.Semaphore(3);asyncio.run(main())
