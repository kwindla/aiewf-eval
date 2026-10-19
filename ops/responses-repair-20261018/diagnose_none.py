"""Bounded, excluded native-manual versus cloud replay at the repaired none failure."""
import asyncio,json
from openai import AsyncOpenAI
from common import HERE,ROOT,cfg,collect,dump
async def main():
 folder=HERE/'excluded-none-diagnostic';folder.mkdir(exist_ok=True)
 c=cfg('luna-none');run=ROOT/c['paths']['run_output_root']/'excluded-smoke'
 audit=[json.loads(line) for line in (run/'responses-history-audit.jsonl').read_text().splitlines()]
 ids=[row['id'] for row in audit if row['event']=='completed']
 async with AsyncOpenAI(api_key=collect.extract_credential(c)) as client:
  responses=[];inputs=[]
  for index,rid in enumerate(ids):
   path=folder/f'retrieved-{index+1:02d}.json'
   if path.exists():data=json.loads(path.read_text())
   else:
    response=await client.responses.retrieve(rid)
    items=await client.responses.input_items.list(rid,order='asc',limit=100)
    assert not items.has_more
    data={'response':response.model_dump(mode='json'),'input':[i.model_dump(mode='json',exclude_none=True) for i in items.data]};dump(path,data)
   responses.append(data['response']);inputs.append(data['input'])
  # input_items may include previous linked inputs, so compare each request's
  # actual structural audit to determine the proper suffix below.
  for i,(response,inp) in enumerate(zip(responses,inputs),1):
   print(i,'inputs',len(inp),'instructions',response.get('instructions'),'output',[(o['type'],o.get('phase'),o.get('name'),[p.get('text') for p in o.get('content',[])]) for o in response['output']],flush=True)
  history=[]
  for i in range(6):
   # External request inputs are system+user initially, then one user each.
   expected=2 if i==0 else 1
   external=[item for item in inputs[i] if item.get('role') in ['system','developer','user']]
   assert len(external)>=expected
   history.extend(external[-expected:]);history.extend(responses[i]['output'])
  final_input=[i for i in inputs[6] if i.get('role')=='user'][-1:]
  params={'model':c['model'],'reasoning':{'effort':'none'},'service_tier':'priority','store':True,'tools':responses[0]['tools']}
  for mode in ['cloud','native']:
   for repeat in range(1,4):
    path=folder/f'{mode}-{repeat}.json'
    if path.exists():continue
    request={**params,'input':final_input if mode=='cloud' else history+final_input}
    if mode=='cloud':request['previous_response_id']=ids[5]
    response=await client.responses.create(**request)
    dump(path,{'request':request,'response':response.model_dump(mode='json')})
    print(mode,repeat,response.output_text,[(i.type,getattr(i,'name',None)) for i in response.output],flush=True)
if __name__=='__main__':asyncio.run(main())
