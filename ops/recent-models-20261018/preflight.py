"""Live direct probes and excluded full-harness smoke conversations."""
import asyncio, importlib.util, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import websockets
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('portable',ROOT/'ops/aiewf-campaign-template/collect.py');collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect)

def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,default=str)+'\n')
def cfg(name):return collect.validate_configuration(HERE/name/'configuration.json',require_serving_verified=False)

async def direct(name,arm):
 c=cfg(name);key=collect.extract_credential(c);env={**c['common_environment'],**c['arms'][arm]['environment']};path=HERE/name/'smoke'/f'direct-{arm}.json'
 if path.exists():return
 tool={'name':'probe','description':'Call this tool with the number 323.','parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}}
 prompt='Call the probe tool with value 323. After its result, say OK.'
 request={};start=time.monotonic()
 if name=='grok-voice':
  url=c['endpoint']['url']+'?model='+c['model']
  async with websockets.connect(url,additional_headers={'Authorization':'Bearer '+key},open_timeout=30) as ws:
   # Read initial session, then configure explicit reasoning and a manual text turn.
   request={'type':'session.update','session':{'instructions':prompt,'reasoning':{'effort':'none'},'turn_detection':None,'tools':[{'type':'function','function':tool}]}}
   await ws.send(json.dumps(request));events=[];echo=None;outputs=[]
   async with asyncio.timeout(120):
    async for raw in ws:
     e=json.loads(raw);events.append(e['type'])
     if e['type']=='error':raise RuntimeError(str(e))
     if e['type']=='session.updated':echo=e['session'];break
    assert echo.get('reasoning',{}).get('effort')=='none',echo
    await ws.send(json.dumps({'type':'conversation.item.create','item':{'type':'message','role':'user','content':[{'type':'input_text','text':prompt}]}}))
    await ws.send(json.dumps({'type':'response.create'}))
    async for raw in ws:
     e=json.loads(raw);events.append(e['type'])
     if e['type']=='error':raise RuntimeError(str(e))
     if e['type']=='response.done':outputs=e['response'].get('output',[]);break
   assert any(i.get('type')=='function_call' and i.get('name')=='probe' for i in outputs),outputs
   dump(path,{'request':request,'session_model':echo.get('model'),'session_reasoning':echo.get('reasoning'),'events':events,'output':outputs,'seconds':time.monotonic()-start})
  return
 if name in {'sonnet','haiku'}:
  client=AsyncAnthropic(api_key=key,timeout=180,max_retries=0)
  thinking={'type':env['MTE_ANTHROPIC_THINKING']}
  if thinking['type']=='adaptive':thinking['display']='omitted'
  anthropic_tool={'name':tool['name'],'description':tool['description'],'input_schema':tool['parameters']}
  request={'model':c['model'],'max_tokens':16384,'thinking':thinking,'output_config':{'effort':'low'},'tools':[anthropic_tool],'messages':[{'role':'user','content':prompt}]}
  r=await client.messages.create(**request);calls=[b for b in r.content if b.type=='tool_use'];assert calls and calls[0].name=='probe',r
  if thinking['type']=='disabled':assert not any(b.type=='thinking' for b in r.content)
  request2={**request,'messages':[ *request['messages'],{'role':'assistant','content':r.content},{'role':'user','content':[{'type':'tool_result','tool_use_id':calls[0].id,'content':'Success'}]}]}
  r2=await client.messages.create(**request2)
  assert any(b.type=='text' and 'OK' in b.text.upper() for b in r2.content),r2
  data={'request':request,'response':r.model_dump(mode='json'),'continuation':r2.model_dump(mode='json')};await client.close()
 elif name=='luna':
  client=AsyncOpenAI(api_key=key,timeout=180,max_retries=0)
  request={'model':c['model'],'input':[{'role':'user','content':prompt}],'reasoning':{'effort':arm},'service_tier':'priority','tools':[{'type':'function',**tool}]}
  r=await client.responses.create(**request);calls=[i for i in r.output if i.type=='function_call'];assert calls and calls[0].name=='probe',r
  r2=await client.responses.create(**{**request,'input':[ *request['input'],*[i.model_dump(exclude_none=True) for i in r.output if i.type=='function_call'],{'type':'function_call_output','call_id':calls[0].call_id,'output':'Success'}]})
  assert 'OK' in r2.output_text.upper(),r2
  data={'request':request,'response':r.model_dump(mode='json'),'continuation':r2.model_dump(mode='json')};await client.close()
 else:
  client=AsyncOpenAI(api_key=key,base_url=c['endpoint']['url'],timeout=180,max_retries=0)
  extra={'reasoning':{'effort':arm},'provider':{'only':['Meta'],'allow_fallbacks':False,'require_parameters':True}} if name=='spark13' else {'reasoning_effort':'low'}
  request={'model':c['model'],'messages':[{'role':'user','content':prompt}],'tools':[{'type':'function','function':tool}],'max_tokens':16384,'extra_body':extra}
  if name=='spark-open':request.update(temperature=1.0,top_p=1.0)
  r=await client.chat.completions.create(**request);calls=r.choices[0].message.tool_calls;assert calls and calls[0].function.name=='probe',r
  message=r.choices[0].message.model_dump(exclude_none=True)
  # Replay only the assistant's public text and actual calls; hosted private reasoning is not available.
  message={k:v for k,v in message.items() if k in {'role','content','tool_calls'}}
  await asyncio.sleep(3 if name=='spark-open' else 0)
  r2=await client.chat.completions.create(**{**request,'messages':[*request['messages'],message,{'role':'tool','tool_call_id':calls[0].id,'content':'Success'}]})
  assert 'OK' in (r2.choices[0].message.content or '').upper(),r2
  data={'request':request,'response':r.model_dump(mode='json'),'continuation':r2.model_dump(mode='json')};await client.close()
 data['seconds']=time.monotonic()-start;dump(path,data)
 # Preserve versioned response IDs separately from the requested ID.
 models={data[k]['model'] for k in ['response','continuation']}
 raw=json.loads(c['_config_path'].read_text());raw['accepted_response_model_ids']=sorted(set(raw['accepted_response_model_ids'])|models);dump(c['_config_path'],raw)
 print(name,arm,'direct verified',sorted(models),flush=True)

async def harness(name,arm):
 c=cfg(name);record=HERE/name/'smoke'/f'harness-{arm}.json'
 if record.exists():return
 run=ROOT/c['paths']['run_output_root']/('smoke-'+arm)
 env=collect.child_environment(c,arm,collect.extract_credential(c))
 if not run.exists():
  log=HERE/name/'smoke'/f'harness-{arm}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:
   p=await asyncio.create_subprocess_exec(sys.executable,str(collect.RUN_ONE),'--config',str(c['_config_path']),'--arm',arm,'--run-dir',str(run),cwd=ROOT,env=env,stdout=f,stderr=asyncio.subprocess.STDOUT)
   rc=await asyncio.wait_for(p.wait(),1800)
  assert rc==0,(name,arm,rc)
 stats=collect.validate_transcript(c,run);collect.validate_run_provenance(c,arm,run)
 rows=collect.read_jsonl(run/'transcript.jsonl');scheduled=[r for r in rows if not r.get('recovery_turn')]
 assert stats['turns']==stats['response_turns'],(name,arm,stats)
 assert any(t.get('name')=='end_session' for r in rows for t in r.get('tool_calls',[])),(name,arm,'missing end_session')
 assert all('</think>' not in r.get('assistant_text','') for r in rows),(name,arm,'reasoning leaked')
 if name!='grok-voice':
  assert all(isinstance(r.get('ttfb_ms'),(int,float)) and r['ttfb_ms']>0 for r in scheduled),(name,arm,'missing TTFAT')
 else:assert (run/'conversation.wav').exists()
 dump(record,{'run_dir':str(run.relative_to(ROOT)),**stats,'verified':True,
              'note':'Early end_session is model behavior and is retained; no quality-based smoke retry.'})
 print(name,arm,'full harness verified',flush=True)

async def lane(names,phase):
 for name in names:
  for arm in cfg(name)['arms']:
   await (direct if phase=='direct' else harness)(name,arm)
  if phase=='harness':
   c=cfg(name);raw=json.loads(c['_config_path'].read_text());raw['serving']={'verified':True,'notes':'Direct tool round-trip and excluded full-task harness smoke checked every registered mode; any early end_session retained as model behavior. See smoke artifacts for observed completion.'};dump(c['_config_path'],raw)

async def main(phase):
 results=await asyncio.gather(*(lane(names,phase) for names in [['luna'],['sonnet','haiku'],['spark13'],['spark-open'],['grok-voice']]),return_exceptions=True)
 failures=[str(r) for r in results if isinstance(r,BaseException)]
 if failures:raise RuntimeError(failures)
if __name__=='__main__':asyncio.run(main(sys.argv[1]))
