"""Excluded high-mode protocol/tool probe and full-task harness smoke."""
import asyncio,json,time
import websockets
from common import HERE,ROOT,collect,cfg,dump,base_preflight
async def direct():
 path=HERE/'grok-voice/smoke/direct-high.json'
 if path.exists():return
 c=cfg();key=collect.extract_credential(c)
 prompt='Call the probe tool with value 323. After its result, say OK.'
 tool={'name':'probe','description':'Call this tool with the number 323.','parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}}
 request={'type':'session.update','session':{'instructions':prompt,'reasoning':{'effort':'high'},'turn_detection':None,'tools':[{'type':'function','function':tool}]}}
 events=[];outputs=[];start=time.monotonic()
 async with websockets.connect(c['endpoint']['url']+'?model='+c['model'],additional_headers={'Authorization':'Bearer '+key},open_timeout=30) as ws:
  await ws.send(json.dumps(request))
  async def until(kind):
   async with asyncio.timeout(180):
    async for raw in ws:
     event=json.loads(raw);events.append(event['type'])
     if event['type']=='error':raise RuntimeError(str(event))
     if event['type']==kind:return event
   raise RuntimeError('Closed before '+kind)
  echo=(await until('session.updated'))['session']
  assert echo.get('reasoning',{}).get('effort')=='high',echo
  assert echo.get('model')==c['model'],echo
  await ws.send(json.dumps({'type':'conversation.item.create','item':{'type':'message','role':'user','content':[{'type':'input_text','text':prompt}]}}))
  await ws.send(json.dumps({'type':'response.create'}))
  response=(await until('response.done'))['response'];outputs=response.get('output',[])
  call=next(i for i in outputs if i.get('type')=='function_call' and i.get('name')=='probe')
  assert json.loads(call['arguments'])=={'value':323}
  await ws.send(json.dumps({'type':'conversation.item.create','item':{'type':'function_call_output','call_id':call['call_id'],'output':'Success'}}))
  await ws.send(json.dumps({'type':'response.create'}))
  continuation=(await until('response.done'))['response']
  assert 'OK' in json.dumps(continuation).upper(),continuation
 dump(path,{'request':request,'session_model':echo['model'],'session_reasoning':echo['reasoning'],'events':events,'output':outputs,'continuation':continuation,'seconds':time.monotonic()-start})
 print('High session echo and tool round-trip verified',flush=True)
async def main():
 await direct()
 base_preflight.HERE=HERE;base_preflight.cfg=cfg
 await base_preflight.harness('grok-voice','high')
 record=json.loads((HERE/'grok-voice/smoke/harness-high.json').read_text());assert record['turns']==record['response_turns']==30,record
 from audio import measure
 measure('smoke-high',ROOT/record['run_dir'])
 c=cfg();raw=json.loads(c['_config_path'].read_text());raw['serving']={'verified':True,'notes':'Explicit high echoed by server; direct tool round-trip and excluded full 30-turn speech-v2 smoke verified.'};dump(c['_config_path'],raw)
if __name__=='__main__':asyncio.run(main())
