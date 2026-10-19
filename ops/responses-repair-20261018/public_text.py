"""Recover only hash-verified public output omitted by the cancellation recorder."""
import asyncio,importlib,json,hashlib
from copy import deepcopy
from openai import AsyncOpenAI
from common import HERE,ROOT,RAW,NAMES,cfg,collect,dump,sha
from wire_audit import digest,summary

MANIFEST=HERE/'PUBLIC-TEXT-RESTORATIONS.json'
def verify_output_items(response,event):
 actual=[summary(i) for i in response['output']];expected=event['output_items']
 assert len(actual)==len(expected),('Provider output count changed',event['id'])
 omitted=[]
 for item,captured in zip(actual,expected):
  if item==captured:continue
  # Persisted-response retrieval omits encrypted reasoning and rejects its
  # include parameter. Require exact public/native tool items; never claim the
  # unavailable private item's full hash was verified or use it as evidence.
  assert item['type']==captured['type']=='reasoning'
  assert captured['has_encrypted_reasoning'] and not item['has_encrypted_reasoning']
  keys=set(item)|set(captured)
  assert all(item.get(k)==captured.get(k) for k in keys-{'sha256','has_encrypted_reasoning'})
  omitted.append(item['id'])
 return {'native_output_items_verified':not omitted,'native_public_output_items_verified':True,
         'unretrieved_encrypted_reasoning_item_ids':omitted}
def candidates(final=True):
 found=[]
 for name in NAMES:
  c=cfg(name);path=HERE/name/'COLLECTED.json'
  if final:
   rows=json.loads(path.read_text())['cases']
  else:
   manifest=HERE/name/'artifacts/canonical.tsv';rows=collect.read_tsv(manifest) if manifest.exists() else []
  for row in rows:
   run=ROOT/row['run_dir'];records=collect.read_jsonl(run/'transcript.jsonl')
   endings=[r for r in records if not r.get('recovery_turn') and any(t.get('name')=='end_session' for t in r.get('tool_calls',[]))]
   audit=collect.read_jsonl(run/'responses-history-audit.jsonl')
   output=[r for r in audit if r['event']=='completed' and any(i.get('name')=='end_session' for i in r['output_items'])]
   if not endings or not output:continue
   record=endings[-1];event=output[-1]
   if record.get('assistant_text','').strip() or not any(i.get('role')=='assistant' and i.get('content_characters',0)>0 for i in event['output_items']):continue
   found.append((name,row,run,record,event))
 return found
async def prepare(final=True):
 items=candidates(final);semaphore=asyncio.Semaphore(4)
 async with AsyncOpenAI(api_key=collect.extract_credential(cfg(NAMES[0]))) as client:
  async def restore(candidate):
   name,row,run,record,event=candidate
   async with semaphore:
    cache=RAW/'provider-output-retrievals'/(event['id']+'.json')
    if not cache.exists():
     response=await client.responses.retrieve(event['id'])
     dump(cache,response.model_dump(mode='json',exclude_none=True))
    response=json.loads(cache.read_text());assert response['id']==event['id']
    assert response['model'] in cfg(name)['accepted_response_model_ids']
    verification=verify_output_items(response,event)
    text=''.join(part['text'] for item in response['output'] if item['type']=='message' for part in item.get('content',[]) if part['type']=='output_text')
    assert text.strip()
    return {'name':name,'slot':int(row['slot']),'turn':record['turn'],'source':str((run/'transcript.jsonl').relative_to(ROOT)),
            'source_sha256':sha(run/'transcript.jsonl'),'record_sha256':digest(record),'source_text_sha256':hashlib.sha256(record['assistant_text'].encode()).hexdigest(),
            'provider_response_id':event['id'],'provider_response':str(cache.relative_to(ROOT)),'provider_response_sha256':sha(cache),
            'public_text':text,'public_text_sha256':hashlib.sha256(text.encode()).hexdigest(),**verification}
  entries=await asyncio.gather(*(restore(item) for item in items))
 if final:
  data={'schema_version':1,'amendment_sha256':sha(HERE/'PUBLIC-TEXT-AMENDMENT.md'),'entries':entries}
  if MANIFEST.exists():assert json.loads(MANIFEST.read_text())==data
  else:dump(MANIFEST,data)
 print('Verified public output retrievals:',len(entries),'final' if final else 'prefetch only',flush=True)
 return entries

def install(base):
 """Normalize a derived judge view; raw transcript snapshots remain identical."""
 judge_module=importlib.import_module('multi_turn_eval.judging.claude_judge')
 if getattr(judge_module,'_public_text_manifest_sha256',None):
  assert judge_module._public_text_manifest_sha256==sha(MANIFEST);base.prepare_judge_records=judge_module.prepare_judge_records;return
 data=json.loads(MANIFEST.read_text());assert data['amendment_sha256']==sha(HERE/'PUBLIC-TEXT-AMENDMENT.md')
 entries={entry['record_sha256']:entry for entry in data['entries']};assert len(entries)==len(data['entries'])
 for entry in entries.values():
  assert sha(ROOT/entry['source'])==entry['source_sha256']
  assert sha(ROOT/entry['provider_response'])==entry['provider_response_sha256']
  assert hashlib.sha256(entry['public_text'].encode()).hexdigest()==entry['public_text_sha256'] and entry['native_public_output_items_verified']
 original=judge_module.prepare_judge_records
 def normalize(records):
  changed=[];restored=[]
  for record in records:
   entry=entries.get(digest(record))
   if entry:
    assert not record['assistant_text'].strip() and not record.get('recovery_turn')
    restored.append((record,entry));record={**record,'assistant_text':entry['public_text']}
   changed.append(record)
  prepared,changes=original(changed)
  for record,entry in restored:
   result=next(r for r in prepared if r['turn']==record['turn'] and not r.get('recovery_turn'))
   changes.append({'turn':record['turn'],'source_text_sha256':__import__('hashlib').sha256(record['assistant_text'].encode()).hexdigest(),
                   'judge_text_sha256':__import__('hashlib').sha256(result['assistant_text'].encode()).hexdigest(),
                   'restored_public_characters':len(entry['public_text']),'provider_response_id':entry['provider_response_id'],
                   'provider_response_sha256':entry['provider_response_sha256'],'reason':'verified_public_output_restore'})
  return prepared,changes
 judge_module.prepare_judge_records=normalize;base.prepare_judge_records=normalize
 judge_module.REASONING_NORMALIZATION='verified-public-output-and-think-sections-v1'
 judge_module._public_text_manifest_sha256=sha(MANIFEST)
if __name__=='__main__':asyncio.run(prepare(final=__import__('sys').argv[1:]!=['--prefetch']))
