"""Replace only provider-stream-invalid assignments; preserve original manifests."""
import asyncio,json,sys
from common import HERE,ROOT,NAMES,cfg,collect,dump,environment,sha
from audit_capture import validate

def provider_failure(run):
 audit=[json.loads(line) for line in (run/'responses-history-audit.jsonl').read_text().splitlines() if line.strip()]
 requested={r['sequence'] for r in audit if r['event']=='request'}
 completed={r['sequence'] for r in audit if r['event']=='completed'}
 log=(run/'run.log').read_text()
 evidence=[line for line in log.splitlines() if ('_streaming.py:' in line and 'Error during completion:' in line) or ('Responses API error event:' in line and any(k in line for k in ['response.failed','response.incomplete','response.error']))]
 assert requested-completed and max(requested-completed)<max(requested) and evidence,('Unexplained wire failure; review rather than silently retry',run)
 return {'uncompleted_requests':sorted(requested-completed),'provider_stream_error_evidence':evidence,
         'raw_sha256':{name:sha(run/name) for name in ['transcript.jsonl','responses-history-audit.jsonl','run.log']}}

async def repair():
 ledger=HERE/'OPERATIONAL-REPLACEMENTS.json'
 state=json.loads(ledger.read_text()) if ledger.exists() else {'amendment_sha256':sha(HERE/'OPERATIONAL-AMENDMENT.md'),'replacements':[]}
 assert state['amendment_sha256']==sha(HERE/'OPERATIONAL-AMENDMENT.md')
 for name in NAMES:
  c=cfg(name);path=HERE/name/'COLLECTED.json';original=HERE/name/'COLLECTED-ORIGINAL.json'
  if not original.exists():original.write_bytes(path.read_bytes())
  old=json.loads(original.read_text());rows=[]
  for row in old['cases']:
   run=ROOT/row['run_dir']
   try:validate(run,c)
   except AssertionError:
    evidence=provider_failure(run)
    entry=next((r for r in state['replacements'] if r['name']==name and r['slot']==int(row['slot'])),None)
    if entry is None:
     entry={'name':name,'slot':int(row['slot']),'original':row,'evidence':evidence,'original_collected_sha256':sha(original),'portable_canonical_sha256':sha(HERE/name/'artifacts/canonical.tsv'),'attempts':[]}
     state['replacements'].append(entry);dump(ledger,state)
    else:assert entry['evidence']==evidence
    accepted=entry.get('accepted')
    if accepted is None:
     for number in range(int(row['attempt'])+1,4):
      candidate=ROOT/c['paths']['run_output_root']/f'operational-slot{int(row["slot"]):03d}-attempt{number:02d}'
      collect.verify_source_hashes(c)
      if not candidate.exists():
       with (HERE/name/f'operational-slot{int(row["slot"]):03d}-attempt{number:02d}.log').open('w') as out:
        proc=await asyncio.create_subprocess_exec(sys.executable,str(HERE/'run_one.py'),'--config',str(c['_config_path']),'--arm','standard','--run-dir',str(candidate),cwd=ROOT,env=environment(c),stdout=out,stderr=asyncio.subprocess.STDOUT)
        try:code=await asyncio.wait_for(proc.wait(),1800)
        except asyncio.TimeoutError:proc.terminate();await proc.wait();code=124
      else:code=None
      try:
       stats=collect.validate_transcript(c,candidate);collect.validate_run_provenance(c,'standard',candidate);wire=validate(candidate,c)
       accepted={**row,**stats,'attempt':str(number),'run_dir':str(candidate.relative_to(ROOT))}
       entry['attempts'].append({'attempt':number,'exit_code':code,'accepted':True,'run_dir':accepted['run_dir'],'wire_audit':wire})
       entry['accepted']=accepted;dump(ledger,state)
       print('Operational replacement',name,row['slot'],'attempt',number,'retained',stats['turns'],'turns',flush=True)
       break
      except Exception as exc:
       entry['attempts'].append({'attempt':number,'exit_code':code,'accepted':False,'run_dir':str(candidate.relative_to(ROOT)),'error':str(exc)});dump(ledger,state)
     assert accepted is not None,('Three operational attempts exhausted',name,row['slot'])
    validate(ROOT/accepted['run_dir'],c);rows.append(accepted)
   else:rows.append(row)
  updated={**old,'cases':rows,'operational_supersession_ledger':'ops/responses-repair-20261018/OPERATIONAL-REPLACEMENTS.json'}
  if not ledger.exists():dump(ledger,state)
  dump(path,updated)
 dump(HERE/'OPERATIONAL-REPLACEMENTS-COMPLETE.json',{'verified':True,'replaced_assignments':len(state['replacements']),'ledger_sha256':sha(ledger)})
 return state
if __name__=='__main__':asyncio.run(repair())
