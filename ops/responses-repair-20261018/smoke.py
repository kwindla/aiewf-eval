import asyncio,json,sys
from common import HERE,ROOT,cfg,collect,dump,environment
async def smoke(name):
 c=cfg(name);run=ROOT/c['paths']['run_output_root']/'excluded-validation'
 if not run.exists():
  with (HERE/name/'smoke.log').open('w') as out:
   proc=await asyncio.create_subprocess_exec(sys.executable,str(HERE/'run_one.py'),'--config',str(c['_config_path']),'--arm','standard','--run-dir',str(run),cwd=ROOT,env=environment(c),stdout=out,stderr=asyncio.subprocess.STDOUT)
   code=await asyncio.wait_for(proc.wait(),1800)
  assert code==0,(name,code)
 stats=collect.validate_transcript(c,run);collect.validate_run_provenance(c,'standard',run)
 assert stats['turns']==stats['response_turns'] and stats['turns']>0,(name,stats)
 if name=='luna-low':assert stats['turns']==30,(name,stats)
 audit=[json.loads(line) for line in (run/'responses-history-audit.jsonl').read_text().splitlines()]
 last=None;requests=0;outputs=0
 for row in audit:
  if row['event']=='request':
   assert row['previous_response_id']==last,(name,row['sequence'],'cloud pointer mismatch')
   if last:assert not any(i.get('role') in ['assistant','system'] or i['type'] in ['function_call','reasoning'] for i in row['input_items'])
   requests+=1
  else:last=row['id'];outputs+=len(row['output_items'])
 assert requests>=stats['turns'] and outputs>=stats['turns']
 dump(HERE/name/'SMOKE.json',{'verified':True,'run_dir':str(run.relative_to(ROOT)),**stats,'requests':requests,'native_output_items':outputs,'cloud_lineage_verified':True,'runtime_sha256':{str(p.relative_to(ROOT)):__import__('common').sha(p) for p in [ROOT/'src/multi_turn_eval/services/openai_responses.py',ROOT/'src/multi_turn_eval/pipelines/base.py']}})
 print(name,f"{stats['turns']}/30 turns; native/cloud history verified; early endings retained",flush=True)
async def main():
 await asyncio.gather(smoke('luna-none'),smoke('luna-low'))
 dump(HERE/'SMOKE-COMPLETE.json',{'verified':True,'modes':['none','low']})
if __name__=='__main__':asyncio.run(main())
