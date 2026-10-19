"""Fixed 30-assignment cohorts, four configurations at a time, preserving failures."""
import asyncio,fcntl,json,sys
from common import HERE,ROOT,NAMES,cfg,collect,dump,sha
async def lane(names):
 for name in names:
  c=cfg(name)
  with (HERE/name/'collection.log').open('a') as out:
   proc=await asyncio.create_subprocess_exec(sys.executable,str(HERE/'collect_one.py'),'--config',str(c['_config_path']),'--execute',cwd=ROOT,stdout=out,stderr=asyncio.subprocess.STDOUT)
   code=await proc.wait()
  assert code==0,(name,code)
  _,rows=collect.validate_manifests(c,collect.validate_schedule(c));collect.verify_source_hashes(c)
  dump(HERE/name/'COLLECTED.json',{'name':name,'at':collect.utc_now(),'cases':rows})
  print(name,len(rows),'assigned captures collected',flush=True)
async def main():
 assert json.loads((HERE/'SMOKE-COMPLETE.json').read_text())['verified']
 assert all((HERE/name/'DIRECT.json').exists() for name in NAMES)
 snapshot_hashes={}
 for name in NAMES:
  c=cfg(name);raw=json.loads(c['_config_path'].read_text())
  if not raw['serving']['verified']:
   raw['serving']={'verified':True,'notes':'Direct tool continuation passed. Final Luna integration smoke checked cloud lineage; low completed 30, none early endings retained as model behavior. See SMOKE.json and bounded native/cloud diagnostic.'};dump(c['_config_path'],raw)
  # Preserve exact generation sources before any canonical request.
  text=collect.source_hashes(cfg(name));path=HERE/name/'artifacts/source-sha256.txt'
  if path.exists():assert path.read_text()==text
  else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
  for line in text.splitlines():
   digest,relative=line.split(maxsplit=1);snapshot_hashes[relative]=digest;source=ROOT/relative;target=HERE/'runtime-snapshot'/relative
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():assert target.read_bytes()==source.read_bytes()
   else:target.write_bytes(source.read_bytes())
 dump(HERE/'runtime-sha256.json',snapshot_hashes)
 await asyncio.gather(*(lane(NAMES[offset::4]) for offset in range(4)))
 dump(HERE/'COLLECTION-COMPLETE.json',{'at':collect.utc_now(),'configurations':NAMES,'assigned':360})
if __name__=='__main__':
 with (HERE/'.collection.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
