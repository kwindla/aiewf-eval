"""Use the unchanged Opus 5.5 Medium judge and credit cooldown policy."""
import asyncio,fcntl,importlib,json,os,sys
from dotenv import dotenv_values
from common import HERE,ROOT,RAW,NAMES,cfg,collect,dump,sha
sys.path.insert(0,str(ROOT/'ops/recent-models-20261018'))
base=importlib.import_module('judge_campaign')
sys.path.pop(0)
base.HERE=HERE;base.ROOT=ROOT;base.RAW=RAW/'judging';base.NAMES=NAMES;base.cfg=cfg;base.collect=collect
from public_text import install,MANIFEST
if MANIFEST.exists():install(base)
PATHS=[HERE/x for x in ['judge.py','common.py','PROTOCOL.md','JUDGING-SCHEDULE-AMENDMENT.md','impact-inventory.json','runtime-sha256.json','collection-driver-sha256.json','audit_capture.py','repair_captures.py','OPERATIONAL-AMENDMENT.md','OPERATIONAL-REPLACEMENTS.json','public_text.py','PUBLIC-TEXT-AMENDMENT.md','PUBLIC-TEXT-RESTORATIONS.json','judging/inputs.json']]
def identity():return {str(p.relative_to(ROOT)):sha(p) for p in PATHS}
def verify():assert json.loads((HERE/'judging/driver-sha256.json').read_text())==identity()
async def grade_all(cases):
 for path in ['/home/khkramer/src/aiewf-eval/.env','/home/khkramer/src/gb-benchmarks/.env']:
  key=dotenv_values(path).get('ANTHROPIC_API_KEY')
  if key:os.environ.setdefault('ANTHROPIC_API_KEY',key)
 cooldown=base.retry.CreditCooldown();semaphore=asyncio.Semaphore(8)
 async def one(case):
  async with semaphore:
   if not base.valid(case) and not base.blocked_path(case).exists():await base.grade(case,cooldown)
 base.log('scheduler',concurrency=8,policy='continuous queue; unchanged per-case grader')
 await asyncio.gather(*(one(case) for case in cases))
 dump(HERE/'judging/COMPLETE.json',{'at':collect.utc_now(),'conversations':len(cases),'graded':sum(base.valid(c) for c in cases),'provider_blocked':sum(base.blocked_path(c).exists() for c in cases)})
async def main():
 while not (HERE/'COLLECTION-COMPLETE.json').exists():await asyncio.sleep(10)
 from repair_captures import repair
 await repair()
 from public_text import prepare
 await prepare()
 install(base)
 from audit_capture import validate
 for name in NAMES:
  c=cfg(name)
  for case in json.loads((HERE/name/'COLLECTED.json').read_text())['cases']:
   validate(ROOT/case['run_dir'],c)
 for relative,digest in json.loads((HERE/'collection-driver-sha256.json').read_text()).items():assert sha(ROOT/relative)==digest
 cases=base.freeze();path=HERE/'judging/driver-sha256.json'
 if path.exists():verify()
 else:dump(path,identity())
 await grade_all(cases);verify()
if __name__=='__main__':
 (HERE/'judging').mkdir(exist_ok=True)
 with (HERE/'judging/.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
