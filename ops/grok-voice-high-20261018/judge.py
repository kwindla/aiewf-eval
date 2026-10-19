"""Bind the unchanged Opus judge to the high-only cohort; freeze this driver too."""
import asyncio,fcntl,importlib,sys
from common import HERE,ROOT,OLD,RAW,cfg,collect,dump,sha,verify_baseline
sys.path.insert(0,str(OLD))
base=importlib.import_module('judge_campaign')
sys.path.pop(0)
base.HERE=HERE;base.RAW=RAW/'judging';base.NAMES=['grok-voice'];base.cfg=cfg;base.collect=collect
DRIVER_PATHS=[HERE/'judge.py',HERE/'common.py',HERE/'PROTOCOL.md',HERE/'baseline-sha256.json',HERE/'grok-voice/configuration.json',HERE/'grok-voice/artifacts/source-sha256.txt']
def driver_identity():
 return {str(p.relative_to(ROOT)):sha(p) for p in DRIVER_PATHS+[HERE/'judging/inputs.json']}
def verify_driver():
 assert __import__('json').loads((HERE/'judging/driver-sha256.json').read_text())==driver_identity()
 verify_baseline()
async def main():
 while not (HERE/'COLLECTION-COMPLETE.json').exists():await asyncio.sleep(10)
 verify_baseline();base.freeze();identity=driver_identity();path=HERE/'judging/driver-sha256.json'
 if path.exists():verify_driver()
 else:dump(path,identity)
 await base.main()
 verify_driver()
if __name__=='__main__':
 (HERE/'judging').mkdir(exist_ok=True)
 with (HERE/'judging/.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
