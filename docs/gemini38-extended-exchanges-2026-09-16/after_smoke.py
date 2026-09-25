"""Process smoke; launch exactly 25 attempts only after all validation checks pass."""
import json,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for _ in range(110):
 m=re.search(r'^Output directory: (.+)$',(HERE/'smoke.log').read_text(),re.M)
 if m:
  run=ROOT/m[1]
  if (run/'runtime.json').exists() and (run/'conversation.wav').exists():break
 time.sleep(10)
else:raise RuntimeError('Smoke did not finalize; cohort not launched')
# Finish event stream and recorder cleanup before reading immutable capture.
time.sleep(3)
for name,args in [('smoke-processing',[HERE/'process_run.py',run]),('smoke-report',[HERE/'report.py',run])]:
 with (HERE/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
result=json.loads((run/'exchange-validation/complete.json').read_text())
if not result['approved']:
 (HERE/'cohort-held.json').write_text(json.dumps(result,indent=2)+'\n');raise SystemExit('Smoke validation failed; review required before cohort')
print('Smoke validation passed. Launching 25 fresh attempts.',flush=True)
with (HERE/'campaign.log').open('w') as f:
 subprocess.run([sys.executable,str(HERE/'campaign.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
with (HERE/'cohort-report.log').open('w') as f:
 subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
print('Cohort and report finished.',flush=True)
