"""Launch the authorized experiment only after pilot measurement checks pass."""
import json,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for _ in range(180):
 text=(HERE/'finish-pilot.log').read_text() if (HERE/'finish-pilot.log').exists() else ''
 if 'Pilot validation pipeline finished:' in text:break
 time.sleep(10)
else:raise RuntimeError('Pilot validation has not finished; cohort not launched')
probes=json.loads((HERE/'coordinator-probes.json').read_text())
assert len(probes)==4 and all(r['expected_ready']==r['actual_ready'] for r in probes), 'Coordinator behavior needs review'
m=re.search(r'^Output directory: (.+)$',(HERE/'pilot-full-v1.log').read_text(),re.M);run=ROOT/m[1]
result=json.loads((run/'experimental-validation/complete.json').read_text())
if not result['approved']:
 (HERE/'cohort-held.json').write_text(json.dumps(dict(reason='Pilot measurement validation needs review',details=result),indent=2)+'\n');raise SystemExit('Pilot needs review; cohort not launched')
# Complete 30 exchanges and preserve every actual spoken fragment. No quality-score threshold.
transcript=[json.loads(l) for l in (run/'transcript.jsonl').read_text().splitlines()];assert {r['turn'] for r in transcript}==set(range(30))
with (HERE/'campaign.log').open('w') as log:
 code=subprocess.call([sys.executable,str(HERE/'campaign.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
if code:raise SystemExit(f'Campaign exited {code}; inspect preserved attempts')
with (HERE/'report-cohort.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'report_cohort.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Experimental cohort and report finished.',flush=True)
