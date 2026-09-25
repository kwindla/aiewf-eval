"""Generate audited results after capture, judging and audio validation complete."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
while not (HERE/'complete.json').exists():
 time.sleep(10)
for script in ['audit_live_gate.py','analyze.py']:
 with (HERE/(script.removesuffix('.py')+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
a=json.loads((HERE/'aggregate.json').read_text());m=a['models'][0]
result=dict(attempts=m['attempts'],complete=m['complete'],strict_percent=m['strict_percent'],canonical_eligible=m['canonical_eligible'],judge_conflict_flags=len(a['explicit_judge_conflict_flags']),report=str(HERE/'REPORT.md'),finished_unix=time.time())
(HERE/'validation-complete.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
