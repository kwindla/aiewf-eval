"""Read-only experiment progress."""
import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def rows(p):
 if not p.exists():return []
 result=[]
 for l in p.read_text().splitlines():
  try:result.append(json.loads(l))
  except json.JSONDecodeError:pass
 return result
for name in ['pilot-routing-v1','pilot-full-v1']:
 log=HERE/(name+'.log')
 if not log.exists():continue
 m=re.search(r'^Output directory: (.+)$',log.read_text(),re.M)
 if not m:continue
 run=ROOT/m[1];ev=rows(run/'live-events.jsonl')
 print(name, 'completed exchanges',sum(e['type']=='turn_complete' for e in ev),'coordinator errors',sum(e['type']=='coordinator_error' for e in ev),'run',run.name)
 if (run/'experimental-validation/complete.json').exists():print((run/'experimental-validation/complete.json').read_text())
cohort=HERE/'cohort';attempts=rows(cohort/'attempts.jsonl');processed=rows(cohort/'processed.jsonl');started=rows(cohort/'started.jsonl')
print('Cohort: started',len(started),'captured',len(attempts),'full',sum(r['full'] for r in attempts),'processed',len(processed),'complete',(cohort/'complete.json').exists())
if (HERE/'cohort-held.json').exists():print((HERE/'cohort-held.json').read_text())
