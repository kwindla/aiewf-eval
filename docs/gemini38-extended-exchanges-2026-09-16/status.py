import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def rows(p):
 if not p.exists():return []
 out=[]
 for l in p.read_text().splitlines():
  try:out.append(json.loads(l))
  except json.JSONDecodeError:pass
 return out
m=re.search(r'^Output directory: (.+)$',(HERE/'smoke.log').read_text(),re.M)
if m:
 r=ROOT/m[1];ev=rows(r/'gemini-events.jsonl');print('Smoke',r.name,'commits',sum(e['type']=='exchange_commit' for e in ev),'IN_PROGRESS phases',sum(e['type']=='phase_boundary' and e.get('status')=='IN_PROGRESS' for e in ev))
 p=r/'exchange-validation/complete.json'
 if p.exists():print(p.read_text())
cohort=HERE/'cohort';records=rows(cohort/'attempts.jsonl');print('Cohort started',len(rows(cohort/'started.jsonl')),'captured',len(records),'full',sum(r['full'] for r in records),'processed',len(rows(cohort/'processed.jsonl')))
if (HERE/'cohort-held.json').exists():print('HELD:',(HERE/'cohort-held.json').read_text())
