"""Read-only progress and live timing audit; safe while captures are running."""
import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def rows(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:pass
 return out
records=rows(HERE/'capture/attempts.jsonl');finished={r['index'] for r in records}
print(f"Captures {len(records)}/25; full {sum(r['full'] for r in records)}; processed {len(list((HERE/'processed').glob('*/done.json')))}; campaign complete {(HERE/'complete.json').exists()}")
violations=[];errors=[]
for log in sorted((HERE/'capture').glob('live-*.log')):
 m=re.search(r'^Output directory: (.+)$',log.read_text(),re.M)
 if not m:continue
 run=ROOT/m[1];ev=rows(run/'live-events.jsonl');ends={e['turn']:e['t'] for e in ev if e['type']=='input_end'};commits=[e for e in ev if e['type']=='turn_complete']
 for e in commits:
  if e['turn'] not in ends or e['t']-ends[e['turn']]<2.95:violations.append((log.stem,e['turn']))
 errors.extend((log.stem,e) for e in ev if e['type']=='pipeline_error' or e.get('event',{}).get('type')=='error')
 if int(log.stem.split('-')[1]) not in finished:print(log.stem,'completed turns',len(commits),'input ends',len(ends))
print('Premature commits:',len(violations),'; explicit errors:',len(errors))
if violations:print(violations)
if errors:print(errors)
