"""Detect commits before the intended post-input quiet interval; no score changes."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
records=[json.loads(l) for l in (HERE/'capture/attempts.jsonl').read_text().splitlines() if l]
checked=[];violations=[]
for r in sorted([r for r in records if r['arm']=='live'],key=lambda r:r['index']):
 if not r.get('run_dir'):continue
 run=ROOT/r['run_dir'];events=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()];ends={};turns=[]
 for e in events:
  if e['type']=='input_end':ends[e['turn']]=e['t']
  if e['type']=='turn_complete' and e['turn'] in ends:
   gap=e['t']-ends[e['turn']]
   row=dict(attempt=r['index'],turn=e['turn'],input_end_to_commit_ms=gap*1000,assistant=e.get('transcript'),run_dir=r['run_dir'])
   turns.append(row)
   if gap<2.95:violations.append(row)
 checked.append(dict(attempt=r['index'],turns=len(turns),minimum_gap_ms=min((t['input_end_to_commit_ms'] for t in turns),default=None)))
result=dict(checked=checked,violations=violations,criterion='Commit less than 2.95 seconds after input_end violates the intended 3-second post-input quiet interval (50ms tolerance). This is a harness validity audit, not a model/API failure.')
(HERE/'live-gate-fix/audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
