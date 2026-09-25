"""Summarize functional smoke evidence, without assigning benchmark scores."""
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);rows=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
turns=[];tools=[];errors=[];closed=[];backend=[]
for row in rows:
 e=row.get('event',{})
 if row['type']=='input_start':turns.append(dict(turn=row['turn'],user='',assistant=''))
 if turns and e.get('type')=='session.input_transcript.delta':turns[-1]['user']+=e['delta']
 if turns and e.get('type')=='session.output_transcript.delta':turns[-1]['assistant']+=e['delta']
 if row['type']=='tool':tools.append(row)
 if row['type']=='pipeline_error' or e.get('type')=='error':errors.append(row)
 if e.get('type')=='session.closed':closed.append(e)
 if e.get('type')=='response.event' and e.get('event',{}).get('type')=='response.completed':
  response=e['event']['response'];backend.append({k:response.get(k) for k in ['model','reasoning','status','usage']})
summary=dict(turns=turns,tools=tools,errors=errors,closed=closed,backend=backend)
(out/'review.json').write_text(json.dumps(summary,indent=2))
for t in turns:print(t)
print('tools',tools,'errors',errors,'closed',bool(closed))
assert len(turns)==4 and all(t['assistant'].strip() for t in turns)
assert len(tools)==1 and tools[0]['name']=='submit_session_suggestion'
assert tools[0]['arguments']['name']=='Jennifer Smith'
assert 'trac' in tools[0]['arguments']['suggestion_text'].lower()
assert not errors and closed
assert all(b['status']=='completed' for b in backend)
