"""Validate actual request settings and every native/cloud continuation boundary."""
import json

def validate(run,config):
 rows=[json.loads(line) for line in (run/'responses-history-audit.jsonl').read_text().splitlines() if line.strip()]
 env={**config['common_environment'],**config['arms']['standard']['environment']}
 filler='MTE_FILLER_DOTS' in env
 last=None;native=[];requests=0;completed=0
 for row in rows:
  if row['event']=='request':
   requests+=1
   assert row['model']==config['model']
   assert row['service_tier']=='priority' and row['store'] is True
   effort=env.get('MTE_OPENAI_RESPONSES_REASONING_EFFORT')
   assert row['reasoning']==({'effort':effort} if effort else None)
   assert not row['sampling'],(run,row['sampling'])
   if filler:
    assert row['previous_response_id'] is None
    hashes={i['sha256'] for i in row['input_items']}
    assert all(i['sha256'] in hashes for i in native),(run,'native output replay changed')
   else:
    assert row['previous_response_id']==last,(run,'cloud pointer mismatch')
    if last:
     assert not any(i.get('role') in ['assistant','system'] or i['type'] in ['function_call','reasoning'] for i in row['input_items'])
     expected={i['call_id'] for i in native if i['type']=='function_call' and i.get('name')!='end_session'}
     actual={i['call_id'] for i in row['input_items'] if i['type']=='function_call_output'}
     assert expected<=actual,(run,'tool result missing',expected,actual)
  else:
   completed+=1;last=row['id']
   assert row['model'] in config['accepted_response_model_ids']
   if filler:native.extend(row['output_items'])
   else:native=row['output_items']
 requested_sequences={r['sequence'] for r in rows if r['event']=='request'}
 completed_sequences={r['sequence'] for r in rows if r['event']=='completed'}
 unanswered=sorted(requested_sequences-completed_sequences)
 terminal_timeout=False
 if unanswered:
  # A final unanswered request does not corrupt any subsequent history. Retain
  # the incomplete assignment, rather than selecting away a serving stall.
  log=(run/'run.log').read_text();lines=log.splitlines()
  queued=[i for i,line in enumerate(lines) if 'queue_turn:' in line]
  terminal_timeout=(len(unanswered)==1 and unanswered[0]==max(requested_sequences)
                    and rows[-1]['event']=='request' and queued
                    and 'Idle timeout detected.' in log
                    and not any('Recorded turn ' in line for line in lines[queued[-1]+1:]))
 assert requests>0 and (requests==completed or terminal_timeout),(run,requests,completed)
 return {'requests':requests,'completed':completed,'native_or_cloud_history_verified':True,
         'terminal_unanswered_timeout':bool(terminal_timeout),'uncompleted_requests':unanswered}

