"""Live progress for the high-only follow-up."""
import json,time
from common import HERE,RAW,collect

def main():
 while True:
  attempts=[json.loads(p.read_text()) for p in (HERE/'grok-voice/artifacts').glob('attempt-*.json')]
  complete=sum(a['complete'] for a in attempts);active=[]
  for run in sorted(RAW.glob('attempt-*')):
   if (HERE/'grok-voice/artifacts'/(run.name+'.json')).exists():continue
   path=run/'transcript.jsonl';n=0
   if path.exists():
    for line in path.read_text().splitlines():
     try:n+=not json.loads(line).get('recovery_turn',False)
     except json.JSONDecodeError:pass
   active.append(f'{run.name}: {n}/30 turns')
  smoke=HERE/'grok-voice/smoke/harness-high.json';gate='verified' if (HERE/'runtime-sha256.json').exists() else 'running'
  grades=len(list((HERE/'judging/results').glob('*.json')));verified=(HERE/'VERIFICATION.json').exists()
  text=f'# Grok Voice 3.0 high progress\n\nUpdated {collect.utc_now()}\n\nSmoke gate: **{gate}**. Complete captures: **{complete}/25** from {len(attempts)} finished attempts.\n\nOpus 5.5 Medium grades: **{grades}/25**. Final verification: **{"passed" if verified else "pending"}**.\n\n'
  if active:text+='Active captures: '+', '.join(active)+'.\n\n'
  if gate=='running':
   path=RAW/'smoke-high/transcript.jsonl'
   if path.exists():text+=f'Excluded smoke: {len(path.read_text().splitlines())}/30 recorded turns.\n\n'
  text+='[Protocol](PROTOCOL.md) · [Results](REPORT.md)\n'
  (HERE/'STATUS.md').write_text(text)
  if verified:return
  time.sleep(30)
if __name__=='__main__':main()
