"""Write a live, credential-free campaign progress page."""
import csv,json,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
while True:
 rows=[]
 for name,target in [('luna',60),('sonnet',60),('haiku',60),('spark13',60),('spark-open',30)]:
  config=json.loads((HERE/name/'configuration.json').read_text());path=HERE/name/'artifacts/canonical.tsv'
  captures=list(csv.DictReader(path.open(),delimiter='\t')) if path.exists() else []
  complete=sum(r['classification']=='complete_30' for r in captures)
  rows.append(f'| {name} | {"verified" if config["serving"]["verified"] else "smoke running"} | {len(captures)}/{target} | {complete} |')
 config=json.loads((HERE/'grok-voice/configuration.json').read_text())
 attempts=[json.loads(p.read_text()) for p in sorted((HERE/'grok-voice/artifacts').glob('attempt-*.json'))]
 active=[]
 for run in sorted((HERE.parents[1]/'runs/recent-models-20261018/grok-voice').glob('attempt-*')):
  if not (HERE/'grok-voice/artifacts'/(run.name+'.json')).exists():
   transcript=run/'transcript.jsonl';active.append(f'{run.name}: {len(transcript.read_text().splitlines()) if transcript.exists() else 0}/30 turns')
 rows.append(f'| Grok Voice 3.0 off | {"verified" if config["serving"]["verified"] else "smoke running"} | {len(attempts)} finished, {len(active)} active | {sum(a["complete"] for a in attempts)}/25 |')
 grades=len(list((HERE/'judging/results').glob('*.json')));blocks=len(list((HERE/'judging/provider-blocked').glob('*.json')))
 text='# Recent-model campaign progress\n\nUpdated '+datetime.now(timezone.utc).isoformat()+'\n\n| Model | Smoke gate | Captured / assigned | Complete |\n|---|---|---:|---:|\n'+'\n'.join(rows)+f'\n\nOpus 5.5 Medium grades: **{grades}/295**. Provider-blocked judge cases: {blocks}.\n\n[Protocol](PROTOCOL.md) · [Results](REPORT.md)\n'
 if active:text+='\nActive voice captures: '+', '.join(active)+'.\n'
 if (HERE/'COLLECTION-ERRORS.json').exists():text+='\nCollection needs review; see COLLECTION-ERRORS.json.\n'
 if (HERE/'PUBLICATION.json').exists():text+='\nResults published to README and the standalone leaderboard.\n'
 (HERE/'STATUS.md').write_text(text)
 if (HERE/'VERIFICATION.json').exists():break
 time.sleep(30)
