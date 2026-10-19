"""Publish concise progress in both worktrees while frozen voice capture runs."""
import json,time
from pathlib import Path
from common import HERE,ROOT,NAMES,cfg,collect
PUBLIC=Path('/home/khkramer/src/aiewf-eval-wt/gemini38-flash/ops/responses-repair-20261018')
PUBLIC.mkdir(parents=True,exist_ok=True)
while True:
 rows=[];total=0;completed=0
 for name in NAMES:
  c=cfg(name);p=HERE/name/'artifacts/canonical.tsv';records=collect.read_tsv(p) if p.exists() else []
  good=sum(r.get('classification')=='complete_30' for r in records)
  active=[]
  attempts=HERE/name/'artifacts/attempts.tsv';finished={r['run_dir'] for r in collect.read_tsv(attempts)} if attempts.exists() else set()
  for folder in sorted((ROOT/c['paths']['run_output_root']).glob('*slot*')):
   if str(folder.relative_to(ROOT)) in finished:continue
   transcript=folder/'transcript.jsonl';turns=sum(not json.loads(line).get('recovery_turn') for line in transcript.read_text().splitlines() if line.strip()) if transcript.exists() else 0
   active.append(f'{turns}/30')
  total+=len(records);completed+=good
  rows.append(f"| {name} | {len(records)}/30 | {good} | {', '.join(active) or '—'} |")
 grades=len(list((HERE/'judging/results').glob('*.json')))
 source='# Responses adapter refresh progress\n\nUpdated '+collect.utc_now()+f'\n\nAssigned captures: **{total}/360**; complete conversations so far: **{completed}**. Opus 5.5 Medium grades: **{grades}/360**.\n\n'
 source+='The repaired runtime runs in an isolated worktree while Grok high uses its frozen runtime. Fixed assignments retain early endings; incomplete conversations are not topped up.\n\n| Configuration | Assigned captured | Complete | Active recorded turns |\n|---|---:|---:|---|\n'+'\n'.join(rows)+'\n'
 for dest in [HERE/'STATUS.md',PUBLIC/'STATUS.md']:
  temp=dest.with_suffix('.tmp');temp.write_text(source);temp.replace(dest)
 if (HERE/'aggregate.json').exists():break
 time.sleep(30)
