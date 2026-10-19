"""Replace only the twelve affected current text configurations."""
import json,re,shutil,subprocess,sys
from common import HERE,ROOT,dump
DIMS=('tool_use_correct','instruction_following','kb_grounding')
def pct(value):return '—' if value is None else f'{value:.1f}%'
def ms(value):return '—' if value is None else f'{value:.0f}ms'
def replace_note(source,begin,end,note):
 if begin in source:
  a=source.index(begin);b=source.index(end,a)+len(end);return source[:a]+note+source[b:]
 point=source.index('Speech-to-speech models:\n');return source[:point]+note+'\n\n'+source[point:]
def main():
 data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==360
 models=data['text_models'];assert len(models)==12
 labels={m['label'] for m in models};ranked=[m for m in models if m['complete']>0];unranked=[m for m in models if not m['complete']]
 source=(ROOT/'README.md').read_text();prior=HERE/'publication-prior'
 if not prior.exists():
  for name in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md']:
   p=prior/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
 start=source.index('| Model | Pass Rate | Any Error |');end=source.index('\n\n',start);lines=source[start:end].splitlines()
 rows=[[c.strip() for c in line.strip('|').split('|')] for line in lines[2:] if line.split('|')[1].strip() not in labels]
 for m in ranked:
  rows.append([m['label'],pct(m['strict_percent']),pct(100-m['strict_percent']),*[pct(m['dimension_error_percent'][d]) for d in DIMS],*[ms(m['latency']['positive'][k]) for k in ['p50_ms','p95_ms','max_ms']],m['provider'],str(m['complete'])])
 rows.sort(key=lambda r:(float('inf') if r[1]=='—' else -float(r[1].rstrip('%')),float('inf') if r[6]=='—' else float(r[6].removesuffix('ms')),r[0]))
 source=source[:start]+'\n'.join(lines[:2]+['| '+' | '.join(row)+' |' for row in rows])+source[end:]
 source=re.sub(r'(?:Fourteen|Thirteen|Thirteen current|\d+ ranked) rows have fewer than 30;',f'{sum(int(row[-1])<30 for row in rows)} ranked rows have fewer than 30;',source)
 # Remove obsolete zero-completion summaries from the earlier publication note.
 old_begin='<!-- recent-models-20261018:start -->';old_end='<!-- recent-models-20261018:end -->'
 if old_begin in source:
  a=source.index(old_begin);b=source.index(old_end,a)
  note=source[a:b]
  if '\nConfigurations with **zero complete conversations**' in note:
   note=note.split('\nConfigurations with **zero complete conversations**',1)[0].rstrip()+'\n'
   source=source[:a]+note+source[b:]
 source=source.replace('Fresh Ultra and the October 18 new-model cohorts are prospective;', 'Fresh Ultra, the October 18 new-model cohorts, and the Responses adapter refresh are prospective;')
 begin='<!-- luna-responses-integration-20261018:start -->';finish='<!-- luna-responses-integration-20261018:end -->'
 notice=begin+'\nThe earlier GPT-6 Luna captures used a defective Responses history adapter. Current affected OpenAI rows now use the [repaired adapter refresh](ops/responses-repair-20261018/REPORT.md); the [original investigation](ops/luna-completion-investigation-20261018/REPORT.md) remains available. The repair preserves native phases and reasoning, but early model endings can still occur.\n'+finish
 source=replace_note(source,begin,finish,notice)
 begin='<!-- responses-repair-20261018:start -->';finish='<!-- responses-repair-20261018:end -->'
 note=begin+'\nThe [Responses adapter refresh](ops/responses-repair-20261018/REPORT.md) replaces twelve affected configurations using 30 fresh assignments each, Opus 5.5 Medium judging, and recalculated latency. Ordinary conversations use cloud-linked native history; the +96 dots arm replays native outputs with earlier user text filler-free. These are independent prospective cohorts, explicitly requesting Priority; older tiers varied or were unlogged. The table remains conditional on completion. All early endings and missing future turns are retained in the report’s fixed-900 scores.\n\n'
 note+='| Configuration | Complete / assigned | Fixed-900 pass |\n|---|---:|---:|\n'+'\n'.join(f"| {m['label']} | {m['complete']}/30 | {pct(m['all_assigned']['strict_percent'])} |" for m in models)+'\n'
 if unranked:note+='\nZero-completion configurations are unranked; completed-only quality and latency are unavailable.\n'
 note+=finish
 source=replace_note(source,begin,finish,note)
 (ROOT/'README.md').write_text(source)
 text=source.split('Text mode models:\n',1)[1].split('Speech-to-speech models:\n',1)[0]
 (ROOT/'leaderboard-medium-context.md').write_text('# aiwf_medium_context text leaderboard\n\nText mode models:\n'+text.rstrip()+'\n')
 audit=ROOT/'docs/text-cohort-sizes-2026-10-17.md';body=audit.read_text()
 body=re.sub(r'> Responses adapter publication, October 18:.*\n\n','',body)
 body=re.sub(r'> (?:Earlier )?new-model publication, October 18:.*\n\n','',body,flags=re.I)
 point=body.index('\n\n')+2
 notice=f"> Responses adapter publication, October 18: the current table has {len(rows)} ranked configurations, {sum(int(r[-1]) for r in rows):,} complete conversations, and {sum(int(r[-1])<30 for r in rows)} shortfalls. Twelve affected configurations received 30 prospective assignments each; {len(unranked)} have zero completes and remain unranked. See [fresh cohort results](../ops/responses-repair-20261018/REPORT.md). The inventories below are historical.\n\n"
 audit.write_text(body[:point]+notice+body[point:])
 subprocess.run([sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py')],cwd=ROOT,check=True)
 dump(HERE/'PUBLICATION.json',{'replaced_labels':sorted(labels),'ranked_labels':[m['label'] for m in ranked],'unranked_labels':[m['label'] for m in unranked],'assigned_conversations':360,'complete_conversations':sum(m['complete'] for m in models),'current_ranked_rows':len(rows),'current_complete_conversations':sum(int(r[-1]) for r in rows),'current_shortfalls':sum(int(r[-1])<30 for r in rows),'aggregate_sha256':__import__('common').sha(HERE/'aggregate.json')})
 print('Published repaired Responses cohorts',len(ranked),'ranked;',len(unranked),'unranked',flush=True)
if __name__=='__main__':main()
