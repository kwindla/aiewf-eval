"""Publish audited recent-model rows while retaining historical results."""
import json, re, shutil, subprocess, sys
from pathlib import Path
from preflight import HERE,ROOT,dump
from analyze import pct,ms
DIMS=('tool_use_correct','instruction_following','kb_grounding')

def main():
 data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete']
 # Never fabricate completed-conversation latency for a configuration with no completes.
 ranked=[m for m in data['text_models'] if m['complete']>0]
 unranked=[m for m in data['text_models'] if m['complete']==0]
 source=(ROOT/'README.md').read_text();prior=HERE/'publication-prior'
 if not prior.exists():
  prior.mkdir()
  for name in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md']:
   p=prior/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
 start=source.index('| Model | Pass Rate | Any Error |');end=source.index('\n\n',start);lines=source[start:end].splitlines()
 labels={m['label'] for m in data['text_models']}
 rows=[[c.strip() for c in line.strip('|').split('|')] for line in lines[2:] if line.split('|')[1].strip() not in labels]
 for m in ranked:
  rows.append([m['label'],pct(m['strict_percent']),pct(100-m['strict_percent']),*[pct(m['dimension_error_percent'][d]) for d in DIMS],*[ms(m['latency']['positive'][k]) for k in ('p50_ms','p95_ms','max_ms')],m['provider'],str(m['complete'])])
 rows.sort(key=lambda c:(float('inf') if c[1]=='—' else -float(c[1].rstrip('%')),float(c[6].removesuffix('ms')),c[0]))
 table='\n'.join(lines[:2]+['| '+' | '.join(r)+' |' for r in rows]);source=source[:start]+table+source[end:]
 source=source.replace('Except for the fresh prospective Ultra cohort below, we selected the most recent matching complete conversations, up to 30 per configuration.',
     'The historical refresh selected the most recent matching complete conversations, up to 30 per configuration. Fresh Ultra and the October 18 new-model cohorts are prospective; their linked reports give assignment and completion policies.')
 shortfalls=sum(int(row[-1])<30 for row in rows)
 source=re.sub(r'(?:Thirteen current|\d+ ranked) rows have fewer than 30;',f'{shortfalls} ranked rows have fewer than 30;',source)
 note_start='<!-- recent-models-20261018:start -->';note_end='<!-- recent-models-20261018:end -->'
 if note_start in source:
  a=source.index(note_start);b=source.index(note_end,a)+len(note_end);source=source[:a].rstrip()+'\n\n'+source[b:].lstrip()
 note=note_start+'\nThe [October 18 new-model campaign](ops/recent-models-20261018/REPORT.md) adds GPT-6 Luna, Sonnet/Haiku 5.5, Muse Spark 1.3, and Muse Spark Open at the approved lowest reasoning settings. Each text configuration has 30 assigned conversations. The table reports completed-conversation quality and latency; the campaign report also gives completion counts and fixed-denominator scores, retaining premature endings. Sonnet’s “no up-front thinking” uses `between_tools`, with low effort; it can still emit tool progress updates. Spark 1.3 is routed through OpenRouter pinned to Meta; Spark Open uses Baseten NVFP4. OpenAI requested Priority; direct probes reported effective Fast service.\n'
 if unranked:
  note+='\nConfigurations with **zero complete conversations** are unranked, with completed-only quality and latency unavailable:\n\n'
  note+='| Configuration | Complete / assigned | Fixed-denominator pass |\n|---|---:|---:|\n'
  note+='\n'.join('| '+m['label']+' | 0/30 | '+pct(m['all_assigned']['strict_percent'])+' |' for m in unranked)+'\n'
 note+=note_end+'\n\n'
 source=source.replace('Speech-to-speech models:\n',note+'Speech-to-speech models:\n',1)
 voice=data['voice_models'][0];den=30*voice['complete'];counts=voice['complete_only']['dimension_passes']
 row=[voice['label'],str(voice['complete']),f"{voice['failed_conversations']}/{voice['eligible_attempts']} ({voice['failure_rate_percent']:.1f}%)",pct(voice['strict_percent']),*[f'{counts[d]}/{den}' for d in DIMS],f"{voice['turn_taking_passes']}/{den}" if voice['turn_taking_scored'] else '—',ms(voice['non_tool']['median_ms']),ms(voice['non_tool']['max_ms']),ms(voice['tool']['mean_ms']),ms(voice['silence']['mean_ms'])]
 start=source.index('| Model |',source.index('Speech-to-speech models:\n'));end=source.index('\n\n',start);lines=source[start:end].splitlines();vrows=[line for line in lines[2:] if voice['label'] not in line];vrows.append('| '+' | '.join(row)+' |');vrows.sort(key=lambda line:-float(line.split('|')[4].strip().rstrip('%')))
 source=source[:start]+'\n'.join(lines[:2]+vrows)+source[end:]
 voice_note='Grok Voice 3.0 is pinned to `grok-voice-3.0` with session `reasoning.effort=none`, using the same speech-v2 tasks and Opus 5.5 Medium voice rubric. The [new-model report](ops/recent-models-20261018/REPORT.md) retains unsuccessful attempts and waveform latency diagnostics.'
 if voice_note not in source:
  position=source.index('\n\n',source.index('| Model |',source.index('Speech-to-speech models:\n')));source=source[:position]+ '\n\n'+voice_note+source[position:]
 (ROOT/'README.md').write_text(source)
 text=source.split('Text mode models:\n',1)[1].split('Speech-to-speech models:\n',1)[0]
 (ROOT/'leaderboard-medium-context.md').write_text('# aiwf_medium_context text leaderboard\n\nText mode models:\n'+text.rstrip()+'\n')
 audit=ROOT/'docs/text-cohort-sizes-2026-10-17.md';body=audit.read_text()
 body=body.replace('> Publication update, October 18: the current leaderboard has 42 text configurations,',
     '> Earlier publication update, October 18, before the new-model campaign: the leaderboard had 42 text configurations,')
 body=re.sub(r'> New-model publication, October 18:.*\n\n','',body)
 notice=f'> New-model publication, October 18: the ranked leaderboard now has {len(rows)} text configurations and {sum(int(r[-1]) for r in rows):,} complete conversations; {len(unranked)} new configurations have zero completes and are reported separately. The inventory below preserves the earlier refresh; see [new-model results](../ops/recent-models-20261018/REPORT.md).\n\n'
 first=body.index('\n\n')+2;audit.write_text(body[:first]+notice+body[first:])
 subprocess.run([sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py')],cwd=ROOT,check=True)
 dump(HERE/'PUBLICATION.json',{'ranked_labels':[m['label'] for m in ranked],'unranked_labels':[m['label'] for m in unranked],'text_rows_added':len(ranked),'complete_text_conversations_added':sum(m['complete'] for m in ranked),'voice_rows_added':1,'text_shortfalls_added':sum(m['shortfall']>0 for m in ranked)})
 print('Published',len(ranked),'ranked text rows,',len(unranked),'unranked text configurations, and Grok Voice 3.0')
if __name__=='__main__':main()
