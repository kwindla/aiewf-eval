"""Publish both 25-attempt arms and audited Gemini low latency; preserve text scores."""
import copy,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD=ROOT/'docs/speech-v2-canonical-2026-09-15';REV=OLD/'rejudge-policy-compliance-1'
def main():
 new=json.loads((HERE/'aggregate.json').read_text());previous=json.loads((REV/'aggregate.json').read_text());audio=json.loads((HERE/'audio-audit/aggregate.json').read_text())
 assert len(new['models'])==2 and all(m['attempts']==25 for m in new['models'])
 assert not any(g['failed_audits'] for g in audio['groups'].values()),'Resolve audio audit failures before publication'
 models=copy.deepcopy(previous['models'])+[m for m in new['models'] if m['canonical_eligible']]
 low=next(m for m in models if m['arm']['key']=='gemini38')
 for key in ['all','non_tool','tool']:low['latency'][key]=audio['groups']['gemini38low']['latency'][key]
 def ms(v):return 'N/A' if v is None else f'{v:.0f}ms'
 lines=['| Model | Complete | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m in sorted(models,key=lambda m:-m['strict_percent']):
  a=m['arm'];name='gpt-live-1 + gpt-5.6-terra (medium)' if a['key']=='live' else a['model']+(f" ({a.get('reasoning') or a.get('thinking')})" if a.get('reasoning') or a.get('thinking') else '')
  if m['complete']<25:name+='*'
  if a['key']=='gemini38':name+='†'
  s=m['scores'];l=m['latency'];lines.append(f"| {name} | {m['complete']}/25 | {m['strict_percent']:.1f}% | {s['tool_use_correct']}/750 | {s['instruction_following']}/750 | {s['kb_grounding']}/750 | {s['turn_taking']}/750 | {ms(l['non_tool']['p50_ms'])} | {ms(l['non_tool']['max_ms'])} | {ms(l['tool']['mean_ms'])} | {ms(l['silence']['mean_ms'])} |")
 table='\n'.join(lines)+'\n';(HERE/'combined-table.md').write_text(table)
 readme=ROOT/'README.md';before=readme.read_text();prefix,section=before.split('Speech-to-speech models:',1)
 saved=json.loads((OLD/'preserved-text-inputs.json').read_text());assert hashlib.sha256(prefix.encode()).hexdigest()==saved['README_text_section']
 for p,h in saved.items():
  if p!='README_text_section':assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 section=re.sub(r'\| Model \| Complete \|.*?(?=\n(?:Judge limitation:|Pass Rate))',table,section,count=1,flags=re.S)
 section=re.sub(r'\nJudge limitation:.*?\n(?=\nPass Rate)', '\nJudge limitations and raw score/explanation disagreements are documented in the original and extension audits; scores are retained without selective favorable rejudgments.\n',section,count=1,flags=re.S)
 marker='Text-model prompts and scores above are unchanged.'
 addition='''

† **Gemini 3.8 low:** all 25 attempts have 30 transcript entries, but an input-sequence
audit found that attempt 24 recorded a startup greeting in place of the first user
question. Its frozen accuracy scores are retained with this harness caveat; this is
not a provider API error. The other 124 original attempts passed the sequence check
for their recorded input span.

The [GPT-Live and Gemini medium extension](docs/speech-v2-live-medium-2026-09-15/REPORT.md)
documents the GPT-Live five-run pilot plus 20 new attempts and 25 new Gemini medium
attempts. Gemini medium is added to the table. GPT-Live is held as a diagnostic
cohort because the playback gate can advance too early after a backchannel; the
[isolated fix and evidence](docs/speech-v2-live-medium-2026-09-15/live-gate-fix/README.md)
are available for review. The same full speech prompt and revised
Opus 5 judge apply. Gemini low and medium V2V use the
[waveform-validated input-window audit](docs/speech-v2-live-medium-2026-09-15/audio-audit/REPORT.md),
which corrects missing-marker turn association while preserving original audio and
scores. Measurement coverage, silent closing turns, and an older low-run startup
caveat are documented there. GPT-Live uses its validated input/commit-window analysis;
packet-based silence padding is not meaningful for its continuous audio stream.
'''
 section=section.replace(marker,marker+addition,1)
 section=section.replace('No explicit API or rate-limit errors were logged in this corrected cohort.', 'The original five-model cohort logged no explicit API or rate-limit errors; extension reliability is reported separately.')
 for m in new['models']:
  if m['complete']<25:
   section=section.replace('Pass Rate requires',f"* **{m['arm']['model']}:** {25-m['complete']}/25 incomplete attempts; see the extension report for retained failure details.\n\nPass Rate requires",1)
 # Replace earlier pilot-only completion text with the final 25-attempt summary.
 live=next(m for m in new['models'] if m['arm']['key']=='live');lat=live['latency']['all'];rs=live['reliability'];api=sum(bool(r['evidence']['api_error']) or bool(r.get('live_api_errors')) for r in rs)
 start=section.index('Five concurrent full GPT-Live conversations completed')
 end=section.index('\n## Quick Start',start)
 section=section[:start]+f"GPT-Live has **25 captured conversations** (five pilot plus 20 additions), with {live['complete']}/25 reaching all 30 transcript turns. A turn-gate defect can advance immediately after input ends when an early backchannel has already gone quiet. These captures are retained as **diagnostics**, not a canonical score. See the [capture report](docs/speech-v2-live-medium-2026-09-15/REPORT.md) and [isolated gate fix](docs/speech-v2-live-medium-2026-09-15/live-gate-fix/README.md). A fresh uniform cohort with the corrected gate is recommended.\n\nA separate **GPT-Live prompt-optimized** experiment is drafted in the [design document](docs/gpt-live-prompt-optimized-design-2026-09-15/DESIGN.md). Its frontend/backend prompt split is not active. Both future standard and optimized cohorts should use the same corrected gate.\n"+section[end:]
 readme.write_text(prefix+'Speech-to-speech models:'+section)
 assert hashlib.sha256(readme.read_text().split('Speech-to-speech models:',1)[0].encode()).hexdigest()==saved['README_text_section']
 print(table)
if __name__=='__main__':main()
