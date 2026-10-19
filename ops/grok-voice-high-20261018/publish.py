"""Publish only the new high row and its mode-comparison note."""
import json
from common import HERE,ROOT,dump,verify_baseline
LABEL='Grok Voice 3.0 (reasoning high)'
def luna_notice(source):
 report=ROOT/'ops/luna-completion-investigation-20261018/REPORT.md'
 if not report.exists():return source
 begin='<!-- luna-responses-integration-20261018:start -->';end='<!-- luna-responses-integration-20261018:end -->'
 note=begin+'\n**Luna integration finding:** the GPT-6 Luna captures above are affected by a confirmed conversation-history preservation defect in our Responses adapter. Treat their completion figures and fixed-denominator scores as integration diagnostics pending corrected runs. See the [investigation and controlled replay](ops/luna-completion-investigation-20261018/REPORT.md).\n'+end
 if begin in source:
  a=source.index(begin);b=source.index(end,a)+len(end);return source[:a]+note+source[b:]
 point=source.find('Speech-to-speech models:\n')
 if point<0:return source.rstrip()+'\n\n'+note+'\n'
 return source[:point]+note+'\n\n'+source[point:]
def main():
 verify_baseline();data=json.loads((HERE/'aggregate.json').read_text());assert data['all_quality_complete'] and data['judged_conversations']==25
 m=data['voice_models'][0];assert m['label']==LABEL and m['complete']==25
 den=30*m['complete'];counts=m['complete_only']['dimension_passes']
 ms=lambda v:'—' if v is None else f'{v:.0f}ms'
 row=[LABEL,str(m['complete']),f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",f"{m['strict_percent']:.1f}%",*[f'{counts[d]}/{den}' for d in ['tool_use_correct','instruction_following','kb_grounding']],f"{m['turn_taking_passes']}/{den}",ms(m['non_tool']['median_ms']),ms(m['non_tool']['max_ms']),ms(m['tool']['mean_ms']),ms(m['silence']['mean_ms'])]
 path=ROOT/'README.md';source=path.read_text();backup=HERE/'README-before-high.md'
 if not backup.exists():backup.write_text(source)
 start=source.index('| Model |',source.index('Speech-to-speech models:\n'));end=source.index('\n\n',start);lines=source[start:end].splitlines()
 rows=[line for line in lines[2:] if line.split('|')[1].strip()!=LABEL];rows.append('| '+' | '.join(row)+' |');rows.sort(key=lambda line:-float(line.split('|')[4].strip().rstrip('%')))
 source=source[:start]+'\n'.join(lines[:2]+rows)+source[end:]
 begin='<!-- grok-voice-high-20261018:start -->';finish='<!-- grok-voice-high-20261018:end -->'
 note=begin+'\nGrok Voice 3.0 also has a fresh `reasoning.effort=high` cohort, with the first 25 complete conversations, unchanged speech-v2 tasks and Opus 5.5 Medium judging. The [high versus none comparison](ops/grok-voice-high-20261018/REPORT.md) gives completion, quality, recorded-waveform latency, and all unsuccessful attempts.\n'+finish
 if begin in source:
  a=source.index(begin);b=source.index(finish,a)+len(finish);source=source[:a]+note+source[b:]
 else:
  point=source.index('\n\n',source.index('| Model |',source.index('Speech-to-speech models:\n')));source=source[:point]+'\n\n'+note+source[point:]
 source=luna_notice(source)
 path.write_text(source)
 standalone=ROOT/'leaderboard-medium-context.md';standalone.write_text(luna_notice(standalone.read_text()))
 dump(HERE/'PUBLICATION.json',{'label':LABEL,'voice_rows_added':1,'complete_conversations':25,'judge_model':data['judge_model'],'judge_effort':data['judge_effort']})
 print('Published Grok Voice 3.0 high; all previous rows retained',flush=True)
if __name__=='__main__':main()
