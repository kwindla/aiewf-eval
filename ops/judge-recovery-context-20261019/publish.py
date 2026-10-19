"""Canonical publication entry point for the recovery-context migration."""
import subprocess
import sys
from common import *

FILES=['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md']

def pct(value):
    return '—' if value is None else f'{value:.1f}%'

def replace_section(source,begin,end,body):
    if begin in source:
        start=source.index(begin)
        stop=source.index(end,start)+len(end)
        return source[:start]+body+source[stop:]
    point=source.index('Speech-to-speech models:\n')
    return source[:point]+body+'\n\n'+source[point:]

def main():
    d=read(HERE/'aggregate.json')
    assert read(HERE/'VALIDATION.json')['passed'] and d['all_quality_complete']
    assert d['inventory_sha256']==sha(HERE/'inventory.json')
    models={m['label']:m for m in d['text_models']+d['voice_models']}
    source=(ROOT/'README.md').read_text()
    start=source.index('| Model | Pass Rate | Any Error |')
    stop=source.index('\n\n',start)
    lines=source[start:stop].splitlines()
    rows=[]
    for line in lines[2:]:
        row=[c.strip() for c in line.strip('|').split('|')]
        m=models[row[0]]
        assert m['mode']=='text' and m['complete']>0 and int(row[-1])==m['complete']
        row[1:6]=[pct(m['strict_percent']),pct(100-m['strict_percent']),*[pct(m['dimension_error_percent'][dim]) for dim in DIMS]]
        rows.append(row)
    assert len(rows)==50
    rows.sort(key=lambda r:(-float(r[1].rstrip('%')),float(r[6].removesuffix('ms')),r[0]))
    source=source[:start]+'\n'.join(lines[:2]+['| '+' | '.join(row)+' |' for row in rows])+source[stop:]
    # Update current prospective fixed-900 cells too; preserve historical studies.
    updated=[]
    for line in source.splitlines():
        if line.startswith('| '):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if len(cells)==3 and cells[0] in models and '/' in cells[1]:
                m=models[cells[0]]
                assert int(cells[1].split('/')[1])==m['assigned']
                cells[2]=pct(m['all_assigned']['strict_percent'])
                line='| '+' | '.join(cells)+' |'
        updated.append(line)
    source='\n'.join(updated)+'\n'
    begin='<!-- recovery-context-20261019:start -->'
    end='<!-- recovery-context-20261019:end -->'
    note=begin+'\nThe [recovery-context migration](ops/judge-recovery-context-20261019/REPORT.md) retains all public recovery exchanges and recorded tool outcomes as unscored chronological evidence. Opus 5.5 Medium rejudged all 698 affected active assignments; 988 byte-identical requests reuse their verified prior grades. Scoring still uses only the original scripted turns, and later recovery does not erase an earlier unnecessary deferral. The 24-conversation text/voice validation used independent Astra/Fable reference labels and two diagnostic repetitions; these are model reviews, not human ground truth or a population accuracy estimate. Generation cohorts, completion counts, latency and audio evidence are unchanged. Current exact scores and confidence intervals are in the [reconciled aggregate](ops/judge-recovery-context-20261019/aggregate.json); earlier linked studies preserve historical grades.\n'+end
    source=replace_section(source,begin,end,note)
    (ROOT/'README.md').write_text(source)
    text=source.split('Text mode models:\n',1)[1].split('Speech-to-speech models:\n',1)[0]
    (ROOT/'leaderboard-medium-context.md').write_text('# aiwf_medium_context text leaderboard\n\nText mode models:\n'+text.rstrip()+'\n')
    audit=ROOT/'docs/text-cohort-sizes-2026-10-17.md'
    body=audit.read_text()
    notice='> Recovery-context migration, October 19: quality now uses the [reconciled grades](../ops/judge-recovery-context-20261019/REPORT.md). Cohort membership, completion counts and recorded latency are unchanged. Inventories below remain historical.\n\n'
    if notice not in body:
        point=body.index('\n\n')+2
        audit.write_text(body[:point]+notice+body[point:])
    subprocess.run([sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py')],cwd=ROOT,check=True)
    dump(HERE/'PUBLICATION.json',{'aggregate_sha256':sha(HERE/'aggregate.json'),'ranked_text_rows':50,'voice_rows':11,
                                'complete_conversations':1619,'canonical_entry_point':'ops/judge-recovery-context-20261019/publish.py',
                                'artifacts_sha256':{name:sha(ROOT/name) for name in FILES}})
    print('Published reconciled recovery-context grades',flush=True)

if __name__=='__main__':main()
