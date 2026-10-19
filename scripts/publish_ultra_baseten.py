"""Publish the approved fresh thinking-on Ultra row without changing study grades."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018'
LABEL='nemotron-3-ultra (thinking on)'
RETIRED={'nemotron-3-ultra (128)','nemotron-3-ultra (96)'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=json.loads((STUDY/'aggregate.json').read_text());arm=data['arms']['on']
    verification=json.loads((STUDY/'VERIFICATION.json').read_text())
    assert verification['all_checks_passed'] and verification['all_quality_complete']
    assert verification['artifacts_sha256']['aggregate.json']==sha(STUDY/'aggregate.json')
    assert arm['complete_conversations']==arm['conversations']==30
    assert arm['observed_scripted_turns']==arm['fixed_denominator']==900
    assert arm['all_quality_complete'] and arm['reasoning_effort']=='high'
    row=[LABEL,f"{arm['strict_percent']:.1f}%",f"{100-arm['strict_percent']:.1f}%"]
    row += [f"{arm['observed_dimension_error_percent'][d]:.1f}%" for d in ['tool_use_correct','instruction_following','kb_grounding']]
    row += [f"{arm['ttfat'][k]:.0f}ms" for k in ['p50_ms','p95_ms','max_ms']]
    row += ['Baseten','30']
    original=(ROOT/'README.md').read_text()
    archive=STUDY/'publication-prior'
    if not archive.exists():
        archive.mkdir()
        for filename in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg']:
            target=archive/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/filename,target)
        shutil.copy2(ROOT/'ops/judge-refresh-opus55-20261017/VERIFICATION.json',archive/'refresh-VERIFICATION.json')
    start=original.index('| Model | Pass Rate | Any Error |')
    end=original.index('\n\n',start)
    lines=original[start:end].splitlines()
    kept=[]
    for line in lines[2:]:
        cells=[c.strip() for c in line.strip('|').split('|')]
        if cells[0].startswith('nemotron-3-ultra'):continue
        kept.append(cells)
    kept.append(row)
    kept.sort(key=lambda cells:(float('inf') if cells[1]=='—' else -float(cells[1].rstrip('%')),float(cells[6].removesuffix('ms')),cells[0]))
    table='\n'.join(lines[:2]+['| '+' | '.join(cells)+' |' for cells in kept])
    updated=original[:start]+table+original[end:]
    updated=updated.replace('Fourteen rows have fewer than 30;', 'Thirteen current rows have fewer than 30;')
    updated=updated.replace('We selected the most recent matching complete conversations, up to 30 per configuration.',
        'Except for the fresh prospective Ultra cohort below, we selected the most recent matching complete conversations, up to 30 per configuration.')
    old_start=updated.index('The four Nemotron 3 rows now use explicitly mapped, content-aware captures:') if 'The four Nemotron 3 rows now use explicitly mapped, content-aware captures:' in updated else updated.index('The published Nemotron 3 Ultra row uses')
    old_end=updated.index('\n\n',old_start)
    note='''The published Nemotron 3 Ultra row uses **30 fresh thinking-on conversations on Baseten**, with `reasoning_effort=high`, temperature 1.0, top-p 0.95, and max_tokens 16384; all 30 completed. No numeric thinking budget was requested. This replaces the older Modal 128/96 rows. Thinking-off is omitted from the current leaderboard; the completed comparison, original grades, and early-ending runs remain in the [Ultra study](ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/REPORT.md). Three-second request pacing occurs before latency measurement. Super/Nano 512 retain their mapped **NVFP4 on AWS B200** cohorts. The [failure audit](ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/failure-analysis/REPORT.md) and [same-output rubric comparison](ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/rubric-comparison/REPORT.md) explain the current score. The [earlier configuration inventory](ops/judge-refresh-opus55-20261017/configurations.json) and [source audit](ops/judge-refresh-opus55-20261017/SOURCE-AUDIT.md) preserve the retired Ultra cohorts; they are historical evidence.'''
    updated=updated[:old_start]+note+updated[old_end:]
    (ROOT/'README.md').write_text(updated)
    text=updated.split('Text mode models:\n',1)[1].split('Speech-to-speech models:\n',1)[0]
    (ROOT/'leaderboard-medium-context.md').write_text('# aiwf_medium_context text leaderboard\n\nText mode models:\n'+text.rstrip()+'\n')
    audit=ROOT/'docs/text-cohort-sizes-2026-10-17.md'
    body=audit.read_text()
    notice='> Publication update, October 18: the current leaderboard has 42 text configurations, 1,072 complete conversations, and 13 shortfalls. Fresh Baseten Ultra thinking-on (N=30) replaces the historical Modal 128/96 rows. The inventory below preserves the original 43-configuration refresh.\n\n'
    if notice not in body:
        first=body.index('\n\n')+2;audit.write_text(body[:first]+notice+body[first:])
    subprocess.run([sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py')],cwd=ROOT,check=True)
    publication={'published_at':datetime.now(timezone.utc).isoformat(),'selected_arm':'on','label':LABEL,
        'retired_labels':sorted(RETIRED),'published_row':row,'source_aggregate_sha256':sha(STUDY/'aggregate.json'),
        'source_aggregate':'aggregate.json','retained_current_rubric_grades':True,
        'generation_testing_complete':True,'further_budget_experiments':False,
        'artifacts_sha256':{p:sha(ROOT/p) for p in ['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg','docs/text-cohort-sizes-2026-10-17.md','scripts/publish_ultra_baseten.py']}}
    (STUDY/'PUBLICATION.json').write_text(json.dumps(publication,indent=2)+'\n')
    print('Published: '+' | '.join(row))


if __name__=='__main__':main()
