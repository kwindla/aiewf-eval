"""Add only the verified Nova row and its explicit reference-layout note."""
import hashlib,json
from common import HERE,ROOT,sha,dump
from analyze import LABEL,ms

BEGIN='<!-- nova25-user-context-20261019:start -->'
END='<!-- nova25-user-context-20261019:end -->'

def render(before,data):
    m=data['voice_models'][0];counts=m['complete_only']['dimension_passes']
    cells=[LABEL,'25',f"{m['failed_conversations']}/{m['eligible_attempts']} ({m['failure_rate_percent']:.1f}%)",
        f"{m['strict_percent']:.1f}%",*[f'{counts[d]}/750' for d in ['tool_use_correct','instruction_following','kb_grounding']],
        f"{m['turn_taking_passes']}/750",ms(m['non_tool']['median_ms']),ms(m['non_tool']['max_ms']),
        ms(m['tool']['mean_ms']),ms(m['silence']['mean_ms'])]
    row='| '+' | '.join(cells)+' |'
    start=before.index('| Model |',before.index('Speech-to-speech models:\n'));end=before.index('\n\n',start)
    lines=before[start:end].splitlines();index=len(lines)
    for i,line in enumerate(lines[2:],2):
        if float(line.split('|')[4].strip().rstrip('%'))<m['strict_percent']:index=i;break
    lines.insert(index,row)
    note=(BEGIN+'\n**Nova 2.5 Sonic content-filter refusals:** AWS rejected the full standard SYSTEM prompt. '
        'Excluded diagnostic probes also blocked short SYSTEM prompts containing `Replicate` or the session title '
        '“Voice Cloning and Synthesis: Ethics and Technology”, while benign text of the full prompt’s length passed. '
        'The scored configuration keeps the unchanged speech-v2 policy in SYSTEM and the complete unchanged knowledge base in USER history; '
        'no reference facts were removed. This provider message-layout difference is explicit. '
        'The prompt-level filter rejections are excluded diagnostics, separate from the 25/25 completed scored conversations.\n\n'
        '**Nova failure breakdown:** Of 105 failing turns (86.0% strict pass across 750 turns), '
        '74 (70.5%) omitted required tool calls: 25 session endings, 17 votes, and 32 session suggestions, dietary requests or support tickets. '
        'Many responses claimed success without executing the action, also failing instruction following. '
        'Eight of 25 conversations made no tool calls. The remaining 31 failing turns failed only instruction following; '
        'KB grounding and audio turn-taking each passed 750/750.\n\n'
        'The [Nova report](ops/nova-2-5-sonic-20261019/user-context-v2/REPORT.md) records the excluded smoke, first 25 complete conversations, '
        'all attempts, Opus 5.5 Medium grades, and verified waveform latency; the [filter investigation](ops/nova-2-5-sonic-20261019/filter-diagnostics/REPORT.md) '
        'retains the earlier blocked probes. Three captures from the earlier adapter profile are retained as harness exclusions after a tool-result delivery defect; the scored cohort uses the repaired profile.\n'+END)
    return before[:start]+'\n'.join(lines)+'\n\n'+note+before[end:]

def main():
    data=json.loads((HERE/'aggregate.json').read_text())
    assert data['all_quality_complete'] and data['judged_conversations']==25
    assert data['voice_models'][0]['complete']==25
    baseline=HERE/'prior/README.md';before=baseline.read_text();after=render(before,data)
    current=ROOT/'README.md';assert current.read_text() in (before,after),'README changed outside this publication'
    current.write_text(after)
    dump(HERE/'PUBLICATION.json',{'label':LABEL,'voice_rows_added':1,'complete_conversations':25,
        'judge_model':data['judge_model'],'judge_effort':data['judge_effort'],
        'prior_artifacts_sha256':{'README.md':sha(baseline)},
        'published_artifacts_sha256':{'README.md':sha(current)},'aggregate_sha256':sha(HERE/'aggregate.json'),
        'scope':'One additive voice row and reference-layout note; all prior publication content retained.'})
    print('Published Nova 2.5 Sonic with explicit USER reference layout',flush=True)

if __name__=='__main__':main()
