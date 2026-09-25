"""Compare every revised turn to its preserved first pass; never modify scores."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'rejudge-policy-compliance-1'
DIMS=['tool_use_correct','instruction_following','kb_grounding','turn_taking']

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert (OUT/'complete.json').exists()
    groups=defaultdict(Counter);changes=[];question=[];reason_conflicts=[]
    for r in protocol['runs']:
        run=ROOT/r['run_dir'];new=OUT/r['key'];g=groups[r['arm']]
        for name,h in r['original_hashes'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
        summary=json.loads((new/'claude_summary.json').read_text())
        assert summary['judge_prompt_sha256']==protocol['judge_prompt_sha256']
        before={t['turn']:t for t in map(json.loads,(run/'claude_judged.jsonl').read_text().splitlines())}
        after={t['turn']:t for t in map(json.loads,(new/'claude_judged.jsonl').read_text().splitlines())}
        assert before.keys()==after.keys()
        g['conversations']+=1;g['planned']+=30;g['observed']+=len(after)
        for turn,b in before.items():
            a=after[turn];bs=b['scores'];s=a['scores']
            assert b['assistant_text']==a['assistant_text']
            assert bs['turn_taking']==s['turn_taking']
            old=all(bs[k] for k in DIMS[:3]);current=all(s[k] for k in DIMS[:3])
            g['before_strict']+=old;g['after_strict']+=current
            g['pass_to_fail']+=old and not current;g['fail_to_pass']+=current and not old
            for dim in DIMS:
                if bs[dim]!=s[dim]:g[dim+'_changed']+=1
            detail=dict(key=r['key'],arm=r['arm'],turn=turn,assistant=a['assistant_text'],before=bs,after=s,before_reason=b['claude_reasoning'],after_reason=a['claude_reasoning'])
            for dim in DIMS[:3]:
                pattern=dim.replace('_',r'[_ -]')+r'\s*(?:is|=|:|should be|must be)\s*(true|false)'
                for match in re.finditer(pattern,a['claude_reasoning'],re.I):
                    if s[dim]!=(match.group(1).lower()=='true'):reason_conflicts.append(dict(**detail,dimension=dim,matched_text=match.group(0)))
            if bs!=s:changes.append(detail)
            if turn==4:
                detail['numeric_pattern']=bool(re.search(r'\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:(?:main|conference|scheduled|listed|MCP|track)\s+){0,3}(?:sessions|talks|workshops|keynotes)\b',a['assistant_text'],re.I))
                question.append(detail)
    clusters=defaultdict(list)
    for q in question:clusters[re.sub(r'[^a-z0-9]+',' ',q['assistant'].lower()).strip()].append(q)
    disagreements=[dict(answer=k,judgments=v) for k,v in clusters.items() if len({r['after']['instruction_following'] for r in v})>1]
    retry_logs=[str(p.relative_to(OUT)) for p in OUT.glob('*/attempt-*.log') if p.name!='attempt-1.log']
    result=dict(retry_logs=retry_logs,explicit_reason_score_conflicts=reason_conflicts,exact_answer_instruction_disagreements=disagreements,protocol=protocol,groups={k:dict(v) for k,v in groups.items()},changed_turns=changes,session_count_question=question,note='Diagnostic patterns are for human review only. All revised judgments retained without score-based retries or manual overrides. Non-target changes include normal judge variability as well as rubric effects.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Shared speech judge: before/after audit','','All 131 conversations received one complete revised Opus 5 judgment. Original transcripts, model prompt provenance, and first-pass judgments passed hash verification. The only rubric change requires policy compliance for instruction following and independent factual assessment across the full KB. Model/frontend prompts and text-model judging are unchanged.','','| Arm | Conversations | Original strict | Revised strict | Pass → fail | Fail → pass |','|---|---:|---:|---:|---:|---:|']
    for arm,g in groups.items():
        n=g['planned'];lines.append(f"| {arm} | {g['conversations']} | {g['before_strict']}/{n} ({g['before_strict']/n*100:.1f}%) | {g['after_strict']}/{n} ({g['after_strict']/n*100:.1f}%) | {g['pass_to_fail']} | {g['fail_to_pass']} |")
    lines+=['',f'Execution/parse retries: **{len(retry_logs)}**. These are judge-service failures, separate from model reliability.', '',f'Explicit score/explanation conflict flags: **{len(reason_conflicts)}**. These diagnostic flags are preserved for review, without manual score correction.','', 'The denominator includes every planned turn, including the six missing turns after the Gemini 3.1 timeout. GPT-Live here combines its single validation and five-run pilot; it is not a 25-attempt canonical cohort.','','## MCP session-count question','','| Arm | Instruction failures before | Instruction failures after | Numeric-pattern answers with instruction pass after |','|---|---:|---:|---:|']
    for arm in groups:
        q=[r for r in question if r['arm']==arm]
        lines.append(f"| {arm} | {sum(not r['before']['instruction_following'] for r in q)}/{len(q)} | {sum(not r['after']['instruction_following'] for r in q)}/{len(q)} | {sum(r['numeric_pattern'] and r['after']['instruction_following'] for r in q)} |")
    lines+=['','Exact-answer disagreements are also recorded in the JSON audit. Numeric patterns are diagnostic, not automatic failures: qualified subsection descriptions require contextual review. See [all 131 answers and every changed judgment](audit.json). No manual score overrides or favorable selective rejudgments were used.','','## Limits','','Opus judgments are stochastic. Changes outside the motivating question can reflect both the revised rubric and ordinary judge variation; this is not a controlled estimate of rubric effect alone. Audio metrics and all per-turn audio pass/fail assessments were held fixed. Original scores remain in the capture directories and [original aggregate](../aggregate.json). The [revision protocol](protocol.json) fixes the full cohort and prompt hash.']
    lines+=['','## Explicit score/explanation flags','']
    for r in reason_conflicts:
        lines.append(f"- `{r['key']}`, turn {r['turn']}, `{r['dimension']}`: Boolean `{r['after'][r['dimension']]}`, explanation says `{r['matched_text']}`. [Full judgment]({r['key']}/claude_analysis.md).")
    if reason_conflicts:
        impact=Counter(r['arm'] for r in reason_conflicts if all(r['after'][k] for k in DIMS[:3]) and r['matched_text'].lower().endswith('false'))
        lines+=['','Sensitivity check only (not score overrides):']
        for arm,n in impact.items():
            g=groups[arm];lines.append(f"- If the flagged strict passes followed their explicit failure explanations, `{arm}` would be {(g['after_strict']-n)/g['planned']*100:.1f}% instead of {g['after_strict']/g['planned']*100:.1f}%.")
    if not reason_conflicts:lines.append('None detected by the explicit-label check; this is not an exhaustive semantic correctness guarantee.')
    (OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
