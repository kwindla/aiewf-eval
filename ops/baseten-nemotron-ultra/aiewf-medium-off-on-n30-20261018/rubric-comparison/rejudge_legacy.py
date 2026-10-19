"""Paired legacy/current rubric diagnostic on the same 30 fresh thinking-on runs."""

import argparse
import ast
import asyncio
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import dotenv_values
import numpy as np

HERE=Path(__file__).resolve().parent
PRIMARY=HERE.parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(PRIMARY))
import judge


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_rubric():
    source=subprocess.check_output(['git','show','5dfcd96:src/multi_turn_eval/judging/claude_judge.py'],cwd=ROOT,text=True)
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUDGE_SYSTEM_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('Historical rubric not found')


def summarize(cases,manifest):
    counts=Counter();by_turn=Counter();paired=[];disagreements=[];legacy_cost=0
    for case in cases:
        new=json.loads((PRIMARY/'judging/results'/f"{case['job_id']}.json").read_text())
        old=json.loads(judge.result_path(case).read_text())
        assert judge.valid(case)
        assert old['metadata']['judge_input_sha256']==new['metadata']['judge_input_sha256']
        assert old['transcript_sha256']==new['transcript_sha256']==case['transcript_sha256']
        legacy_cost+=old['metadata']['judge_cost_usd']
        change=0
        for turn in map(str,range(30)):
            old_pass=all(old['judgments'][turn]['scores'][d] for d in judge.DIMS)
            new_pass=all(new['judgments'][turn]['scores'][d] for d in judge.DIMS)
            counts['legacy_passes']+=old_pass;counts['current_passes']+=new_pass
            counts[f"legacy_{'pass' if old_pass else 'fail'}_current_{'pass' if new_pass else 'fail'}"]+=1
            change+=int(old_pass)-int(new_pass)
            if old_pass!=new_pass:
                by_turn[int(turn)]+=1
                disagreements.append({'job_id':case['job_id'],'turn':int(turn),'legacy_pass':old_pass,'current_pass':new_pass,
                    'legacy':old['judgments'][turn],'current':new['judgments'][turn]})
        paired.append(change)
    samples=np.random.default_rng(20261018).choice(np.asarray(paired,dtype=float),size=(20000,30),replace=True).mean(axis=1)/30*100
    data={'completed_at':judge.now(),'conversations':30,'turns':900,'judge_model':judge.MODEL,'judge_effort':judge.EFFORT,
          'legacy_rubric_sha256':manifest['legacy_rubric_sha256'],'current_rubric_sha256':manifest['current_rubric_sha256'],
          'counts':dict(counts),'legacy_percent':counts['legacy_passes']/9,'current_percent':counts['current_passes']/9,
          'legacy_minus_current_percentage_points':sum(paired)/9,
          'paired_conversation_bootstrap_ci95':np.percentile(samples,[2.5,97.5]).tolist(),
          'legacy_judge_cost_usd':legacy_cost,'disagreement_counts_by_turn':dict(by_turn),'disagreements':disagreements}
    judge.dump(HERE/'aggregate.json',data)
    report=f'''# Same Ultra transcripts, two rubric packages

The same 30 fresh Baseten thinking-on conversations were judged twice with
Opus 5.5 Medium. No new Ultra conversations were generated. The configured
difference is the system rubric: the legacy v4 rubric from git `5dfcd96` versus
the current policy-grounded rubric, which also supplies the full assistant policy
and knowledge base. The formatted conversation prompt and transcript hashes match
exactly for every pair; tools and project settings are disabled in both arms.

| Rubric | Strict passes | Pass rate |
|---|---:|---:|
| Legacy v4 | {counts['legacy_passes']}/900 | {data['legacy_percent']:.1f}% |
| Current policy + full KB | {counts['current_passes']}/900 | {data['current_percent']:.1f}% |

Legacy minus current: {data['legacy_minus_current_percentage_points']:.1f} percentage
points; 95% paired-conversation bootstrap interval
{data['paired_conversation_bootstrap_ci95'][0]:.1f}–{data['paired_conversation_bootstrap_ci95'][1]:.1f}
points (20,000 resamples). Legacy passes/current failures:
{counts['legacy_pass_current_fail']}; legacy failures/current passes:
{counts['legacy_fail_current_pass']}.

This tests rubric/reference sensitivity on identical fresh outputs. One grade
per rubric/input means individual differences also include judge variability.
The old near-perfect published Ultra rows used a different cohort, serving/budget
configuration, and historical judge version. This is not a reconstruction of
that exact historical experiment or proof of a model regression. The reference
KB/policy addition is part of the rubric package; this does not separate its
effect from the wording changes.

Only first valid grades are retained; credit failures wait at least 90 seconds.
Original current-rubric grades remain unchanged. Legacy SDK-reported cost:
**${legacy_cost:.2f}**. See [frozen inputs](inputs.json), [full paired results](aggregate.json),
and [legacy grades](judging/results/).
'''
    (HERE/'REPORT.md').write_text(report)
    paths=[HERE/'inputs.json',HERE/'legacy-rubric.txt',HERE/'aggregate.json',HERE/'REPORT.md',Path(__file__)]
    paths+=list((HERE/'judging/results').glob('*.json'))
    judge.dump(HERE/'VERIFICATION.json',{'completed_at':judge.now(),'all_pairs_valid':True,
        'artifacts_sha256':{str(p.relative_to(HERE)):digest(p) for p in paths}})
    print(json.dumps({k:data[k] for k in ['legacy_percent','current_percent','legacy_minus_current_percentage_points','legacy_judge_cost_usd']}),flush=True)


async def main():
    original=judge.freeze()
    cases=[c for c in original['cases'] if c['arm']=='on']
    assert len(cases)==30 and all(c['observed_turns']==list(range(30)) for c in cases)
    rubric=legacy_rubric()
    manifest={'judge_model':judge.MODEL,'judge_effort':judge.EFFORT,'legacy_git_ref':'5dfcd96',
        'legacy_rubric_sha256':hashlib.sha256(rubric.encode()).hexdigest(),'current_rubric_sha256':original['rubric_sha256'],
        'cases':cases,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),PRIMARY/'judge.py',ROOT/'src/multi_turn_eval/judging/claude_judge.py']}}
    frozen=HERE/'inputs.json'
    if frozen.exists():assert json.loads(frozen.read_text())==manifest,'Diagnostic inputs or implementation changed'
    else:judge.dump(frozen,manifest)
    (HERE/'legacy-rubric.txt').write_text(rubric)
    judge.HERE=HERE;judge.RUBRIC=rubric;judge.RAW=ROOT/'runs/baseten-ultra-20261018-paced/legacy-rubric'
    (HERE/'judging').mkdir(exist_ok=True)
    for path in [Path('/home/khkramer/src/aiewf-eval/.env'),Path('/home/khkramer/src/gb-benchmarks/.env')]:
        key=dotenv_values(path).get('ANTHROPIC_API_KEY')
        if key:os.environ.setdefault('ANTHROPIC_API_KEY',key)
    if not os.getenv('ANTHROPIC_API_KEY'):raise RuntimeError('Anthropic credential unavailable')
    cooldown=judge.retry.CreditCooldown()
    pending=[c for c in cases if not judge.valid(c)]
    while pending:
        await asyncio.gather(*(judge.grade(c,cooldown) for c in pending[:4]))
        if any(judge.blocked_path(c).exists() for c in pending):raise RuntimeError('Provider rejected a diagnostic input; no imputed grade')
        pending=[c for c in cases if not judge.valid(c)]
    summarize(cases,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.exit(message='Pass --execute to judge the 30 existing transcripts with the legacy rubric.\n')
    with (HERE/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main())
