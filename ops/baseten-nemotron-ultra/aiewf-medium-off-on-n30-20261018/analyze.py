"""Publish the fresh off/on experiment without changing the README leaderboard."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

import judge

HERE,ROOT=judge.HERE,judge.ROOT
DIMS=judge.DIMS


def positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def metrics(values):
    return {'n':len(values),'p50_ms':statistics.median(values) if values else None,
            'p95_ms':float(np.percentile(values,95,method='linear')) if values else None,
            'max_ms':max(values) if values else None}


def interval(values,denominator=30,seed=20261018):
    if not values:return None
    rng=np.random.default_rng(seed)
    samples=rng.choice(np.asarray(values,dtype=float),size=(20000,len(values)),replace=True).mean(axis=1)/denominator*100
    return [float(x) for x in np.percentile(samples,[2.5,97.5])]


def token_totals(rows):
    totals=Counter();missing=0
    for row in rows:
        usage=row.get('tokens')
        if not usage or any(type(usage.get(k)) is not int for k in ('prompt_tokens','completion_tokens')):
            missing+=1;continue
        totals['prompt_tokens']+=usage['prompt_tokens']
        totals['completion_tokens']+=usage['completion_tokens']
        for key in ('cache_read_input_tokens','thinking_tokens'):
            value=usage.get(key)
            if type(value) is int:totals[key]+=value
            else:totals[f'{key}_missing_records']+=1
    return {**totals,'missing_usage_records':missing}


def pct(value):return '—' if value is None else f'{value:.1f}%'
def ms(value):return '—' if value is None else f'{value:.0f}ms'
def dollars(value):return 'unavailable' if value is None else f'${value:.2f}'
def ci(value):return 'unavailable' if value is None else f'{value[0]:.1f}–{value[1]:.1f}%'
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join('---' for _ in header)+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    assert (HERE/'judging/COMPLETE.json').exists()
    manifest=judge.freeze();arms={};conversations=[]
    attempts=judge.collect.read_tsv(HERE/'artifacts/attempts.tsv')
    operations={}
    for attempt in attempts:
        if not attempt.get('log'):
            continue
        path=ROOT/attempt['log']
        text=path.read_text() if path.exists() else ''
        operations.setdefault(int(attempt['slot']), []).append({
            'attempt':int(attempt['attempt']), 'log':attempt['log'],
            'explicit_429_retries':text.count('Baseten HTTP 429 before stream: retry'),
            'terminal_http_429': 'Error code: 429' in text,
        })
    for arm in ['off','on']:
        cases=[c for c in manifest['cases'] if c['arm']==arm]
        assert len(cases)==30
        counts=Counter();known_strict=0;complete_passes=[];passes=[]
        latencies=[];raw_latencies=[];complete_latencies=[];all_records=[]
        absent_latency=0;nonpositive_latency=0;nonpositive_raw=0;absent_raw=0
        blocked=[];complete=0;observed=0;recovery=0;judge_cost=0;missing_judge_cost=0
        early_end_turns=Counter();interest_turn_tools=Counter();failed_by_turn=Counter()
        for case in cases:
            path=ROOT/case['source'];data=path.read_bytes()
            assert hashlib.sha256(data).hexdigest()==case['transcript_sha256']
            assert (ROOT/case['input']).read_bytes()==data
            records=[json.loads(s) for s in data.splitlines() if s.strip()]
            scripted=[r for r in records if not r.get('recovery_turn')]
            assert [r['turn'] for r in scripted]==case['observed_turns']
            assert all(r['model_name']=='nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B' for r in scripted)
            is_complete=len(scripted)==30 and all(r.get('assistant_text','').strip() or r.get('tool_calls') for r in scripted)
            complete+=is_complete;observed+=len(scripted);recovery+=len(records)-len(scripted)
            for r in scripted:
                for tool in r.get('tool_calls',[]):
                    if tool['name']=='end_session' and r['turn']<29:
                        early_end_turns[r['turn']]+=1
                    if r['turn']==6:
                        interest_turn_tools[tool['name']]+=1
            all_records+=records
            eligible=[r for r in records if r.get('recovery_turn') or r['turn']!=0]
            values=[r['ttfb_ms'] for r in eligible if positive(r.get('ttfb_ms'))]
            latencies+=values
            if is_complete:complete_latencies+=values
            raw_latencies+=[r['raw_ttfb_ms'] for r in eligible if positive(r.get('raw_ttfb_ms'))]
            absent_latency+=sum(r.get('ttfb_ms') is None for r in eligible)
            nonpositive_latency+=sum(r.get('ttfb_ms') is not None and not positive(r['ttfb_ms']) for r in eligible)
            absent_raw+=sum(r.get('raw_ttfb_ms') is None for r in eligible)
            nonpositive_raw+=sum(r.get('raw_ttfb_ms') is not None and not positive(r['raw_ttfb_ms']) for r in eligible)
            score=None;analysis_path=None
            if judge.valid(case):
                grade=json.loads(judge.result_path(case).read_text())
                if grade.get('raw_output'):
                    analysis_path=grade['raw_output']+'/claude_analysis.md'
                score=0
                for t,j in grade['judgments'].items():
                    for dim in DIMS:counts[dim]+=j['scores'][dim]
                    passed=all(j['scores'][dim] for dim in DIMS)
                    score+=passed
                    if not passed:failed_by_turn[int(t)]+=1
                known_strict+=score;passes.append(score)
                if is_complete:complete_passes.append(score)
                cost=grade['metadata'].get('judge_cost_usd')
                if type(cost) in (int,float):judge_cost+=cost
                else:missing_judge_cost+=1
            else:
                assert judge.blocked_path(case).exists()
                blocked.append(case['job_id'])
            conversations.append({**case,'complete':is_complete,'strict_passes':score,
                                  'missing_future_turns':30-len(scripted),'ttfat':metrics(values),
                                  'judge_analysis':analysis_path,
                                  'transport_attempts':operations.get(case['slot'],[])})
        usage=token_totals(all_records)
        catalog=json.loads((HERE/'smoke/model-catalog.json').read_text())
        entry=next(m for m in catalog['data'] if m['id']=='nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B')
        rates={k:float(entry['pricing'][k]) for k in ('prompt','completion','input_cache_read')}
        cached=usage.get('cache_read_input_tokens',0)
        known_usage=not usage['missing_usage_records'] and not usage.get('cache_read_input_tokens_missing_records')
        estimate=((usage.get('prompt_tokens',0)-cached)*rates['prompt']+
                  cached*rates['input_cache_read']+usage.get('completion_tokens',0)*rates['completion']) if known_usage else None
        arms[arm]={
            'reasoning_effort':'none' if arm=='off' else 'high', 'conversations':30,
            'complete_conversations':complete,'incomplete_conversations':30-complete,
            'observed_scripted_turns':observed,'missing_future_turns':900-observed,'recovery_turns':recovery,
            'fixed_denominator':900,'known_strict_passes':known_strict,
            'strict_percent':100*known_strict/900 if not blocked else None,
            'strict_ci95':interval(passes) if not blocked else None,
            'complete_only_strict_percent':100*sum(complete_passes)/(30*complete) if complete and not blocked else None,
            'complete_only_ci95':interval(complete_passes) if not blocked else None,
            'dimension_passes':dict(counts),
            'observed_dimension_error_percent':{d:100*(observed-counts[d])/observed if not blocked else None for d in DIMS},
            'provider_blocked':blocked,'all_quality_complete':not blocked,
            'ttfat':metrics(latencies),'raw_ttft':metrics(raw_latencies),
            'complete_only_ttfat':metrics(complete_latencies),
            'missing_ttfat':absent_latency,'nonpositive_ttfat':nonpositive_latency,
            'missing_raw_ttft':absent_raw,'nonpositive_raw_ttft':nonpositive_raw,
            'usage_including_recovery':usage,'catalog_per_token_rates':rates,
            'generation_cost_estimate_usd':estimate,'successful_judge_cost_usd':judge_cost,
            'judge_cost_missing_requests':missing_judge_cost,
            'early_end_session_turn_indices':dict(early_end_turns),
            'interest_turn_tool_calls':dict(interest_turn_tools),
            'observed_strict_failures_by_turn':dict(failed_by_turn),
            'explicit_429_retries':sum(o['explicit_429_retries'] for c in conversations if c['arm']==arm for o in c['transport_attempts']),
            'terminal_429_conversations':sum(any(o['terminal_http_429'] for o in c['transport_attempts']) for c in conversations if c['arm']==arm),
        }
    assert len(conversations)==60
    paired=[]
    for pair in range(1,31):
        group={c['arm']:c for c in conversations if (c['slot']+1)//2==pair}
        assert set(group)=={'off','on'}
        if all(c['strict_passes'] is not None for c in group.values()):
            paired.append(group['on']['strict_passes']-group['off']['strict_passes'])
    diff={'on_minus_off_percentage_points':statistics.mean(paired)/30*100 if len(paired)==30 else None,
          'paired_conversation_bootstrap_ci95':interval(paired) if len(paired)==30 else None}
    data={'as_of':datetime.now(timezone.utc).isoformat(),'campaign':'aiewf-medium-ultra-baseten-off-on-n30-20261018',
          'all_quality_complete':all(a['all_quality_complete'] for a in arms.values()),
          'source_manifest':'judging/inputs.json','arms':arms,'comparison':diff,
          'generation_attempts':len(attempts),'excluded_infrastructure_attempts':len(attempts)-len(conversations),
          'diagnostics_excluded':['smoke-off','smoke-on','direct-off','direct-on'],
          'unpaced_pilot_excluded_in_full':'../pilot-unpaced-20261018/PILOT.md',
          'conversations':conversations}
    judge.dump(HERE/'aggregate.json',data)
    comparison=[]
    for arm,a in arms.items():
        comparison.append([arm,a['reasoning_effort'],f"{a['complete_conversations']}/30",f"{a['observed_scripted_turns']}/900",
            f"{a['known_strict_passes']}/900 ({pct(a['strict_percent'])})",pct(a['complete_only_strict_percent']),
            ms(a['ttfat']['p50_ms']),ms(a['ttfat']['p95_ms']),ms(a['ttfat']['max_ms'])])
    details=[]
    for c in sorted(conversations,key=lambda c:c['slot']):
        evidence=f"[grade](judging/results/{c['job_id']}.json)"
        if c['judge_analysis']:
            evidence+=f" · [failures](../../../{c['judge_analysis']})"
        details.append([c['slot'],c['arm'],len(c['observed_turns']),c['missing_future_turns'],
                        'unscored' if c['strict_passes'] is None else f"{c['strict_passes']}/30",
                        ms(c['ttfat']['p50_ms']),evidence])
    judge_total=sum(a['successful_judge_cost_usd'] for a in arms.values())
    component_rows=[[arm,*[pct(a['observed_dimension_error_percent'][d]) for d in DIMS]] for arm,a in arms.items()]
    tool_rows=[[arm,', '.join(f'{name}: {n}' for name,n in sorted(a['interest_turn_tool_calls'].items())) or 'none'] for arm,a in arms.items()]
    early_labels={arm:', '.join(f"user turn {int(t)+1}: {n}" for t,n in sorted(a['early_end_session_turn_indices'].items())) or 'none' for arm,a in arms.items()}
    hotspots=[[int(t)+1,judge.BenchmarkConfig.turns[int(t)]['input'],f'{n}/30'] for t,n in sorted(arms['on']['observed_strict_failures_by_turn'].items(),key=lambda item:(-item[1],int(item[0])))[:5]]
    difference='unavailable' if diff['on_minus_off_percentage_points'] is None else f"{diff['on_minus_off_percentage_points']:.1f}"
    paired_ci=diff['paired_conversation_bootstrap_ci95']
    paired_bounds='unavailable' if paired_ci is None else f'{paired_ci[0]:.1f}–{paired_ci[1]:.1f} percentage points'
    report=f'''# Fresh Nemotron Ultra on Baseten: thinking off and on

Collected 30 newly generated conversations per mode using Baseten's released
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`, with no historical transcripts reused.
Completed analysis: {data['as_of']}.

{table(['Mode','API effort','Complete','Observed scripted turns','Strict, fixed 900 turns','Strict, complete conversations only','TTFAT P50','TTFAT P95','TTFAT max'],comparison)}

The fixed denominator retains all 30 assigned conversations per mode. A complete
conversation has all 30 scripted turns with assistant text or an actual tool call
at each turn. Missing
future turns after early termination count as failures. Completion-conditional
quality uses only complete conversations and should not be read as reliability.
Provider-rejected judge inputs remain explicitly unscored, if any.

Error rates among observed scripted turns (missing future turns are additional
failures in the primary score above):

{table(['Mode','Tool-use errors','Instruction errors','KB grounding errors'],component_rows)}

Most frequent failed thinking-on turns (one-based user turn numbers; per-case
grades below contain the specific reasons):

{table(['User turn','Question','Failed conversations'],hotspots) if hotspots else 'No observed strict failures.'}

## Observed tool behavior

The seventh scripted user turn (index 6) says: "I'm particularly interested in
Voice AI and in techniques for building reliable agents." It asks for no action.
The following tool calls were recorded on that turn, independently of judge scores:

{table(['Mode','Recorded tools on the interests turn'],tool_rows)}

Premature end_session counts by one-based user turn were
**off: {early_labels['off']}**; **on: {early_labels['on']}**. Raw transcripts remain
unchanged. These observations identify API-visible behavior; they do not isolate
the cause within model weights, chat templates, or provider tool parsing.

## Design and serving checks

The [frozen protocol](PROTOCOL.md), [configuration](configuration.json), and
[schedule](schedule.tsv) specify 30 randomized temporal pairs, one off and one on
conversation per pair. Collection was strictly sequential at the shared Baseten
endpoint, with temperature 1.0, top-p 0.95, max_tokens 16384, no filler, the same
30 scripted turns, and the same benchmark instructions/knowledge base/tools.
No numeric reasoning budget was requested. The API model catalog reports FP4
weights and a deployed context limit of 202800 tokens; exact provider engine and
weight revisions are not exposed by this catalog. Its snapshot is in
[smoke/model-catalog.json](smoke/model-catalog.json).

Direct streaming verified zero reasoning tokens for off and separate reasoning
for on. Full-harness diagnostics are preserved in smoke/harness-off.json and
smoke/harness-on.json and excluded from the registered cohorts. Early termination
was not used to tune or rerun smokes.
There were {data['generation_attempts']} registered collection attempts.
All attempts, including exclusions, are recorded in [artifacts/attempts.tsv](artifacts/attempts.tsv).

The entire initial unpaced [operational pilot](../pilot-unpaced-20261018/PILOT.md)
was stopped after an explicit provider HTTP 429, before any judging. All its
cases were excluded together. Effective limits were verified at 120 RPM and
500,000 combined TPM. The registered restart uses the same generation settings
with three-second request spacing before latency measurement and up to three
additional HTTP 429 retries before streaming (10/20/40 seconds). Retry waits
remain in measured latency. Registered off/on explicit retry counts were
{arms['off']['explicit_429_retries']}/{arms['on']['explicit_429_retries']};
terminal rate-limit affected conversation counts were
{arms['off']['terminal_429_conversations']}/{arms['on']['terminal_429_conversations']}.

## Judging and uncertainty

Every observed scripted turn was graded with Opus 5.5, medium effort, adaptive
thinking, no tools/project settings, and the frozen README text rubric.
Generating-model identity, mode, paths, and prior scores were withheld. Explicit
reasoning sections are excluded by the standard judge formatter; raw captures
remain intact. Recovery records are not additional scored turns and were omitted
from judge input, matching the existing formatter. First valid grades are kept;
credit failures wait at least 90 seconds before retrying.

Whole-conversation bootstrap 95% intervals (20,000 resamples):

- Off, fixed denominator: {ci(arms['off']['strict_ci95'])}.
- On, fixed denominator: {ci(arms['on']['strict_ci95'])}.
- On minus off: {difference} percentage points;
  paired-conversation interval {paired_bounds}.

Intervals account for conversation-level clustering, not judge uncertainty or
future provider changes. Per-conversation grades and source hashes are linked
below and frozen in [judging/inputs.json](judging/inputs.json).

## Latency and usage

TTFAT is recorded time to first answer text or tool-call token; raw TTFT is the
first stream chunk. First scripted responses are excluded; recorded recoveries
are included. Positive finite values supply P50, linear P95, and maximum. No
missing timing is replaced with zero. Early-ending runs cover fewer turns, so
latency alone must not be interpreted as a quality improvement. Counts and
complete-only latency/quality are in [aggregate.json](aggregate.json).

SDK-reported successful judge cost: **${judge_total:.2f}**. Registered-conversation
generation cost estimates, using the frozen API catalog's per-token rates and
recorded cache/usage, are {dollars(arms['off']['generation_cost_estimate_usd'])} off and
{dollars(arms['on']['generation_cost_estimate_usd'])} on. These are estimates, not billing
receipts, and exclude diagnostic smokes and any unrecorded failed-request usage.

## Per-conversation results

{table(['Slot','Mode','Observed turns','Missing future turns','Strict passes / fixed 30','TTFAT P50','Evidence'],details)}

The existing README Ultra rows are unchanged pending review of this experiment.
'''
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps({'arms':{arm:{k:a[k] for k in ['complete_conversations','observed_scripted_turns',
                     'strict_percent','complete_only_strict_percent','ttfat','successful_judge_cost_usd']}
                     for arm,a in arms.items()},'comparison':diff},indent=2))


if __name__=='__main__':main()
