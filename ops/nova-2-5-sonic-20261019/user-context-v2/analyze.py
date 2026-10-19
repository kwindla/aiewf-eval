"""Reconcile every score and waveform measurement to the frozen Nova cohort."""
import json,math,statistics
import numpy as np
from common import HERE,ROOT,dump,sha,now
from audio import measure
import judge

LABEL='Nova 2.5 Sonic (reference in USER context)'

def metrics(values):
    return {'n':len(values),'mean_ms':statistics.mean(values) if values else None,
        'median_ms':statistics.median(values) if values else None,'p50_ms':statistics.median(values) if values else None,
        'p95_ms':float(np.percentile(values,95,method='linear')) if values else None,
        'max_ms':max(values) if values else None}
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def ci(values):
    rng=np.random.default_rng(20261018)
    samples=rng.choice(values,size=(20000,len(values)),replace=True).mean(axis=1)/30*100
    return list(map(float,np.percentile(samples,[2.5,97.5])))
def ms(value):return '—' if value is None else f'{value:.0f}ms'

def main():
    assert json.loads((HERE/'judging/COMPLETE.json').read_text())['graded']==25
    cases=judge.freeze();counts={d:0 for d in judge.DIMS};strict=0;taking=0;per=[];cost=0
    signed=[];positive=[];tool=[];non_tool=[];pads=[];inputs=[]
    for case in cases:
        assert judge.valid(case);grade=json.loads(judge.result_path(case).read_text())
        passed=0
        for judgment in grade['judgments'].values():
            scores=judgment['scores'];passed+=all(scores[d] for d in judge.DIMS)
            for d in judge.DIMS:counts[d]+=scores[d]
            taking+=scores['turn_taking']
        strict+=passed;per.append(passed);cost+=grade['metadata'].get('judge_cost_usd',0) or 0
        run=(ROOT/case['source']).parent;data,wave=measure(case['job_id'],run)
        inputs.append({'job':case['job_id'],'source_wav':str((run/'conversation.wav').relative_to(ROOT)),
            'wav_sha256':sha(run/'conversation.wav'),'metrics':str((HERE/'audio'/(case['job_id']+'.json')).relative_to(ROOT)),
            'input_waveform_audit':str((HERE/'audio'/(case['job_id']+'-input-waveform.json')).relative_to(ROOT))})
        for row in data['turns']:
            v=row.get('wav_v2v_ms')
            if finite(v):
                signed.append(v)
                if v>0:positive.append(v);(tool if row.get('has_tool_call') else non_tool).append(v)
            pad=row.get('silent_pad_silero_ms')
            if finite(pad) and pad>=0:pads.append(pad)
    attempts=json.loads((HERE/'attempts.json').read_text());failed=sum(not a['complete'] for a in attempts)
    assert sum(a['complete'] for a in attempts)==25
    events=[json.loads(line) for line in (HERE/'judging/attempts.jsonl').read_text().splitlines() if line.strip()]
    model={'label':LABEL,'model':'amazon.nova-2-5-sonic','provider':'AWS Bedrock','complete':25,
        'eligible_attempts':len(attempts),'failed_conversations':failed,'failure_rate_percent':100*failed/len(attempts),
        'complete_only':{'strict_passes':strict,'dimension_passes':counts,'denominator':750},
        'strict_percent':100*strict/750,'strict_ci95':ci(per),'per_conversation_strict_passes':per,
        'turn_taking_passes':taking,'turn_taking_scored':True,'first_speech_signed':metrics(signed),
        'first_speech_positive':metrics(positive),'non_tool':metrics(non_tool),'tool':metrics(tool),'silence':metrics(pads),
        'nonpositive_n':sum(v<=0 for v in signed),'missing_latency_n':750-len(signed),'audio_inputs':inputs}
    excluded=json.loads((HERE/'prior-harness-exclusions.json').read_text())
    model.update(harness_excluded=excluded['count'],harness_exclusion_details=excluded['attempts'])
    serving=json.loads((HERE/'serving-tools-check.json').read_text())
    assert serving['checked']==25 and serving['all_five_tool_declarations_identical']
    model['conversations_with_no_tool_calls']=sum(c['observed_calls']==0 for c in serving['cases'])
    data={'at':now(),'judge_model':judge.MODEL,'judge_effort':judge.EFFORT,'all_quality_complete':True,
        'judged_conversations':25,'successful_judge_cost_usd':cost,'judge_credit_cooldowns':sum(e['event']=='credit_cooldown' for e in events),
        'text_models':[],'voice_models':[model],'cases':cases}
    dump(HERE/'aggregate.json',data)
    report=f'''# Nova 2.5 Sonic: full-reference USER-context benchmark

**{model['strict_percent']:.1f}% strict pass**, 25 complete 30-turn conversations
(750 turns), Opus 5.5 Medium. The 95% conversation-bootstrap interval is
{model['strict_ci95'][0]:.1f}–{model['strict_ci95'][1]:.1f}% (20,000 draws; seed 20261018).

| Measure | Result |
|---|---:|
| Strict quality | {strict}/750 |
| Tool use | {counts['tool_use_correct']}/750 |
| Instruction following | {counts['instruction_following']}/750 |
| KB grounding | {counts['kb_grounding']}/750 |
| Audio turn-taking | {taking}/750 |
| Unsuccessful canonical attempts | {failed}/{len(attempts)} ({model['failure_rate_percent']:.1f}%) |
| Earlier adapter-defect captures, excluded from canonical denominator | {model['harness_excluded']} |
| First-speech positive P50 / P95 | {ms(model['first_speech_positive']['p50_ms'])} / {ms(model['first_speech_positive']['p95_ms'])} |
| Non-tool median / maximum | {ms(model['non_tool']['median_ms'])} / {ms(model['non_tool']['max_ms'])} |
| Tool mean | {ms(model['tool']['mean_ms'])} |
| Silence pad mean | {ms(model['silence']['mean_ms'])} |
| Positive / nonpositive / missing timing samples | {len(positive)} / {model['nonpositive_n']} / {model['missing_latency_n']} |

The exact speech-v2 policy is sent as SYSTEM, followed by the complete unchanged
knowledge base as a noninteractive USER history message. Their concatenation is
the original 41,161-character prompt. No facts, tools, scripted turns or input
audio were removed. The [filter investigation](../filter-diagnostics/REPORT.md)
explains why placing the whole reference in SYSTEM was blocked. This message
placement is an explicit provider configuration difference from standard rows.

For a single reproduction with AWS credentials configured:

```bash
MTE_NOVA_KB_CONTEXT_ROLE=user MTE_ENABLE_RECOVERY=0 uv run multi-turn-eval run aiwf_medium_context_speech_v2 --model amazon.nova-2-5-sonic --pipeline nova-sonic
```

The corrected excluded full-task smoke passed before generation was frozen. The
first 30-turn capture lacked local VAD timing markers; a second excluded capture
was stopped after confirming a startup clock offset. Those captures, their source
versions, and failed audits remain preserved. The final smoke validates the shared
VAD/logger wiring and synchronized startup. Selection uses
the first 25 complete conversations in attempt order; every unsuccessful attempt
is retained. There is no quality-based replacement, generation recovery, prompt
truncation, or grading retry based on score. Accuracy is conditional on completion.
The original blocked smoke and diagnostic probes are excluded from canonical
attempt counts and remain available in the parent study.

{model['conversations_with_no_tool_calls']} of the 25 complete conversations made no tool calls, although all 25
requests declared the identical five tools. These conversations remain included;
completion is separate from correct tool use and instruction following. The
[serving-tool audit](serving-tools-check.json) records every case.

The first frozen adapter profile exposed deferred native tool-result delivery in
three captures. Those are retained as harness exclusions in the
[exclusion ledger](prior-harness-exclusions.json), with raw artifact hashes and
their original source freeze. The repaired profile sends each native tool result
immediately and marks its ID completed, preventing a later context push from
resending it. A new excluded full-task smoke passed before this profile's freeze.
Only the repaired profile supplies the 25 scored conversations.

Latency and turn-taking use the actual stereo recordings and unchanged analyzer.
Nova explicitly arms one recording marker per sequential input WAV, so an early
response cannot lose its marker while waiting for local VAD-stop. The earlier
28-marker smoke and its failed audit are retained; this annotation repair was
validated by the final full-task smoke before the canonical source freeze.
All 30 user input clips in each complete capture pass the existing three-landmark
waveform correlation and alignment checks. Signed and positive latency statistics
are retained in the aggregate. Comparison with other rows is descriptive across
independent cohorts; the confidence interval describes these sampled conversations.

Successful Opus judge requests cost ${cost:.6f}. Judging credit cooldown events:
{data['judge_credit_cooldowns']}, each waiting at least 90 seconds. Private
reasoning is not used as grading evidence.

[Protocol](PROTOCOL.md) · [Configuration](configuration.json) ·
[Attempts](attempts.json) · [Frozen inputs](judging/inputs.json) ·
[Aggregate](aggregate.json) · [Capture hashes](generation-sha256.json)
'''
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps({'strict_percent':model['strict_percent'],'cost_usd':cost,'attempts':len(attempts)}))

if __name__=='__main__':main()
