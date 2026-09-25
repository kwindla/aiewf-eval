"""Fixed-denominator report for both new arms, retaining every attempted capture."""
import hashlib,json,math,re,statistics,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD=ROOT/'docs/speech-v2-canonical-2026-09-15'
sys.path.insert(0,str(OLD))
from audit_reliability import audit
DIMS=['tool_use_correct','instruction_following','kb_grounding','turn_taking']
def readlines(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
def stats(xs):
 xs=sorted(xs)
 return dict(n=len(xs),mean_ms=statistics.mean(xs),p50_ms=statistics.median(xs),p90_ms=float(np.percentile(xs,90)),p95_ms=xs[math.ceil(.95*len(xs))-1],min_ms=min(xs),max_ms=max(xs)) if xs else dict(n=0,mean_ms=None,p50_ms=None,p90_ms=None,p95_ms=None,min_ms=None,max_ms=None)
def main():
 assert (HERE/'complete.json').exists()
 protocol=json.loads((HERE/'protocol.json').read_text());records=readlines(HERE/'capture/attempts.jsonl');assert len(records)==25
 models=[];flags=[]
 for arm in protocol['arms']:
  selected=sorted([r for r in records if r['arm']==arm['key']],key=lambda r:r['index']);assert [r['index'] for r in selected]==list(range(1,26))
  counts=Counter();lat={k:[] for k in ['all','non_tool','tool','silence']};signed=[];runs=[];failures=[];reliability=[];per_turn=[Counter() for _ in range(30)];passes=[]
  for r in selected:
   key=f"{arm['key']}-{r['index']:02d}";done=json.loads((HERE/'processed'/key/'done.json').read_text());assert done['record']==r
   rel=audit(r);reliability.append(rel);judged={};measured={};metrics={};events=[]
   if r.get('run_dir'):
    run=ROOT/r['run_dir'];meta=json.loads((run/'benchmark-provenance.json').read_text())
    if arm['key']=='live':
     assert meta['frontend-prompt_sha256']==meta['system-prompt_sha256']==protocol['prompt_sha256']
     assert meta['backend']==arm['backend'] and meta['reasoning']==arm['reasoning']
     assert meta['implementation_sha256']==protocol['source_hashes']['src/multi_turn_eval/pipelines/openai_live.py']
     events=readlines(run/'live-events.jsonl')
     starts=[e['event']['session'] for e in events if e.get('event',{}).get('type')=='session.started']
     for s in starts:
      assert hashlib.sha256(s['instructions'].encode()).hexdigest()==protocol['prompt_sha256']
      assert hashlib.sha256(s['delegation']['responses']['instructions'].encode()).hexdigest()==protocol['prompt_sha256']
     rel['live_api_errors']=[e for e in events if e.get('event',{}).get('type')=='error' or e['type']=='pipeline_error']
     rel['session_closed']=any(e.get('event',{}).get('type')=='session.closed' for e in events)
    else:
     assert meta['prompt_sha256']==protocol['prompt_sha256'] and meta['thinking']=='medium'
     input_indices=[int(n) for n in re.findall(r'\[USER_AUDIO_QUEUED\] turn=(\d+)',(run/'run.log').read_text())]
     rel['scripted_inputs']=input_indices
     rel['input_sequence_valid']=input_indices==list(range(len(input_indices)))
     assert rel['input_sequence_valid'],f'Input sequence anomaly requires audit: {key}'
     if r['full']:assert input_indices==list(range(30)),f'Incomplete actual input sequence: {key}'
   if r['turns']:
    assert done['judge'],f'Judge incomplete: {key}'
    path=Path(done['score_dir']);s=json.loads((path/'claude_summary.json').read_text());assert s['judge_model']==protocol['judge_model'] and s['judge_prompt_sha256']==protocol['judge_prompt_sha256']
    judged={t['turn']:t for t in readlines(path/'claude_judged.jsonl') if not t.get('recovery_turn')}
    assert len(judged)==r['turns']
   n=0
   for i in range(30):
    t=judged.get(i,{});sc=t.get('scores',{})
    for dim in DIMS:counts[dim]+=sc.get(dim) is True;per_turn[i][dim]+=sc.get(dim) is True
    strict=all(sc.get(k) is True for k in DIMS[:3]);n+=strict;per_turn[i]['strict']+=strict
    if not strict and t:failures.append(dict(key=key,turn=i,assistant=t.get('assistant_text'),scores=sc,reason=t.get('claude_reasoning')))
    for dim in DIMS[:3]:
     for m in re.finditer(dim.replace('_',r'[_ -]')+r'\s*(?:is|=|:|should be|must be)\s*(true|false)',t.get('claude_reasoning',''),re.I):
      if sc[dim]!=(m.group(1).lower()=='true'):flags.append(dict(key=key,turn=i,dimension=dim,score=sc[dim],reason=t['claude_reasoning']))
   counts['strict']+=n;passes.append(n)
   if done['audio']:
    metrics=json.loads(Path(done['audio_path']).read_text())
    if arm['key']=='gemini38medium':
     raw=metrics
     for t in raw['turns']:
      if t.get('silent_pad_silero_ms') is not None:lat['silence'].append(t['silent_pad_silero_ms'])
     audit_path=HERE/'audio-audit'/f'{key}.done.json'
     association=json.loads(audit_path.read_text())
     done['input_window_audit']=association
     metrics=json.loads(Path(association['metrics']).read_text()) if association['audit'] else dict(turns=[],issues=[dict(reason='input-window waveform audit failed')])
    for t in metrics['turns']:
     idx=t.get('turn_index',t.get('turn'));v=t.get('wav_v2v_ms');measured[idx]=v
     if v is not None:signed.append(v)
     if v is not None and v>0:
      lat['all'].append(v);lat['tool' if t['has_tool_call'] else 'non_tool'].append(v)
     pad=t.get('silent_pad_silero_ms')
     if pad is not None:lat['silence'].append(pad)
   gaps=[i for i in range(30) if measured.get(i) is None]
   nonpositive=[dict(turn=i,v2v_ms=v) for i,v in measured.items() if v is not None and v<=0]
   runs.append(dict(attempt=r,strict_pass=n,judged_turns=len(judged),processing=done,latency_missing_or_unrepresented=gaps,nonpositive_latencies=nonpositive,audio_issues=metrics.get('issues',[])))
  rng=np.random.default_rng(20260915);boot=np.array(passes)[rng.integers(0,25,size=(20000,25))].sum(axis=1)/750*100
  models.append(dict(arm=arm,attempts=25,complete=sum(r['full'] for r in selected),denominator=750,observed_turns=sum(r['turns'] for r in selected),scores=dict(counts),strict_percent=counts['strict']/7.5,strict_ci95=np.quantile(boot,[.025,.975]).tolist(),latency={k:stats(v) for k,v in lat.items()},signed_latency=stats(signed),runs=runs,reliability=reliability,per_turn=[dict(c) for c in per_turn],failure_examples=failures))
 gate=json.loads((HERE/'live-gate-fix/audit.json').read_text())
 for m in models:
  m['canonical_eligible']=(not gate['violations'] and len(gate['checked'])==25 and all(r['processing'].get('alignment') is True for r in m['runs']))
  if not m['canonical_eligible']:m['validity_issue']='Premature turn commit after early backchannel; current cohort retained as diagnostics pending a fresh uniform cohort with the input-end gate fix.'
 aggregate=dict(protocol=protocol,models=models,explicit_judge_conflict_flags=flags,live_gate_audit=gate)
 (HERE/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
 def fmt(v):return 'N/A' if v is None else f'{v:.0f}ms'
 lines=['# GPT-Live corrected-gate standard cohort: 25-attempt results','','| Model | Complete | Strict pass | Non-tool V2V median | Tool V2V mean | Overall V2V mean | Overall p90 | Measured positive responses |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m in models:
  name='GPT-Live + GPT-5.6-Terra (medium), corrected gate' if m['arm']['key']=='live' else 'Gemini 3.8 Extended Thinking (medium)';l=m['latency'];lines.append(f"| {name} | {m['complete']}/25 | {m['scores']['strict']}/750 ({m['strict_percent']:.1f}%) | {fmt(l['non_tool']['p50_ms'])} | {fmt(l['tool']['mean_ms'])} | {fmt(l['all']['mean_ms'])} | {fmt(l['all']['p90_ms'])} | {l['all']['n']}/750 |")
 lines+=['',('**Validity:** all recorded turn commits passed the post-input quiet-interval audit.' if not gate['violations'] else '**Validity hold:** turn-gate violations require investigation; do not publish as canonical.'),'', 'Every planned turn remains in the accuracy denominator; incomplete attempts are retained. Strict pass combines tool use, instruction following, and factual grounding. Audio turn-taking is separate. V2V measures actual user-speech end to first actual assistant speech, including acknowledgments. It is not time to the substantive answer or tool completion. Missing measurements are not treated as zero.','','## Reliability and measurement coverage','']
 for m in models:
  arm=m['arm']['key'];rs=m['reliability'];l=m['latency'];missing=sum(len(r['latency_missing_or_unrepresented']) for r in m['runs']);nonpositive=sum(len(r['nonpositive_latencies']) for r in m['runs']);reasons=Counter(r['failure_reason'] for r in rs if r['failure_reason']);api=sum(bool(r['evidence']['api_error']) or bool(r.get('live_api_errors')) for r in rs);quota=sum(bool(r['evidence']['rate_limit']) for r in rs)
  lines.append(f"- **{arm}:** {m['complete']}/25 complete; failure reasons {dict(reasons) or 'none'}; explicit API-error evidence in {api} attempts; rate-limit evidence in {quota} attempts. {l['all']['n']} positive V2V measurements, {missing} missing/unrepresented planned turns, {nonpositive} nonpositive measurements. Strict 95% conversation-bootstrap interval {m['strict_ci95'][0]:.1f}–{m['strict_ci95'][1]:.1f}%.")
  if arm=='live':lines.append(f"  Confirmed closed sessions: {sum(r.get('session_closed',False) for r in rs)}/25. Waveform alignment checks passed for {sum(r['processing'].get('alignment') is True for r in m['runs'])}/25 attempts.")
 lines+=['','## Judge audit','',f'{len(flags)} explicit score/explanation conflict flags; raw outputs retained without manual overrides. See `aggregate.json` for every flag and failed turn. Retries are limited to judge execution/parse failures.','','## Cohort and provenance','','This cohort contains 25 fresh attempts with the corrected input-end gate, concurrency five, and no imported pilot conversations. All previous captures remain diagnostic and are excluded. The exact shared full speech-v2 prompt is used in both frontend and backend; the revised Opus 5 rubric is unchanged. Every capture freezes its implementation and prompt hashes.','','See [protocol](PROTOCOL.md), [frozen settings](protocol.json), [all attempts](capture/attempts.jsonl), [gate audit](live-gate-fix/audit.json), and [aggregate](aggregate.json). V2V uses canonical Silero speech detection and input/commit windows; waveform alignment is independently audited. Missing and signed measurements are retained explicitly. The text benchmark and prompt-optimized experiment are untouched.']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
