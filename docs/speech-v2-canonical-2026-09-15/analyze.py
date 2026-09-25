"""Aggregate a frozen attempt-based voice cohort; never select successful runs."""
import hashlib,json,math,statistics,sys
from collections import Counter
from pathlib import Path
import numpy as np
from audit_reliability import audit
ROOT=Path.cwd();HERE=ROOT/'docs/speech-v2-canonical-2026-09-15'
REJUDGED='--rejudged' in sys.argv
REVISION=HERE/'rejudge-policy-compliance-1'
DEST=REVISION if REJUDGED else HERE
DIMS=['tool_use_correct','instruction_following','kb_grounding','turn_taking']
def stats(xs):
 if not xs:return {'n':0,'p50_ms':None,'p95_ms':None,'max_ms':None,'mean_ms':None}
 xs=sorted(xs);return {'n':len(xs),'p50_ms':statistics.median(xs),'p95_ms':xs[math.ceil(.95*len(xs))-1],'max_ms':max(xs),'mean_ms':statistics.mean(xs)}
def main():
 frozen=json.loads((HERE/'protocol.json').read_text());records=[json.loads(x) for x in (HERE/'capture/attempts.jsonl').read_text().splitlines()]
 assert len(records)==125 and len({(r['arm'],r['index']) for r in records})==125
 revision=json.loads((REVISION/'protocol.json').read_text()) if REJUDGED else None
 if REJUDGED:assert (REVISION/'complete.json').exists()
 models=[]
 for arm in frozen['arms']:
  selected=sorted((r for r in records if r['arm']==arm['key']),key=lambda r:r['index'])
  assert len(selected)==25 and {r['index'] for r in selected}==set(range(1,26))
  counts=Counter();per_run=[];reliability=[];per_turn=[Counter() for _ in range(30)];failure_examples=[];latency={'non_tool':[],'tool':[],'silence':[]};details=[]
  for r in selected:
   reliability.append(audit(r))
   processed=HERE/'processed'/f"{arm['key']}-{r['index']:02d}"
   done=json.loads((processed/'done.json').read_text());assert done['record']==r
   judged={}
   if r.get('run_dir') and r['turns']:
    path=ROOT/r['run_dir'];score_path=REVISION/f"{arm['key']}-{r['index']:02d}" if REJUDGED else path
    summary=json.loads((score_path/'claude_summary.json').read_text());assert summary['judge_prompt_sha256']==(revision or frozen)['judge_prompt_sha256']
    provenance=json.loads((path/'benchmark-provenance.json').read_text());assert provenance['prompt_sha256']==frozen['prompt_sha256']
    assert provenance['model']==arm['model']
    assert provenance['thinking']==arm.get('thinking')
    for line in (score_path/'claude_judged.jsonl').read_text().splitlines():
     x=json.loads(line)
     if not x.get('recovery_turn') and 0<=x['turn']<30:
      assert x['turn'] not in judged
      judged[x['turn']]=x
   passes=0
   for turn in range(30):
    scores=judged.get(turn,{}).get('scores',{})
    for dim in DIMS:
     counts[dim]+=scores.get(dim) is True
     per_turn[turn][dim]+=scores.get(dim) is True
    ok=all(scores.get(k) is True for k in DIMS[:3]);passes+=ok;per_turn[turn]['strict']+=ok
    per_turn[turn]['observed']+=turn in judged
    if not ok and turn in judged:failure_examples.append({'run_dir':r['run_dir'],'turn':turn,'reason':judged[turn].get('claude_reasoning'),'assistant_text':judged[turn].get('assistant_text'),'scores':scores})
   counts['strict']+=passes;per_run.append(passes)
   if done['audio']:
    metrics=json.loads((processed/'metrics.json').read_text())
    for t in metrics['turns']:
     v=t.get('wav_v2v_ms')
     if v is not None and v>0:latency['tool' if t.get('has_tool_call') else 'non_tool'].append(v)
     s=t.get('silent_pad_silero_ms')
     if s is not None:latency['silence'].append(s)
   details.append({'attempt':r,'strict_pass':passes,'judged_turns':len(judged)})
  rng=np.random.default_rng(20260915);boot=np.array(per_run)[rng.integers(0,25,size=(20000,25))].sum(axis=1)/750*100
  models.append({'arm':arm,'attempts':25,'complete':sum(r['full'] for r in selected),'denominator':750,'scores':dict(counts),'strict_percent':counts['strict']/750*100,'strict_ci95':np.quantile(boot,[.025,.975]).tolist(),'latency':{k:stats(v) for k,v in latency.items()},'runs':details,'reliability':reliability,'observed_turns':sum(d['judged_turns'] for d in details),'per_turn':[dict(x) for x in per_turn],'failure_examples':failure_examples})
 out={'protocol':frozen,'judging_revision':revision,'models':models};(DEST/'aggregate.json').write_text(json.dumps(out,indent=2)+'\n')
 def ms(x):return 'N/A' if x is None else f'{x:.0f}ms'
 header='| Model | Complete | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |'
 lines=[header,'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m in sorted(models,key=lambda m:-m['strict_percent']):
  s=m['scores'];l=m['latency'];name=m['arm']['model'];effort=m['arm'].get('reasoning') or m['arm'].get('thinking');name+=f' ({effort})' if effort else ''
  if m['complete']<25:name+='*'
  lines.append(f"| {name} | {m['complete']}/25 | {m['strict_percent']:.1f}% | {s['tool_use_correct']}/750 | {s['instruction_following']}/750 | {s['kb_grounding']}/750 | {s['turn_taking']}/750 | {ms(l['non_tool']['p50_ms'])} | {ms(l['non_tool']['max_ms'])} | {ms(l['tool']['mean_ms'])} | {ms(l['silence']['mean_ms'])} |")
 table='\n'.join(lines)+'\n';(DEST/'table.md').write_text(table)
 report='# Canonical speech-to-speech benchmark: voice policy v2\n\n'+table+'\n25 attempted conversations per model; every row uses the fixed 750-turn denominator. Missing future turns count as failures. Strict pass is the conjunction of tool use, instruction following, and KB grounding. Audio turn-taking is reported separately. V2V is conditional on measurable responses and includes the first scripted response, matching the historical speech table convention.\n\n'
 report+='## Uncertainty and latency coverage\n\n'
 for m in models:
  lo,hi=m['strict_ci95'];report+=f"- {m['arm']['model']}: strict 95% conversation-bootstrap interval {lo:.1f}–{hi:.1f}%; V2V observations {m['latency']['non_tool']['n']} non-tool / {m['latency']['tool']['n']} tool.\n"
 report+='\n## Reliability and observed-turn accuracy\n\n* marks a model with incomplete conversations. Failures in this corrected cohort are counted for every model under the same policy. The interrupted pre-fix cohort is excluded in full; its tool-delivery timeouts were harness failures, not API errors.\n\n'
 for m in models:
  reasons=Counter(r['failure_reason'] for r in m['reliability'] if r['failure_reason'])
  observed=m['observed_turns'];rate=m['scores']['strict']/observed*100 if observed else 0
  api=sum(bool(r['evidence']['api_error']) for r in m['reliability']);quota=sum(bool(r['evidence']['rate_limit']) for r in m['reliability'])
  report+=f"- {m['arm']['model']}: {m['complete']}/25 complete; failure reasons {dict(reasons) or 'none'}; API-error evidence in {api} attempts; quota-error evidence in {quota} attempts. Observed-turn strict accuracy: {m['scores']['strict']}/{observed} ({rate:.1f}%).\n"
 report+='\nHistorical reliability note: the [August investigation](../async-gemini-live-comparison-2026-08-04/root-cause-probes/README.md) reported 18/20 no-response-free Gemini 3.1 conversations and documented a WebSocket 1006 reconnect in an additional current attempt. Those historical observations are separate from the corrected cohort counts above.\n'
 report+='\n## Provenance\n\nSee [protocol](PROTOCOL.md), [frozen settings](protocol.json), [aggregate with all attempts](aggregate.json), and [capture ledger](capture/attempts.jsonl). These results supersede the speech-to-speech reference table only. The legacy text prompt, shared benchmark data, and README text-model scores were preserved.\n'
 if REJUDGED:
  report=report.replace('(../async-', '(../../async-').replace('(PROTOCOL.md)', '(../PROTOCOL.md)').replace('(protocol.json)', '(../protocol.json)').replace('(capture/attempts.jsonl)', '(../capture/attempts.jsonl)')
  report+='\n## Revised judging\n\nAll 131 saved conversations were rejudged uniformly with Opus 5 using `speech-v2-policy-compliance-1`. Model prompts, source judgments, recordings, audio metrics, and text scoring are unchanged. See [revision protocol](protocol.json), [before/after audit](AUDIT.md), and [original first-pass scores](../aggregate.json).\n'
 (DEST/'REPORT.md').write_text(report);print(table)
if __name__=='__main__':main()
