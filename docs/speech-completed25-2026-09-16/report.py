"""Completed-conversation accuracy plus attempt reliability; immutable originals."""
import hashlib,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
JUDGE_HASH='c5a8a3b3c3240172cc392529a1784b59d2913a32e2ec2ba843a456415b75ef27'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).exists() else []
def stats(xs):
 return dict(n=len(xs),mean_ms=statistics.mean(xs) if xs else None,median_ms=statistics.median(xs) if xs else None,max_ms=max(xs) if xs else None)
cohorts=[]
def cohort(key,label,entries):cohorts.append(dict(key=key,label=label,entries=entries))
def entry(attempt,score,audio,exclude=None):
 r=dict(attempt);return dict(attempt=r,score_dir=str(score),audio_path=str(audio),harness_exclusion=exclude)
original=ROOT/'docs/speech-v2-canonical-2026-09-15';revision=original/'rejudge-policy-compliance-1';audio_root=ROOT/'docs/speech-v2-live-medium-2026-09-15/audio-audit'
labels={'grok':'Grok Voice Think Fast','openai21':'GPT-Realtime-2.1 low','openai15':'GPT-Realtime-1.5','gemini31':'Gemini 3.1 minimal','gemini38':'Gemini 3.8 low'}
for m in read(revision/'aggregate.json')['models']:
 k=m['arm']['key'];es=[]
 for r in m['runs']:
  a=r['attempt'];tag=f"{k}-{a['index']:02d}";audio=audio_root/f"gemini38low-{a['index']:02d}.json" if k=='gemini38' else original/'processed'/tag/'metrics.json'
  es.append(entry(a,revision/tag,audio,'Startup greeting replaced first scripted exchange' if k=='gemini38' and a['index']==24 else None))
 cohort('gemini38low' if k=='gemini38' else k,labels[k],es)
extension=ROOT/'docs/speech-v2-live-medium-2026-09-15'
for m in read(extension/'aggregate.json')['models']:
 if m['arm']['key']!='gemini38medium':continue
 cohort('gemini38medium_old','Gemini 3.8 medium (earlier harness)',[entry(r['attempt'],ROOT/r['attempt']['run_dir'],audio_root/f"gemini38medium-{r['attempt']['index']:02d}.json") for r in m['runs']])
standard=ROOT/'docs/gpt-live-fixed-gate-n25-2026-09-15'
for m in read(standard/'aggregate.json')['models']:
 es=[]
 for r in m['runs']:
  a=r['attempt'];run=ROOT/a['run_dir'];ap=HERE/'standard-live-alignment-review.json' if a['index']==12 else run/'live-audio-metrics.json';es.append(entry(a,run,ap))
 cohort('live_standard','GPT-Live standard',es)
for key,label,folder in [('gemini38medium_new','Gemini 3.8 medium (interaction-aware)','gemini38-extended-exchanges-2026-09-16'),('live_optimized','GPT-Live prompt-optimized','gpt-live-optimized-quiet5-2026-09-16')]:
 es=[]
 for r in read(ROOT/'docs'/folder/'cohort/aggregate.json')['runs']:
  a=r['attempt'];run=ROOT/a['run_dir'];es.append(entry(a,run/'reconstructed',run/('exchange-validation/audio.json' if key.startswith('gemini') else 'live-audio-metrics.json')))
 cohort(key,label,es)
for r in rows(HERE/'attempts.jsonl'):
 c=next(c for c in cohorts if c['key']==r['arm']);run=ROOT/r['run_dir'] if r.get('run_dir') else None
 if run:
  score=run/'reconstructed' if r['arm'] in ['gemini38medium_new','live_optimized'] else run
  ap=run/('exchange-validation/audio.json' if r['arm']=='gemini38medium_new' else 'live-audio-metrics.json' if r['arm']=='live_optimized' else 'topup-validation/audio.json')
  e=entry(r,score,ap);e['topup']=True;c['entries'].append(e)
results=[]
for c in cohorts:
 eligible=[e for e in c['entries'] if not e['harness_exclusion']];complete=[e for e in eligible if e['attempt']['full']];failed=[e for e in eligible if not e['attempt']['full']]
 selected=complete[:25];assert len(complete)<=25,'Unexpected extra completed captures; inspect before selecting'
 counts={k:0 for k in ['tool_use_correct','instruction_following','kb_grounding','turn_taking','strict']};signed=[];non_tool=[];tool=[];silence=[];failures=[];missing=[]
 for e in selected:
  score=Path(e['score_dir']);summary=read(score/'claude_summary.json');assert summary['judge_prompt_sha256']==JUDGE_HASH and summary['judge_model']=='claude-opus-5'
  js=rows(score/'claude_judged.jsonl');assert len(js)==30 and {t['turn'] for t in js}==set(range(30))
  for t in js:
   for k in counts:
    if k!='strict':counts[k]+=t['scores'].get(k) is True
   counts['strict']+=all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding'])
  audio=read(e['audio_path']);seen=set()
  for t in audio['turns']:
   value=t.get('wav_v2v_ms');idx=t.get('turn_index',t.get('turn'));seen.add(idx) if value is not None else None
   if value is not None:
    signed.append(value)
    if value>0:(tool if t.get('has_tool_call') else non_tool).append(value)
   if t.get('silent_pad_silero_ms') is not None:silence.append(t['silent_pad_silero_ms'])
  if len(seen)<30:missing.append(dict(run_dir=e['attempt']['run_dir'],turn_indices=sorted(set(range(30))-seen)))
 for e in failed:
  a=e['attempt'];runtime=read(ROOT/a['run_dir']/'runtime.json') if a.get('run_dir') and (ROOT/a['run_dir']/'runtime.json').exists() else {};f=runtime.get('failure') or a.get('failure') or {};idx=f.get('turn');failures.append(dict(run_dir=a.get('run_dir'),completed_turns=a.get('turns'),reason=f.get('reason','capture_failure'),detail=f.get('detail'),turn_index=idx,turn_number=idx+1 if idx is not None else None))
 result=dict(key=c['key'],label=c['label'],complete=len(selected),accuracy_denominator=30*len(selected),scores=counts,strict_percent=counts['strict']/(30*len(selected))*100 if selected else None,eligible_attempts=len(eligible),failed_conversations=len(failed),failure_rate_percent=100*len(failed)/len(eligible),harness_excluded=sum(bool(e['harness_exclusion']) for e in c['entries']),failure_details=failures,first_speech_positive=stats([v for v in signed if v>0]),first_speech_signed=stats(signed),non_tool=stats(non_tool),tool=stats(tool),silence=stats(silence),nonpositive_n=sum(v<=0 for v in signed),missing_audio=missing,selected_runs=selected,all_attempts=c['entries'])
 results.append(result)
ready=all(r['complete']==25 for r in results)
(HERE/'aggregate.json').write_text(json.dumps(dict(policy='25 completed conversations; failures reported separately',ready=ready,models=results),indent=2)+'\n')
def fmt(x):return '—' if x is None else f'{x:.0f}'
lines=['# Speech benchmark: 25 completed conversations per cohort','',f"Status: {'complete' if ready else 'top-ups in progress'}. Original attempts and old fixed-denominator reports are preserved.",'','Accuracy below is conditional on completed conversations. Failed conversations—including all their partial turns—are excluded from accuracy and counted separately. Harness-invalid captures are separately excluded, never called provider failures. Stop at the first 25 valid completions without selecting on answer quality.','', '| Model / cohort | Scored conversations | Accuracy | Failed conversations | Harness excluded | First-speech mean / median | Positive latency coverage |','|---|---:|---:|---:|---:|---:|---:|']
for r in results:
 lines.append(f"| {r['label']} | {r['complete']} | {r['scores']['strict']}/{r['accuracy_denominator']} ({r['strict_percent']:.1f}%) | {r['failed_conversations']}/{r['eligible_attempts']} ({r['failure_rate_percent']:.1f}%) | {r['harness_excluded']} | {fmt(r['first_speech_positive']['mean_ms'])} / {fmt(r['first_speech_positive']['median_ms'])} ms | {r['first_speech_positive']['n']}/{r['accuracy_denominator']} |")
lines+=['','## Failed conversations','','Turn numbers below are **one-based**, out of 30; raw zero-based indices are retained in aggregate.json. A no-response timeout is not labeled an explicit API error.','']
for r in results:
 for f in r['failure_details']:lines.append(f"- **{r['label']}**: {f['reason']} on turn **{f['turn_number']}**, after {f['completed_turns']} completed turns. Run `{f['run_dir']}`.")
lines+=['','## Measurement and harness notes','','- Gemini 3.8 low: original attempt 24 replaced because a startup greeting occupied turn zero. The old capture is preserved as a harness exclusion. Replacement uses a startup guard; model settings and prompt are unchanged.','- GPT-Live standard: one recording has shifted input windows; matching three waveform landmarks per turn recovers input locations without changing any measured V2V latency. See standard-live-alignment-review.json.','- GPT-Live optimized: one short recorded answer was missed by canonical VAD. Keep its latency missing and the completed conversation eligible; see optimized-live-vad-review.json.','- First-speech metrics include interim acknowledgments. Negative and missing observations are retained in coverage, not silently converted to positive latencies. Substantive-answer latency is not established.','- Earlier and interaction-aware Gemini medium cohorts remain separate; they used different harness protocols. Failed diagnostic cohorts and smoke tests are not pooled into current reliability rates.','- Quality uses the same shared Opus 5 rubric and prompts as before. No score-driven recapture or rejudging. Existing empty closing-turn audio limitations remain disclosed in original reports.','']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Ready',ready,[(r['key'],r['complete'],r['failed_conversations']) for r in results])
