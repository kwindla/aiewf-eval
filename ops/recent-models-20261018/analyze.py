"""Audit all grades, report completion and both fixed/conditional quality."""
import asyncio, hashlib, json, math, statistics, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np
import judge_campaign as judge
from preflight import HERE,ROOT,cfg,collect,dump
from verify_capture_sources import captured_sources
LABELS={('luna','none'):'gpt-6-luna (none)',('luna','low'):'gpt-6-luna (low)',('sonnet','between-tools'):'claude-sonnet-5-5 (no up-front thinking, low)',('sonnet','low'):'claude-sonnet-5-5 (low)',('haiku','off'):'claude-haiku-5-5 (thinking off, low)',('haiku','low'):'claude-haiku-5-5 (low)',('spark13','minimal'):'muse-spark-1.3 (minimal)',('spark13','low'):'muse-spark-1.3 (low)',('spark-open','low'):'muse-spark-open (low, NVFP4)',('grok-voice','none'):'Grok Voice 3.0 (reasoning off)'}
PROVIDERS={'luna':'OpenAI','sonnet':'Anthropic','haiku':'Anthropic','spark13':'OpenRouter / Meta','spark-open':'Baseten','grok-voice':'xAI'}
def metrics(values):
 return {'n':len(values),'mean_ms':statistics.mean(values) if values else None,'median_ms':statistics.median(values) if values else None,'p50_ms':statistics.median(values) if values else None,'p95_ms':float(np.percentile(values,95,method='linear')) if values else None,'max_ms':max(values) if values else None}
def finite(v):return type(v) in (float,int) and math.isfinite(v)
def ci(values):
 if not values:return None
 rng=np.random.default_rng(20261018);s=rng.choice(values,size=(20000,len(values)),replace=True).mean(axis=1)/30*100
 return list(map(float,np.percentile(s,[2.5,97.5])))
def latency(records):
 eligible=[r for r in records if r.get('recovery_turn') or r['turn']!=0];v=[r.get('ttfb_ms') for r in eligible]
 return {'positive':metrics([x for x in v if finite(x) and x>0]),'signed':metrics([x for x in v if finite(x)]),'missing_n':sum(not finite(x) for x in v),'nonpositive_n':sum(finite(x) and x<=0 for x in v),'eligible_response_n':len(v),'raw_ttft':metrics([r['raw_ttfb_ms'] for r in eligible if finite(r.get('raw_ttfb_ms')) and r['raw_ttfb_ms']>0])}
def score(cases,grades):
 counts=Counter();strict=0;per=[]
 for c in cases:
  r=grades.get(c['job_id'])
  if r is None:continue
  p=0
  for j in r['judgments'].values():
   for dim in judge.DIMS:counts[dim]+=j['scores'][dim]
   p+=all(j['scores'][d] for d in judge.DIMS)
  strict+=p;per.append(p)
 denominator=30*len(cases);known=len(per)==len(cases) and denominator>0
 return {'strict_passes':strict,'dimension_passes':dict(counts),'denominator':denominator,'strict_percent':100*strict/denominator if known else None,'strict_ci95':ci(per) if known else None,'dimension_error_percent':{d:100*(denominator-counts[d])/denominator if known else None for d in judge.DIMS}}
def pct(x):return '—' if x is None else f'{x:.1f}%'
def ms(x):return '—' if x is None else f'{x:.0f}ms'
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join('---' for _ in header)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 assert (HERE/'judging/COMPLETE.json').exists()
 assert (HERE/'COLLECTION-COMPLETE.json').exists()
 with captured_sources(collect,HERE) as resolutions:cases=judge.freeze()
 dump(HERE/'capture-source-resolution.json',{'all_captured_sources_verified':True,'archived_versions_used':resolutions})
 grades={};raws={};complete=set();cost=0
 for c in cases:
  raw=(ROOT/c['source']).read_bytes();assert judge.sha(raw)==c['transcript_sha256'];records=[json.loads(s) for s in raw.splitlines() if s.strip()];raws[c['job_id']]=records
  scheduled=[r for r in records if not r.get('recovery_turn')]
  if len(scheduled)==30 and all(r.get('assistant_text','').strip() or r.get('tool_calls') for r in scheduled):complete.add(c['job_id'])
  if judge.valid(c):
   grade=json.loads(judge.result_path(c).read_text());grades[c['job_id']]=grade;cost+=grade['metadata'].get('judge_cost_usd',0) or 0
  else:assert judge.blocked_path(c).exists()
 models=[]
 for key,label in LABELS.items():
  name,arm=key;selected=[c for c in cases if (c['name'],c['arm'])==key];done=[c for c in selected if c['job_id'] in complete]
  all_score=score(selected,grades);completed_score=score(done,grades)
  all_records=[r for c in selected for r in raws[c['job_id']]];done_records=[r for c in done for r in raws[c['job_id']]]
  m={'key':name+'-'+arm,'name':name,'arm':arm,'label':label,'provider':PROVIDERS[name],'assigned':len(selected),'complete':len(done),'incomplete':len(selected)-len(done),'shortfall':30-len(done) if name!='grok-voice' else 25-len(done),'all_assigned':all_score,'complete_only':completed_score,'strict_percent':completed_score['strict_percent'],'strict_ci95':completed_score['strict_ci95'],'dimension_error_percent':completed_score['dimension_error_percent'],'latency':latency(done_records),'all_attempt_latency':latency(all_records),'judge_blocked':[c['job_id'] for c in selected if c['job_id'] not in grades],'cases':[c['job_id'] for c in selected]}
  logs=[((ROOT/c['source']).parent/'run.log').read_text() for c in selected]
  m['observed_scripted_turns']=sum(len(c['observed_turns']) for c in selected)
  m['missing_future_turns']=30*len(selected)-m['observed_scripted_turns']
  m['early_end_session_turns']=dict(Counter(r['turn'] for r in all_records if not r.get('recovery_turn') and r['turn']<29 and any(t.get('name')=='end_session' for t in r.get('tool_calls',[]))))
  m['generation_credit_retries']=sum(t.count('Anthropic credit balance error before stream; retry in') for t in logs)
  m['generation_prestream_429_retries']=sum(t.count('Baseten HTTP 429 before stream: retry') for t in logs)
  if name=='grok-voice':
   from voice_waveform import inputs as verify_audio_inputs, audit as audit_input_waveform
   verify_audio_inputs()
   attempts=json.loads((HERE/name/'attempts.json').read_text());failed=sum(not a['complete'] for a in attempts)
   m.update(eligible_attempts=len(attempts),failed_conversations=failed,failure_rate_percent=100*failed/len(attempts))
   waveform=[];pads=[];tool=[];non_tool=[];signed=[];audio_inputs=[];tts=[];taking_ok=True
   for c in done:
    path=HERE/name/'audio'/(c['job_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    run=(ROOT/c['source']).parent
    if not path.exists():
     r=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_turn_metrics.py'),str(run),'--json'],cwd=ROOT,capture_output=True,text=True,timeout=180)
     if r.returncode:raise RuntimeError(r.stderr[-2000:])
     dump(path,json.loads(r.stdout))
    data=json.loads(path.read_text());rows=data['turns'];assert len(rows)==30
    if not data.get('alignment',{}).get('alignment_ok'):
     raise RuntimeError(f'Hold complete voice capture for alignment review: {c["job_id"]}')
    waveform_path=path.with_name(path.stem+'-input-waveform.json')
    if not waveform_path.exists():dump(waveform_path,audit_input_waveform(run))
    if json.loads(waveform_path.read_text())['issues']:
     raise RuntimeError(f'Hold complete voice capture for input waveform review: {c["job_id"]}')
    audio_inputs.append({'job':c['job_id'],'source_wav':str((run/'conversation.wav').relative_to(ROOT)),'wav_sha256':hashlib.file_digest((run/'conversation.wav').open('rb'),'sha256').hexdigest(),'metrics':str(path.relative_to(ROOT)),'metrics_sha256':judge.sha(path.read_bytes()),'input_waveform_audit':str(waveform_path.relative_to(ROOT)),'input_waveform_audit_sha256':judge.sha(waveform_path.read_bytes())})
    for r in rows:
     v=r.get('wav_v2v_ms')
     if finite(v):
      signed.append(v)
      if v>0:waveform.append(v);(tool if r.get('has_tool_call') else non_tool).append(v)
     pad=r.get('silent_pad_silero_ms')
     if finite(pad) and pad>=0:pads.append(pad)
    ta=grades.get(c['job_id'],{}).get('turn_taking_analysis')
    if not ta or ta.get('error'):taking_ok=False
    else:tts.extend(j['scores'].get('turn_taking') for j in grades[c['job_id']]['judgments'].values())
   m.update(first_speech_positive=metrics(waveform),first_speech_signed=metrics(signed),tool=metrics(tool),non_tool=metrics(non_tool),silence=metrics(pads),nonpositive_n=sum(v<=0 for v in signed),missing_latency_n=30*len(done)-len(signed),turn_taking_scored=taking_ok,turn_taking_passes=sum(v is True for v in tts) if taking_ok else None,audio_inputs=audio_inputs)
  models.append(m)
 events=[json.loads(line) for line in (HERE/'judging/attempts.jsonl').read_text().splitlines() if line.strip()]
 data={'at':collect.utc_now(),'judge_model':judge.MODEL,'judge_effort':judge.EFFORT,'text_models':[m for m in models if m['name']!='grok-voice'],'voice_models':[m for m in models if m['name']=='grok-voice'],'all_quality_complete':len(grades)==len(cases),'judged_conversations':len(grades),'successful_judge_cost_usd':cost,'judge_credit_cooldowns':sum(e['event']=='credit_cooldown' for e in events),'cases':cases}
 dump(HERE/'aggregate.json',data)
 rows=[[m['label'],str(m['complete'])+'/'+str(m['assigned']),pct(m['all_assigned']['strict_percent']),pct(m['strict_percent']),ms(m['latency']['positive']['p50_ms']),ms(m['latency']['positive']['p95_ms'])] for m in data['text_models']]
 voice=data['voice_models'][0]
 report='# Recent-model benchmark results\n\n'+table(['Configuration','Complete / assigned','Fixed-denominator pass','Completed-only pass','Completed TTFAT P50','P95'],rows)+'\n\n'
 report+='The fixed-denominator column retains early-ending conversations and counts missing future turns as failures. Completed-only quality and latency match the current README convention. Smoke runs are excluded; all attempts and original captures remain available. See [protocol](PROTOCOL.md) and [aggregate](aggregate.json).\n\n'
 report+='The [mode settings](MODE-SETTINGS.md) document the explicit controls and the Sonnet 5.5 SDK-default caveat.\n\n'
 report+='Capture-time sources are verified against their unchanged manifests. When current library code evolves after capture, the original bytes must match the pre-capture runtime snapshot; see [source verification](capture-source-resolution.json).\n\n'
 observed=[[m['label'],m['observed_scripted_turns'],m['missing_future_turns'],ms(m['all_attempt_latency']['positive']['p50_ms']),ms(m['all_attempt_latency']['positive']['p95_ms']),m['all_attempt_latency']['positive']['n']] for m in data['text_models']]
 report+='Observed timing across all assigned conversations, including incomplete prefixes. These timings use the observed responses only; they do not imply the missing future turns were answered. As in the main table, scripted turn zero is excluded and later recovery responses are included.\n\n'
 report+=table(['Configuration','Observed scripted turns','Missing future turns','Observed TTFAT P50','P95','Timing N'],observed)+'\n\n'
 report+='The [Luna completion audit](luna/COMPLETION.md) inventories every premature end_session and the excluded streaming check. No tool-name translation defect was found in that diagnostic.\n\n'
 report+=f"Grok Voice 3.0 off: {voice['complete']} complete conversations; {voice['failed_conversations']}/{voice['eligible_attempts']} unsuccessful attempts; {pct(voice['strict_percent'])} strict pass. First-speech P50 {ms(voice['first_speech_positive']['p50_ms'])}, P95 {ms(voice['first_speech_positive']['p95_ms'])}.\n\n"
 if voice['failed_conversations']:report+='See the [incomplete-session review](grok-voice/STALL-REVIEW.md) and [all voice attempt records](grok-voice/attempts.json) for operational failures excluded from the completed-conversation quality cohort.\n\n'
 report+=f"All grades use Opus 5.5 Medium, the existing text or speech rubric, and unchanged task prompts. Successful judge request cost: ${cost:.6f}. Provider-blocked cases: {len(cases)-len(grades)}.\n"
 report+=f"\nCredit reload retries: {sum(m['generation_credit_retries'] for m in models)} during generation; {data['judge_credit_cooldowns']} judging cooldown events. Every credit retry waits at least 90 seconds. Generation credit waits remain included in affected latency observations.\n"
 (HERE/'REPORT.md').write_text(report)
 print(json.dumps({'configurations':len(models),'conversations':len(cases),'graded':len(grades),'judge_cost_usd':cost}))
if __name__=='__main__':main()
