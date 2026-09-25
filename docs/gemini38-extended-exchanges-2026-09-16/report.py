"""Fixed-denominator report for a smoke run or the 25-attempt cohort."""
import json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def stat(xs):return dict(n=len(xs),mean_ms=statistics.mean(xs),median_ms=statistics.median(xs),max_ms=max(xs)) if xs else dict(n=0,mean_ms=None,median_ms=None,max_ms=None)
if len(sys.argv)>1:
 run=Path(sys.argv[1]);records=[dict(index=1,run_dir=str(run),full=len(rows(run/'transcript.jsonl'))==30)];out=HERE/'smoke-results'
else:
 out=HERE/'cohort';assert (out/'complete.json').exists();records=rows(out/'attempts.jsonl');assert len(records)==25
out.mkdir(exist_ok=True);denom=len(records)*30;strict=observed=interim=0;lat=[];completion=[];tool=[];nontool=[];flags=[];details=[];silent=[];failures=[];judging_incomplete=[]
for r in records:
 run=ROOT/r['run_dir'] if r.get('run_dir') else None
 judged=rows(run/'reconstructed/claude_judged.jsonl') if run else [];observed+=len(judged)
 if run and len(judged)!=len(rows(run/'transcript.jsonl')):judging_incomplete.append(r['index'])
 strict+=sum(all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding']) for t in judged)
 validation=json.loads((run/'exchange-validation/complete.json').read_text()) if run and (run/'exchange-validation/complete.json').exists() else {}
 if not validation.get('approved'):flags.append(r['index'])
 mapping=json.loads((run/'reconstructed/mapping.json').read_text()) if run and (run/'reconstructed/mapping.json').exists() else {};interim+=mapping.get('interim_phase_count',0)
 audio=json.loads((run/'exchange-validation/audio.json').read_text()) if run and (run/'exchange-validation/audio.json').exists() else {}
 for t in audio.get('turns',[]):
  v=t['wav_v2v_ms'];lat.append(v);completion.append(t['bot_silero_end_ms']-t['user_end_ms'])
  if v>0:(tool if t['has_tool_call'] else nontool).append(v)
 silent.extend(dict(attempt=r['index'],turn=t) for t in validation.get('silent_closing_turns',[]))
 runtime=json.loads((run/'runtime.json').read_text()) if run and (run/'runtime.json').exists() else {}
 if runtime.get('failure'):failures.append(dict(attempt=r['index'],failure=runtime['failure']))
 details.append(dict(attempt=r,validation=validation,strict_pass=sum(all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding']) for t in judged),interim_phases=mapping.get('interim_phase_count'),failures=[t for t in judged if not all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding'])]))
result=dict(model='gemini-3.8-live-extended-thinking',thinking='medium',method='gemini-idle-exchanges-v1',attempts=len(records),complete=sum(r['full'] for r in records),denominator=denom,judged_turns=observed,strict_pass=strict,strict_percent=100*strict/denom if not judging_incomplete else None,judging_incomplete_attempts=judging_incomplete,validation_flagged_attempts=flags,failure_records=failures,interim_phases=interim,first_speech_signed=stat(lat),first_speech_positive=stat([v for v in lat if v>0]),non_tool_positive=stat(nontool),tool_positive=stat(tool),speech_completion=stat(completion),missing_measurements=denom-len(lat),nonpositive_measurements=sum(v<=0 for v in lat),silent_closing_turns=silent,substantive_answer_latency=None,runs=details)
(out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
f=result['first_speech_positive'];end=result['speech_completion']
score_text=f"**{strict}/{denom} ({result['strict_percent']:.1f}%)**" if result['strict_percent'] is not None else '**unavailable: recorded exchanges still need valid judgments**'
lines=['# Gemini 3.8 Extended Thinking medium: exchange audit','',f"Completed {result['complete']}/{len(records)} attempted conversations. Opus 5 strict spoken-answer score: {score_text}; {observed} turns judged after reconstruction.",'',f"Validation-flagged attempts: {flags or 'none'}. Provider IN_PROGRESS phase boundaries observed: {interim}. All planned turns remain in the score denominator.",'',f"First-speech V2V: mean {f['mean_ms']}ms, median {f['median_ms']}ms, maximum {f['max_ms']}ms ({f['n']} positive measurements). Missing measurements: {denom-len(lat)}; nonpositive: {result['nonpositive_measurements']}.",'',f"User-speech end to final spoken-audio end: mean {end['mean_ms']}ms, median {end['median_ms']}ms. This is speech completion, not task correctness or substantive-answer onset.",'','First speech includes interim acknowledgments. Substantive-answer onset is not claimed: provider receipt timestamps and IDLE timestamps are not audio word alignment. Silent end_session tool-only turns and other measurement gaps remain explicit in the aggregate.','','Shared full speech-v2 prompt and revised Opus 5 quality rubric are unchanged. The adapter uses provider IDLE and playback drain, not an external completion classifier. All output-transcription fragments, including interim speech, are retained; thought text is not substituted as speech. Original transcripts and audio are preserved.','','[Protocol](../PROTOCOL.md) · [Full aggregate](aggregate.json)']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
