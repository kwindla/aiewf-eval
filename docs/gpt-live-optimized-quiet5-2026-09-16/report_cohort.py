"""Fixed denominator experimental report; never substitute backend for spoken score."""
import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'cohort'
assert (OUT/'complete.json').exists()
records=[json.loads(l) for l in (OUT/'attempts.jsonl').read_text().splitlines()];assert len(records)==25
strict=judged_count=0;latencies=[];missing=[];invalid=[];api=[];runs=[]
for r in sorted(records,key=lambda r:r['index']):
 run=ROOT/r['run_dir'] if r['run_dir'] else None;score=run/'reconstructed/claude_judged.jsonl' if run else None
 judged=[json.loads(l) for l in score.read_text().splitlines()] if score and score.exists() else []
 assert len({t['turn'] for t in judged})==len(judged)
 strict+=sum(all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding']) for t in judged);judged_count+=len(judged)
 validation=json.loads((run/'experimental-validation/complete.json').read_text()) if run and (run/'experimental-validation/complete.json').exists() else {}
 if not validation.get('approved'):invalid.append(r['index'])
 if len(judged)!=r['turns']:missing.append(r['index'])
 audio=json.loads((run/'live-audio-metrics.json').read_text()) if run and (run/'live-audio-metrics.json').exists() else {}
 signed=[t['wav_v2v_ms'] for t in audio.get('turns',[])];latencies+=signed
 events=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()] if run and (run/'live-events.jsonl').exists() else []
 errors=[e for e in events if e.get('event',{}).get('type')=='error' or e['type']=='pipeline_error']
 if errors:api.append(dict(attempt=r['index'],errors=errors))
 runs.append(dict(attempt=r,validation=validation,audio_issues=audio.get('issues',[]),signed_latencies=signed))
positive=[v for v in latencies if v>0]
result=dict(label='GPT-Live prompt-optimized',method='live-playback-exchanges-v2',attempts=25,complete=sum(r['full'] for r in records),planned_turns=750,judged_turns=judged_count,strict_pass=strict,strict_percent=strict/7.5 if not missing else None,unjudged_attempts=missing,validation_flagged_attempts=invalid,explicit_error_attempts=api,positive_v2v_n=len(positive),signed_v2v_n=len(latencies),first_speech_mean_ms=statistics.mean(positive) if positive else None,first_speech_median_ms=statistics.median(positive) if positive else None,substantive_response_latency=None,runs=runs)
(OUT/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
score_text=f'{strict}/750 ({strict/7.5:.1f}%)' if not missing else 'unavailable: recorded exchanges remain unjudged'
lines=['# GPT-Live prompt-optimized: 25-attempt experimental cohort','',f"Completed {result['complete']}/25 conversations; {judged_count}/750 planned turns judged. Strict spoken-answer pass: {score_text}.",'',f"Validation-flagged attempts: {invalid or 'none'}. Unjudged recorded-turn attempts: {missing or 'none'}. Retain flags and resolve before treating results as validated.",'',f"First-speech V2V (acknowledgments included): mean {result['first_speech_mean_ms']}ms, median {result['first_speech_median_ms']}ms across {len(positive)} positive observations; {len(latencies)-len(positive)} nonpositive measurements retained separately.",'','Substantive-answer latency is not validated and is not represented by these first-speech figures. The pilot ASR word timing exposed an inconsistency; do not substitute transcript receipt or backend completion time.','','Scores use reconstructed spoken exchanges and the shared Opus 5 quality rubric. Backend messages are diagnostic only. Audio interaction is measured separately; judge default turn-taking Booleans do not establish success. Missing planned turns remain failures; recordings and failures are not replaced.','','This experiment uses optimized prompts and the standard Live event/playback gate, with no semantic completion classifier. Pilot conversations are excluded.','','See [protocol](protocol.json), [attempts](attempts.jsonl), [detailed aggregate](aggregate.json), and the [experiment protocol](../PROTOCOL.md).']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
