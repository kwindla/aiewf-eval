"""Finalize the standard-prompt full validation and five-attempt load pilot."""
import json, re, hashlib, sys, os
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
REJUDGED='--rejudged' in sys.argv
REVISION=Path('docs/speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1')
DEST=HERE/'rejudged' if REJUDGED else HERE
DEST.mkdir(exist_ok=True)
revision=json.loads((REVISION/'protocol.json').read_text()) if REJUDGED else None
if REJUDGED:assert (REVISION/'complete.json').exists()
protocol=json.loads(Path('docs/speech-v2-canonical-2026-09-15/protocol.json').read_text())
first=re.search(r'^Output directory: (.+)$',(HERE/'standard-prompt-full-validation.log').read_text(),re.M).group(1)
attempts=[json.loads(l) for l in (HERE/'concurrency-pilot/attempts.jsonl').read_text().splitlines()]
assert len(attempts)==5 and {a['index'] for a in attempts}==set(range(1,6))
for a in attempts:
 p=json.loads((HERE/f"concurrency-pilot/processed-{a['index']}.json").read_text())
 assert p['judge_exit']==p['audio_exit']==p['alignment_exit']==0,p
runs=[('Single validation',Path(first))]+[(f"Concurrent {a['index']}",Path(a['run_dir'])) for a in sorted(attempts,key=lambda a:a['index'])]
rows=[]
for label,run in runs:
 score_path=REVISION/('live-single' if label=='Single validation' else f"live-{int(label.split()[-1]):02d}") if REJUDGED else run
 runtime=json.loads((run/'runtime.json').read_text());judge=json.loads((score_path/'claude_summary.json').read_text());audio=json.loads((run/'live-audio-metrics.json').read_text());alignment=json.loads((run/'audio-alignment-audit.json').read_text());meta=json.loads((run/'benchmark-provenance.json').read_text())
 events=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()]
 assert meta['frontend-prompt_sha256']==meta['system-prompt_sha256']==protocol['prompt_sha256']
 assert meta['judge-prompt_sha256']==protocol['judge_prompt_sha256']
 assert judge['judge_prompt_sha256']==(revision or protocol)['judge_prompt_sha256']
 started=next(e['event']['session'] for e in events if e.get('event',{}).get('type')=='session.started')
 assert hashlib.sha256(started['instructions'].encode()).hexdigest()==protocol['prompt_sha256']
 assert hashlib.sha256(started['delegation']['responses']['instructions'].encode()).hexdigest()==protocol['prompt_sha256']
 api_errors=[e for e in events if e.get('event',{}).get('type')=='error' or e['type']=='pipeline_error']
 closed=any(e.get('event',{}).get('type')=='session.closed' for e in events)
 bad=[]
 for l in (score_path/'claude_judged.jsonl').read_text().splitlines():
  d=json.loads(l)
  if not all(d['scores'][k] for k in ['tool_use_correct','instruction_following','kb_grounding']):bad.append({'turn':d['turn'],'scores':d['scores'],'reason':d['claude_reasoning']})
 rows.append(dict(label=label,judge_dir=str(score_path),run_dir=str(run),runtime=runtime,judge=judge,audio=audio,alignment_issues=alignment['issues'],api_errors=api_errors,session_closed=closed,failed_turns=bad))
def stats(values):
 return dict(n=len(values),mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p90_ms=float(np.percentile(values,90)),min_ms=float(min(values)),max_ms=float(max(values))) if values else {'n':0}
pilot=rows[1:];latencies=[t['wav_v2v_ms'] for r in pilot for t in r['audio']['turns']]
summary=dict(attempts=5,complete=sum(r['runtime']['valid'] and r['runtime']['turns']==30 for r in pilot),planned_turns=150,strict_passes=sum(r['judge']['turn_pass']['count'] for r in pilot),api_errors=sum(len(r['api_errors']) for r in pilot),closed_sessions=sum(r['session_closed'] for r in pilot),v2v=stats(latencies),tool_v2v=stats([t['wav_v2v_ms'] for r in pilot for t in r['audio']['turns'] if t['has_tool_call'] and t['wav_v2v_ms']>0]),non_tool_v2v=stats([t['wav_v2v_ms'] for r in pilot for t in r['audio']['turns'] if not t['has_tool_call'] and t['wav_v2v_ms']>0]),overlap_turns=sum(r['audio']['summary']['overlap_turns'] for r in pilot),boundary_issues=sum(len(r['audio']['issues']) for r in pilot),alignment_issues=sum(len(r['alignment_issues']) for r in pilot))
summary['strict_percent']=100*summary['strict_passes']/150
(DEST/'standard-validation-results.json').write_text(json.dumps(dict(pilot=summary,runs=rows),indent=2))
lines=['# GPT-Live standard-prompt validation','', '**GPT-Live-1 + GPT-5.6-Terra, medium reasoning; Pipecat 1.10.0 / OpenAI SDK 3.14.0.**', '', 'Frontend and backend both received the full, byte-identical shared speech-v2 prompt and knowledge base. All six runs used the unchanged Opus 5 rubric, original audio, and original tool schemas. No recovery nudges, prompt adaptations, or replacement attempts.', '', '| Run | Turns | Strict accuracy | Mean V2V | Median V2V | p90 V2V |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 s=r['audio']['summary']['all_signed_v2v'];score=r['judge']['turn_pass']['count'];link=os.path.relpath(Path(r['judge_dir'])/'claude_analysis.md',DEST)
 lines.append(f"| [{r['label']}]({link}) | {r['runtime']['turns']}/30 | {score}/30 ({100*score/30:.1f}%) | {s['mean_ms']/1000:.2f} s | {s['median_ms']/1000:.2f} s | {s['p90_ms']/1000:.2f} s |")
lines += ['', '## Five-conversation concurrency pilot', '', f"Completed {summary['complete']}/5 attempts; {summary['strict_passes']}/150 strict passes ({summary['strict_percent']:.1f}%). All planned turns are in the denominator. API/pipeline errors: {summary['api_errors']}; confirmed closed sessions: {summary['closed_sessions']}/5.", '', f"Pooled V2V: mean {summary['v2v']['mean_ms']/1000:.2f} s, median {summary['v2v']['median_ms']/1000:.2f} s, p90 {summary['v2v']['p90_ms']/1000:.2f} s. Speech-overlap turns: {summary['overlap_turns']}; turn-boundary issues: {summary['boundary_issues']}; waveform alignment issues: {summary['alignment_issues']}.", '', 'V2V is measured from actual user-speech end to first actual assistant speech in the stereo WAV, including acknowledgements. The detector/thresholds are shared with the canonical benchmark; Live turn association uses recorded input/commit windows because the API emits continuous silence audio. These figures do not measure time to a substantive answer or tool completion.', '', 'The five Live sessions overlapped the tail of the existing OpenAI Realtime campaign; see `concurrency-pilot/background-load.json`. This is evidence for N=5 under the tested load, not a guarantee about sustained quotas or a canonical 25-attempt score.', '', '## Errors and provenance', '']
for r in rows:
 for fail in r['failed_turns']:lines.append(f"- {r['label']}, turn {fail['turn']}: {fail['reason']}")
lines += ['', 'All earlier custom-frontend probes are diagnostic only and excluded in full, including the silent name-turn failure and the interrupted unauthorized prompt revision. The user required the same frontend prompt for every model; all results in this report comply. The other five models retained their frozen prompt/rubric throughout.', '', 'See `standard-validation-results.json` for per-turn audio metrics, complete judge summaries, error evidence, and every run path. Prompt improvements belong in model-specific README notes and were not applied to these benchmark captures.']
if REJUDGED:
 lines=[l.replace('All six runs used the unchanged Opus 5 rubric, original audio, and original tool schemas.', 'All six saved conversations were rejudged with the same clarified Opus 5 speech rubric used for the other 125 attempts; original audio and tool schemas are unchanged.').replace('The other five models retained their frozen prompt/rubric throughout.', 'The other five models retained their frozen model prompts throughout and received the identical judge revision.') for l in lines]
 lines.insert(2, 'Judge revision: `speech-v2-policy-compliance-1`. Original judgments remain in the capture directories; revised judgments and the full before/after audit are in [the shared revision directory](../../speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/AUDIT.md).')
(DEST/'VALIDATION.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
