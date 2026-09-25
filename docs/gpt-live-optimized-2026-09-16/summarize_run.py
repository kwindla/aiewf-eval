"""Describe validated pilot/experiment without conflating backend and spoken results."""
import json,statistics,sys
from pathlib import Path
run=Path(sys.argv[1]);out=run/'experimental-validation';out.mkdir(exist_ok=True)
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
events=rows(run/'live-events.jsonl');judged=rows(run/'reconstructed/claude_judged.jsonl');transcript=rows(run/'transcript.jsonl');audio=json.loads((run/'live-audio-metrics.json').read_text()) if (run/'live-audio-metrics.json').exists() else {};provenance=json.loads((run/'benchmark-provenance.json').read_text());checks=json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else {}
delays=[];delegations={};coordinator=[]
for e in events:
 if e['type']=='coordinator_result':coordinator.append(e)
 v=e.get('event',{})
 if v.get('type')=='session.delegation.created':delegations[v['delegation']['id']]=dict(turn=e['turn'],start=e['t'],first_text=None,completed=None)
 if v.get('type')=='response.event' and v.get('delegation_id') in delegations:
  d=delegations[v['delegation_id']];inner=v['event']
  if inner['type']=='response.output_text.delta' and inner.get('delta') and d['first_text'] is None:d['first_text']=e['t']
  if inner['type']=='response.completed':d['completed']=e['t']
for key,d in delegations.items():delays.append(dict(delegation_id=key,turn=d['turn'],first_backend_text_ms=(d['first_text']-d['start'])*1000 if d['first_text'] is not None else None,backend_last_completion_ms=(d['completed']-d['start'])*1000 if d['completed'] is not None else None))
strict=sum(all(t['scores'].get(k) is True for k in ['tool_use_correct','instruction_following','kb_grounding']) for t in judged)
result=dict(label=provenance['label'],run=str(run),planned_turns=len(provenance['turns']),recorded_turns=len(transcript),judged_turns=len(judged),strict_pass=strict,validation=checks,audio_summary=audio.get('summary'),backend_latencies=delays,coordinator_checks=len(coordinator),coordinator_progress_holds=sum(not e['decision']['ready'] for e in coordinator),coordinator_mean_overhead_ms=statistics.mean(e['elapsed_ms'] for e in coordinator) if coordinator else None,substantive_latency_status='Not available as validated measurement. Independent-ASR sample timestamps require audio alignment; do not substitute first-speech or backend completion latency.',listening_status='Recordings available; no human listening audit claimed.')
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# GPT-Live prompt-optimized: '+provenance['label'],'',f"Recorded {len(transcript)}/{len(provenance['turns'])} planned exchanges. Opus 5 strict quality: {strict}/{len(provenance['turns'])}; {len(judged)} exchanges judged after reconstruction.",'','This is an experimental protocol with a separate exchange coordinator. It is not a prompt-only comparison against the standard timing-only cohort. Backend text never substitutes for spoken answers.','', '## Validation','', '```json',json.dumps(checks,indent=2),'```','','## First-speech metrics','', 'Includes interim speech; does not measure time to the substantive answer.','', '```json',json.dumps(audio.get('summary'),indent=2),'```','','Substantive-response latency remains unvalidated. Independent ASR can aid review, but sample word timestamps include a timing contradiction; they are not published as answer latency.','','## Artifacts','','- [Stereo conversation](../conversation.wav): user left, assistant right.','- [Reconstructed exchanges](../reconstructed/transcript.jsonl) and [mapping](../reconstructed/mapping.json).','- [Separate backend messages and actual tools](../reconstructed/backend-diagnostics.json).','- [Detailed results and backend timing](summary.json).','- [Independent audio review](../audio-review/): timestamped ASR and channel clips, when generated.','','Coordinator inference overhead is harness time, reported separately and excluded from first-speech V2V. Audio turn-taking is evaluated quantitatively; judge default turn-taking Booleans are not published as audio success.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
