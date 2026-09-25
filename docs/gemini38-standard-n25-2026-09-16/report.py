"""Summarize preserved attempts without treating missing measurements as zero."""
import json, statistics, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def build():
 rows=[json.loads(l) for l in (HERE/'attempts.jsonl').read_text().splitlines()] if (HERE/'attempts.jsonl').exists() else []
 review=json.loads((HERE/'harness-review.json').read_text()) if (HERE/'harness-review.json').exists() else {}; excluded={r['index'] for r in review.get('excluded_attempts',[])}
 accepted=[r for r in rows if r['accepted']];failures=[r for r in rows if not r['full'] and r.get('failure') and r['index'] not in excluded]
 scores={k:0 for k in ['turn_taking','tool_use_correct','instruction_following','kb_grounding']};strict=0;metrics=[]
 for r in accepted:
  run=ROOT/r['run_dir'];s=json.loads((run/'claude_summary.json').read_text());assert s['turns_scored']==30
  strict+=s['turn_pass']['count']
  for k in scores:scores[k]+=s['claude_passes'][k]
  metrics.extend(json.loads((run/'topup-validation/audio.json').read_text())['turns'])
 def latency(ts):
  xs=[t['wav_v2v_ms'] for t in ts if t['wav_v2v_ms']>0]
  return dict(n=len(xs),mean_ms=statistics.mean(xs) if xs else None,median_ms=statistics.median(xs) if xs else None,max_ms=max(xs) if xs else None)
 a=dict(model='gemini-3.8-live',thinking=None,complete=len(accepted)==25,completed_conversations=len(accepted),attempts=len(rows),harness_excluded=len(excluded),eligible_attempts=len(rows)-len(excluded),failures=failures,strict_passes=strict,judged_turns=len(accepted)*30,scores=scores,latency=latency(metrics),tool_latency=latency([t for t in metrics if t['has_tool_call']]),non_tool_latency=latency([t for t in metrics if not t['has_tool_call']]),measured_turns=len(metrics),nonpositive_turns=sum(t['wav_v2v_ms']<=0 for t in metrics),missing_turns=len(accepted)*30-len(metrics),all_attempts=rows)
 (HERE/'aggregate.json').write_text(json.dumps(a,indent=2)+'\n')
 lines=['# Gemini 3.8 Live — regular model','',f"Status: {'complete' if a['complete'] else 'in progress / review holds recorded in campaign.log'}",'',f"Completed conversations: {len(accepted)}/25. Finished attempts: {len(rows)}. Model/service failures: {len(failures)}. Harness-excluded attempts: {len(excluded)}.",'',f"Strict passes: {strict}/{a['judged_turns']}.",'',f"Judged criteria: {json.dumps(scores)}",'',f"Positive first-speech latency: {json.dumps(a['latency'])}",'',f"Missing latency: {a['missing_turns']}; nonpositive latency: {a['nonpositive_turns']}.",'','## Failed conversations','']
 for r in failures:lines.append(f"- Attempt {r['index']}: {r['turns']} recorded turns; failure {json.dumps(r['failure'])}; {r['run_dir']}")
 lines+=['','Accuracy includes only completed validated conversations. Failed attempts are preserved separately. Regular Gemini 3.8 does not support a thinking level. Standard frontend prompt and shared Opus 5 rubric unchanged. See PROTOCOL.md.']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 return a['complete']
if __name__=='__main__':
 import sys
 if '--watch' in sys.argv:
  while not build():time.sleep(15)
 else:build()
