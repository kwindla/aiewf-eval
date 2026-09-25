"""Report the independent Gemini low/medium turn-association audit."""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'audio-audit'
sys.path.insert(0,str(HERE))
from analyze import stats

def main():
 assert (OUT/'complete.json').exists()
 groups={}
 for arm in ['gemini38low','gemini38medium']:
  rows=[];lat={k:[] for k in ['all','tool','non_tool']};signed=[];reasons=Counter();bad=[];missing=0
  for i in range(1,26):
   key=f'{arm}-{i:02d}';done=json.loads((OUT/f'{key}.done.json').read_text())
   if not done['audit']:bad.append(key);missing+=30;rows.append(dict(key=key,audit=done));continue
   data=json.loads(Path(done['metrics']).read_text());alignment=json.loads(Path(done['alignment']).read_text())
   assert not alignment['issues'];seen=set()
   for t in data['turns']:
    seen.add(t['turn']);v=t['wav_v2v_ms'];signed.append(v)
    if v>0:lat['all'].append(v);lat['tool' if t['has_tool_call'] else 'non_tool'].append(v)
   missing+=30-len(seen);reasons.update(t['reason'] for t in data['issues'])
   rows.append(dict(key=key,audit=done,missing_input_windows=sorted(set(range(30))-{int(k) for k in data['windows']}),missing_turns=sorted(set(range(30))-seen),issues=data['issues'],max_user_alignment_offset_ms=max((abs(x) for c in alignment['checks'] for x in c['offset_ms']),default=0)))
  groups[arm]=dict(latency={k:stats(v) for k,v in lat.items()},signed_latency=stats(signed),missing_turns=missing,nonpositive=sum(x<=0 for x in signed),issue_reasons=dict(reasons),failed_audits=bad,runs=rows)
 original=json.loads((ROOT/'docs/speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/aggregate.json').read_text());low=next(m for m in original['models'] if m['arm']['key']=='gemini38')
 result=dict(groups=groups,original_low_latency=low['latency'],note='Audio-only association correction. Same recordings and canonical VAD detector/thresholds; no model prompts, semantic scores, or original artifacts changed.')
 (OUT/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Gemini 3.8 audio association audit','','The legacy analyzer assigns turn indices by bot audio-marker order. A marker is not guaranteed for every scripted turn, so a missing marker can shift subsequent tool/non-tool labels and omit an otherwise measurable response. It also stops emitting rows at the number of detected markers.','','This independent analysis assigns actual recorded speech to logged input windows and verifies those windows by normalized waveform correlation at three speech landmarks per user utterance. It uses the same canonical Silero detector and thresholds. Source recordings, original metrics, model prompts, and semantic judgments remain untouched.','','| Setting | Positive V2V observations | Overall mean | Overall median | Overall p90 | Non-tool median | Tool mean | Missing planned turns |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm,g in groups.items():
  l=g['latency'];f=lambda v:'N/A' if v is None else f'{v:.0f}ms'
  lines.append(f"| {arm} | {l['all']['n']}/750 | {f(l['all']['mean_ms'])} | {f(l['all']['p50_ms'])} | {f(l['all']['p90_ms'])} | {f(l['non_tool']['p50_ms'])} | {f(l['tool']['mean_ms'])} | {g['missing_turns']} |")
 lines+=['','## Coverage and limitations','']
 for arm,g in groups.items():lines.append(f"- **{arm}:** audit failures {g['failed_audits'] or 'none'}; boundary/detection issues {g['issue_reasons'] or 'none'}; nonpositive measurements {g['nonpositive']}. Missing bot speech can be a silent tool-only turn, not necessarily an API error; inspect the corresponding transcript before classifying it.")
 lines+=['','The low cohort has seven silent `end_session` turns and one absent logged input window (attempt 24, turn 0). That attempt recorded a startup greeting as turn 0 and sent the next question during greeting handling. This is a harness/startup caveat, not a provider API error. Its source transcript and accuracy scores are retained unchanged; the unmeasurable turn contributes no latency. A read-only check of all 125 original input sequences found no other such anomaly; see [sequence audit](original-input-sequences.json). New medium captures are checked for the same anomaly.','', 'Input-window association fixes the marker-order assumption. It does not change the definition of V2V: first actual assistant speech after user speech, including acknowledgments, rather than time to the substantive answer. Missing measurements are excluded from latency statistics and explicitly counted. All planned turns remain in accuracy denominators.','','See [full audit data](aggregate.json) and the preserved original tag-based measurements under the canonical and extension campaign `processed/` directories.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
