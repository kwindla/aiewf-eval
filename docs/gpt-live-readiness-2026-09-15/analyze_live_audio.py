"""Audio-only Live latency, using the canonical Silero detector and parameters.

Live streams silence continuously, so packet/tag arrival is not speech onset.
Use logged input/commit windows to associate WAV speech with each scripted turn.
"""
import argparse,json,wave
from pathlib import Path
import numpy as np
from scripts.analyze_turn_metrics import load_silero_vad,run_silero_vad

p=argparse.ArgumentParser();p.add_argument('run_dir');args=p.parse_args();run=Path(args.run_dir)
rows=[json.loads(l) for l in (run/'live-events.jsonl').read_text().splitlines()]
transcript={r['turn']:r for r in [json.loads(l) for l in (run/'transcript.jsonl').read_text().splitlines()]}
baseline=next(r['t'] for r in rows if r['type']=='recording_start')
windows={}
for r in rows:
 if r['type']=='input_start':windows[r['turn']]={'start_ms':(r['t']-baseline)*1000}
 if r['type']=='input_end':windows[r['turn']]['input_end_ms']=(r['t']-baseline)*1000
 if r['type']=='turn_complete':windows[r['turn']]['end_ms']=(r['t']-baseline)*1000
model,get_timestamps=load_silero_vad();users,bots=run_silero_vad(run/'conversation.wav',model,get_timestamps)
metrics=[];issues=[]
for idx,w in windows.items():
 if 'end_ms' not in w:continue
 us=[s for s in users if s['end_ms']>w['start_ms'] and s['start_ms']<w['input_end_ms']+250]
 bs=[s for s in bots if s['end_ms']>w['start_ms'] and s['start_ms']<w['end_ms']]
 if not us or not bs:issues.append({'turn':idx,'reason':'missing user or bot speech'});continue
 ue=max(s['end_ms'] for s in us);first=min(s['start_ms'] for s in bs);last=max(s['end_ms'] for s in bs)
 overlap=sum(max(0,min(u['end_ms'],b['end_ms'])-max(u['start_ms'],b['start_ms'])) for u in us for b in bs)
 gaps=[max(0,b['start_ms']-a['end_ms']) for a,b in zip(bs,bs[1:])]
 metrics.append(dict(turn_index=idx,user_start_ms=min(s['start_ms'] for s in us),user_end_ms=ue,bot_silero_start_ms=first,bot_silero_end_ms=last,wav_v2v_ms=first-ue,has_tool_call=bool(transcript[idx]['tool_calls']),overlap_ms=overlap,bot_speech_segments=len(bs),max_within_answer_silence_ms=max(gaps,default=0),completion_quiet_ms=w['end_ms']-last))
 # Any response speech crossing the chosen boundary means the gate cut it off.
 if last>w['end_ms']+100:issues.append({'turn':idx,'reason':'bot speech crosses turn commit','overrun_ms':last-w['end_ms']})
def stats(values):
 return {'n':len(values),'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),'p90_ms':float(np.percentile(values,90)),'min_ms':float(min(values)),'max_ms':float(max(values))} if values else {'n':0}
summary={'all_signed_v2v':stats([m['wav_v2v_ms'] for m in metrics]),'positive_v2v':stats([m['wav_v2v_ms'] for m in metrics if m['wav_v2v_ms']>0]),'tool_positive_v2v':stats([m['wav_v2v_ms'] for m in metrics if m['wav_v2v_ms']>0 and m['has_tool_call']]),'non_tool_positive_v2v':stats([m['wav_v2v_ms'] for m in metrics if m['wav_v2v_ms']>0 and not m['has_tool_call']]),'overlap_turns':sum(m['overlap_ms']>=100 for m in metrics),'negative_v2v_turns':sum(m['wav_v2v_ms']<0 for m in metrics),'turns':len(metrics)}
result=dict(method='Canonical Silero detector/parameters, stereo WAV; per-turn windows from input/commit events. First speech includes spoken acknowledgements; this is not latency to the substantive answer.',summary=summary,turns=metrics,issues=issues,user_segments=users,bot_segments=bots)
(run/'live-audio-metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2));print('Boundary issues:',issues)
if issues or len(metrics)!=len(transcript):raise SystemExit('Audio validation incomplete or boundary audit failed')
