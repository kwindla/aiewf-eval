"""Independent input-window V2V audit; never edits captures or original metrics.

Associate actual speech with logged scripted input windows instead of assuming one
bot packet tag per turn. The VAD detector and parameters remain canonical.
"""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analyze_turn_metrics import load_silero_vad,run_silero_vad,parse_recording_baseline_from_log
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('output',type=Path);args=p.parse_args();run=args.run_dir
text=(run/'run.log').read_text();base=parse_recording_baseline_from_log(run/'run.log');assert base is not None
windows={}
for m in re.finditer(r'\[USER_AUDIO_QUEUED\] turn=(\d+) predicted_end=([0-9.]+) duration=([0-9.]+)',text):
 idx=int(m[1]);assert idx not in windows,'Repeated input requires separate recovery audit'
 windows[idx]=dict(start_ms=(float(m[2])-float(m[3])-base)*1000,input_end_ms=(float(m[2])-base)*1000)
transcript={r['turn']:r for r in map(json.loads,(run/'transcript.jsonl').read_text().splitlines()) if not r.get('recovery_turn')}
model,fn=load_silero_vad();users,bots=run_silero_vad(run/'conversation.wav',model,fn)
metrics=[];issues=[];indices=sorted(windows)
for at,i in enumerate(indices):
 w=windows[i];end=windows[indices[at+1]]['start_ms'] if at+1<len(indices) else float('inf')
 us=[s for s in users if s['end_ms']>w['start_ms'] and s['start_ms']<w['input_end_ms']+250]
 bs=[s for s in bots if s['end_ms']>w['start_ms'] and s['start_ms']<end]
 if not us or not bs:
  issues.append(dict(turn=i,reason='missing user speech' if not us else 'missing bot speech'));continue
 ue=max(s['end_ms'] for s in us);first=min(s['start_ms'] for s in bs);last=max(s['end_ms'] for s in bs)
 overlap=sum(max(0,min(u['end_ms'],b['end_ms'])-max(u['start_ms'],b['start_ms'])) for u in us for b in bs)
 if first<w['start_ms']-100:issues.append(dict(turn=i,reason='bot segment starts before input window',offset_ms=first-w['start_ms']))
 if last>end+100:issues.append(dict(turn=i,reason='bot segment crosses next input boundary',overrun_ms=last-end))
 metrics.append(dict(turn=i,user_start_ms=min(s['start_ms'] for s in us),user_end_ms=ue,bot_silero_start_ms=first,bot_silero_end_ms=last,wav_v2v_ms=first-ue,has_tool_call=bool(transcript.get(i,{}).get('tool_calls')),overlap_ms=overlap,bot_speech_segments=len(bs)))
args.output.parent.mkdir(parents=True,exist_ok=True)
result=dict(method='Independent input-window association with unchanged canonical Silero VAD; no bot-tag ordinal assumption. User window timestamps require waveform alignment validation.',run_dir=str(run),windows=windows,turns=metrics,issues=issues,user_segments=users,bot_segments=bots)
args.output.write_text(json.dumps(result,indent=2)+'\n')
xs=[t['wav_v2v_ms'] for t in metrics];print(json.dumps(dict(n=len(xs),mean_ms=float(np.mean(xs)),median_ms=float(np.median(xs)),issues=issues),indent=2))
