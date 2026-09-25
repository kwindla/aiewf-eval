"""Derived measurement review; never alters raw recordings or prior metrics."""
import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
standard=Path('runs/aiwf_medium_context_speech_live/20260915T223812_gpt-live-1_88e6d5e8')
audit=json.loads((standard/'audio-alignment-audit.json').read_text());audio=json.loads((standard/'live-audio-metrics.json').read_text());events=[json.loads(l) for l in (standard/'live-events.jsonl').read_text().splitlines()]
assert all(c['min_correlation']>.99 and c['within_turn_drift_ms']<1 for c in audit['checks'])
offsets={c['turn']:statistics.median(c['offset_ms']) for c in audit['checks']};base=next(e['t'] for e in events if e['type']=='recording_start');windows={}
for e in events:
 i=e['turn']
 if e['type']=='input_start':windows[i]=dict(start_ms=(e['t']-base)*1000+offsets[i])
 if e['type']=='input_end':windows[i]['input_end_ms']=(e['t']-base)*1000+offsets[i]
 if e['type']=='turn_complete':windows[i]['end_ms']=(e['t']-base)*1000
metrics=[];issues=[]
for i,w in windows.items():
 us=[s for s in audio['user_segments'] if s['end_ms']>w['start_ms'] and s['start_ms']<w['input_end_ms']+250]
 bs=[s for s in audio['bot_segments'] if s['end_ms']>w['start_ms'] and s['start_ms']<w['end_ms']]
 if not us or not bs:issues.append(dict(turn=i,reason='missing speech'));continue
 old=next(t for t in audio['turns'] if t['turn_index']==i);new=dict(old,user_start_ms=min(s['start_ms'] for s in us),user_end_ms=max(s['end_ms'] for s in us),bot_silero_start_ms=min(s['start_ms'] for s in bs),bot_silero_end_ms=max(s['end_ms'] for s in bs));new['wav_v2v_ms']=new['bot_silero_start_ms']-new['user_end_ms'];metrics.append(new)
 if new['bot_silero_end_ms']>w['end_ms']+100:issues.append(dict(turn=i,reason='speech crosses commit'))
changed=[m['turn_index'] for m in metrics if m['wav_v2v_ms']!=next(t['wav_v2v_ms'] for t in audio['turns'] if t['turn_index']==m['turn_index'])]
result=dict(run=str(standard),method='User input windows located from three high-correlation recorded waveform landmarks per turn; canonical stored VAD segments unchanged.',approved=not issues and len(metrics)==30,offsets_ms=offsets,windows=windows,turns=metrics,issues=issues,changed_latency_turns=changed)
(HERE/'standard-live-alignment-review.json').write_text(json.dumps(result,indent=2)+'\n')
optimized=Path('runs/aiwf_medium_context_speech_live_optimized/20260916T131943_gpt-live-1_b298f449');asr=json.loads((optimized/'audio-review/turn-21-1.json').read_text());o=dict(run=str(optimized),turn=21,classification='VAD missed brief recorded speech; conversation is complete',api_transcript='In Juniper.',independent_asr=asr['transcription']['text'],recording_clip=str(optimized/'audio-review/turn-21-1.wav'),conversation_eligible=True,latency_eligible_for_this_turn=False,measurement_policy='Keep missing VAD boundary in coverage. Do not invent a latency or substitute ASR word times.')
(HERE/'optimized-live-vad-review.json').write_text(json.dumps(o,indent=2)+'\n');print('Standard alignment',result['approved'],'changed latency turns',changed,'issues',issues);print('Optimized:',o['classification'])
