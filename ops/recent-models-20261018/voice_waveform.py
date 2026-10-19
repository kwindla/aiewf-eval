"""Verify recorded Grok user audio against the unchanged tracked input WAVs.

Uses the existing speech audit's three-landmark normalized-correlation method.
The generic realtime capture supplies enqueue timestamps in run.log.
"""
import hashlib,json,re,subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate,resample_poly
from preflight import HERE,ROOT,dump
from benchmarks.aiwf_medium_context.config import BenchmarkConfig

def audit(run):
 audio,sr=sf.read(run/'conversation.wav',dtype='float64');assert sr==24000 and audio.shape[1]==2
 lines=(run/'run.log').read_text().splitlines();base=next(datetime.fromisoformat(l[:23]) for l in lines if 'PacedInputTransport#0: Recording baseline set at monotonic=' in l)
 enqueues=[]
 for line in lines:
  match=re.search(r'enqueue_wav_file path=.*turn_(\d+)\.wav frames=',line)
  if match:enqueues.append((int(match[1]),datetime.fromisoformat(line[:23])))
 assert [i for i,t in enqueues]==list(range(30)),enqueues
 checks=[]
 for turn,stamp in enqueues:
  source,rate=sf.read(BenchmarkConfig.get_audio_path(turn),dtype='float64')
  if rate!=sr:source=resample_poly(source,sr,rate)
  expected=int((stamp-base).total_seconds()*sr);offsets=[];scores=[]
  for part in np.array_split(np.arange(max(0,len(source)-2400)),3):
   choices=part[::240]
   if not len(choices):continue
   pos=max(choices,key=lambda p:float(np.sum(source[p:p+2400]**2)));patch=source[pos:pos+2400];energy=float(np.sum(patch**2))
   if energy<1e-5:continue
   lo=max(0,expected+pos-24000);hi=min(len(audio),expected+pos+24000+len(patch));region=audio[lo:hi,0]
   corr=correlate(region,patch,mode='valid',method='fft');cum=np.concatenate(([0.0],np.cumsum(region**2)));energies=cum[len(patch):]-cum[:-len(patch)]
   normalized=corr/np.sqrt(np.maximum(energy*energies,1e-20));at=int(np.argmax(normalized));offsets.append((lo+at-pos-expected)/sr*1000);scores.append(float(normalized[at]))
  checks.append({'turn':turn,'offset_ms':offsets,'min_correlation':min(scores,default=0),'within_turn_drift_ms':max(offsets,default=0)-min(offsets,default=0)})
 issues=[c for c in checks if c['min_correlation']<0.95 or max(map(abs,c['offset_ms']),default=0)>250 or c['within_turn_drift_ms']>100]
 return {'checks':checks,'issues':issues,'method':'existing speech three-landmark waveform correlation; enqueue timestamp vs recording baseline','thresholds':{'minimum_correlation':0.95,'maximum_absolute_offset_ms':250,'maximum_within_turn_drift_ms':100}}

def inputs():
 manifest={}
 for turn in range(30):
  p=BenchmarkConfig.get_audio_path(turn);relative=str(p.relative_to(ROOT));raw=p.read_bytes();prior=subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)
  assert raw==prior,relative
  manifest[relative]=hashlib.sha256(raw).hexdigest()
 path=HERE/'grok-voice/input-audio-sha256.json'
 if path.exists():assert json.loads(path.read_text())==manifest
 else:dump(path,manifest)
 return manifest
if __name__=='__main__':
 import sys
 inputs();result=audit(Path(sys.argv[1]));dump(Path(sys.argv[2]),result)
 print(json.dumps({'turns':len(result['checks']),'issues':result['issues'],'minimum_correlation':min(c['min_correlation'] for c in result['checks'])}))
 if result['issues']:raise SystemExit('Hold capture for input waveform alignment review')
