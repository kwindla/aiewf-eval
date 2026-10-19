"""Audit waveform data as each stable completed-attempt prefix becomes available."""
import json,time
from common import HERE,ROOT
from audio import measure

def main():
 while True:
  attempts=[json.loads(p.read_text()) for p in sorted((HERE/'grok-voice/artifacts').glob('attempt-*.json'))]
  by_number={a['attempt']:a for a in attempts};prefix=[]
  for number in range(1,max(by_number,default=0)+1):
   if number not in by_number:break
   prefix.append(by_number[number])
  cases=[a for a in prefix if a['complete']][:25]
  for slot,c in enumerate(cases,1):
   job=f'grok-voice-high-{slot:03d}';proof=HERE/'grok-voice/audio'/(job+'-source.json')
   if not proof.exists():
    measure(job,ROOT/c['run_dir']);print(job,'waveform verified',flush=True)
  if len(cases)==25:return
  time.sleep(30)
if __name__=='__main__':main()
