"""Precompute derived waveform audits for completed voice captures only."""
import json,subprocess,sys,time
from pathlib import Path
from voice_waveform import audit,inputs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def dump(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 inputs();folder=HERE/'grok-voice/audio';folder.mkdir(exist_ok=True)
 while True:
  attempts=[json.loads(p.read_text()) for p in sorted((HERE/'grok-voice/artifacts').glob('attempt-*.json'))]
  by_number={a['attempt']:a for a in attempts};prefix=[]
  for number in range(1,max(by_number,default=0)+1):
   if number not in by_number:break
   prefix.append(by_number[number])
  cases=[a for a in prefix if a['complete']][:25]
  for slot,c in enumerate(cases,1):
   job=f'grok-voice-none-{slot:03d}';run=ROOT/c['run_dir'];path=folder/(job+'.json')
   if not path.exists():
    r=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_turn_metrics.py'),str(run),'--json'],cwd=ROOT,capture_output=True,text=True,timeout=180,check=True)
    data=json.loads(r.stdout);assert len(data['turns'])==30
    assert data.get('alignment',{}).get('alignment_ok'),job
    dump(path,data)
   waveform=folder/(job+'-input-waveform.json')
   if not waveform.exists():
    data=audit(run);assert not data['issues'],(job,data['issues']);dump(waveform,data)
   print(job,'waveform checked',flush=True)
  if len(cases)==25:return
  time.sleep(30)
if __name__=='__main__':main()
