"""Derived waveform caches bound to each exact raw capture and analysis source."""
import importlib.util,json,subprocess,sys
from common import HERE,ROOT,OLD,sha,dump
spec=importlib.util.spec_from_file_location('grok_high_waveform',OLD/'voice_waveform.py')
sys.path.insert(0,str(OLD))
try:
 wave=importlib.util.module_from_spec(spec);spec.loader.exec_module(wave)
finally:
 sys.path.pop(0)
wave.HERE=HERE

def measure(job,run):
 folder=HERE/'grok-voice/audio';folder.mkdir(parents=True,exist_ok=True)
 path=folder/(job+'.json');wp=folder/(job+'-input-waveform.json');proof=folder/(job+'-source.json')
 identity={'run_dir':str(run.relative_to(ROOT)),'raw_sha256':{name:sha(run/name) for name in ['conversation.wav','run.log','transcript.jsonl']},'analysis_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/analyze_turn_metrics.py',OLD/'voice_waveform.py']}}
 if proof.exists():
  old=json.loads(proof.read_text());assert old['identity']==identity,job
  assert old['metrics_sha256']==sha(path) and old['input_waveform_sha256']==sha(wp),job
  return json.loads(path.read_text()),json.loads(wp.read_text())
 wave.inputs()
 result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_turn_metrics.py'),str(run),'--json'],cwd=ROOT,capture_output=True,text=True,timeout=180,check=True)
 data=json.loads(result.stdout);assert len(data['turns'])==30 and data.get('alignment',{}).get('alignment_ok'),job
 waveform=wave.audit(run);assert not waveform['issues'],(job,waveform['issues'])
 dump(path,data);dump(wp,waveform);dump(proof,{'identity':identity,'metrics_sha256':sha(path),'input_waveform_sha256':sha(wp)})
 return data,waveform
