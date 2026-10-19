"""Use the unchanged waveform metrics and three-landmark correlation audit."""
import importlib.util,json,subprocess,sys
from common import HERE,ROOT,sha,dump
OLD=ROOT/'ops/recent-models-20261018'
sys.path.insert(0,str(OLD))
try:
    spec=importlib.util.spec_from_file_location('nova_waveform',OLD/'voice_waveform.py')
    wave=importlib.util.module_from_spec(spec);spec.loader.exec_module(wave)
finally:sys.path.pop(0)

def measure(job,run):
    folder=HERE/'audio';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(job+'.json');wp=folder/(job+'-input-waveform.json');proof=folder/(job+'-source.json')
    identity={'run_dir':str(run.relative_to(ROOT)),
        'raw_sha256':{name:sha(run/name) for name in ['conversation.wav','run.log','transcript.jsonl']},
        'analysis_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/analyze_turn_metrics.py',OLD/'voice_waveform.py',HERE/'audio.py']}}
    if proof.exists():
        p=json.loads(proof.read_text());assert p['identity']==identity
        assert p['metrics_sha256']==sha(path) and p['input_waveform_sha256']==sha(wp)
        return json.loads(path.read_text()),json.loads(wp.read_text())
    result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_turn_metrics.py'),str(run),'--json'],cwd=ROOT,capture_output=True,text=True,timeout=180,check=True)
    data=json.loads(result.stdout);dump(path,data)
    assert len(data['turns'])==30 and data.get('alignment',{}).get('alignment_ok'),data.get('alignment')
    waveform=wave.audit(run);dump(wp,waveform)
    assert not waveform['issues'],waveform['issues']
    dump(proof,{'identity':identity,'metrics_sha256':sha(path),'input_waveform_sha256':sha(wp)})
    return data,waveform
