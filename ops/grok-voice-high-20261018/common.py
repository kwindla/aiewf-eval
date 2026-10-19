"""Paths and the unchanged portable collector shared with the completed off study."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'ops/recent-models-20261018'
RAW=ROOT/'runs/grok-voice-high-20261018'
spec=importlib.util.spec_from_file_location('grok_high_portable_preflight',OLD/'preflight.py')
base_preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(base_preflight)
collect=base_preflight.collect
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def cfg(name='grok-voice'):
 assert name=='grok-voice'
 return collect.validate_configuration(HERE/name/'configuration.json',require_serving_verified=False)
def verify_baseline():
 for name,digest in json.loads((HERE/'baseline-sha256.json').read_text()).items():assert sha(ROOT/name)==digest,name
