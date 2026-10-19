import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=ROOT/'runs/responses-repair-20261018'
spec=importlib.util.spec_from_file_location('repair_portable',ROOT/'ops/aiewf-campaign-template/collect.py')
collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect)
collect.RUN_ONE=HERE/'run_one.py'
NAMES=[r['key'] for r in json.loads((HERE/'impact-inventory.json').read_text())['rows'] if r['rerun_candidate']]
def cfg(name):return collect.validate_configuration(HERE/name/'configuration.json',require_serving_verified=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def environment(c):
 env=collect.child_environment(c,'standard',collect.extract_credential(c))
 env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
 return env
