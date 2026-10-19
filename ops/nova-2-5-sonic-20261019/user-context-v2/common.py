"""Nova campaign paths, private credential loading and frozen source identity."""
import hashlib, json, os
from datetime import datetime, UTC
from pathlib import Path
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT/'runs/nova-2-5-sonic-20261019-user-context-v2'
MODEL = 'amazon.nova-2-5-sonic'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(UTC).isoformat()
def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def env():
    result={k:v for k,v in os.environ.items() if not k.startswith('MTE_')}
    for key,value in dotenv_values('/home/khkramer/src/aiewf-eval/.env').items():
        if key.startswith('AWS_') and value: result[key]=value
    result.update(AWS_REGION='us-east-1',MTE_ENABLE_RECOVERY='0',
                  MTE_NOVA_KB_CONTEXT_ROLE='user',MTE_DEDUPE_TOOL_CALLS='1',MTE_TOOL_RESULT_RUN_LLM='0',
                  PYTHONPATH=str(ROOT/'src')+':'+str(ROOT))
    return result
def config(): return json.loads((HERE/'configuration.json').read_text())
def sources():
    paths=[ROOT/'pyproject.toml',ROOT/'uv.lock',*[HERE/n for n in ['common.py','collect.py','direct.py','audio.py','PROTOCOL.md','configuration.json','schedule.tsv','dependencies.json','session-continuation-policy.json','prior-harness-exclusions.json']],ROOT/'ops/aiewf-campaign-template/run_one.py',ROOT/'ops/recent-models-20261018/voice_waveform.py',ROOT/'scripts/analyze_turn_metrics.py']
    for folder in ['src/multi_turn_eval','benchmarks/_shared','benchmarks/aiwf_medium_context','benchmarks/aiwf_medium_context_speech_v2']:
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.extend((ROOT/'benchmarks/_shared/audio').glob('turn_*.wav'))
    paths.extend((ROOT/'benchmarks/aiwf_medium_context/data').glob('*.txt'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def freeze():
    identity=sources();path=HERE/'generation-sha256.json'
    if path.exists(): assert json.loads(path.read_text())==identity,'Generation sources changed'
    else:
        for name in identity:
            target=HERE/'runtime-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        dump(path,identity)
    return identity
