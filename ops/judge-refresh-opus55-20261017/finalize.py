"""Finish metric publication and validation after both paid modes complete."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT)
RAW = ROOT/'runs'/HERE.name


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while not all((HERE/f'{mode}-COMPLETE.json').exists() for mode in ('text','voice')):
        if not args.watch:
            raise RuntimeError('Both modes must finish before publication')
        if list(HERE.glob('*-operational-failures.json')):
            raise RuntimeError('Paid worker stopped; inspect operational failures before finalization')
        time.sleep(20)
    commands=[
        [sys.executable,str(REL/'analyze.py'),'--final'],
        [sys.executable,str(REL/'publish.py')],
        [sys.executable,'scripts/render_text_model_voice_readiness.py'],
        [sys.executable,'-m','pytest','-q'],
        [sys.executable,'scripts/render_text_model_voice_readiness.py','--check'],
        ['git','diff','--check'],
    ]
    records=[]
    logdir=RAW/'validation'; logdir.mkdir(parents=True,exist_ok=True)
    for number,command in enumerate(commands,1):
        began=now()
        print('Running:', ' '.join(command),flush=True)
        result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        logfile=logdir/f'{number:02d}.log';logfile.write_text(result.stdout)
        records.append({'command':command,'started_at':began,'finished_at':now(),'exit_code':result.returncode,
                        'log':str(logfile.relative_to(ROOT)), 'output_tail':result.stdout.splitlines()[-3:]})
        if result.returncode:
            print(result.stdout[-8000:],flush=True)
            raise RuntimeError(f'Validation failed; see {logfile}')
        print('\n'.join(result.stdout.splitlines()[-3:]),flush=True)
    aggregate=json.loads((HERE/'aggregate.json').read_text())
    assert aggregate['processing_complete'] is True
    grades=sorted((HERE/'results').glob('*.json'))
    blocked=sorted((HERE/"provider-blocked").glob("*.json"))
    assert len(grades)+len(blocked)==1316
    fingerprint=hashlib.sha256()
    for path in grades:
        fingerprint.update(path.name.encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
    blocked_fingerprint=hashlib.sha256()
    for path in blocked:
        blocked_fingerprint.update(path.name.encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
    outputs=['README.md','leaderboard-medium-context.md','docs/text-model-voice-readiness.svg',
             str(REL/'aggregate.json'),str(REL/'latency-observations.json'),str(REL/'REPORT.md')]
    evidence={'completed_at':now(),'all_steps_passed':True,'commands':records,
              'grade_files':len(grades),'provider_blocked_files':len(blocked),
              'all_quality_complete':aggregate['all_quality_complete'],
              'grade_set_sha256':fingerprint.hexdigest(),
              'provider_blocked_set_sha256':blocked_fingerprint.hexdigest(),
              'published_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in outputs}}
    (HERE/'VERIFICATION.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('All publication and validation steps passed.',flush=True)


if __name__=='__main__':main()
