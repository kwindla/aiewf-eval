"""Publish and validate only after every registered conversation is judged."""

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=ROOT/'runs/baseten-ultra-20261018-paced/validation'


def main():
    while not (HERE/'judging/COMPLETE.json').exists():time.sleep(15)
    RAW.mkdir(parents=True,exist_ok=True)
    commands=[
        [sys.executable,str(HERE/'analyze.py')],
        [sys.executable,'-m','pytest','-q'],
        [sys.executable,'scripts/render_text_model_voice_readiness.py','--check'],
        ['git','diff','--check'],
    ]
    records=[]
    for n,command in enumerate(commands,1):
        print('Running: '+' '.join(command),flush=True)
        process=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=RAW/f'{n:02d}.log';log.write_text(process.stdout)
        records.append({'command':command,'exit_code':process.returncode,'log':str(log.relative_to(ROOT)),
                        'output_tail':process.stdout.splitlines()[-3:]})
        if process.returncode:
            print(process.stdout[-6000:],flush=True)
            raise RuntimeError(f'Validation failed: {log}')
        print('\n'.join(process.stdout.splitlines()[-3:]),flush=True)
    aggregate=json.loads((HERE/'aggregate.json').read_text())
    assert len(aggregate['conversations'])==60
    assert all(a['conversations']==30 and a['fixed_denominator']==900 for a in aggregate['arms'].values())
    artifacts=[HERE/'configuration.json',HERE/'schedule.tsv',HERE/'pairs.json',HERE/'PROTOCOL.md',
               HERE/'preflight.py',HERE/'judge.py',HERE/'analyze.py',HERE/'finish.py',
               HERE/'artifacts/canonical.tsv',HERE/'artifacts/attempts.tsv',HERE/'artifacts/source-sha256.txt',
               HERE/'judging/inputs.json',HERE/'judging/rubric.txt',HERE/'aggregate.json',HERE/'REPORT.md']
    artifacts+=sorted((HERE/'smoke').glob('*.json'))
    artifacts+=sorted((HERE/'judging/results').glob('*.json'))
    artifacts+=sorted((HERE/'judging/provider-blocked').glob('*.json'))
    for path in HERE.rglob('*.json'):json.loads(path.read_text())
    verification={'completed_at':datetime.now(timezone.utc).isoformat(),'all_checks_passed':True,
                  'all_quality_complete':aggregate['all_quality_complete'],'commands':records,
                  'artifacts_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (HERE/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('Report and validation complete: '+str(HERE/'REPORT.md'),flush=True)


if __name__=='__main__':main()
