"""Publish only after complete canonical coverage and all verification passes."""
import subprocess
import sys
import time
from common import *
from publish import FILES

def run(name,command):
    with (HERE/(name+'.log')).open('w') as out:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('Passed',name,flush=True)

def main():
    while not (HERE/'canonical-COMPLETE.json').exists():
        time.sleep(5)
    done=read(HERE/'canonical-COMPLETE.json')
    assert done['conversations']==done['graded']==698
    backup=HERE/'publication-before'
    for name in FILES:
        target=backup/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    try:
        for name in ['analyze','report','publish','verify']:
            run(name,[sys.executable,str(HERE/(name+'.py'))])
        run('tests-final',[sys.executable,'-m','pytest','-q'])
        run('whitespace-final',['git','diff','--check'])
    except BaseException:
        for name in FILES:
            (ROOT/name).write_bytes((backup/name).read_bytes())
        publication=HERE/'PUBLICATION.json'
        if publication.exists():
            publication.rename(HERE/f'PUBLICATION-failed-{int(time.time())}.json')
        print('Restored previous publication after a failed final check',flush=True)
        raise
    verify_prior()
    verification=read(HERE/'VERIFICATION.json')
    assert verification['passed']
    dump(HERE/'COMPLETE.json',{'completed_at_epoch':time.time(),'canonical_conversations':698,'reused_conversations':988,
         'active_assignments':1686,'published_complete_conversations':1619,'validation_passed':True,
         'verification_sha256':sha(HERE/'VERIFICATION.json'),'aggregate_sha256':sha(HERE/'aggregate.json'),
         'publication_sha256':sha(HERE/'PUBLICATION.json'),'final_tests_sha256':sha(HERE/'tests-final.log'),
         'source_sha256':{name:sha(HERE/name) for name in ['analyze.py','report.py','publish.py','verify.py','finish.py']},
         'all_required_checks_passed':True})
    print('Migration complete',flush=True)

if __name__=='__main__':main()
