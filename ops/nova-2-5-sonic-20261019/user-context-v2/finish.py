"""Judge, publish and verify after all canonical captures complete."""
import json,subprocess,sys,time
from common import HERE,ROOT,sha,dump,now,freeze

def run(name,arguments):
    with (HERE/(name+'.log')).open('w') as out:
        subprocess.run(arguments,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

def prior_evidence():
    parent=HERE.parent
    for name,digest in json.loads((parent/'evidence-sha256.json').read_text()).items():assert sha(parent/name)==digest
    for name,digest in json.loads((parent/'filter-diagnostics/evidence-sha256.json').read_text()).items():
        assert sha(parent/'filter-diagnostics'/name)==digest
    excluded=json.loads((HERE/'prior-harness-exclusions.json').read_text())
    assert excluded['count']==3 and not excluded['judged']
    assert sha(parent/'user-context/generation-sha256.json')==excluded['profile_generation_manifest_sha256']
    for name,digest in json.loads((parent/'user-context/generation-sha256.json').read_text()).items():
        assert sha(parent/'user-context/runtime-snapshot'/name)==digest
    for row in excluded['attempts']:
        assert row['classification']=='harness_excluded'
        for name,digest in row['raw_sha256'].items():assert sha(ROOT/row['run_dir']/name)==digest
    resolutions=[]
    # The diagnostic freezes bind their original adapter; preserve those bytes
    # in this campaign's pre-change archive when current serving code evolves.
    for manifest in sorted((parent/'filter-diagnostics/requests').glob('*-freeze.json')):
        for name,digest in json.loads(manifest.read_text())['sources_sha256'].items():
            path=ROOT/name
            if sha(path)!=digest:
                path=HERE/'prior'/name;assert sha(path)==digest,(manifest,name)
                resolutions.append({'manifest':str(manifest.relative_to(ROOT)),'source':name,
                    'archive':str(path.relative_to(ROOT)),'sha256':digest})
    return resolutions

def main():
    while not (HERE/'COLLECTION-COMPLETE.json').exists():time.sleep(15)
    freeze();resolutions=prior_evidence()
    for script in ['judge.py','analyze.py','publish.py']:run(script.removesuffix('.py'),[sys.executable,str(HERE/script)])
    published=[ROOT/'README.md',HERE/'PUBLICATION.json']
    before={str(p):p.read_bytes() for p in published}
    run('publish-repeat',[sys.executable,str(HERE/'publish.py')])
    assert all(p.read_bytes()==before[str(p)] for p in published),'Publication not idempotent'
    for name,command in [('tests-final',[sys.executable,'-m','pytest','-q','--disable-warnings']),
        ('chart-final',[sys.executable,str(ROOT/'scripts/render_text_model_voice_readiness.py'),'--check']),
        ('whitespace-final',['git','diff','--check'])]:run(name,command)
    freeze();prior_evidence()
    data=json.loads((HERE/'aggregate.json').read_text())
    artifacts=['aggregate.json','REPORT.md','PUBLICATION.json','COLLECTED.json','COLLECTION-COMPLETE.json',
        'attempts.json','generation-sha256.json','judging/inputs.json','judging/COMPLETE.json','session-continuation-policy.json']
    dump(HERE/'VERIFICATION.json',{'at':now(),'all_checks_passed':True,'complete_conversations':25,
        'judged_conversations':25,'publisher_idempotent':True,'prior_evidence_unchanged':True,
        'prior_source_resolution':resolutions,'tests_summary':(HERE/'tests-final.log').read_text().strip().splitlines()[-1],
        'successful_judge_cost_usd':data['successful_judge_cost_usd'],
        'artifacts_sha256':{name:sha(HERE/name) for name in artifacts},
        'published_artifacts_sha256':{'README.md':sha(ROOT/'README.md')},
        'result_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'judging/results').glob('*.json'))},
        'audio_proof_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'audio').glob('*-source.json'))}})
    dump(HERE/'SESSION.json',{'at':now(),'status':'complete','remaining':[],'no_commit_or_push':True})
    print('Nova 2.5 Sonic collection, judging, publication and verification complete',flush=True)

if __name__=='__main__':main()
