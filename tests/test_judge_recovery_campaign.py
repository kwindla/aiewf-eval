"""Corrected grades must retain fixed membership and immutable prior evidence."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'ops/judge-recovery-context-20261019'
DIMS=('tool_use_correct','instruction_following','kb_grounding')
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def common():
    spec=importlib.util.spec_from_file_location('recovery_campaign_common_test',HERE/'common.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.local_evidence
def test_all_active_requests_have_full_historical_equivalence_proofs():
    data=read(HERE/'inventory.json')
    assert not data['holds'] and data['counts']=={'reuse':988,'rejudge':698}
    assert len(data['cases'])==1686 and len(data['models'])==62
    assert sum(c['complete'] for c in data['cases'])==1619
    assert sum(c['recovery_records'] for c in data['cases'])==1460
    assert sum(len(c['inferred_parents']) for c in data['cases'])==108
    for c in data['cases']:
        grade=read(ROOT/c['old_grade'])
        assert sha(ROOT/c['old_grade'])==c['old_grade_sha256']
        assert c['historical_request']['user_sha256']==grade['metadata']['judge_input_sha256']
        assert (c['historical_request']==c['candidate_request']) is (c['disposition']=='reuse')
        assert bool(c['recovery_records']) is (c['disposition']=='rejudge')

def test_historical_archive_integrity_cannot_be_waived(tmp_path,monkeypatch):
    m=common()
    monkeypatch.setattr(m,'HERE',tmp_path)
    monkeypatch.setattr(m,'ROOT',tmp_path)
    (tmp_path/'legacy').mkdir()
    archive=tmp_path/'legacy/judge.py'
    archive.write_text('frozen')
    (tmp_path/'legacy-sha256.json').write_text(json.dumps({'legacy/judge.py':sha(archive)}))
    (tmp_path/'prior-evidence-sha256.json').write_text('{}')
    m.verify_prior()
    archive.write_text('corrupted')
    with pytest.raises(AssertionError,match='Prior evidence changed'):
        m.verify_prior()

@pytest.mark.local_evidence
def test_verified_restorations_precede_context_formatting_and_preserve_raw_bytes():
    m=common()
    data=read(HERE/'inventory.json')
    c=next(c for c in data['cases'] if c['restorations_applied'] and c['recovery_records'])
    original=(ROOT/c['source']).read_bytes()
    restored,applied=m.restored_records(c)
    assert applied==c['restorations_applied']
    from multi_turn_eval.judging import claude_judge as candidate
    request,_=asyncio.run(m.request_identity(candidate,c))
    assert request==c['candidate_request']
    assert (ROOT/c['source']).read_bytes()==original
    assert any(r['assistant_text'].strip() for r in restored)

@pytest.mark.local_evidence
def test_diagnostic_samples_are_complete_disjoint_and_preregistered():
    selection=read(HERE/'validation-selection.json')
    inventory={c['job_id']:c for c in read(HERE/'inventory.json')['cases']}
    assert len(selection['primary'])==len(selection['reserve'])==24
    assert len({e['source_sha256'] for k in ['primary','reserve'] for e in selection[k]})==48
    for kind in ['primary','reserve']:
        for e in selection[kind]:
            assert inventory[e['job_id']]['observed_turns']==list(range(30))
            assert len(set(e['targets']))==2
            assert e['source_sha256'] not in selection['exclusions']

@pytest.mark.parametrize('failure',['before_commit','after_commit'])
@pytest.mark.local_evidence
def test_publication_io_failure_never_buys_a_second_draw(tmp_path,failure):
    script=r'''
import asyncio,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import canonical,driver
from common import read,HERE
case=next(c for c in read(HERE/'inventory.json')['cases'] if c['disposition']=='rejudge')
driver.HERE=Path(sys.argv[2]);canonical.HERE=driver.HERE;driver.ROOT=driver.HERE
calls=0
async def fake_query(**kwargs):
 global calls
 calls+=1
 result={'phase1_analysis':[],'realignment_notes':'','function_call_tracking':{},'final_judgments':[
  {'turn':turn,'tool_use_correct':True,'instruction_following':True,'kb_grounding':True} for turn in case['observed_turns']]}
 yield SimpleNamespace(model=kwargs['options'].model,content=[SimpleNamespace(text=json.dumps(result))],message_id='TEST-ONLY-no-API')
canonical.ORIGINAL_QUERY=fake_query
driver.ORIGINAL_QUERY=canonical.identified_query
actual_dump=canonical.ORIGINAL_DUMP
actual_log=canonical.ORIGINAL_LOG
def broken_dump(path,value):
 if Path(path).name=='result.json':raise OSError('injected atomic commit failure')
 return actual_dump(path,value)
def broken_log(event,**kwargs):
 if event=='done':raise OSError('injected post-commit log failure')
 return actual_log(event,**kwargs)
canonical.ORIGINAL_DUMP=broken_dump if sys.argv[3]=='before_commit' else actual_dump
canonical.ORIGINAL_LOG=broken_log if sys.argv[3]=='after_commit' else actual_log
driver.dump=canonical.durable_dump
driver.log=canonical.durable_log
driver.valid_result=canonical.readable_result
async def test():
 try:
  await canonical.first_valid_after_restart(case,'fault-test',driver.Cooldown(),asyncio.Semaphore(1))
 except canonical.EvidencePersistenceFailure:pass
 else:raise AssertionError('Expected injected persistence failure')
 assert calls==1
 pending=list(driver.HERE.glob('judging/fault-test/*/attempt-*/candidate-result.json'))
 assert len(pending)==1 and driver.valid_result(pending[0],case)
 canonical.ORIGINAL_DUMP=actual_dump;canonical.ORIGINAL_LOG=actual_log
 await canonical.first_valid_after_restart(case,'fault-test',driver.Cooldown(),asyncio.Semaphore(1))
 assert calls==1,'Resume bought a second draw'
 assert driver.valid_result(driver.HERE/'judging/fault-test'/case['job_id']/'result.json',case)
asyncio.run(test())
'''
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    result=subprocess.run([sys.executable,'-c',script,str(HERE),str(tmp_path),failure],cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.mark.local_evidence
def test_canonical_coverage_and_publication_match_all_per_turn_grades():
    if not (HERE/'PUBLICATION.json').exists():
        pytest.skip('Migration has not passed validation and canonical publication')
    data=read(HERE/'aggregate.json')
    assert read(HERE/'VALIDATION.json')['passed']
    assert data['counts']['assignments']==1686 and data['counts']['rejudged']==698 and data['counts']['reused']==988
    ledger={e['job_id']:e for e in data['ledger']}
    inventory=read(HERE/'inventory.json')
    cases={c['job_id']:c for c in inventory['cases']}
    assert set(ledger)==set(cases)
    for model in data['text_models']+data['voice_models']:
        counts={d:0 for d in DIMS}
        strict=0
        complete=0
        all_strict=0
        baseline=next(m for m in inventory['models'] if m['label']==model['label'])
        assert model.get('latency')==baseline.get('latency')
        for job in model['cases']:
            e=ledger[job]
            assert sha(ROOT/e['grade_path'])==e['grade_sha256']
            grade=read(ROOT/e['grade_path'])
            c=cases[job]
            assert sha(ROOT/c['source'])==c['transcript_sha256']
            assert (ROOT/c['source']).read_bytes()==(ROOT/c['input']).read_bytes()
            assert set(grade['judgments'])==set(map(str,c['observed_turns']))
            if e['disposition']=='reuse':
                assert e['grade_path']==c['old_grade'] and e['grade_sha256']==c['old_grade_sha256']
            else:
                assert grade['phase']=='canonical'
                assert grade['metadata']['judge_input_sha256']==c['candidate_request']['user_sha256']
            for j in grade['judgments'].values():
                value=all(j['scores'][d] for d in DIMS)
                all_strict+=value
                if c['complete']:
                    strict+=value
                    for d in DIMS:counts[d]+=j['scores'][d]
            complete+=c['complete']
        assert model['complete']==complete
        assert model['all_assigned']['strict_passes']==all_strict
        assert model['all_assigned']['denominator']==30*model['assigned']
        assert model['complete_only']['dimension_passes']==counts
        assert model['complete_only']['strict_passes']==strict
        assert model['complete_only']['denominator']==30*complete
    for name,expected in read(HERE/'PUBLICATION.json')['artifacts_sha256'].items():
        nova=ROOT/'ops/nova-2-5-sonic-20261019/user-context-v2'
        publication=read(nova/'PUBLICATION.json') if (nova/'PUBLICATION.json').exists() else {}
        if name in publication.get('prior_artifacts_sha256',{}):
            assert publication['prior_artifacts_sha256'][name]==expected
            assert sha(nova/'prior'/name)==expected
            expected = publication['published_artifacts_sha256'][name]
            entry = read(ROOT/'docs/model-evaluations-2026-10/manifest.json')['public_entrypoints'][name]
            assert entry['source_sha256'] == expected
            assert sha(ROOT/name) == entry['published_sha256']
        else:
            entry = read(ROOT/'docs/model-evaluations-2026-10/manifest.json')['public_entrypoints'][name]
            assert entry['source_sha256'] == expected
            assert sha(ROOT/name) == entry['published_sha256']
    common().verify_prior()
