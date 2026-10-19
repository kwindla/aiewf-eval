"""Verify fixed coverage, immutability, table cells and first-valid ownership."""
from collections import Counter
import subprocess
import sys
import numpy as np
from common import *
from driver import valid_result, frozen_identity
from publish import FILES

def table_rows(source,header):
    body=source.split(header,1)[1].split('\n\n',1)[0]
    return {cells[0]:cells for line in body.splitlines() if line.startswith('| ') and
            (cells:=[x.strip() for x in line.strip('|').split('|')])}

def main():
    from acceptance import verify_acceptance
    verify_acceptance()
    verify_prior()
    assert read(HERE/'canonical-driver-sha256.json')==frozen_identity()
    registration=read(HERE/'canonical-registration.json')
    assert registration['canonical_entry_sha256']==sha(HERE/'canonical.py')
    data=read(HERE/'aggregate.json')
    inventory=read(HERE/'inventory.json')
    cases={c['job_id']:c for c in inventory['cases']}
    assert len(cases)==len(data['ledger'])==1686
    assert {e['job_id'] for e in data['ledger']}==set(cases)
    assert len(data['text_models'])==51 and len(data['voice_models'])==11
    assert sum(m['complete'] for m in data['text_models'])==1344
    assert sum(m['complete'] for m in data['voice_models'])==275
    assert data['counts']['reused']==988 and data['counts']['rejudged']==698
    assert read(HERE/'VALIDATION.json')['passed']
    dispositions=Counter()
    for e in data['ledger']:
        c=cases[e['job_id']]
        records(c)
        path=ROOT/e['grade_path']
        assert sha(path)==e['grade_sha256']
        assert e['disposition']==c['disposition']
        dispositions[e['disposition']]+=1
        if e['disposition']=='reuse':
            assert c['historical_request']==c['candidate_request']
            assert path==ROOT/c['old_grade'] and e['grade_sha256']==c['old_grade_sha256']
        else:
            assert valid_result(path,c)
            r=read(path)
            assert r['phase']=='canonical'
            selected=ROOT/r['raw_output']
            prior=[p for p in selected.parent.glob('attempt-*/candidate-result.json') if int(p.parent.name.split('-')[1])<int(selected.name.split('-')[1])]
            for p in prior:
                try:
                    valid=valid_result(p,c)
                except (AssertionError,ValueError,KeyError):
                    valid=False
                assert not valid,('Later draw displaced a first valid result',c['job_id'])
            identity_file=selected/'response-identities.jsonl'
            assert identity_file.exists()
            identities=[json.loads(line) for line in identity_file.read_text().splitlines()]
            assert any(row.get('message_id') for row in identities),'Missing actual judge response message IDs'
    ledger={e['job_id']:e for e in data['ledger']}
    for model in data['text_models']+data['voice_models']:
        baseline=next(m for m in inventory['models'] if m['label']==model['label'])
        assert model['cases']==baseline['cases']
        assert model['graded_conversations']==len(model['cases'])
        assert model['graded_turns']==model['all_assigned']['observed_scripted_turns']
        assert model['accuracy_denominator']==model['complete_only']['denominator']
        if 'source_cohort_manifest' in model:
            assert (ROOT/model['source_cohort_manifest']).is_file()
        if model['mode']=='voice':
            expected_audio=(baseline['scores']['turn_taking'] if 'scores' in baseline else baseline['turn_taking_passes'])
            assert model['scores']['turn_taking']==expected_audio
            assert model['turn_taking_scored']==baseline['turn_taking_scored']
        for complete_only in [False,True]:
            selected=[cases[job] for job in model['cases'] if not complete_only or cases[job]['complete']]
            counts={dim:0 for dim in DIMS}
            strict=0
            observed=0
            conversation_passes=[]
            for case in selected:
                grade=read(ROOT/ledger[case['job_id']]['grade_path'])
                conversation_passes.append(sum(all(j['scores'][dim] for dim in DIMS) for j in grade['judgments'].values()))
                for judgment in grade['judgments'].values():
                    observed+=1
                    strict+=all(judgment['scores'][dim] for dim in DIMS)
                    for dim in DIMS:
                        counts[dim]+=judgment['scores'][dim]
            denominator=30*len(selected)
            actual=model['complete_only' if complete_only else 'all_assigned']
            assert actual['strict_passes']==strict and actual['dimension_passes']==counts
            assert actual['denominator']==denominator and actual['observed_scripted_turns']==observed
            assert actual['missing_future_turns']==denominator-observed
            assert actual['strict_percent']==(100*strict/denominator if denominator else None)
            assert actual['dimension_error_percent']=={dim:100*(denominator-counts[dim])/denominator if denominator else None for dim in DIMS}
            assert actual['conversation_strict_passes']==conversation_passes
            if selected:
                rng=np.random.default_rng(20261019)
                samples=rng.integers(0,len(selected),size=(20000,len(selected)))
                expected_ci=np.percentile(np.asarray(conversation_passes)[samples].sum(axis=1)*100/denominator,[2.5,97.5])
                assert np.allclose(actual['strict_ci95'],expected_ci,rtol=0,atol=1e-10)
            else:
                assert actual['strict_ci95'] is None
        assert model['strict_percent']==model['complete_only']['strict_percent']
        assert model['dimension_error_percent']==model['complete_only']['dimension_error_percent']
    source=(ROOT/'README.md').read_text()
    rows=table_rows(source,'| Model | Pass Rate | Any Error |')
    for m in data['text_models']:
        if not m['complete']:
            assert m['label'] not in rows
            continue
        row=rows[m['label']]
        assert row[1:6]==[f"{m['strict_percent']:.1f}%",f"{100-m['strict_percent']:.1f}%",*[f"{m['dimension_error_percent'][dim]:.1f}%" for dim in DIMS]]
        assert int(row[-1])==m['complete']
    assert len(rows)==50
    model_lookup={m['label']:m for m in data['text_models']+data['voice_models']}
    fixed_cells=0
    for line in source.splitlines():
        if line.startswith('| '):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if len(cells)==3 and cells[0] in model_lookup and '/' in cells[1]:
                model=model_lookup[cells[0]]
                assert cells[1]==f"{model['complete']}/{model['assigned']}"
                assert cells[2]==f"{model['all_assigned']['strict_percent']:.1f}%"
                fixed_cells+=1
    assert fixed_cells==12,'Expected all Responses fixed-900 cells'
    # Voice requests and every corresponding publication cell remain unchanged.
    old=(HERE/'legacy/publication/README.md').read_text().split('Speech-to-speech models:\n',1)[1]
    now=source.split('Speech-to-speech models:\n',1)[1]
    assert old==now
    standalone=(ROOT/'leaderboard-medium-context.md').read_text()
    assert table_rows(standalone,'| Model | Pass Rate | Any Error |')==rows
    before={name:sha(ROOT/name) for name in FILES}
    subprocess.run([sys.executable,str(HERE/'publish.py')],cwd=ROOT,check=True)
    assert before=={name:sha(ROOT/name) for name in FILES},'Publication is not idempotent'
    assert read(HERE/'PUBLICATION.json')['artifacts_sha256']==before
    verification={'passed':True,'assignments':1686,'complete_conversations':1619,'incomplete_assignments':67,
                  'dispositions':dict(dispositions),'prior_evidence_files_verified':len(read(HERE/'prior-evidence-sha256.json')),
                  'legacy_files_verified':len(read(HERE/'legacy-sha256.json')),
                  'voice_publication_unchanged':True,'publication_idempotent':True,
                  'inventory_sha256':sha(HERE/'inventory.json'),'aggregate_sha256':sha(HERE/'aggregate.json'),
                  'validation_sha256':sha(HERE/'VALIDATION.json'),'publication_sha256':sha(HERE/'PUBLICATION.json')}
    dump(HERE/'VERIFICATION.json',verification)
    print(json.dumps(verification),flush=True)

if __name__=='__main__':main()
