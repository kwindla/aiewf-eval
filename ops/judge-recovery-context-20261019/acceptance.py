"""Authenticate the registered references and every accepted diagnostic result."""
from common import *
import driver

def evidence():
    registration=read(HERE/'validation-request-registration.json')
    assert registration['reference_freeze_sha256']==sha(HERE/'references/FROZEN.json')
    assert registration['selection_sha256']==sha(HERE/'validation-selection.json')
    assert registration['driver_sha256']==sha(HERE/'driver-validation-v1-sha256.json')
    assert read(HERE/'driver-sha256.json')==read(HERE/'driver-validation-v1-sha256.json')
    for name,expected in read(HERE/'driver-validation-v1-sha256.json').items():
        path=HERE/('driver-validation-v1.py' if name=='driver.py' else name)
        assert sha(path)==expected,('Validation source changed',name)
    selection=read(HERE/'validation-selection.json')
    freeze=read(HERE/'validation-freeze.json')
    assert freeze['selection_sha256']==sha(HERE/'validation-selection.json')
    for name,expected in freeze['packets'].items():
        assert sha(HERE/'references'/name)==expected
    refs=read(HERE/'references/FROZEN.json')
    for name,expected in refs['files'].items():
        assert sha(HERE/name)==expected
        batch=Path(name).stem.removeprefix('batch-')
        assert read(HERE/name)['packet_sha256']==freeze['packets'][f'packet-{batch}.json']
    inventory=read(HERE/'inventory.json')
    inventory_reconciliation=None
    if selection['inventory_sha256']!=sha(HERE/'inventory.json'):
        # An asynchronous preflight finished after selection read its predecessor.
        # The exact original bytes are recoverable by their preregistered hash:
        # only the CLI report-filter source hash differs, not one case/request.
        earlier=read(HERE/'inventory-selection-v1.json')
        assert sha(HERE/'inventory-selection-v1.json')==selection['inventory_sha256']
        assert earlier['judge_source_sha256']==sha(HERE/'judge-selection-v1.py')
        left,right=dict(earlier),dict(inventory)
        left.pop('judge_source_sha256');right.pop('judge_source_sha256')
        assert left==right,'Selection cohort or request identities changed'
        old=(HERE/'judge-selection-v1.py').read_text()
        new=(ROOT/'src/multi_turn_eval/judging/claude_judge.py').read_text()
        assert old.replace('records = [r for r in records if r["turn"] in only_turns]',
                           'records = [r for r in records if r.get(\'recovery_turn\') or r["turn"] in only_turns]')==new
        inventory_reconciliation={'selection_inventory_sha256':sha(HERE/'inventory-selection-v1.json'),
                                  'diagnostic_inventory_sha256':sha(HERE/'inventory.json'),
                                  'only_difference':'judge_source_sha256 for CLI output-side --only-turns filter; all models/cases/effective requests byte-identical'}
    cases={c['job_id']:c for c in inventory['cases']}
    aliases={e['case']:e['job_id'] for e in selection['primary']}
    diagnostic={}
    bindings={}
    for phase in ['validation-A','validation-B','development-A','development-B']:
        ids=aliases.values() if phase.startswith('validation') else [c['job_id'] for c in driver.development_cases(inventory)]
        diagnostic[phase]={}
        for job in ids:
            path=HERE/'judging'/phase/job/'result.json'
            assert driver.valid_result(path,cases[job])
            diagnostic[phase][job]=read(path)
            bindings[str(path.relative_to(HERE))]=sha(path)
    v=read(HERE/'VALIDATION.json')
    assert v['passed'] and v['reference_counts']==refs['counts']
    assert refs['minimum_gate_passed']
    for phase in ['validation-A','validation-B']:
        matches=0
        for row in refs['consensus']:
            j=diagnostic[phase][aliases[row['case']]]['judgments'][str(row['turn'])]
            matches+=j['scores'][row['dimension']]==row['expected']
        assert matches==v['repetitions'][phase]['counts']['overall']['matched']
        assert len(refs['consensus'])==v['repetitions'][phase]['counts']['overall']['total']
        assert matches/len(refs['consensus'])>=.95
    disagreements=0
    for job in aliases.values():
        a,b=diagnostic['validation-A'][job]['judgments'],diagnostic['validation-B'][job]['judgments']
        disagreements+=sum(a[str(t)]['scores'][d]!=b[str(t)]['scores'][d] for t in range(30) for d in DIMS)
    assert disagreements==v['repeatability']['discordant'] and disagreements/2160<=.02
    for e in v['development']['checks']:
        job=('refresh--gemini-3.5-flash-023-68432ef89f55' if e['case']=='support' else 'refresh--deepseek-v4-flash-0731-high-023-c4c7afd555bf')
        actual=diagnostic[e['phase']][job]['judgments'][str(e['turn'])]['scores'][e['dimension']]
        assert actual==e['actual']==e['expected']
    assert v['development']['passed']
    assert read(HERE/'RECOVERY-PUBLIC-TEXT-RESTORATIONS-v1.json')['entries']==[], 'Nonempty appendix requires a new versioned restoration policy'
    return {'validation_sha256':sha(HERE/'VALIDATION.json'),'registration_sha256':sha(HERE/'validation-request-registration.json'),
            'selection_sha256':sha(HERE/'validation-selection.json'),'reference_freeze_sha256':sha(HERE/'references/FROZEN.json'),
            'diagnostic_results':bindings,'validator_source_sha256':sha(HERE/'validate.py'),
            'acceptance_source_sha256':sha(HERE/'acceptance.py'),'frozen_validation_driver_sha256':sha(HERE/'driver-validation-v1.py'),
            'canonical_driver_sha256':sha(HERE/'driver.py'),'passed':True,
            'inventory_reconciliation':inventory_reconciliation,
            'note':'Additive authentication of registered unchanged reference/diagnostic evidence. No labels, thresholds or requests regenerated.'}

def verify_acceptance(create=False):
    current=evidence()
    path=HERE/'VALIDATION-ACCEPTED.json'
    if path.exists():assert read(path)==current,'Accepted validation chain changed'
    elif create:dump(path,current)
    else:raise AssertionError('No accepted validation chain')
    return current

if __name__=='__main__':
    verify_acceptance(create=True)
    print('Registered reference/diagnostic evidence authenticated',flush=True)
