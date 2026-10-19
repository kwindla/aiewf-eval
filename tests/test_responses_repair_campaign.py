"""The prospective replacement must preserve assignment, raw history and grades."""
import hashlib,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'ops/responses-repair-20261018'
DIMS=('tool_use_correct','instruction_following','kb_grounding')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.mark.local_evidence
def test_repair_scope_matches_actual_transport_audit_and_explicit_settings():
    inventory=read(HERE/'impact-inventory.json')
    configurations=[read(p) for p in HERE.glob('*/configuration.json')]
    assert len(configurations)==inventory['rerun_candidate_count']==12
    assert sum(c['target_eligible_runs'] for c in configurations)==360
    candidates={r['key']:r for r in inventory['rows'] if r['rerun_candidate']}
    for path in HERE.glob('*/configuration.json'):
        c=read(path);r=candidates[path.parent.name]['reproduction']
        assert c['model']==r['model']
        assert {**c['common_environment'],**c['arms']['standard']['environment']}==r['environment']
        assert (ROOT/c['schedule_path']).read_text().splitlines()==['slot\tarm']+[f'{i}\tstandard' for i in range(1,31)]
        assert c['eligibility']['policy']=='first_valid_response'
    assert not any(r['rerun_candidate'] for r in inventory['voice_profiles'])


@pytest.mark.local_evidence
def test_fresh_results_reconcile_every_assignment_and_preserve_native_continuity():
    if not (HERE/'aggregate.json').exists():pytest.skip('Fresh study still running')
    data=read(HERE/'aggregate.json');inputs=read(HERE/'judging/inputs.json')
    assert data['all_quality_complete'] and data['judged_conversations']==360
    cases={c['job_id']:c for c in inputs['cases']};assert len(cases)==360
    total_cost=0;seen=set()
    for model in data['text_models']:
        strict=0;complete_strict=0;complete_n=0;dimensions={d:0 for d in DIMS};all_dimensions={d:0 for d in DIMS}
        for job in model['cases']:
            assert job not in seen;seen.add(job)
            c=cases[job];assert c['name']==model['key']
            source=ROOT/c['source'];assert sha(source)==c['transcript_sha256']==sha(ROOT/c['input'])
            records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            scheduled=[r for r in records if not r.get('recovery_turn')]
            is_complete=len(scheduled)==30 and all(r.get('assistant_text','').strip() or r.get('tool_calls') for r in scheduled)
            complete_n+=is_complete
            grade=read(HERE/'judging/results'/(job+'.json'));meta=grade['metadata']
            assert grade['transcript_sha256']==c['transcript_sha256']
            assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
            assert meta['judge_response_models']==['claude-opus-5-5'] and meta['judge_tools']==[]
            assert set(grade['judgments'])==set(map(str,c['observed_turns']))
            total_cost+=meta.get('judge_cost_usd',0) or 0
            for turn in grade['judgments'].values():
                scores=turn['scores'];assert all(type(scores[d]) is bool for d in DIMS)
                strict+=all(scores[d] for d in DIMS)
                for d in DIMS:all_dimensions[d]+=scores[d]
                if is_complete:
                    complete_strict+=all(scores[d] for d in DIMS)
                    for d in DIMS:dimensions[d]+=scores[d]
            audit=[json.loads(line) for line in (source.parent/'responses-history-audit.jsonl').read_text().splitlines()]
            last=None;prior_outputs=[]
            for row in audit:
                if row['event']=='request':
                    assert row['store'] is True
                    if model['key']=='gpt-5.4-none-dots96':
                        assert row['previous_response_id'] is None
                        # Every earlier native output item is replayed unchanged.
                        wire_hashes={i['sha256'] for i in row['input_items']}
                        assert all(i['sha256'] in wire_hashes for i in prior_outputs)
                    else:
                        assert row['previous_response_id']==last
                        if last:
                            assert not any(i.get('role') in ['assistant','system'] or i['type'] in ['function_call','reasoning'] for i in row['input_items'])
                            expected={i['call_id'] for i in prior_outputs if i['type']=='function_call' and i.get('name')!='end_session'}
                            actual={i['call_id'] for i in row['input_items'] if i['type']=='function_call_output'}
                            assert expected<=actual,(job,expected,actual)
                else:
                    last=row['id']
                    if model['key']=='gpt-5.4-none-dots96':prior_outputs.extend(row['output_items'])
                    else:prior_outputs=row['output_items']
        assert len(model['cases'])==model['assigned']==30
        assert complete_n==model['complete']
        assert model['all_assigned']['strict_passes']==strict
        assert model['all_assigned']['dimension_passes']==all_dimensions
        assert model['all_assigned']['denominator']==900
        assert abs(model['all_assigned']['strict_percent']-strict/9)<1e-10
        assert model['complete_only']['strict_passes']==complete_strict
        assert model['complete_only']['denominator']==30*complete_n
        assert all(model['complete_only']['dimension_passes'].get(d,0)==dimensions[d] for d in DIMS)
        if not complete_n:assert model['strict_percent'] is None
    assert seen==set(cases)
    assert abs(total_cost-data['successful_judge_cost_usd'])<1e-8


@pytest.mark.local_evidence
def test_operational_replacements_preserve_original_evidence_and_fixed_assignments():
    if not (HERE/'OPERATIONAL-REPLACEMENTS-COMPLETE.json').exists() or not (HERE/'judging/inputs.json').exists():pytest.skip('Operational audit awaits collection')
    ledger=read(HERE/'OPERATIONAL-REPLACEMENTS.json')
    assert ledger['amendment_sha256']==sha(HERE/'OPERATIONAL-AMENDMENT.md')
    cases={ (c['name'],int(c['slot'])):c for c in read(HERE/'judging/inputs.json')['cases'] }
    for entry in ledger['replacements']:
        original=entry['original'];accepted=entry['accepted']
        assert original['run_dir']!=accepted['run_dir']
        assert entry['evidence']['uncompleted_requests']
        assert entry['evidence']['provider_stream_error_evidence']
        assert sha(HERE/entry['name']/'COLLECTED-ORIGINAL.json')==entry['original_collected_sha256']
        assert sha(HERE/entry['name']/'artifacts/canonical.tsv')==entry['portable_canonical_sha256']
        for name,digest in entry['evidence']['raw_sha256'].items():
            assert sha(ROOT/original['run_dir']/name)==digest
        assert int(original['attempt'])<int(accepted['attempt'])<=3
        assert int(original['slot'])==int(accepted['slot'])==entry['slot']
        assert cases[(entry['name'],entry['slot'])]['source']==accepted['run_dir']+'/transcript.jsonl'
    assert read(HERE/'OPERATIONAL-REPLACEMENTS-COMPLETE.json')['ledger_sha256']==sha(HERE/'OPERATIONAL-REPLACEMENTS.json')


@pytest.mark.parametrize('terminal_timeout', [True,False])
def test_wire_audit_retains_terminal_stall_but_rejects_continuing_after_unfinished_response(tmp_path,terminal_timeout):
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_capture_test',HERE/'audit_capture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def request(sequence,parent):
        return {'event':'request','sequence':sequence,'model':'gpt-6-luna','store':True,
                'service_tier':'priority','reasoning':{'effort':'none'},'sampling':{},
                'previous_response_id':parent,'input_items':[{'type':'message','role':'user'}]}
    def complete(sequence,id):
        return {'event':'completed','sequence':sequence,'model':'gpt-6-luna','id':id,'output_items':[]}
    rows=[request(1,None),complete(1,'r1'),request(2,'r1')]
    if not terminal_timeout:rows += [request(3,'r1'),complete(3,'r3')]
    (tmp_path/'responses-history-audit.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (tmp_path/'run.log').write_text('queue_turn: turn_idx=0\nRecorded turn 0\nqueue_turn: turn_idx=1\nIdle timeout detected.\n')
    config={'model':'gpt-6-luna','accepted_response_model_ids':['gpt-6-luna'],
            'common_environment':{'MTE_OPENAI_RESPONSES_REASONING_EFFORT':'none'},'arms':{'standard':{'environment':{}}}}
    if terminal_timeout:
        result=module.validate(tmp_path,config)
        assert result['native_or_cloud_history_verified'] and result['terminal_unanswered_timeout']
        assert result['uncompleted_requests']==[2]
    else:
        with pytest.raises(AssertionError):module.validate(tmp_path,config)


@pytest.mark.local_evidence
def test_public_text_restorations_are_exact_provider_output_with_unchanged_raw_sources():
    if not (HERE/'PUBLIC-TEXT-RESTORATIONS.json').exists() or not (HERE/'judging/COMPLETE.json').exists():pytest.skip('Public-output restoration awaits final collection')
    manifest=read(HERE/'PUBLIC-TEXT-RESTORATIONS.json')
    assert manifest['amendment_sha256']==sha(HERE/'PUBLIC-TEXT-AMENDMENT.md')
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
    for entry in manifest['entries']:
        source=ROOT/entry['source'];cache=ROOT/entry['provider_response']
        assert sha(source)==entry['source_sha256'] and sha(cache)==entry['provider_response_sha256']
        records=[json.loads(line) for line in source.read_text().splitlines()]
        record=next(r for r in records if digest(r)==entry['record_sha256'])
        assert not record['assistant_text'].strip() and not record.get('recovery_turn')
        assert hashlib.sha256(record['assistant_text'].encode()).hexdigest()==entry['source_text_sha256']
        provider=read(cache);assert provider['id']==entry['provider_response_id']
        audit=[json.loads(line) for line in (source.parent/'responses-history-audit.jsonl').read_text().splitlines()]
        event=next(e for e in audit if e['event']=='completed' and e['id']==provider['id'])
        assert len(provider['output'])==len(event['output_items'])
        omitted=[]
        for item,captured in zip(provider['output'],event['output_items']):
            if digest(item)==captured['sha256']:continue
            assert item['type']==captured['type']=='reasoning'
            assert captured['has_encrypted_reasoning'] and not item.get('encrypted_content')
            assert item['id']==captured['id']
            omitted.append(item['id'])
        assert entry['native_public_output_items_verified']
        assert entry['unretrieved_encrypted_reasoning_item_ids']==omitted
        assert entry['native_output_items_verified']==(not omitted)
        public=''.join(part['text'] for item in provider['output'] if item['type']=='message' for part in item.get('content',[]) if part['type']=='output_text')
        assert public==entry['public_text']
        assert hashlib.sha256(public.encode()).hexdigest()==entry['public_text_sha256']
        grade=read(HERE/'judging/results'/(entry['name']+'-standard-'+f"{entry['slot']:03d}"+'.json'))
        assert grade['metadata']['judge_input_normalization']=='verified-public-output-and-think-sections-v1'
        assert any(c.get('provider_response_id')==provider['id'] and c['reason']=='verified_public_output_restore' for c in grade['metadata']['judge_normalized_turns'])
