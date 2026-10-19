"""Reconcile the published Nova cohort to immutable captures, grades and WAVs."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'ops/nova-2-5-sonic-20261019/user-context-v2'
DIMS=('tool_use_correct','instruction_following','kb_grounding')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_nova_reference_layout_is_explicit_and_preserves_all_benchmark_text():
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
    config=read(HERE/'configuration.json')
    assert config['model']=='amazon.nova-2-5-sonic' and config['target_complete']==25
    assert config['common_environment']=={'MTE_NOVA_KB_CONTEXT_ROLE':'user','MTE_ENABLE_RECOVERY':'0'}
    assert BenchmarkConfig.system_policy+BenchmarkConfig.reference_context==BenchmarkConfig.system_instruction
    assert sha(ROOT/'benchmarks/aiwf_medium_context_speech_v2/prompts/system.py')==read(HERE.parent/'preflight-source-sha256.json')['benchmarks/aiwf_medium_context_speech_v2/prompts/system.py']

def test_published_nova_reconciles_all_scores_and_preserves_the_previous_publication():
    if not (HERE/'PUBLICATION.json').exists():pytest.skip('Nova collection/judging has not completed')
    data=read(HERE/'aggregate.json');m=data['voice_models'][0];inputs=read(HERE/'judging/inputs.json')
    assert data['all_quality_complete'] and data['judged_conversations']==25
    assert len(inputs['cases'])==25 and m['complete']==25
    counts={d:0 for d in DIMS};strict=0;turn_taking=0;cost=0
    for case in inputs['cases']:
        assert sha(ROOT/case['source'])==case['transcript_sha256']
        assert (ROOT/case['source']).read_bytes()==(ROOT/case['input']).read_bytes()
        grade=read(HERE/'judging/results'/(case['job_id']+'.json'));meta=grade['metadata']
        assert grade['transcript_sha256']==case['transcript_sha256']
        assert meta['judge_model']=='claude-opus-5-5' and meta['judge_effort']=='medium'
        assert meta['judge_response_models']==['claude-opus-5-5']
        assert meta['judge_tools']==[] and meta['judge_setting_sources']==[]
        assert meta['judge_prompt_sha256']==inputs['rubric_sha256']
        assert set(grade['judgments'])==set(map(str,range(30)))
        cost+=meta['judge_cost_usd']
        for row in grade['judgments'].values():
            scores=row['scores'];assert all(type(scores[d]) is bool for d in DIMS+('turn_taking',))
            strict+=all(scores[d] for d in DIMS);turn_taking+=scores['turn_taking']
            for d in DIMS:counts[d]+=scores[d]
        proof=read(HERE/'audio'/(case['job_id']+'-source.json'))
        for name,digest in proof['identity']['raw_sha256'].items():assert sha((ROOT/case['source']).parent/name)==digest
        assert sha(HERE/'audio'/(case['job_id']+'.json'))==proof['metrics_sha256']
        wave=HERE/'audio'/(case['job_id']+'-input-waveform.json')
        assert sha(wave)==proof['input_waveform_sha256'] and not read(wave)['issues']
    assert counts==m['complete_only']['dimension_passes'] and strict==m['complete_only']['strict_passes']
    assert m['strict_percent']==100*strict/750 and turn_taking==m['turn_taking_passes']
    assert abs(cost-data['successful_judge_cost_usd'])<1e-9
    assert m['first_speech_positive']['n']+m['nonpositive_n']+m['missing_latency_n']==750
    attempts=read(HERE/'attempts.json');assert m['eligible_attempts']==len(attempts)
    assert m['failed_conversations']==sum(not a['complete'] for a in attempts)
    # Campaign drivers intentionally have local common/analyze module names.
    # Isolate their imports from other campaigns loaded by the full test suite.
    script='''
import json,sys
from pathlib import Path
here=Path(sys.argv[1]);sys.path.insert(0,str(here))
import publish
root=here.parents[2]
from scripts.campaign_publication_links import public_links
assert (root/'README.md').read_text()==public_links(publish.render((here/'prior/README.md').read_text(),json.loads((here/'aggregate.json').read_text())))
'''
    environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    check=subprocess.run([sys.executable,'-c',script,str(HERE)],cwd=ROOT,env=environment,capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    for name,digest in read(HERE/'generation-sha256.json').items():
        assert sha(ROOT/name)==digest
        assert sha(HERE/'runtime-snapshot'/name)==digest

# These reconcile immutable full captures, not just the public summary.
pytestmark = pytest.mark.local_evidence
