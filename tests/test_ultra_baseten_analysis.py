"""Offline check of early termination and recovery accounting in the fresh study."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_early_end_stays_in_fixed_denominator_and_recovery_only_changes_latency(tmp_path, monkeypatch):
    folder=Path(__file__).resolve().parents[1]/'ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('ultra_analysis',folder/'analyze.py')
    analysis=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    here=tmp_path/'study';here.mkdir()
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    monkeypatch.setattr(analysis,'HERE',here)
    monkeypatch.setattr(analysis.judge,'ROOT',tmp_path)
    monkeypatch.setattr(analysis.judge,'HERE',here)
    cases=[]
    for slot in range(1,61):
        arm='off' if slot%2 else 'on'
        n=7 if slot==1 else 30
        rows=[{'turn':t,'model_name':'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B',
               'assistant_text':'Answer','tool_calls':[], 'ttfb_ms':100,'raw_ttfb_ms':10,
               'tokens':{'prompt_tokens':100,'completion_tokens':10,
                         'cache_read_input_tokens':50,'thinking_tokens':0}} for t in range(n)]
        if slot==1:
            rows.append({**rows[-1],'turn':30,'recovery_turn':True,'ttfb_ms':200})
        raw=('\n'.join(json.dumps(r) for r in rows)+'\n').encode()
        source=tmp_path/f'source-{slot}.jsonl';source.write_bytes(raw)
        snapshot=tmp_path/f'input-{slot}.jsonl';snapshot.write_bytes(raw)
        case={'job_id':f'job-{slot}','slot':slot,'arm':arm,'source':source.name,
              'input':snapshot.name,'transcript_sha256':hashlib.sha256(raw).hexdigest(),
              'observed_turns':list(range(n))}
        cases.append(case)
        grade={'judgments':{str(t):{'scores':{d:True for d in analysis.DIMS}} for t in range(n)},
               'metadata':{'judge_cost_usd':.1}}
        analysis.judge.dump(analysis.judge.result_path(case),grade)
    monkeypatch.setattr(analysis.judge,'freeze',lambda:{'cases':cases})
    monkeypatch.setattr(analysis.judge,'valid',lambda _:True)
    monkeypatch.setattr(analysis.judge.collect,'read_tsv',lambda _:[{} for _ in range(60)])
    (here/'judging/COMPLETE.json').write_text('{}')
    (here/'smoke').mkdir()
    (here/'smoke/model-catalog.json').write_text(json.dumps({'data':[{
        'id':'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B',
        'pricing':{'prompt':'0.0000006','completion':'0.0000024','input_cache_read':'0.00000012'},
    }]}))
    analysis.main()
    data=json.loads((here/'aggregate.json').read_text())
    off,on=data['arms']['off'],data['arms']['on']
    assert off['complete_conversations']==29 and off['incomplete_conversations']==1
    assert off['observed_scripted_turns']==877 and off['missing_future_turns']==23
    assert off['known_strict_passes']==877
    assert off['strict_percent']==pytest.approx(877/900*100)
    assert off['complete_only_strict_percent']==100
    assert off['recovery_turns']==1 and off['ttfat']['n']==848
    assert off['ttfat']['max_ms']==200 and off['complete_only_ttfat']['n']==841
    assert on['complete_conversations']==30 and on['strict_percent']==100
    assert data['comparison']['on_minus_off_percentage_points']==pytest.approx(23/900*100)
    assert sum(a['successful_judge_cost_usd'] for a in (off,on))==pytest.approx(6)
    assert len(data['conversations'])==60
