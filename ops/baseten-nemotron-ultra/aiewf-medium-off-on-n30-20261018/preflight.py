"""Validate direct streaming and full Pipecat conversations for both Ultra arms."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('portable_collect', ROOT/'ops/aiewf-campaign-template/collect.py')
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    config=collect.validate_configuration(HERE/'configuration.json', require_serving_verified=False)
    collect.validate_schedule(config)
    if not args.execute:
        print('Preflight configuration valid; pass --execute for direct and full-harness requests')
        return
    key=collect.extract_credential(config)
    client=OpenAI(api_key=key,base_url=config['endpoint']['url'],timeout=180,max_retries=2)
    smoke=HERE/'smoke'
    smoke.mkdir(exist_ok=True)
    catalog=smoke/'model-catalog.json'
    if not catalog.exists():
        data=client.models.list().model_dump(mode='json')
        assert any(m['id']==config['model'] for m in data['data']), 'Requested released model missing'
        dump(catalog, data)
    for arm, effort in [('off','none'),('on','high')]:
        direct=smoke/f'direct-{arm}.json'
        if not direct.exists():
            request={
                'model':config['model'], 'messages':[{'role':'user','content':'What is 17 times 19?'}],
                'reasoning_effort':effort,'temperature':1.0,'top_p':0.95,'max_tokens':16384,
                'stream':True,'stream_options':{'include_usage':True},
            }
            content=[]; reasoning=[]; models=set(); finish=[]; usage=None
            start=time.monotonic(); first_chunk=None; first_answer=None
            for chunk in client.chat.completions.create(**request):
                if chunk.model:models.add(chunk.model)
                if chunk.usage:usage=chunk.usage.model_dump(mode='json')
                if not chunk.choices:continue
                if first_chunk is None:first_chunk=1000*(time.monotonic()-start)
                choice=chunk.choices[0]; delta=choice.delta
                thought=getattr(delta,'reasoning_content',None) or getattr(delta,'reasoning',None)
                if thought:reasoning.append(thought)
                if delta.content:
                    if first_answer is None:first_answer=1000*(time.monotonic()-start)
                    content.append(delta.content)
                if choice.finish_reason:finish.append(choice.finish_reason)
            answer=''.join(content)
            assert '323' in answer and '</think>' not in answer
            assert models=={config['model']}, models
            assert finish==['stop'], finish
            assert bool(reasoning)==(arm=='on'), (arm, 'unexpected reasoning separation')
            dump(direct, {'request':request,'response_models':sorted(models),'content':answer,
                          'reasoning_characters':sum(map(len,reasoning)), 'usage':usage,
                          'finish_reasons':finish,'raw_ttfb_ms':first_chunk,'ttfat_ms':first_answer})
        run=ROOT/config['paths']['run_output_root']/f'smoke-{arm}'
        result=smoke/f'harness-{arm}.json'
        if not result.exists():
            if not run.exists():
                env=collect.child_environment(config,arm,key)
                with (smoke/f'harness-{arm}.log').open('w') as log:
                    process=subprocess.run([sys.executable,str(collect.RUN_ONE),'--config',str(HERE/'configuration.json'),
                                            '--arm',arm,'--run-dir',str(run)],cwd=ROOT,env=env,
                                           stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                assert process.returncode==0, f'Inspect {arm} harness log'
            stats=collect.validate_transcript(config,run)
            collect.validate_run_provenance(config,arm,run)
            assert stats['turns']==stats['response_turns'], stats
            rows=collect.read_jsonl(run/'transcript.jsonl')
            scheduled=[r for r in rows if not r.get('recovery_turn')]
            assert all('</think>' not in r['assistant_text'] for r in scheduled)
            assert any(t['name']=='end_session' for r in scheduled for t in r['tool_calls'])
            assert all(isinstance(r.get('ttfb_ms'),(int,float)) and r['ttfb_ms']>0 for r in scheduled)
            dump(result, {'arm':arm,'run_dir':str(run.relative_to(ROOT)),**stats,
                          'positive_ttfat_turns':len(scheduled),'answer_reasoning_markers':0,
                          'note':'An early end_session is model behavior, not a reason to retry for a better smoke'})
        print(f'{arm}: direct streaming and harness smoke verified; see observed completion count',flush=True)


if __name__=='__main__':main()
