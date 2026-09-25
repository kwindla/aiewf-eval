"""Uniform, resumable Opus 5 rejudge; source captures/first-pass outputs are read-only."""
import asyncio
import hashlib
import json
import os
import time
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'rejudge-policy-compliance-1'
LIVE = ROOT / 'docs/gpt-live-readiness-2026-09-15'
sys.path.insert(0, str(ROOT))
from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig
from benchmarks.aiwf_medium_context_speech_v2.judge import JUDGE_REVISION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=True)
    config = BenchmarkConfig()
    frozen = json.loads((HERE / 'protocol.json').read_text())
    assert hashlib.sha256(config.system_instruction.encode()).hexdigest() == frozen['prompt_sha256']
    records = [json.loads(l) for l in (HERE / 'capture/attempts.jsonl').read_text().splitlines()]
    runs = [dict(key=f"{r['arm']}-{r['index']:02d}", arm=r['arm'], run_dir=r['run_dir']) for r in records]
    first = re.search(r'^Output directory: (.+)$', (LIVE / 'standard-prompt-full-validation.log').read_text(), re.M).group(1)
    runs.append(dict(key='live-single', arm='live', run_dir=first))
    for r in sorted([json.loads(l) for l in (LIVE / 'concurrency-pilot/attempts.jsonl').read_text().splitlines()], key=lambda r:r['index']):
        runs.append(dict(key=f"live-{r['index']:02d}", arm='live', run_dir=r['run_dir']))
    assert len(runs) == 131 and len({r['run_dir'] for r in runs}) == 131
    for r in runs:
        source = ROOT / r['run_dir']
        r['original_hashes'] = {name:digest(source/name) for name in ['transcript.jsonl', 'claude_judged.jsonl', 'claude_summary.json', 'claude_analysis.md', 'benchmark-provenance.json']}
        old = json.loads((source/'claude_summary.json').read_text())
        assert old['judge_prompt_sha256'] == frozen['judge_prompt_sha256']
    protocol = dict(revision=JUDGE_REVISION, judge_model='claude-opus-5', judge_prompt_sha256=hashlib.sha256(config.judge_system_prompt.encode()).hexdigest(), original_judge_prompt_sha256=frozen['judge_prompt_sha256'], model_prompt_sha256=frozen['prompt_sha256'], runs=runs, concurrency=5, policy='One complete successful judgment per conversation; retry execution/parse failures only. No score-based retries or overrides. Reuse original per-turn audio assessment; quantitative audio metrics unchanged.')
    target=OUT/'protocol.json'
    if target.exists(): assert json.loads(target.read_text()) == protocol
    else: target.write_text(json.dumps(protocol, indent=2)+'\n')
    (OUT/'judge-prompt.txt').write_text(config.judge_system_prompt)
    return protocol


async def worker(key):
    from dotenv import load_dotenv
    from multi_turn_eval.judging.claude_judge import judge_with_claude, load_transcript, write_outputs
    import multi_turn_eval.judging.turn_taking as timing
    load_dotenv(ROOT/'.env')
    protocol=json.loads((OUT/'protocol.json').read_text())
    record=next(r for r in protocol['runs'] if r['key']==key)
    source=ROOT/record['run_dir']; dest=OUT/key;dest.mkdir(exist_ok=True)
    old=[json.loads(l) for l in (source/'claude_judged.jsonl').read_text().splitlines()]
    # Use the identical precomputed per-turn audio evidence supplied by the original
    # judge. This worker is a separate process, so the substitution is local.
    cached=timing.TurnTakingAnalysis(run_dir=str(source))
    for t in old:
        ok=t['scores']['turn_taking']
        cached.per_turn[t['turn']]=timing.TurnTakingResult(t['turn'],ok,t.get('turn_taking_issues',[]))
        if not ok: cached.failed_turns.append(t['turn'])
    cached.overall_ok=not cached.failed_turns
    timing.analyze_turn_taking=lambda _:cached
    config=BenchmarkConfig()
    assert hashlib.sha256(config.judge_system_prompt.encode()).hexdigest()==protocol['judge_prompt_sha256']
    result=await judge_with_claude(source, expected_turns=config.turns, judge_model=protocol['judge_model'], judge_system_prompt=config.judge_system_prompt)
    write_outputs(dest, load_transcript(source), result['judgments'], result['summary'], result['model_name'], result.get('realignment_notes',''), result.get('function_tracking',{}), result.get('turn_taking_analysis'), judge_metadata=result['judge_metadata'])
    new=[json.loads(l) for l in (dest/'claude_judged.jsonl').read_text().splitlines()]
    assert {t['turn'] for t in new}=={t['turn'] for t in old}
    assert {t['turn']:t['scores']['turn_taking'] for t in new}=={t['turn']:t['scores']['turn_taking'] for t in old}
    for name,expected in record['original_hashes'].items():assert digest(source/name)==expected
    (dest/'done.json').write_text(json.dumps(dict(key=key, run_dir=record['run_dir'], turns=len(new), judge_prompt_sha256=protocol['judge_prompt_sha256']))+'\n')


async def campaign():
    protocol=prepare();concurrency=int(os.environ.get('REJUDGE_CONCURRENCY',protocol['concurrency']));sem=asyncio.Semaphore(concurrency)
    adopted=json.loads((OUT/'adopted-workers.json').read_text()) if (OUT/'adopted-workers.json').exists() else {}
    with (OUT/'execution-history.jsonl').open('a') as f:f.write(json.dumps(dict(start_unix=time.time(),concurrency=concurrency,adopted=adopted))+'\n')
    async def run(record):
        key=record['key'];dest=OUT/key;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():return
        async with sem:
            if key in adopted:
                while Path(f"/proc/{adopted[key]}").exists() and not (dest/'done.json').exists():await asyncio.sleep(2)
                if (dest/'done.json').exists():return
            for attempt in range(1,4):
                log=dest/f'attempt-{attempt}.log'
                if log.exists():continue
                with log.open('w') as f:
                    p=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve()),'--worker',key,stdout=f,stderr=asyncio.subprocess.STDOUT,cwd=ROOT)
                    try:code=await asyncio.wait_for(p.wait(),timeout=900)
                    except asyncio.TimeoutError:
                        p.kill();await p.wait();code=124
                if code==0:
                    s=json.loads((dest/'claude_summary.json').read_text())
                    print('DONE',key,s['turn_pass'],flush=True);return
                print('EXECUTION_FAILURE',key,attempt,code,flush=True)
            raise RuntimeError(f'No complete judgment: {key}')
    results=await asyncio.gather(*(run(r) for r in protocol['runs']),return_exceptions=True)
    errors=[str(r) for r in results if isinstance(r,Exception)]
    if errors:raise RuntimeError(errors)
    (OUT/'complete.json').write_text(json.dumps(dict(conversations=131,judge_prompt_sha256=protocol['judge_prompt_sha256']))+'\n')
    print('COMPLETE 131 conversations',flush=True)

if __name__=='__main__':
    asyncio.run(worker(sys.argv[2]) if len(sys.argv)>1 and sys.argv[1]=='--worker' else campaign())
