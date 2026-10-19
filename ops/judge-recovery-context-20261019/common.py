"""Frozen evidence and request identities for the recovery-context migration."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = 'claude-opus-5-5'
DIMS = ('tool_use_correct', 'instruction_following', 'kb_grounding')

def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()

def sha(path):
    return digest(Path(path).read_bytes())

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def rootpath(path):
    return ROOT / path

def records(case):
    raw = rootpath(case['source']).read_bytes()
    assert digest(raw) == case['transcript_sha256']
    assert raw == rootpath(case['input']).read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

def rubric(case):
    return (HERE / 'legacy' / (case['mode'] + '-rubric.txt')).read_text()

def turns(case):
    return read(HERE / 'legacy' / (case['mode'] + '-turns.json'))

def audio(case):
    value = case.get('turn_taking_data')
    return None if value is None else {int(k): v for k, v in value.items()}

def old_grade(case):
    assert sha(rootpath(case['old_grade'])) == case['old_grade_sha256']
    return read(rootpath(case['old_grade']))

def restoration_entries():
    manifest = ROOT/'ops/responses-repair-20261018/PUBLIC-TEXT-RESTORATIONS.json'
    data = read(manifest)
    assert data['amendment_sha256'] == sha(manifest.parent/'PUBLIC-TEXT-AMENDMENT.md')
    for entry in data['entries']:
        assert sha(rootpath(entry['source'])) == entry['source_sha256']
        assert sha(rootpath(entry['provider_response'])) == entry['provider_response_sha256']
        assert digest(entry['public_text']) == entry['public_text_sha256']
        assert entry['native_public_output_items_verified']
    return data['entries']

def restored_records(case):
    raw = records(case)
    entries = [e for e in restoration_entries() if e['source_sha256'] == case['transcript_sha256']]
    applied = []
    for record in raw:
        fingerprint = digest(json.dumps(record, sort_keys=True, separators=(',', ':')))
        found = [entry for entry in entries if entry['record_sha256'] == fingerprint]
        if found:
            assert len(found) == 1 and not record['assistant_text'].strip()
            entry = found[0]
            assert digest(record['assistant_text']) == entry['source_text_sha256']
            record['assistant_text'] = entry['public_text']
            applied.append(entry['record_sha256'])
    assert sorted(applied) == sorted(e['record_sha256'] for e in entries), 'Unapplied restoration'
    return raw, applied

def load_legacy():
    name = 'multi_turn_eval.judging.recovery_legacy'
    spec = importlib.util.spec_from_file_location(name, HERE/'legacy/src/multi_turn_eval/judging/claude_judge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

async def request_identity(module, case):
    """Stop immediately before the query; never contact a model or touch a source."""
    class Captured(Exception):
        pass
    captured = {}
    async def query(*, prompt, options):
        captured.update(user_sha256=digest(prompt), system_sha256=digest(options.system_prompt),
                        model=options.model, effort=options.effort, thinking=options.thinking,
                        tools=options.tools, setting_sources=options.setting_sources)
        raise Captured()
        yield
    original_query, original_load = module.query, module.load_transcript
    module.query = query
    restored, applied = restored_records(case)
    module.load_transcript = lambda _: deepcopy(restored)
    try:
        try:
            await module.judge_with_claude(
                HERE/'preflight', expected_turns=turns(case), skip_turn_taking=audio(case) is None,
                precomputed_turn_taking=audio(case), judge_model=MODEL, judge_effort='medium',
                judge_system_prompt=rubric(case),
                task_clarification=old_grade(case)['metadata'].get('judge_task_clarification'))
        except Captured:
            pass
        assert captured
        return captured, applied
    finally:
        module.query, module.load_transcript = original_query, original_load

def verify_prior():
    for manifest in ['legacy-sha256.json', 'prior-evidence-sha256.json']:
        data = read(HERE/manifest)
        for name, expected in data.items():
            path = Path(name)
            if not path.is_absolute():
                path = (HERE if manifest == 'legacy-sha256.json' else ROOT)/path
            assert sha(path) == expected, ('Prior evidence changed', str(path))

def credentials():
    from dotenv import dotenv_values
    for path in ['/home/khkramer/src/aiewf-eval/.env', '/home/khkramer/src/gb-benchmarks/.env']:
        key = dotenv_values(path).get('ANTHROPIC_API_KEY')
        if key:
            os.environ.setdefault('ANTHROPIC_API_KEY', key)
