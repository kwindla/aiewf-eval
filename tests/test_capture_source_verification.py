"""Historical analysis must verify real frozen bytes, never waive changed sources."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('capture_source_verification',ROOT/'ops/recent-models-20261018/verify_capture_sources.py')
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

def fixture(tmp_path):
    digest=hashlib.sha256(b'original runtime').hexdigest()
    artifact=tmp_path/'artifacts';artifact.mkdir()
    text=digest+'  src/runtime.py\n'
    (artifact/'source-sha256.txt').write_text(text)
    archive=tmp_path/'runtime-snapshot/src/runtime.py';archive.parent.mkdir(parents=True)
    archive.write_bytes(b'original runtime')
    (tmp_path/'runtime-sha256.json').write_text(json.dumps({'src/runtime.py':digest}))
    return {'_artifact_dir':artifact},archive,text

def test_verified_capture_snapshot_is_used_and_live_checker_restored(tmp_path):
    config,archive,recorded=fixture(tmp_path)
    original=lambda config:'new-hash  src/runtime.py\n'
    collector=SimpleNamespace(source_hashes=original)
    with module.captured_sources(collector,tmp_path) as resolutions:
        assert collector.source_hashes(config)==recorded
        assert resolutions['src/runtime.py']['snapshot']=='runtime-snapshot/src/runtime.py'
    assert collector.source_hashes is original

@pytest.mark.parametrize('failure',['corrupt','missing','extra-path','wrong-hash-map'])
def test_unverifiable_capture_source_changes_are_rejected(tmp_path,failure):
    config,archive,recorded=fixture(tmp_path)
    live='new-hash  src/runtime.py\n'
    if failure=='corrupt':archive.write_bytes(b'corrupted')
    elif failure=='missing':archive.unlink()
    elif failure=='extra-path':live+='other-hash  src/new.py\n'
    else:(tmp_path/'runtime-sha256.json').write_text(json.dumps({'src/runtime.py':'wrong'}))
    with pytest.raises(ValueError):module.resolve(config,live,tmp_path)

def test_unchanged_live_sources_need_no_archived_file(tmp_path):
    config,archive,recorded=fixture(tmp_path)
    archive.unlink()
    text,resolutions=module.resolve(config,recorded,tmp_path)
    assert text==recorded and resolutions==[]
