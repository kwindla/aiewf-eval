"""Verify capture-time source bytes, including archived versions after capture.

Generation always uses the portable collector's strict live-source check. This
analysis-only context accepts a changed current file only when the original bytes
are present in the pre-capture runtime snapshot and match the recorded manifest.
No source manifest is rewritten.
"""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path

def entries(text):
 result={}
 for line in text.splitlines():
  if not line.strip():continue
  digest,name=line.split(maxsplit=1);name=name.strip()
  if name in result:raise ValueError('Duplicate captured source path: '+name)
  result[name]=digest
 return result

def resolve(config,live_text,study):
 manifest=config['_artifact_dir']/'source-sha256.txt'
 recorded=manifest.read_text();expected=entries(recorded);live=entries(live_text)
 if set(live)!=set(expected):raise ValueError('Capture source path set changed')
 snapshot_hashes=json.loads((study/'runtime-sha256.json').read_text());resolutions=[]
 for name,digest in expected.items():
  if live[name]==digest:continue
  relative=Path(name)
  if relative.is_absolute() or '..' in relative.parts:
   raise ValueError('Cannot resolve captured source outside the recorded repository')
  snapshot=study/'runtime-snapshot'/relative
  if snapshot_hashes.get(name)!=digest or not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
   raise ValueError('Changed source has no matching capture-time snapshot: '+name)
  resolutions.append({'path':name,'captured_sha256':digest,'current_sha256':live[name],
                      'snapshot':str(snapshot.relative_to(study))})
 return recorded,resolutions

@contextmanager
def captured_sources(collector,study):
 original=collector.source_hashes;resolutions={}
 def source_hashes(config):
  recorded,items=resolve(config,original(config),study)
  for item in items:resolutions[item['path']]=item
  return recorded
 collector.source_hashes=source_hashes
 try:yield resolutions
 finally:collector.source_hashes=original
