"""Reproduce the compact publication or safely restore excluded local evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from scripts.campaign_publication_links import BUNDLE, ROOT, public_links

MANIFEST = ROOT / BUNDLE / 'manifest.json'
DROP = {'cases', 'ledger', 'conversations', 'inputs', 'attempts', 'jobs', 'astra_reference_usage', 'disagreements'}


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if not (key in DROP and isinstance(item, (list, dict)))}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def render(source, raw, output):
    if source.endswith('.md'):
        import posixpath
        text = public_links(raw.decode(), source, output)
        storage = posixpath.relpath(BUNDLE + '/README.md', posixpath.dirname(output))
        return (text + '\n[Evidence storage and local audit instructions](' + storage + ').\n').encode()
    data = {'publication_summary': True, 'source_artifact': source,
            'source_sha256': hashlib.sha256(raw).hexdigest(),
            'evidence_storage': os.path.relpath(BUNDLE + '/README.md', Path(output).parent),
            'data': compact(json.loads(raw))}
    return (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode()


def default_archive():
    return Path(os.environ.get('AIEWF_EVIDENCE_ARCHIVE',
        str(Path.home() / 'src/aiewf-eval-archive/2026-10-benchmark-refresh/evidence.tar.gz')))


class EvidenceArchive:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads(MANIFEST.read_text())
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.metadata['archive_sha256']:
            raise ValueError('Archive checksum mismatch')
        self.tar = tarfile.open(self.path, 'r:gz')
        member_manifest = self.tar.extractfile('MANIFEST.json').read()
        if hashlib.sha256(member_manifest).hexdigest() != self.metadata['member_manifest_sha256']:
            self.tar.close()
            raise ValueError('Member manifest checksum mismatch')
        self.members = {entry['path']: entry for entry in json.loads(member_manifest)['files']}

    def read(self, name):
        raw = self.tar.extractfile(name).read()
        if hashlib.sha256(raw).hexdigest() != self.members[name]['sha256']:
            raise ValueError('Member checksum mismatch: ' + name)
        return raw

    def close(self):
        self.tar.close()


def restore(archive):
    """Restore only excluded ops evidence; never overwrite a tracked/current file."""
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().strip('\0').split('\0'))
    candidates = [name for name in archive.members if name.startswith('ops/') and name not in tracked]
    # Validate the complete restore first; an existing different file is never replaced.
    for name in candidates:
        destination = ROOT / name
        if not destination.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Archive path escapes repository: ' + name)
        if any(p.is_symlink() for p in [destination, *destination.parents] if p.is_relative_to(ROOT)):
            raise ValueError('Archive path uses symlink: ' + name)
        if destination.exists():
            if destination.read_bytes() != archive.read(name):
                raise ValueError('Existing evidence differs: ' + name)
    for name in candidates:
        destination = ROOT / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.restore-', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(archive.read(name))
                    stream.flush()
                    os.fsync(stream.fileno())
                    # Atomic creation, with no overwrite if another writer won.
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
    return len(candidates)


def regenerate(archive):
    metadata = archive.metadata
    for source, entry in metadata['publication_sources'].items():
        raw = archive.read(source)
        if hashlib.sha256(raw).hexdigest() != entry['source_sha256']:
            raise ValueError('Publication source checksum mismatch: ' + source)
        output = ROOT / entry['path']
        output.write_bytes(render(source, raw, entry['path']))
        entry['published_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    # Historical entry points use relocated links; diagrams remain byte-identical.
    for source, entry in metadata['public_entrypoints'].items():
        raw = archive.read(source)
        if hashlib.sha256(raw).hexdigest() != entry['source_sha256']:
            raise ValueError('Publication source checksum mismatch: ' + source)
        output = public_links(raw.decode(), source).encode() if source.endswith('.md') else raw
        (ROOT / source).write_bytes(output)
        entry['published_sha256'] = hashlib.sha256(output).hexdigest()
    MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=default_archive())
    parser.add_argument('--restore-evidence', action='store_true', help='Restore only excluded ops paths; never replace tracked files.')
    args = parser.parse_args()
    archive = EvidenceArchive(args.archive)
    try:
        if args.restore_evidence:
            print('Verified local evidence paths:', restore(archive))
        else:
            regenerate(archive)
            print('Regenerated compact publication from verified archive bytes')
    finally:
        archive.close()


if __name__ == '__main__':
    main()
