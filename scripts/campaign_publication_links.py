"""Relocate archived publication links without changing their metric values."""

import json
from pathlib import Path
import posixpath
import re

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = 'docs/model-evaluations-2026-10'


def public_links(text, source='README.md', output=None):
    """Render a historical document at an explicitly chosen publication path."""
    mapping = json.loads((ROOT / BUNDLE / 'links.json').read_text())
    manifest = json.loads((ROOT / BUNDLE / 'manifest.json').read_text())
    retained = set(manifest['retained_campaign_files'])
    campaigns = {'/'.join(path.split('/')[:2]) for path in mapping}
    output = output or source
    # These publication claims referred directly to artifacts now kept locally.
    text = text.replace(
        'The [previous publication](ops/judge-refresh-opus55-20261017/historical/README.md) preserves the earlier scores, denominators, serving details, and study links (its relative links are rooted at this repository).',
        'Earlier scores, denominators, serving details and study links are preserved in the [local evidence archive](ops/judge-refresh-opus55-20261017/historical/README.md), under `ops/judge-refresh-opus55-20261017/historical/README.md`.')
    text = text.replace(
        'The [cleanup audit](ops/judge-refresh-opus55-20261017/reasoning-cleanup/manifest.json) records every changed input.',
        'The [cleanup summary](ops/judge-refresh-opus55-20261017/reasoning-cleanup/manifest.json) describes the normalization; the full per-input audit is preserved in the [local evidence archive](docs/model-evaluations-2026-10/README.md).')

    text = text.replace('[All cases and outcomes](aggregate.json)', '[Outcome counts and result hashes](aggregate.json)')
    text = text.replace('[full paired results](aggregate.json)', '[paired-result summary](aggregate.json)')

    def replace(match):
        image, label, url = match.groups()
        if re.match(r'^(?:https?:|mailto:|app:|data:|#)', url):
            return match.group(0)
        path, separator, anchor = url.partition('#')
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        target = mapping.get(resolved, resolved)
        if resolved.startswith('runs/'):
            target = BUNDLE + '/README.md'
            label += ' (local raw runs; unarchived)'
            image = separator = anchor = ''
        elif resolved not in mapping and resolved not in retained and any(resolved.startswith(campaign + '/') for campaign in campaigns):
            target = BUNDLE + '/README.md'
            if 'archive' not in label.lower():
                label += ' (local archive)'
            image = separator = anchor = ''
        relative = posixpath.relpath(target, posixpath.dirname(output) or '.')
        return f'{image}[{label}]({relative}{separator}{anchor})'

    return re.sub(r'(!?)\[([^\]]+)\]\(([^\s)]+)\)', replace, text)
