"""Read-only live status."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


match = re.search(r'^Output directory: (.+)$', (HERE / 'smoke.log').read_text(), re.M)
if match:
    run = ROOT / match[1]
    events = rows(run / 'live-events.jsonl')
    print('Smoke:', run.name, 'completed turns:', len(rows(run / 'transcript.jsonl')))
    decisions = [e for e in events if e['type'] == 'coordinator_result']
    print('Decisions:', len(decisions), 'waiting:', sum(not e['decision']['ready'] for e in decisions))
    for p in [run / 'runtime.json', run / 'experimental-validation/complete.json']:
        if p.exists():
            print(p.name, p.read_text())
out = HERE / 'cohort'
attempts = rows(out / 'attempts.jsonl')
print('Cohort:', len(rows(out / 'started.jsonl')), 'started;', len(attempts),
      'captured;', sum(r['full'] for r in attempts), 'full;',
      len(rows(out / 'processed.jsonl')), 'processed')
for p in [HERE / 'held.json', out / 'held.json']:
    if p.exists():
        print('HELD:', p.read_text())
