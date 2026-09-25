"""Report every standard-prompt validation/pilot attempt, including failures."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
logs = [HERE / 'standard-prompt-full-validation.log']
pilot = HERE / 'concurrency-pilot'
if pilot.exists():
    logs += [pilot / f'capture-{i}.log' for i in range(1, 6)]
rows = []
for log in logs:
    match = re.search(r'^Output directory: (.+)$', log.read_text(), re.M) if log.exists() else None
    run = Path(match.group(1)) if match else None
    row = {'capture_log': str(log), 'run_dir': str(run) if run else None, 'planned_turns': 30}
    if run:
        for key, name in [('runtime', 'runtime.json'), ('judge', 'claude_summary.json'),
                          ('audio', 'live-audio-metrics.json'), ('alignment', 'audio-alignment-audit.json')]:
            path = run / name
            if path.exists():
                row[key] = json.loads(path.read_text())
    if 'judge' in row:
        row['strict_passes'] = row['judge']['turn_pass']['count']
        row['strict_percent_planned'] = 100 * row['strict_passes'] / 30
    rows.append(row)
(HERE / 'validation-results.json').write_text(json.dumps(rows, indent=2))
for row in rows:
    print(row['capture_log'], row.get('runtime'), row.get('judge', {}).get('turn_pass'),
          row.get('audio', {}).get('summary'))
