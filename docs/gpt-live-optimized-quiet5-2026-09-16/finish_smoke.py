"""Validate the full smoke before launching the authorized fresh cohort."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def command(script, args, log):
    with (HERE / log).open('w') as stream:
        subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    for _ in range(130):
        match = re.search(r'^Output directory: (.+)$', (HERE / 'smoke.log').read_text(), re.M)
        run = ROOT / match[1] if match else None
        if run and (run / 'runtime.json').exists() and (run / 'incremental-conversation.wav').exists():
            break
        time.sleep(10)
    else:
        raise RuntimeError('Smoke did not finalize')
    runtime = json.loads((run / 'runtime.json').read_text())
    assert runtime.get('valid') is True and runtime.get('turns') == 30, runtime
    command('process_run.py', [run], 'smoke-processing.log')
    command('summarize_run.py', [run], 'smoke-summary.log')
    result = json.loads((run / 'experimental-validation/complete.json').read_text())
    assert result['approved'], result
    print('Smoke validation passed:', run, flush=True)
    command('campaign.py', [], 'campaign.log')
    command('report_cohort.py', [], 'report-cohort.log')
    print('Fresh experimental cohort finished.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (HERE / 'held.json').write_text(json.dumps({'reason': str(exc)}, indent=2) + '\n')
        raise
