"""Audit all attempts, preserving distinctions between API and response failures."""
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def audit(record):
    log = (ROOT / record['log']).read_text()
    run = ROOT / record['run_dir'] if record.get('run_dir') else None
    runtime = json.loads((run / 'runtime.json').read_text()) if run and (run / 'runtime.json').exists() else {}
    transcript = [json.loads(x) for x in (run / 'transcript.jsonl').read_text().splitlines()] if run and (run / 'transcript.jsonl').exists() else []
    rows = [x for x in transcript if not x.get('recovery_turn')]
    indices = [x['turn'] for x in rows]
    assert len(indices) == len(set(indices)), f'Duplicate turns: {run}'
    full = record['exit_code'] == 0 and set(indices) == set(range(30)) and runtime.get('valid') is not False
    assert full == record['full'], f'Completion disagrees with capture ledger: {run}'
    # Match concrete log errors, not quoted prompt text or generic traceback frames.
    patterns = {
        'rate_limit': r'(?i)(RESOURCE_EXHAUSTED|RateLimitError|rate_limit_error|rate.limit.exceeded|too many requests|HTTP[^\n]*429)',
        'api_error': r'(?i)(received (?:1008|1011|1013)|ConnectionClosedError|no close frame received|internal server error|service unavailable|HTTP[^\n]*50[0234]|APIError|API error|ErrorFrame#)',
        'adapter_error': r'(AttributeError:|TypeError:|KeyError:)',
        'reconnect': r'(\[RECONNECT[^\]]*\]|\[TURN_REPLAY[^\]]*\])',
    }
    evidence = {kind: [line for line in log.splitlines() if re.search(pattern, line)] for kind, pattern in patterns.items()}
    input_pending = False
    input_overlaps = []
    for number, line in enumerate(log.splitlines(), 1):
        if '[USER_AUDIO_QUEUED]' in line:
            if input_pending:
                input_overlaps.append({'line': number, 'event': line})
            input_pending = True
        elif 'FINISHED SENDING AUDIO' in line:
            input_pending = False
    reason = runtime.get('failure', {}).get('reason')
    if not full and not reason:
        reason = 'process_watchdog' if record['exit_code'] == 124 else 'early_exit' if record['exit_code'] == 0 else 'process_error'
    return {
        'arm': record['arm'], 'index': record['index'], 'run_dir': record['run_dir'],
        'complete': full, 'recorded_turns': len(rows), 'missing_turns': sorted(set(range(30)) - set(indices)),
        'failure_reason': reason, 'runtime': runtime, 'evidence': evidence,
        'next_input_queued_before_previous_finished': input_overlaps,
        'transcript_reconnections': max((x.get('reconnection_count') or 0 for x in rows), default=0),
    }


def main():
    records = [json.loads(x) for x in (HERE / 'capture/attempts.jsonl').read_text().splitlines()]
    runs = [audit(r) for r in records]
    summary = {}
    for arm in json.loads((HERE / 'protocol.json').read_text())['arms']:
        group = [r for r in runs if r['arm'] == arm['key']]
        summary[arm['key']] = {
            'attempts_finished': len(group), 'complete': sum(r['complete'] for r in group),
            'failure_reasons': dict(Counter(r['failure_reason'] for r in group if r['failure_reason'])),
            'attempts_with_error_evidence': {kind: sum(bool(r['evidence'][kind]) for r in group) for kind in ['api_error', 'rate_limit', 'adapter_error', 'reconnect']},
        }
    (HERE / 'reliability.json').write_text(json.dumps({'summary': summary, 'attempts': runs}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
