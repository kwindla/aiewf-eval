import importlib.util
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'docs/gpt-live-optimized-deterministic-2026-09-16/audit_gate.py'
spec = importlib.util.spec_from_file_location('live_gate_audit', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_event_replay_blocks_pending_backend_and_accepts_clarification(tmp_path):
    events = [
        dict(t=0, turn=0, type='input_start'),
        dict(t=2, turn=0, type='input_end'),
        dict(t=3, turn=0, type='server', event=dict(type='session.delegation.created', delegation={'id': 'd'})),
        dict(t=4, turn=0, type='server', event=dict(type='session.output_transcript.delta', delta='Did you mean Jane?')),
        dict(t=5, turn=0, type='BotStartedSpeakingFrame'),
        dict(t=6, turn=0, type='BotStoppedSpeakingFrame'),
        dict(t=10, turn=0, type='turn_complete'),
    ]
    path = tmp_path / 'live-events.jsonl'
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    assert not m.audit(tmp_path)['approved']
    events.insert(-1, dict(t=6.5, turn=0, type='server', event=dict(
        type='response.event', delegation_id='d', event={'type': 'response.completed'})))
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    assert m.audit(tmp_path)['approved']
    events[-1]['t'] = 7
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    assert not m.audit(tmp_path)['approved']
