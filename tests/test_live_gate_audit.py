import json
from multi_turn_eval.auditing import live_gate as m


def test_event_replay_blocks_pending_backend_and_accepts_clarification(tmp_path):
    events = [
        dict(t=0, turn=0, type="input_start"),
        dict(t=2, turn=0, type="input_end"),
        dict(
            t=3,
            turn=0,
            type="server",
            event=dict(type="session.delegation.created", delegation={"id": "d"}),
        ),
        dict(
            t=4,
            turn=0,
            type="server",
            event=dict(
                type="session.output_transcript.delta", delta="Did you mean Jane?"
            ),
        ),
        dict(t=5, turn=0, type="BotStartedSpeakingFrame"),
        dict(t=6, turn=0, type="BotStoppedSpeakingFrame"),
        dict(t=12, turn=0, type="turn_complete"),
    ]
    path = tmp_path / "live-events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert not m.audit(tmp_path)["approved"]
    events.insert(
        -1,
        dict(
            t=6.5,
            turn=0,
            type="server",
            event=dict(
                type="response.event",
                delegation_id="d",
                event={"type": "response.completed"},
            ),
        ),
    )
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert m.audit(tmp_path)["approved"]
    events[-1]["t"] = 7
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert not m.audit(tmp_path)["approved"]
