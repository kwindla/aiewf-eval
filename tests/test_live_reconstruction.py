import json
from multi_turn_eval.auditing import live_reconstruction as r


def test_reconstruction_keeps_interim_speech_and_never_substitutes_backend(tmp_path):
    row = dict(
        turn=0, user_text="When?", assistant_text="Checking. Wrong date.", tool_calls=[]
    )
    (tmp_path / "transcript.jsonl").write_text(json.dumps(row) + "\n")
    events = [
        dict(
            t=1,
            turn=0,
            type="server",
            event=dict(
                type="session.output_transcript.delta",
                event_id="a",
                delta="Checking.",
                start_ms=100,
                end_ms=200,
            ),
        ),
        dict(
            t=2,
            turn=0,
            type="server",
            event=dict(
                type="session.output_transcript.delta",
                event_id="b",
                delta=" Wrong date.",
                start_ms=300,
                end_ms=400,
            ),
        ),
        dict(
            t=3,
            turn=0,
            type="server",
            event=dict(
                type="response.event",
                delegation_id="d",
                event=dict(
                    type="response.output_item.done",
                    item=dict(
                        type="message",
                        content=[
                            dict(type="output_text", text="Correct backend date.")
                        ],
                    ),
                ),
            ),
        ),
    ]
    (tmp_path / "live-events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    assert r.reconstruct(tmp_path)
    assert json.loads((tmp_path / "reconstructed/transcript.jsonl").read_text()) == row
    # A late fragment must trigger review, not silently improve the committed answer.
    events.append(
        dict(
            t=4,
            turn=0,
            type="server",
            event=dict(
                type="session.output_transcript.delta",
                event_id="c",
                delta=" Late answer.",
                start_ms=150,
                end_ms=200,
            ),
        )
    )
    (tmp_path / "live-events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    assert not r.reconstruct(tmp_path)
    assert not json.loads((tmp_path / "reconstructed/mapping.json").read_text())[
        "approved_for_judging"
    ]
