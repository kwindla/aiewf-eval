from multi_turn_eval.pipelines import live_playback as m


def backend(s, kind, now, **kwargs):
    s.event(
        dict(type="response.event", delegation_id="d", event=dict(type=kind, **kwargs)),
        now,
    )


def speech(s, text, now):
    s.event(dict(type="session.output_transcript.delta", delta=text), now)


def test_interim_before_backend_result_does_not_satisfy_delivery_opportunity():
    s = m.ResultPlaybackState(quiet_seconds=5)
    s.begin(0)
    s.finish_input(2)
    speech(s, "Checking.", 3)
    backend(s, "response.created", 4)
    backend(s, "response.output_text.delta", 6, delta="Result")
    backend(s, "response.completed", 7)
    assert not s.ready(30)
    speech(s, "Did you mean Jane?", 31)
    assert not s.ready(35.99)
    assert s.ready(36)


def test_streaming_answer_can_start_before_backend_finishes():
    s = m.ResultPlaybackState(quiet_seconds=5)
    s.begin(0)
    s.finish_input(2)
    backend(s, "response.created", 3)
    backend(s, "response.output_text.delta", 4, delta="Result")
    speech(s, "Here is the answer.", 5)
    backend(s, "response.output_text.delta", 6, delta=" more result")
    backend(s, "response.completed", 7)
    assert not s.ready(11.99)
    assert s.ready(12)
    s.speaking = True
    assert not s.ready(20)


def test_tool_continuation_and_direct_response():
    s = m.ResultPlaybackState(quiet_seconds=5)
    s.begin(0)
    s.finish_input(1)
    speech(s, "Welcome.", 2)
    assert s.ready(7)
    s.begin(8)
    s.finish_input(9)
    backend(s, "response.created", 10)
    backend(s, "response.output_item.done", 11, item={"type": "function_call"})
    backend(s, "response.completed", 12)
    speech(s, "Working on that.", 13)
    assert not s.ready(30)
    backend(s, "response.created", 31)
    backend(s, "response.output_text.delta", 32, delta="Done")
    backend(s, "response.completed", 33)
    assert not s.ready(40)
    speech(s, "Done.", 41)
    assert s.ready(46)
