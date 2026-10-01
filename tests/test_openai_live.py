from multi_turn_eval.pipelines.openai_live import LiveTurnState


def event(state, subtype, now, **data):
    state.event(
        {
            "type": "response.event",
            "delegation_id": "d",
            "event": {"type": subtype, **data},
        },
        now,
    )


def test_acknowledgement_and_drained_playback_cannot_end_pending_delegation():
    s = LiveTurnState()
    s.begin(0)
    s.input_finished = True
    s.event({"type": "session.output_transcript.delta", "delta": "Let me check."}, 1)
    s.event({"type": "session.delegation.created", "delegation": {"id": "d"}}, 2)
    assert not s.ready(20)
    event(s, "response.created", 21)
    event(s, "response.completed", 22)
    assert not s.ready(24)
    s.event(
        {"type": "session.output_transcript.delta", "delta": "Here is the answer."}, 24
    )
    s.speaking = True
    assert not s.ready(30)
    s.speaking = False
    s.last_activity = 31
    assert not s.ready(33)
    assert s.ready(34)


def test_tool_response_completion_requires_continuation_completion():
    s = LiveTurnState()
    s.input_finished = True
    s.transcript = "Submitting."
    event(s, "response.created", 1)
    event(s, "response.output_item.done", 2, item={"type": "function_call"})
    event(s, "response.completed", 3)
    assert not s.ready(30)
    event(s, "response.created", 31)
    assert not s.ready(35)
    event(s, "response.completed", 36)
    assert s.ready(39)


def test_concurrent_delegations_and_input_must_all_finish():
    s = LiveTurnState()
    s.transcript = "Yes"
    for d in ["a", "b"]:
        s.event({"type": "session.delegation.created", "delegation": {"id": d}}, 1)
    s.event(
        {
            "type": "response.event",
            "delegation_id": "a",
            "event": {"type": "response.completed"},
        },
        2,
    )
    assert not s.ready(20)
    s.event(
        {
            "type": "response.event",
            "delegation_id": "b",
            "event": {"type": "response.completed"},
        },
        3,
    )
    assert not s.ready(20)
    s.input_finished = True
    assert s.ready(20)


def test_backchannel_requires_fresh_quiet_interval_after_user_finishes():
    s = LiveTurnState()
    s.begin(0)
    s.event({"type": "session.output_transcript.delta", "delta": "Mm-hmm."}, 1)
    s.last_activity = 2
    s.finish_input(7)
    assert not s.ready(7)
    assert not s.ready(9.99)
    assert s.ready(10)


def test_backend_start_and_followup_extend_the_new_interval():
    s = LiveTurnState()
    s.begin(0)
    s.event({"type": "session.output_transcript.delta", "delta": "Okay."}, 1)
    s.finish_input(7)
    s.event({"type": "session.delegation.created", "delegation": {"id": "d"}}, 8)
    assert not s.ready(20)
    s.event(
        {
            "type": "response.event",
            "delegation_id": "d",
            "event": {"type": "response.completed"},
        },
        21,
    )
    s.event({"type": "session.output_transcript.delta", "delta": "The answer."}, 22)
    assert not s.ready(24)
    assert s.ready(25)


def test_live_uses_the_exact_shared_speech_prompt_and_judge():
    from benchmarks.aiwf_medium_context_speech_v2.config import BenchmarkConfig as Shared
    from benchmarks.aiwf_medium_context_speech_live.config import BenchmarkConfig as Live
    assert Live.system_instruction == Shared.system_instruction
    assert Live.judge_system_prompt == Shared.judge_system_prompt
    assert Live.tools_schema is Shared.tools_schema
    assert Live.turns is Shared.turns
