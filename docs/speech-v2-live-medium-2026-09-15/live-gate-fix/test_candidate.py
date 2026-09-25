"""Regression checks against the isolated candidate, never the active capture adapter."""
import importlib.util
import sys
from pathlib import Path
p=Path(__file__).with_name('openai_live_candidate.py')
spec=importlib.util.spec_from_file_location('live_gate_candidate',p)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
LiveTurnState=module.LiveTurnState

def test_old_gate_reproduces_instant_commit_after_early_backchannel():
    from multi_turn_eval.pipelines.openai_live import LiveTurnState as Old
    s=Old();s.begin(0)
    s.event({'type':'session.output_transcript.delta','delta':'Mm-hmm.'},1)
    s.last_activity=2  # Backchannel playback has drained.
    s.input_finished=True  # User finishes a longer utterance at t=7.
    assert s.ready(7)  # The observed defect: no post-input quiet interval.

def test_backchannel_requires_fresh_quiet_interval_after_user_finishes():
    s=LiveTurnState();s.begin(0)
    s.event({'type':'session.output_transcript.delta','delta':'Mm-hmm.'},1)
    s.last_activity=2
    s.finish_input(7)
    assert not s.ready(7)
    assert not s.ready(9.99)
    assert s.ready(10)

def test_backend_start_and_followup_extend_the_new_interval():
    s=LiveTurnState();s.begin(0)
    s.event({'type':'session.output_transcript.delta','delta':'Okay.'},1)
    s.finish_input(7)
    s.event({'type':'session.delegation.created','delegation':{'id':'d'}},8)
    assert not s.ready(20)
    s.event({'type':'response.event','delegation_id':'d','event':{'type':'response.completed'}},21)
    s.event({'type':'session.output_transcript.delta','delta':'The answer.'},22)
    assert not s.ready(24)
    assert s.ready(25)
