"""Replay lifecycle/playback signals; never inspect answer meaning."""

import json
from pathlib import Path
from multi_turn_eval.pipelines.live_playback import ResultPlaybackState


def audit(run):
    events = [
        json.loads(line)
        for line in (Path(run) / "live-events.jsonl").read_text().splitlines()
    ]
    state = ResultPlaybackState(quiet_seconds=5.0)
    issues = []
    commits = []
    for e in events:
        now = e["t"]
        kind = e["type"]
        if kind.startswith("coordinator_"):
            issues.append(dict(turn=e["turn"], reason="Unexpected semantic classifier"))
        if kind == "input_start":
            if state.active or state.speaking:
                issues.append(
                    dict(turn=e["turn"], reason="Input while backend/playback active")
                )
            state.first_text_seen.clear()
            state.result_available = {}
            state.speech_after = None
            state.transcript = ""
            state.input_finished = False
            state.last_activity = now
        elif kind == "input_end":
            state.finish_input(now)
        elif kind == "server":
            state.event(e["event"], now)
        elif kind == "BotStartedSpeakingFrame":
            state.speaking = True
            state.last_activity = now
        elif kind == "BotStoppedSpeakingFrame":
            state.speaking = False
            state.last_activity = now
        elif kind == "turn_complete":
            # BotSpeaking heartbeat frames are not logged; a logged stop is the
            # conservative playback boundary. Allow 50 ms for log-call timing.
            entry = dict(
                turn=e["turn"],
                active=sorted(state.active),
                speaking=state.speaking,
                input_finished=state.input_finished,
                quiet_seconds=now - state.last_activity,
                result_available=state.result_available.copy(),
                last_speech=state.speech_after,
            )
            commits.append(entry)
            if not state.ready(now + 0.05):
                issues.append(dict(**entry, reason="Commit before gate conditions"))
    result = dict(
        approved=not issues,
        issues=issues,
        commits=commits,
        limitation="Quiet interval is a heuristic, not an API answer-delivered event.",
    )
    (Path(run) / "deterministic-gate-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
