"""Event-based opportunity for speech after backend content, without text judging."""
from dataclasses import dataclass, field
from multi_turn_eval.pipelines.openai_live import LiveTurnState


@dataclass
class ResultPlaybackState(LiveTurnState):
    result_available: dict = field(default_factory=dict)
    speech_after: float | None = None
    first_text_seen: set = field(default_factory=set)

    def begin(self, now):
        super().begin(now)
        self.result_available.clear()
        self.first_text_seen.clear()
        self.speech_after = None

    def event(self, e, now):
        super().event(e, now)
        if e['type'] == 'session.output_transcript.delta' and e.get('delta', '').strip():
            self.speech_after = now
        if e['type'] != 'response.event':
            return
        inner = e['event']
        key = e.get('delegation_id', 'uncorrelated')
        if inner['type'] == 'response.created':
            self.first_text_seen.discard(key)
        if inner['type'] == 'response.output_text.delta' and inner.get('delta', '').strip():
            if key not in self.first_text_seen:
                self.first_text_seen.add(key)
                self.result_available[key] = now

    def ready(self, now):
        if not super().ready(now):
            return False
        # Use any subsequent fragment, not a new caption group and not speech
        # beginning after response.completed. These are receipt-time observations,
        # not a claim that the frontend communicated every backend fact correctly.
        return all(self.speech_after is not None and self.speech_after >= t
                   for t in self.result_available.values())
