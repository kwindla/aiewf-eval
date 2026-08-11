# Clever Chatter turn-failure root cause

Date: 2026-08-05

## Verdict

The original dominant failure was an audio-turn boundary incompatibility,
compounded by benchmark integration bugs. That failure mode is fixed. A fresh
20-attempt cohort now isolates a residual provider responsiveness problem: 15
conversations completed all 30 turns, while five inputs were accepted and
transcribed but received no model output within the benchmark's 15-second
first-audio limit.

The evidence rules out a missing client send, a swallowed server event, a
turn-12 session limit, and context-window exhaustion. The remaining failures
are either unusually slow provider responses censored by the intentional
15-second cutoff or provider-side no-output stalls; the benchmark correctly
classifies both as failures and never replays the input.

Clever Chatter's automatic activity detector often treated a natural pause
inside a benchmark WAV as the end of the utterance. The model then began
answering while the rest of that same WAV was still being sent. Repeating this
inside one Live session corrupted turn alignment and eventually left the server
silent on a later input.

The correct policy for this scripted benchmark is:

1. disable Gemini automatic activity detection for Clever Chatter;
2. send one `activity_start` immediately before each complete WAV and one
   `activity_end` immediately after it;
3. do not stream continuous silence outside those explicit boundaries;
4. do not replay a timed-out utterance;
5. finalize a response only on `interaction_status=REQUIRES_ACTION`, never on
   `turnComplete`, `generationComplete`, or an audio-playback pause.

## Evidence

### Existing 20-attempt campaign

The no-replay campaign sent 107 user WAVs before 19 attempts timed out and one
ended early. In 42/107 turns, Clever Chatter began returning model audio before
`PacedInputTransport` had finished sending the current WAV. Seventeen of the 19
timeout attempts had at least one such premature response earlier in the same
session. The other two timed out on turn 0, so they had no prior turn that could
have drifted.

The premature endpoints cluster at turns containing pauses: zero-based turns 0,
1, 5, and 6 are especially common. A concrete turn-6 trace transcribed only two
characters, answered "I'm listening, please feel free to continue," and sent
`REQUIRES_ACTION` before the 6.2-second WAV finished. The next WAV was then
queued into an already misinterpreted conversation.

Rebuild this count with:

```bash
.venv/bin/python \
  docs/clever-chatter-comparison-2026-08-04/root-cause-probes/analyze_auto_vad_campaign.py \
  docs/clever-chatter-comparison-2026-08-04/no-replay-n20-2026-08-04 \
  --output docs/clever-chatter-comparison-2026-08-04/root-cause-probes/results/auto-vad-campaign-analysis.json
```

### Raw SDK boundary probes

The standalone probes bypass Pipecat and use the private Google SDK overlay.
They showed:

| Probe | Result |
|---|---|
| Fresh turn, default automatic VAD | Usually responds, but onset reached 14.4s and one 5-trial response did not finish inside 45s |
| Fresh turn, exact activity frames | Stable roughly 1s onset in ordinary trials; complete, nontruncated input preserved |
| Fresh turn, tuned automatic VAD | 13.5s onset, wrong transcription, no terminal inside the observation window |
| Sequential turns 0-8, automatic VAD | Input truncated on turns 1, 5, and 6; no server output on turn 7 |
| Sequential turns 0-8, exact activity frames | All nine turns completed with complete, nontruncated input transcriptions |
| Same sequential automatic-VAD test on `gemini-3.1-flash-live-preview` | All nine turns completed with full input transcriptions |

This isolates the drift to Clever Chatter's audio activity handling rather than
the source WAVs or the raw send loop.

### Integrated full-conversation validation

Run `20260804T222803_clever-chatter_6f53c595` completed all 30 benchmark turns
under the fixed pipeline:

- `runtime.json`: `status=completed`, `valid=true`, `turns=30`;
- exactly 30 `activity_start` and 30 `activity_end` sends;
- 30 `REQUIRES_ACTION` events and five intermediate `IN_PROGRESS` events;
- zero no-response timeouts, replays, reconnects, or malformed activity pairs;
- both required `submit_session_suggestion` calls and the final `end_session` call;
- first-audio TTFB p50 981ms and maximum 6,255ms;
- response-completion latency p50 13,223ms and maximum 78,247ms;
- a 529.6-second, 48.49 MiB stereo recording.

The run crossed the old turn-16 and turn-25 hotspots without a stall. On turn
11 it paused after `IN_PROGRESS`, resumed with more speech, and emitted further
asynchronous phases; the terminal-aware gate kept the entire exchange in one
benchmark turn. This is the end-to-end confirmation that the implemented
boundary and completion policies address the reproduced failure.

### Fresh 20-attempt exact-boundary cohort

The sequential cohort in `exact-boundary-n20-2026-08-05-v2` ran with explicit
activity boundaries, `interaction_status` terminal handling, a 15-second
first-audio watchdog, no replay, and standard stereo recording on every
attempt.

| Reliability | Result |
|---|---:|
| Attempts | 20 |
| Complete 30-turn conversations | 15 |
| Completion rate (Wilson 95% CI) | 75.0% (53.1–88.8%) |
| `no_audio_timeout` | 5 |
| Failed zero-based turns | 0 (1), 4 (2), 10 (1), 12 (1) |

All 20 attempts passed the independent artifact and exact-boundary audit:

- 485 explicit activity starts and 485 matching activity ends;
- zero model-audio events during explicit user input;
- zero non-empty recorded responses missing first-audio timing;
- zero replays, reconnects, malformed boundaries, or unexpected runtime errors;
- 20 stereo 24 kHz PCM recordings, including each failed attempt;
- five zero-output `REQUIRES_ACTION`/interruption housekeeping pairs during
  open input were ignored, and none advanced a benchmark turn.

For each timed-out input, the raw trace contains a provider input
transcription after `activity_end`, proving the utterance reached the service.
The socket continued to deliver empty SDK/control messages, but no model audio,
text, tool call, `IN_PROGRESS`, or `REQUIRES_ACTION` arrived before the
watchdog. The failures occurred early and at different turns; every one of the
15 conversations that reached turn 13 subsequently completed turn 29. This is
the opposite of the pattern expected from a fixed session-duration or growing
context limit.

Timing across the 480 turns that produced responses—including completed turns
from attempts that later failed—was:

| Metric | P50 | P90 | P95 | Max |
|---|---:|---:|---:|---:|
| First audio | 1.107s | 1.508s | 1.785s | 14.504s |
| Response completion | 14.844s | 29.551s | 37.837s | 151.426s |

The stable roughly one-second first-audio medians across early and late turns
also argue against context-driven prefill growth. The long completion tail is
real: Clever Chatter can start promptly and then spend well over a minute
speaking and reasoning asynchronously.

All 15 complete Clever Chatter conversations were judged, for 450 scored
turns. The Gemini comparison combines its frozen 10-conversation cohort with
five new recovery-free conversations, also totaling 450 judged turns. Three
other current Gemini attempts extend the reliability cohort only: one completed
after a WebSocket 1006 reconnect and turn replay, one terminated at a no-audio
timeout, and one completed cleanly. The reconnect was a transport recovery,
not a no-response event.

No-response reliability is analyzed independently of survivor content quality.
The same rule is applied to both models: the first logged 15-second no-response
event terminates the run and the utterance is not replayed. The Gemini cohort
contains the 10 consecutive frozen March minimal-thinking runs, two August 4
current-model controls, and eight August 5 formal attempts. One August 4
control completed only after replay; it is counterfactually classified as
failed at its first timeout. The August 5 campaign enforced no replay and
terminated its failed run directly.

| Metric | Clever Chatter | Gemini 3.1 Flash Live Preview |
|---|---:|---:|
| No-response-free conversations under the shared policy | 15/20 — 75.0% (Wilson 95% CI 53.1–88.8%) | 18/20 — 90.0% (69.9–97.2%) |
| Zero-based turn of the first timeout in failed conversations | 12, 4, 10, 4, 0 | 25, 12 |
| Strict pass (conversation-bootstrap 95% CI) | 88.9% (87.6–90.2%) | 93.1% (90.9–95.1%) |
| Tool use | 92.2% | 93.3% |
| Instruction following | 90.2% | 93.8% |
| Knowledge grounding | 99.6% | 100.0% |

The first two rows use all 20 reliability attempts per model. The four content
rows use the matched survivor-quality cohorts: 15 complete conversations and
450 judged turns per model.

Thus Gemini 3.1 Flash Live does exhibit the same class of no-response failure;
the old replay policy masked one historical failure, and the earlier 17/17
summary incorrectly omitted it. The matched 20-attempt cohorts suggest fewer
failures for Gemini, but they remain too small to estimate the difference
precisely.

Errors are strongly concentrated in action turns. Across the 15 complete
conversations, the model made all 30 expected session-suggestion calls and all
15 `end_session` calls, but only 3/15 dietary-request calls, 8/15 tech-support
calls, and 4/15 vote calls. The strict judge recorded 12 failures at turn 15
(submit the confirmed dietary request), 10 at turn 17 (request tech support),
and 11 at turn 24 (cast the requested vote).

The legacy offline turn-tag detector marked only 273/450 turns as matched. That
is not evidence of 177 conversational overlaps: the detector assumes one
contiguous bot-audio segment per turn, while Clever Chatter often emits several
speech phases around asynchronous reasoning and tools. On an inspected run it
reported zero audio overlaps but treated later speech phases as unmatched
segments. For this model, the raw explicit-boundary audit above is the reliable
test of whether model output crossed user input.

### Slow output versus missing terminal status

Some short text responses take an extraordinary amount of wall time to stream
as audio. In one exact-boundary turn, first output arrived at 10.42s, the final
audio arrived at 82.26s, `generationComplete` arrived at 82.29s, and
`REQUIRES_ACTION` followed at 82.62s. The response was only 294 transcribed
characters.

This corrects an earlier interpretation: a 45- or 60-second probe that saw
audio but no terminal status was generally ending while audio generation was
still active. We have not reproduced a case where `generationComplete` was
followed by a long or permanently missing `REQUIRES_ACTION` under exact
boundaries. The model/API is very slow and bursty, but the private status field
is behaving consistently once generation actually finishes.

### No swallowed receive or send event

The prior campaign's raw tracer observed 7,881 WebSocket messages. On every
no-audio timeout, zero raw server messages arrived after the final audio send
and before termination. A separate outbound trace confirmed every real-audio
SDK call succeeded and the raw receive remained pending. The failure was
therefore upstream of the client receive converter, not an event dropped by the
benchmark.

## Benchmark integration bugs found and fixed

### Stale pacing clock

Turning off continuous silence exposed a `PacedInputTransport` bug. While the
transport sat idle during the greeting, its pacing baseline did not advance.
The first WAV was consequently scheduled almost all at once. The feeder now
resets its next-chunk clock while idle; a 5.14-second WAV again takes about 5.12
seconds to send.

### Duplicate synthetic activity turn

Pipecat's default universal user aggregator treats a later input transcription
as a new turn. After the transport's correct `activity_end`, it synthesized a
second, zero-duration `UserStartedSpeakingFrame`/`UserStoppedSpeakingFrame`
pair. Gemini therefore received an empty second activity turn.

For exact-boundary models, the aggregator now uses
`ExternalUserTurnStartStrategy` and `ExternalUserTurnStopStrategy`, binding it
to the transport frames and preventing the transcription from creating a new
provider activity pair.

### Playback stop was not a terminal state

`TTSStoppedAssistantTranscriptProcessor` previously flushed on
`BotStoppedSpeakingFrame`, and `TurnGate` could advance once playback drained.
That is unsafe for asynchronous reasoning: Clever Chatter can pause, report
`IN_PROGRESS`, and later emit another spoken phase or tool call.

For Clever Chatter, playback stops no longer flush the transcript. The service
emits the definitive `TTSStoppedFrame` only for `REQUIRES_ACTION`, and the turn
gate requires both that provider terminal signal and drained playback. An
integrated turn-11 test exercised multiple `IN_PROGRESS` phases separated by a
playback stop; the benchmark retained one accumulated transcript and did not
advance early.

### A terminal can precede buffered continuation playback

A later preflight exposed a narrower drain race. The provider terminal arrived,
but a continuation already moving through the output pipeline began playback
roughly 474ms later. The old 500ms drain timer expired about 2ms before that
playback-start signal, so the next user WAV began and interrupted the remaining
assistant speech.

If `TTSStartedFrame` arrives after a provider terminal while the turn is still
draining, `TurnGate` now revokes the provisional completion, preserves the
accumulated transcript, and waits for the continuation's eventual terminal and
playback drain. A subsequent 30-turn validation and the 20-attempt cohort had
zero model-audio events cross into explicit input.

### Zero-output terminal housekeeping during open input

The private API sometimes emits an interruption followed by a zero-output
`REQUIRES_ACTION` while the next explicit user activity is open. Treating that
as the response terminal could advance an input that the model had not yet
answered. The service wrapper now tracks explicit input activity and ignores
these terminal housekeeping events until `activity_end`.

Five such terminal/interruption pairs appeared in the new cohort. All were
logged, none advanced a turn, and the affected conversations passed the strict
event-flow audit.

## Files

- `probe_live_boundaries.py`: fresh-session boundary, prompt, tool, and history matrix.
- `probe_live_sequence.py`: accumulated same-session sequence and Preview control.
- `analyze_auto_vad_campaign.py`: premature-endpoint audit of the 20-attempt campaign.
- `results/`: machine-readable probe results.
- `../exact-boundary-n20-2026-08-05/run_campaign.sh`: resumable strict cohort runner.
- `../exact-boundary-n20-2026-08-05/analyze_cohort.py`: reliability, timing,
  event-flow, and judgment aggregator.
- `../exact-boundary-n20-2026-08-05-v2/cohort-summary.md`: fresh cohort summary.
- `../exact-boundary-n20-2026-08-05-v2/cohort-aggregate.json`: machine-readable
  per-run and per-turn aggregate.
- `../gemini31-topup-n5-2026-08-05/summary.md`: matched-turn Gemini 3.1
  content and no-replay reliability summary.
- `../gemini31-topup-n5-2026-08-05/combined-aggregate.json`:
  machine-readable frozen-plus-top-up Gemini aggregate.
- `src/multi_turn_eval/model_capabilities.py`: model-specific policy.
- `src/multi_turn_eval/transports/paced_input.py`: exact activity frames and pacing fix.
- `src/multi_turn_eval/pipelines/realtime.py`: explicit Gemini VAD configuration,
  external turn strategies, and terminal-aware gate.
- `src/multi_turn_eval/processors/tts_transcript.py`: async-safe transcript flushing.

## Conclusion and remaining validation

The boundary and asynchronous-completion integration is now behaving
correctly, but Clever Chatter does not meet this benchmark's reliability or
tool-use bar. A 75% full-conversation completion rate and concentrated misses
on three required action families are sufficient reasons to keep it out of the
README leaderboard, as planned.

If Google needs the residual timeout split diagnosed beyond the benchmark SLO,
the next probe should preserve the exact-boundary policy but observe reproduced
turns for 60–90 seconds without replay. That would distinguish a response that
arrives after the 15-second benchmark limit from one that remains silent. It
would not change the classification of this cohort.
