# Clever Chatter timeout/replay deep dive

Date: 2026-08-04

> **Reliability-cohort correction, 2026-08-05:** The original Gemini comparison
> below selected 11 completed controls and omitted a separate full August 4
> conversation that encountered two no-response timeouts before replay
> recovered it. The matched no-response/no-replay comparison is now 15/20
> completed Clever Chatter conversations versus 18/20 Gemini 3.1 Flash Live
> conversations. See `root-cause-probes/README.md` and
> `gemini31-topup-n5-2026-08-05/combined-aggregate.json`. The original analysis
> remains below as provenance for the earlier replay-behavior investigation.

> **Root-cause update:** Subsequent raw-SDK and integrated probes identified
> premature automatic-VAD endpointing as the dominant source of accumulated
> session drift and corrected an apparent missing-terminal interpretation.
> See [`root-cause-probes/README.md`](root-cause-probes/README.md) for the
> completed investigation and exact-boundary fix. This document preserves the
> earlier replay-cohort analysis for provenance.

## Verdict

Timeout behavior is dramatically worse for Clever Chatter than for
`gemini-3.1-flash-live-preview`, and the benchmark's replay policy materially
amplifies the problem.

### No-replay validation update

We subsequently changed Clever Chatter to terminate on its first 15-second
no-audio timeout, with no replay, and ran 20 sequential full-conversation
attempts using the private interaction-status SDK. Nineteen attempts timed out
between zero-based turns 0 and 8 (median 6); the remaining attempt ended after
six turns. None completed all 30 turns, so there were no valid conversations to
judge.

This campaign traced 7,881 WebSocket messages immediately after
`WebSocket.recv()` and before the Google SDK converter. In every one of the 19
timeout attempts, no raw server message of any kind arrived after the final
user-audio send completed and before the watchdog fired. There were no raw
sequence gaps, decode failures, socket closes, server error messages, GoAway
messages, or replay log entries. This rules out a delivered server event being
silently discarded by the SDK or benchmark during those observed timeout
windows. It cannot establish whether the service would have responded after
the 15-second boundary.

A separate outbound-instrumented diagnostic then traced the SDK call itself.
On its failed turn, all 278 real-audio chunks (266,880 bytes including final
frame padding) returned successfully from `send_realtime_input()`. Another 998
continuous-audio sends succeeded after the WAV ended, the last one millisecond
before the watchdog fired. There were no send errors and no inbound messages;
the pending raw receive was cancelled after waiting 26.0 seconds. This rules
out a silent client-side audio drop in that reproduced timeout.

All 20 standard stereo, 24 kHz `conversation.wav` recordings were retained.
See `no-replay-n20-2026-08-04/analysis.md` and `analysis.json` for the campaign
audit.

The primary comparison uses 15 complete Clever Chatter conversations and 11
complete Gemini 3.1 Flash Live Preview conversations. The Preview cohort is the
frozen ten-run cohort plus one current-code/current-SDK control.

| Metric | Clever Chatter | Gemini 3.1 Flash Live Preview |
|---|---:|---:|
| Complete conversations | 15 | 11 |
| Scored turns | 450 | 330 |
| No-response timeout events | 217 | 0 |
| Distinct turns with a timeout | 124/450 (27.6%) | 0/330 (0.0%) |
| Median affected turns per conversation | 8 | 0 |
| Forced advances after three replays | 23 | 0 |
| Audio requeues | 195 | 0 |
| Replays where bot audio started before replayed user audio stopped | 76/194 (39.2%) | n/a |
| Median full-run wall time | 700.3s | 524.9s |

Every complete Clever Chatter run had between 7 and 23 timeout events. Its first
affected turn was always at or before zero-based turn 10 (median turn 6). A
whole-conversation bootstrap puts the Clever-minus-Preview difference in affected-turn
rate at 23.6 to 32.2 percentage points (95% interval).

The result is favorable to Clever Chatter because it conditions on complete runs. Only
14 of the 25 new campaign attempts completed all 30 turns.

## What the retry loop does

1. `TurnGate` starts a 15-second timer when local VAD reports that the user stopped.
2. If no `TTSStartedFrame` or bot-speech frame appears, the pipeline clears local
   response state.
3. It waits two more seconds, then enqueues the same WAV again in the existing Live
   session. It does not close or reset that session first.
4. The replay is treated as new speech. Local VAD emits an interruption and the server
   receives a second copy of the utterance while the status of the first model turn is
   unresolved.
5. After three replays, the benchmark records an empty response and force-advances.

Of the 124 affected Clever Chatter turns, 80 recovered after one replay, 18 after two,
3 after three, and 23 did not recover. No model response began during the two-second
grace period before a replay (`Turn retry cancelled` occurred zero times).

The replay is not benign. In 76 of 194 traceable replay windows, bot audio started
while the duplicate user audio was still active according to local VAD. One concrete
example is turn 11 of
`20260804T145352_clever-chatter_9554f41d`: the replay started at 14:59:18.989,
Clever Chatter issued the wrong `submit_dietary_request` call at 14:59:21.070, and the
replayed utterance did not stop until 14:59:22.628. The tool result was therefore sent
while the user utterance was still being replayed.

## A TTFB measurement trap

The 17-20 second TTFB samples seen at many replay starts are not evidence that a late
provider response arrived. Pipecat handles `InterruptionFrame` by calling
`stop_all_metrics()`. Replay VAD therefore forcibly closes the outstanding TTFB timer
at the instant replay speech starts. This occurred in 192 of 194 traceable replay
windows.

Consequences:

- TTFB values on recovery-affected turns are not trustworthy.
- The current Clever Chatter V2V median, maximum, and tool-turn mean mix model latency
  with retry time, duplicated audio, interruption, and forced metric stops.
- Recorded TTFB was 891ms median on unaffected Clever turns versus 5,764ms on affected
  turns, but the latter is not an estimate of model latency.
- Complete Clever runs took 33.4% longer at the median than the Preview controls.

## `interactionStatus` confirms premature advancement

Google reports that `turnComplete=true` is not a terminal signal for asynchronous-
reasoning models. The new `serverContent.interactionStatus` field is terminal only when
it is `REQUIRES_ACTION`; `IN_PROGRESS` means that more audio or tool activity may still
follow.

The public google-genai 2.16 SDK receives this field on the wire but drops it in its
response converter. A diagnostic wrapper logged the raw field without otherwise
changing benchmark behavior. On a four-turn Clever Chatter reproduction, the following
sequence occurred on turn 11:

1. The turn first hit the 15-second no-response timeout and was replayed.
2. Clever Chatter started audio during the replay and made an incorrect
   `vote_for_session` call.
3. At 18:52:44.237, the server sent `turnComplete=true` together with
   `interactionStatus=IN_PROGRESS`.
4. At 18:52:44.245, eight milliseconds later, the benchmark ran `on_turn_end` and
   advanced to turn 12.
5. Clever Chatter's continuation started at 18:52:45.545. Turn 12 user audio then
   started at 18:52:45.902 and interrupted it.
6. At 18:52:52.968, the server finally sent `interactionStatus=REQUIRES_ACTION`. The
   delayed vote confirmation was recorded as the answer to turn 12.

The identical four-turn Preview control had zero timeouts, made both required
`submit_session_suggestion` calls correctly, and completed in 61.6 seconds. Clever
Chatter had two timeout/replays, made neither required call correctly, and took 95.9
seconds. Preview did not emit `interactionStatus`, consistent with its ordinary
turn-completion behavior.

This establishes a second, independent harness incompatibility: even after Clever
Chatter begins responding, the current pipeline can close and advance an asynchronous
turn while the server explicitly says it is still processing.

The status field does not by itself explain every initial no-audio stall. In the
diagnostic turn-10 and turn-11 stalls, no new interaction-status event arrived before
the 15-second timeout. The pipeline therefore needs both status-aware finalization and
a safer first-output watchdog.

## Speaking rate is slower, but is not the dominant delay

On clean non-tool turns, Clever Chatter spoke at a median of about 180 words per minute
versus 206 for Preview, approximately 13% slower. This supports the observed slower
delivery. Clever's responses were also much shorter, however: median 20 words and 6.5
seconds of output, versus 40 words and 11.3 seconds for Preview.

Slower speech can lengthen a response and make pause/segment handling more sensitive,
but it cannot trigger the current `NO_RESPONSE` timer: that timer is cancelled as soon
as the first audio frame starts. It also does not explain the inflated voice-to-voice
onset measurements. Timeout/replay and premature asynchronous-turn advancement remain
the dominant failures.

## What is model-specific and what is harness-induced

The trigger is model/API-specific in this comparison: the current Preview control used
the same code, SDK, audio, system prompt, and tool schema and had zero timeouts in 30
turns. The frozen ten Preview runs also had zero timeouts in 300 turns. Clever Chatter
timed out in all 15 complete runs.

The amplifier is harness-induced: replaying an utterance into the unresolved session
creates duplicate input, interruptions, output/input overlap, and sometimes tool calls
during replayed speech.

The original campaign logs establish the absence of user-visible audio for the timeout
window, but they do not preserve enough raw server-event detail to distinguish among:

- no server response at all;
- a server turn that contains only non-audio/control content;
- a Clever Chatter server-VAD/end-of-activity failure with the continuously streamed
  silence used by this pipeline.

The new raw-wire diagnostic adds one confirmed case of `turnComplete=true` while
`interactionStatus=IN_PROGRESS`, followed by a continuation that the pipeline assigned
to the next turn.

The pattern does not look like a context-window limit. Stalls begin in every complete
run by turn 10, and they are sharply turn-specific rather than increasing monotonically:
turn 16 affected 14/15 runs and turn 25 affected 12/15, while turns 15, 22, and 26 had
no timeout events.

## How much does this confound quality?

It clearly confounds latency and turn-taking. Turn-taking passed on 36.3% of
timeout-affected turns versus 53.1% of immediately unaffected turns. After comparing
affected and unaffected observations at the same turn positions, the gap was still
about 15 percentage points.

It does not explain the strict score through the timeout on that same turn: strict pass
was 58.1% on affected turns and 59.5% on immediately unaffected turns. However, that is
not a clean no-replay comparison. Every Clever Chatter session had already experienced
a timeout/replay by turn 10, so all later tool tests occurred in potentially contaminated
session state. A clean standalone probe with the full prompt, tools, seeded history, and
turn-11 audio called `submit_session_suggestion` correctly, whereas the complete-run
cohort failed that required tool call in 15/15 runs. The replay hypothesis is therefore
plausible for some quality failures but not yet quantified causally.

## Required fix and validation

1. Use the validated private google-genai 2.14.0 EAP wheel through the isolated overlay
   at `/home/khkramer/.cache/aiewf-eval/interaction-status-overlay-private-214`; do not
   downgrade the repository's 2.16 environment in place. The private wheel exposes
   `LiveServerContent.interaction_status`, and a typed live smoke test returned
   `InteractionStatus.REQUIRES_ACTION` for Clever Chatter while Preview returned no
   status field.
2. Teach the Gemini Live service to defer response-final frames, transcript flush, and
   benchmark turn advancement when `turnComplete=true` but interaction status is
   `IN_PROGRESS`. Finalize only on `REQUIRES_ACTION`.
3. Preserve tool calls and intermediate audio while the interaction remains in progress,
   instead of treating them as a completed benchmark turn.
4. Do not replay Clever Chatter turns. On a 15-second no-audio timeout, mark the
   run invalid with the turn and failure reason, persist the recording and raw
   trace, and terminate the attempt.
5. Run an explicit-boundary arm: disable server automatic activity detection and send
   `activity_start`/`activity_end` from the already-running local VAD. This separates
   server-VAD stalls from asynchronous reasoning delay.
6. Re-run the turn-16 and turn-25 hotspots, the four-turn tool sequence, and at least
   three full conversations under each candidate policy, with Preview controls.

Do not use the current recovery-affected Clever Chatter latency values in a leaderboard.
Do not treat the 59.1% strict score as a clean estimate of Clever Chatter's underlying
quality until the no-replay or explicit-boundary arm is run.

## Reproduction

- Analysis script: `analyze_timeout_replay.py`
- Machine-readable output: `timeout-replay-analysis.json`
- Raw-status diagnostic wrapper: `probe_interaction_status.py`
- Speaking-rate analysis: `analyze_speaking_rate.py`
- Speaking-rate output: `speaking-rate-analysis.json`
- Rebuild: `./.venv/bin/python docs/clever-chatter-comparison-2026-08-04/analyze_timeout_replay.py`
