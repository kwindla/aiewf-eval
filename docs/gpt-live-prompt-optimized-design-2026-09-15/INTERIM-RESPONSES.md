# Interim speech, reconstructed exchanges, and audio validation

Draft evaluation design, September 15, 2026. This supplements [DESIGN.md](DESIGN.md).
It does not change the running standard cohort, its prompts, or its judge inputs.

## Recommendation

Allow GPT-Live to acknowledge a delegated request and speak naturally while the
backend works. Judge the whole exchange after its substantive response, clarification,
or honest failure has been delivered. Keep backend quality as a separate diagnostic.
An acknowledgment alone must neither finish the exchange nor earn answer credit.

Proposed frontend wording, replacing the current waiting sentence:

> When backend work leaves the user waiting, briefly acknowledge what you are
> checking, then deliver the result when available. Vary the wording naturally;
> avoid repetitive updates. Do not guess the answer or claim an action succeeded
> before confirmation.

“Let me check the program” is appropriate for this KB task. “Looking that up” must
not imply web access we have not provided. A listening backchannel during user speech
and a progress update after a request are different behaviors; measure both.
The proposed wording is not yet rendered into the prompt files or API-tested.

## What the API provides

GPT-Live handles voice independently of delegated backend work. Its prompt can permit
acknowledgments without injecting prerecorded filler. For application-supplied spoken
progress, `session.commentary.append` carries text the model may paraphrase. In managed
Responses mode, application context uses `delegation_id: null`; a non-null append ID
must identify a client-mode delegation, not a Responses ID. An append acknowledgment
confirms estimated context injection, not that the words were spoken. Start with
natural model-generated updates; add application-triggered updates only if listening
shows a need. Avoid adding both and producing duplicate acknowledgments.

Official sources:

- [Prompting](https://developers.openai.com/api/docs/guides/live-prompting)
- [Delegation and useful progress](https://developers.openai.com/api/docs/guides/live-delegation#keep-updates-accurate-and-useful)
- [Transcript timing and grouping](https://developers.openai.com/api/docs/guides/live-conversations#transcript-deltas)

The docs recommend managed Responses delegation as a starting point, with Terra as
a starting backend. They do not identify a universally best configuration. Keep the
prompt-only experiment's Terra medium, voice, tier and tool settings fixed. A later
best-configuration experiment may tune backend effort/tier, but must be labeled as
an additional configuration change rather than attributing gains to the prompt.

## What we already capture

Every completed Live call has `conversation.wav`: 24 kHz stereo PCM, user left,
assistant right. These are locally captured conversations; provider recording storage
is unnecessary. The raw `live-events.jsonl` retains received events, original transcript
fragments and session timestamps, delegation IDs, nested backend response/item/call
IDs, tool results, input boundaries and playback state. `transcript.jsonl` currently
concatenates the spoken deltas into each scripted turn; backend text is not substituted.

The existing gate holds a turn open through known backend work and tool continuation,
then waits for playback to drain and a quiet interval. The approved fix also starts
that quiet interval at input end. This is still a timing heuristic: no authoritative
Live speech-complete event exists. A delayed delegation or substantive continuation
can escape a simple silence rule. Offline regrouping cannot repair a call in which
the next scripted input actually interrupted or redirected the intended answer.

Audio is finalized during pipeline cleanup, so unfinished calls may not yet have a
WAV. For the experiment, add incremental disk recording to retain partial audio even
if the process is killed. Keep recorder changes out of the currently running cohort.

## Proposed harness and reconstruction

1. **Capture first; keep originals immutable.** Save stereo playback, events and
   receipt order. For new experimental captures also log output sample ranges,
   playback drain markers, and clock anchors. Session transcript milliseconds,
   local receipt time and WAV sample position are different clocks; do not subtract
   them without a validated mapping.
2. **Track exchanges separately from speech fragments.** Associate each scripted
   request with its delegations, responses, tool calls/results and spoken spans.
   Acknowledgment, progress, substantive answer, clarification, refusal, and final
   confirmation are annotations within an exchange. Multiple speech bursts are valid.
3. **Hold the next input.** Known backend work, queued tool continuations and active
   playback keep the exchange open. A status-only reply is not completion. Prototype
   a separate turn coordinator that recognizes a delivered answer or request for user
   input using accumulated spoken content and task state, with fixed timeout and
   auditable decisions. Give it no golden answers, expected tool calls, or judge scores;
   it must accept incorrect substantive answers as completed answers. Its processing
   time is harness delay, not model response latency. Freeze this coordinator before
   any experimental cohort; exercise delayed delegation and status-only failures.
4. **Reconstruct offline.** Resolve late transcript fragments using session intervals
   and verified audio alignment, not the `turn` label assigned at network receipt.
   Preserve exact text and chronological order within each exchange, including interim
   speech, wrong claims, repetition, corrections and interruption. Keep raw event IDs
   and audio sample ranges for every reassignment. Do not move a later answer ahead
   of a real intervening user request. Mark ambiguous cross-request attribution for
   review, not favorable automatic reassignment.
5. **Make a separate judge input.** Write `reconstructed/transcript.jsonl` in the
   existing judge schema plus a mapping manifest and normalization version. Attach
   actual tool calls/results to their causally associated exchange without duplicating
   them. Preserve the fixed planned-turn denominator, failures and uncompleted answers.
   Compare raw versus reconstructed judgments and explain every changed assignment.
6. **Judge after reconstruction.** Reuse the shared task-quality rubric on the complete
   spoken exchange; an interim update is not a separate failed benchmark question.
   Keep backend-only evaluation in a separately labeled diagnostic output. Never award
   end-to-end credit for correct backend content that Live omitted or contradicted.
   Reassembly does not itself require relaxing the quality rubric. If timestamps or
   richer event context require a judge input-format change, version it and validate it
   separately rather than silently mixing the scores into the standard leaderboard.
7. **Audit interaction separately.** Keep actual chronology for overlap, barge-in,
   responsiveness, repetitive filler and premature progression. Reconstruction must
   not erase these defects. Listening sounds should be described separately from
   substantive interruption; any revised audio scoring policy needs its own version.

This is a separate evaluation protocol, not merely a prompt change. A row produced
with reconstructed exchanges or a new coordinator needs a method label. For a causal
prompt comparison, run the standard and optimized prompts through the same revised
coordinator and reconstruction; otherwise report the combined-system result as such.
Do not silently replace the current standard cohort's inputs with these reconstructions.

## Separate measurements

| Measurement | What counts |
|---|---|
| First speech V2V | User speech end to first played assistant speech; includes acknowledgments; signed overlap retained |
| Substantive-response latency | User speech end to first played task-relevant answer, necessary clarification, or grounded limitation; excludes progress-only speech |
| Completion latency | User speech end to delivery of the complete answer or confirmed outcome; exclude harness quiet padding |
| Backend latency | Delegation to first useful backend output and to completion, linked by IDs; not spoken latency |
| Backend quality | Correctness and actual tool use on the delegated task; diagnostic only |
| Delivery fidelity | Which required backend details survive into actual speech; omissions and contradictions |
| Interaction | Backchannels, overlap, interruptions, repeated progress, missing or truncated answers |

An example may yield first speech at 0.8 seconds and a substantive answer at 4 seconds.
Publish both, not a single 0.8-second number described as answer latency. Annotate
substantive spans from transcript plus audio, with confidence and review flags; VAD
alone identifies speech, not whether it answers the question. An answer that never
arrives stays missing/failed rather than being assigned the acknowledgment latency.

## Listening and independent transcription

Use the [example bundle](interim-example.json) and audio excerpt (`interim-example.wav`, local-only; not committed)
from diagnostic pilot attempt 1, scripted turn 4 (the fifth exchange). It contains
an acknowledgment followed by the factual response. It is illustrative, not a newly
scored optimized run. The full recording is linked in the bundle.

Before scoring the experimental pilot, listen to a prespecified sample including
long backend waits, action confirmation, recommendations, interruption, closure,
and all reconstruction ambiguities. Check the audible response against event text.
No listening audit is claimed merely because a WAV exists or alignment passed.

If needed, transcribe each recorded channel independently with timestamps; preserve
pauses/overlap rather than using a diarized mono mix. Do not supply the KB or expected
answers as transcription hints. Align the result to the original fragments and
review disagreements around names, numbers, truncation and turn boundaries. Store
ASR model/version, configuration, original output and every correction. Do not let a
transcriber repair a wrong answer from context. Forced alignment can time verified
words but is not independent evidence that those words were spoken. Transcription
and normalization are preprocessing; the unchanged judge is invoked only afterward.

## Acceptance probes before a scored cohort

- Slow backend with one acknowledgment and later substantive speech: one exchange,
  complete speech preserved, distinct first-speech and substantive latencies.
- Delayed delegation after status speech: next scripted input remains held.
- Multiple function calls/continuations: no early advance, no duplicated call credit.
- Backend correct but voice omits a required detail: backend diagnostic passes that
  fact; end-to-end spoken response still loses credit.
- Late transcript fragment: reassignment follows verified time without reordering
  actual conversation or inventing missing text.
- Interruptions, correction and unresolved timeout: retain actual chronology and
  unsuccessful outcomes, including partial audio.

The running 25-attempt standard cohort remains frozen. Implement and test this richer
experimental protocol separately before deciding whether its scores are comparable
with the standard table.
