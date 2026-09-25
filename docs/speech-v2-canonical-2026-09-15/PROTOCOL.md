# Canonical speech-to-speech refresh: speech-v2

## Frozen design

- Five arms: GPT-Realtime-2.1 low, Grok Voice Think Fast 1.0, GPT-Realtime-1.5,
  Gemini 3.1 Flash Live minimal, Gemini 3.8 Live Extended Thinking low.
- 25 attempted full conversations per arm, 30 scripted turns each, fixed denominator
  750 turns per arm. No replacement of early model exits, provider failures, or stalls.
- Two-turn compatibility probes and the xAI tool-call compatibility probe are excluded.
- The new benchmark ID is `aiwf_medium_context_speech_v2`. Its prompt is independent
  of `aiwf_medium_context`; knowledge base, audio, questions, and tools are shared.
- Provider-neutral prompt contains no scripted user names, turn indices, golden
  responses, or benchmark-specific session IDs. Tools are supplied as actual schemas.
- Judge: Opus 5, voice-v2 rubric including the governing voice policy. Same judge
  and rubric for every arm, semantic equivalence permitted, same strict conjunction
  of tool use, instruction following, and KB grounding. Audio turn-taking is separate.
  The final voice judge receives the same complete knowledge base as the assistant
  to verify alternative recommendations and facts; it is authoritative over golden
  examples. The earlier rubric without this reference is preserved with each
  interrupted cohort and contributes no canonical scores.
- One judge pass per recorded conversation. Retry only malformed/unavailable judge
  responses, preserving attempt logs. No score-based retries or label overrides.
- Missing future turns count as failures in every accuracy dimension. Observed
  audio latency is reported with observation counts, conditional on measurable responses.
- Model-initiated early exit is a behavior failure even if the CLI exits successfully.
- Recovery nudges disabled in all arms; input audio and playback are otherwise the
  existing benchmark policy. Gemini 3.8 uses explicit activity and terminal IDLE;
  Gemini 3.1, xAI, and OpenAI use their existing server-VAD paths. These settings
  are operational configurations, not claims of identical provider turn detectors.
- At most five concurrent sessions per provider, shared across that provider's arms.
  Interleaved submission order; wall-clock duration can differ by response length.
- 15-minute process watchdog per attempt, recorded as incomplete on expiration.
- Judge jobs and offline audio analysis do not reuse the live-session concurrency budget.
- Canonical generation starts only after compatibility failures are resolved. Any
  newly discovered systemic harness bug pauses the campaign for an explicit audit;
  affected runs are not quietly relabeled as model failures or selectively replaced.

## Provenance and isolation

`protocol.json`, `system-prompt.txt`, and `judge-prompt.txt` freeze the policy and
hashes. Each new run also saves `system-prompt.txt` and `benchmark-provenance.json`.
`preserved-text-inputs.json` records checksums for the original text prompt,
questions, tools, knowledge base, and README text section.

The CLI judge-model override now actually reaches the judge, and summaries stamp
the effective judge model and rubric hash. The previous text judge rubric and its
implicit default are unchanged. The speech rubric is selected only for the new ID.

## Prompt rationale

Separate unrelated-topic refusals from honest, contextual declines about event
requests. Allow quoting rooms from the schedule while prohibiting invented routes.
Make reuse of conversation facts mandatory, ask only for genuinely missing arguments,
and tie confirmation of success to tool results. Clarify completion of information
requests and distinguish an acknowledgement from an intention to end the session.

This follows the explicit conversation rules/tool conditions in Google's Live
best practices and xAI's voice prompting guide:
- https://ai.google.dev/gemini-api/docs/live-api/best-practices
- https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech/prompting-guide

## Compatibility findings before canonical capture

Pipecat 1.10 removed the audio-item completion handler and renamed session-update
internals used by the xAI adapter. Its old adapter emitted errors in the excluded
smoke run. The adapter now uses the current session-update method and emits
TTSStoppedFrame at response.done, matching Pipecat's whole-response audio bracketing.

## Interrupted initial cohort: tool-result delivery

The first capture batch was stopped after a confirmed harness integration bug.
All 25 launched attempts, including successful ones, are excluded and preserved
under `interrupted-pre-tool-delivery-fix/`; their run directories remain intact.
This is an invalidated cohort, not selective replacement of model failures.

Pipecat 1.10's universal assistant aggregator gates tool-result delivery on local
speech state. Gemini 3.1 sometimes called a tool before the local VAD marked the
user's turn complete. The result entered context immediately, but the aggregator
did not push it upstream. Realtime mode no longer pushes context on user-turn
completion, so the result remained unsent until shutdown. For example, the vote
in `20260915T125341_gemini-3.1-flash-live-preview_303944bf` completed at 13:00:36.492
local time, but its result was sent at 13:00:52.544, after the 15-second timeout.
This is evidence of a harness failure, not a Gemini API error.

The realtime-only aggregator now returns requested tool results independently of
local speech flags, while retaining result validation, grouping, and explicit
`run_llm=False`. Cascade/text aggregation is unchanged. Regression tests cover
tool results arriving while local user and/or bot speech flags remain set. All
five arms receive the same fix. A fresh canonical cohort follows live tool probes.

Those probes also exposed a transcript ownership race in `TurnGate`: two audio
segments separated by back-to-back stop/start frames could cancel the delayed
completion task before its coroutine started. The previous cancellation handler
could not restore the transcript in that ordering. The gate now retains the
pending transcript until completion commits, and merges subsequent segments.
Regression tests cover cancellation before task start and explicit state clearing.
The affected probes are preserved in `tool-delivery-smoke-before-turngate-fix/`.
All probes are excluded from the canonical cohort.

## Interrupted second cohort: premature completion before tool follow-up

All 30 launched attempts in `interrupted-pre-tool-followup-fix/` are also excluded.
Review of the first judged Grok conversation showed that turn 11's spoken preface
("I'll submit that") drained before the tool confirmation arrived. The harness
advanced to turn 12, then assigned the confirmation to that turn and advanced again
while turn 12's input was still playing. This created artificial answer/tool lag.
The concrete example is `runs/aiwf_medium_context_speech_v2/20260915T131050_grok-voice-think-fast-1.0_d1718011`
(archived `grok-04.log`).

The gate now requires provider completion for all five arms. For OpenAI-compatible
protocols, a tool call also holds the turn until the subsequent response begins and
completes. Drained pre-tool audio alone cannot trigger the next user utterance.
Gemini uses its native completed-turn signal (plus GA interaction-status gating).
This changes capture correctness, not the frozen prompt, task set, or rubric.
Full 30-turn validation conversations for all five arms precede the final launch
and are excluded in `full-validation/`.

The first Grok full-validation probe showed that the legacy provider detector
required "realtime" in the model name, so it did not recognize
`grok-voice-think-fast-1.0`. The Grok pipeline now declares its provider identity
directly, enabling its greeting and post-tool completion behavior regardless of
model naming. That probe was stopped; `grok-provider-fix.log` is its replacement
validation, also excluded from scoring.

## Final launch

All five full validation conversations completed with valid 30-turn transcripts;
see `full-validation/verified.json`. The post-tool wait was exercised in each
OpenAI/Grok validation. The voice judge sanity pass completed all 30 turns and
matched the frozen rubric hash. The final campaign launched at approximately
13:39 Pacific on September 15, after these checks. `implementation-hashes.json`
identifies its code and dependency lock. The original text inputs and README
text-model section were verified unchanged immediately before launch.

## Approved judge-only revision, September 15

The user approved clarifying the shared speech judge and uniformly rejudging all
125 canonical attempts plus the six standard-prompt GPT-Live validation captures.
The revision requires governing-policy compliance for instruction following and
independent factual assessment against the full KB. It changes no model prompt,
recording, tool schema, text judge, or audio measurement.

The original `protocol.json` and all first-pass run outputs remain frozen. The
[separate revision protocol](rejudge-policy-compliance-1/protocol.json) records the
new judge hash and hashes of all original inputs/outputs. Revised outputs live in
that directory, not in the capture directories. Original per-turn audio assessments
are reused, so audio labels and quantitative metrics stay fixed. Each conversation
receives one complete judgment with the revised rubric; only execution/parse failures
may be retried. No score-based retries or manual label overrides are permitted.

Judge processing began at concurrency five, then increased to 15. The scheduler
was replaced while preserving its in-flight worker processes and completed outputs;
no successful judgment was repeated. Execution history is recorded separately from
the immutable cohort manifest. This affects offline judging only, not capture load.
