# Nova 2.5 Sonic — October 19, 2026

Test `amazon.nova-2-5-sonic` on AWS Bedrock in us-east-1, Tiffany voice,
HIGH endpointing, the existing speech-v2 system prompt, all 30 unchanged
scripted turns, audio clips, knowledge base, and tools. Use provider sampling
defaults; no added reasoning control. AWS confirms the exact model ID is active.

First exclude a direct model/protocol/tool round-trip and a full 30-turn smoke.
Repair adapter/protocol defects before collection and record those changes;
never tune the benchmark prompt or select conversations by quality. Freeze
configuration, schedule, audio, sources, dependency versions, and runtime copies
before canonical collection. Preserve the prior adapter and publication bytes.

Collect the first 25 complete 30-turn conversations, in ordered batches of at
most three. Retain every attempt, including partial/failed captures. Thirty-minute
timeout per attempt; review after 50 attempts. No generation recovery, history
truncation, prompt replacement, or replay of interrupted turns. Native Pipecat
session continuation may preserve the conversation at AWS's session boundary;
log it and verify full input continuity. Any failed smoke or completed capture
with structural/timing defects is held for review, not replaced for a better score.

Freeze all 25 raw judge inputs and effective judge source hashes before grading.
Use the current recovery-context-correct Opus 5.5 Medium judge, unchanged voice
rubric, no tools/settings sources, and no private reasoning evidence. API credit
failures wait at least 90 seconds before retrying. Operational errors never count
as quality judgments.

Recompute latency and turn-taking from stereo recordings. Verify every input
clip against the tracked WAV with the existing three-landmark correlation audit:
correlation >= .95, absolute offset <=250ms, within-turn drift <=100ms.
Keep signed/positive first-speech statistics, sample counts, non-tool median,
tool mean, missing timing, turn-taking, and unsuccessful-attempt rate. Bootstrap
by conversation with the existing seed and 20,000 draws. Publish an additive
README speech row and report; preserve all prior cohorts/results. Run relevant
tests, publication checks and public preview. Do not commit or push.

References checked October 19:
- https://aws.amazon.com/about-aws/whats-new/2026/10/amazon-nova-2.5-Sonic/
- https://docs.aws.amazon.com/ai/responsible-ai/nova-2-5-sonic/overview.html
- https://docs.aws.amazon.com/us_en/bedrock/latest/userguide/guardrails-supported.html

## Authorized reference-message layout amendment

Following the 50-request filter diagnosis, the user authorized proceeding with
policy in SYSTEM and the complete unchanged knowledge base in a noninteractive
USER history message. Set MTE_NOVA_KB_CONTEXT_ROLE=user. Assert concatenation of
both parts is byte-identical to the original 41,161-character speech-v2 prompt.
No reference facts, including the voice-cloning session and Replicate listing,
are removed. Three full-reference native readback checks passed. The earlier
blocked smoke and all diagnostic artifacts remain preserved in the parent study.

Use this separate user-context campaign namespace for the new excluded full-task
smoke and all canonical attempts. The original native tool probe is reused by
verified artifact hash. Any harness repair is recorded before the generation
freeze and accompanied by a new excluded smoke identity; do not overwrite failed
smokes or reuse their grades. The current Opus 5.5 Medium recovery-context judge
will score the ordinary scripted user/assistant/tool transcript; the initial KB
is reference context, not an additional scripted turn.

## Local VAD repair before canonical collection

The first USER-reference smoke completed all 30 turns and recorded tool results,
but its waveform analyzer found no user turn markers. The old Nova pipeline put
vad_analyzer on TransportParams, a setting Pipecat 1.12 silently ignores. Move
that unchanged 0.2s/0.8s Silero analyzer onto LLMUserAggregatorParams, matching the
existing realtime pipelines. Preserve that first smoke, failed timing audit, and
its exact serving source separately. Run smoke-v2 in a fresh namespace before
freezing canonical generation. This is a measurement/protocol repair; no prompts
or model-quality behavior are tuned or selected.

The retained first smoke's independent waveform audit also exposed roughly one
second of user-track offset. The Nova adapter released its paced input one second
after the recording baseline, so both the feeder's catch-up and output recorder
filled that initial silence. Release readiness at the shared baseline instead.
The second excluded smoke confirmed the same startup offset and was stopped once
the defect was reproduced. Add the shared realtime LLMFrameLogger and its recording
baseline so the analyzer receives standard VAD timing records. Preserve both
smokes/source versions; smoke-v3 validates all repairs before canonical freezing.

## Tool-result delivery repair and new capture profile

The first frozen USER-context profile exposed a race: the universal assistant
aggregator sometimes held the context update containing a completed tool result
until a later scripted user turn. Raw logs show a recorded success result without
a corresponding native Sending tool result event; that event appears only after
the following scripted audio/context push. This is an adapter defect, not a model
quality judgment. Stop that profile's three attempts and retain them as harness
exclusions, with its original source freeze and WAVs intact. No grades were made.

The new profile sends each native tool result immediately from the registered
handler, then records it through the ordinary result callback. It shares Pipecat's
completed-tool-call ID set so later context updates never resend it. Tests verify
immediate exactly-once delivery and send-error rollback. Preserve the original
prompts, tools, results, sampling, pacing and session continuation. Validate a new
full-task smoke, including native delivery of every recorded result, before a new
freeze and the first 25 complete conversations. Only this repaired profile may
contribute the new leaderboard row; disclose the three older harness exclusions.

The first repaired tool-delivery smoke completed 30 turns with all native results
sent, but the marker-based analyzer counted 28 responses: Nova sometimes starts
audio before local VAD-stop, which the common transport then suppresses as a late
duplicate. For Nova's sequential scripted WAVs, explicitly arm exactly one bot
marker when each WAV is queued; ignore subsequent local stop rearming in this
opt-in mode. Other providers retain their existing VAD-driven tagging. This
changes only recording markers, not model input, prompts or response timing.
Preserve that smoke and its failed marker audit. Smoke-final validates this last
measurement repair before canonical capture is frozen.

[Evidence storage and local audit instructions](../../README.md).
