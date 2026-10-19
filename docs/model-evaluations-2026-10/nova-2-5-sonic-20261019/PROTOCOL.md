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

[Evidence storage and local audit instructions](../README.md).
