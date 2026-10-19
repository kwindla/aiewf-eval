# Grok Voice 3.0 high follow-up — October 18, 2026

User authorized testing high after the completed none-mode study. Collect the
first 25 complete 30-turn speech-v2 conversations. Retain every unsuccessful
attempt. Up to three simultaneous voice sessions, in batches; 30-minute timeout
per attempt; review after 50 attempts. No completion by grade, prompt changes,
recovery, replay, or quality-based replacement. Exclude one direct protocol/tool
probe and one full-task smoke before collection. A failed smoke is held for review.

Pin `grok-voice-3.0`, inject `session.reasoning.effort=high` through the existing
Grok pipeline, and verify the server echo. The none baseline used the identical
speech-v2 tasks, audio inputs, serving pipeline, SDK versions, and waveform
analysis. The shared base factory has an unrelated post-capture Sonnet-default
compatibility repair; Grok serving code and voice behavior are unchanged.
The portable config's endpoint-concurrency field remains 1 for schema validation;
the dedicated voice driver, as in the none study, batches up to three sessions.

Freeze current configuration, schedule, generation sources, and exact runtime
snapshots before canonical capture. Freeze all 25 raw judge inputs and judge
source/driver hashes before paid judging. Reuse the unchanged original judge,
Opus 5.5 Medium, speech-v2 rubric, tool-free SDK settings, and the existing
90-second credit-reload retry. Never use private reasoning as grading evidence.

Recompute actual recorded-waveform speech latency and turn-taking. Require all
30 input clips per capture to match the tracked audio and the existing correlation,
offset, drift, and alignment thresholds. Hold any completed capture with timing
issues for review; never replace it for a better score. Preserve signed/positive
latency distributions and timing sample counts.

Compare high with the existing 25 none captures, without regenerating or
rejudging the baseline. The source/grade/aggregate hashes of that completed study
are recorded before this follow-up. Results are descriptive independent cohorts,
not matched identical responses. Bootstrap by conversation, using the existing
seed and 20,000 samples. Publish a separate high row and a mode comparison report.
Run the full tests, chart freshness, whitespace check, publication idempotence,
and verify the public preview. Do not commit or push unless requested.

API reference verified October 18: https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech

[Evidence storage and local audit instructions](../README.md).
