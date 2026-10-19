# Recent-model low-reasoning campaign — October 18, 2026

Approved matrix: GPT-6 Luna none/low; Sonnet 5.5 between_tools+low/adaptive+low;
Haiku 5.5 disabled+low/adaptive+low; Muse Spark 1.3 minimal/low; Muse Spark Open
low; Grok Voice 3.0 none. Nine text configurations, 30 assigned conversations
per configuration; one voice configuration, first 25 complete conversations.

Use unchanged medium-context tasks, tools, and KB. Text uses the text policy;
voice uses speech-v2 and its shared speech rubric. Judge all with Opus 5.5 Medium,
adaptive thinking, no tools or external settings. Strip only the existing
unambiguous reasoning delimiters via the shared judge's normalization. Preserve
raw transcripts, requests, responses, logs, recordings, all unsuccessful attempts,
and full judge outputs. Credit errors wait at least 90 seconds before retrying.
No selective retry based on quality. Credit reload waits are included in any affected request latency and disclosed as operational outliers; pre-stream generation retries are enabled at 90 seconds with a 600-second idle timeout for Anthropic.

OpenAI uses Responses, standard reasoning mode, explicit none/low, Priority tier.
Anthropic uses explicit thinking modes and low effort; display omitted for adaptive,
16,384 maximum output including thinking, provider-default sampling. between_tools
has no display argument and must not be labeled fully disabled. Muse Spark 1.3 uses
OpenRouter pinned to Meta with no fallbacks and required parameter support: no
first-party Meta credential is installed. Explicit minimal/low, 16,384 output cap,
provider-default sampling. Spark Open uses Baseten's catalog checkpoint
meta-models/Muse-Spark-Open-816B-A42B-NVFP4, low (strength 8), temperature/top-p 1.0,
16,384 output cap, three-second pre-request pacing, bounded pre-stream 429 retries.
Grok pins grok-voice-3.0 and explicitly sends session reasoning.effort none.

Before freezing each configuration, direct probes verify identity and tools;
separate full-conversation smoke runs exercise every mode and are excluded from
canonical cohorts. Record all smoke failures and any harness repairs. Freeze
source/config/schedule hashes before each collection; do not change runtime after
collection starts. Preserve prior runtime snapshots so older studies remain
reconstructible despite these new model-routing changes.

Text collection uses the existing first-valid-response collector: infrastructure
failures before any answer may be replaced, but an early end after valid output is
retained and missing turns fail the fixed 900-turn arm denominator. Also report
complete-conversation-only metrics for comparison with the current README; do not
hide incomplete counts. Alternate two-mode arms in deterministically randomized
pairs, concurrency one per text provider; Sonnet and Haiku share an Anthropic lane.
Voice uses up to three concurrent sessions (the previous speech study allowed five), with no recovery or turn replay; use first 25 complete, harness-valid captures,
retaining incomplete attempts and reporting their failure rate. Never replace a
complete capture for a worse grade. Hold unresolved harness/alignment cases.

Text latency: first visible answer/tool output, exclude scripted index zero,
include later recovery responses; retain raw TTFT and missing diagnostics. Voice:
recorded waveform first speech minus end of recorded user speech, existing Silero
analysis, retain signed/missing observations. P95 uses linear interpolation.
Quality uncertainty: 20,000 conversation-cluster bootstrap resamples.

References: https://developers.openai.com/api/docs/models/gpt-6-luna ;
https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide ;
https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5 ;
https://dev.meta.ai/docs/reasoning ; https://dev.meta.ai/docs/muse-spark-open/prompting ;
https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech

[Evidence storage and local audit instructions](../README.md).
