# Speech-v2 extension: GPT-Live and Gemini 3.8 medium

Authorized scope: add 20 GPT-Live conversations to the existing five-run concurrency
pilot, and capture 25 Gemini 3.8 Live Extended Thinking conversations at medium.
Each arm has 25 attempted conversations and a fixed 750 planned-turn denominator.
Incomplete conversations and provider errors are retained; no replacement captures.
The separate single GPT-Live validation conversation is excluded from this cohort.

## Frozen configuration

- GPT-Live: `gpt-live-1`, delegated `gpt-5.6-terra` backend at medium reasoning,
  Marin voice, unchanged frontend/backend full speech-v2 prompt and pipeline.
- Gemini: `gemini-3.8-live-extended-thinking`, `thinking_level=medium`, same voice
  benchmark configuration as the prior low arm except for thinking level.
- Both use the exact common speech-v2 model prompt, KB, questions, input audio, and
  tools. No model-specific prompt changes, recovery nudges, or replacement attempts.
- The same revised Opus 5 speech judge (`speech-v2-policy-compliance-1`) applies to
  every conversation. The five imported pilot attempts use their preserved uniform
  rejudgments; new captures receive that same judge. Retry only execution/parse
  failures, never unfavorable scores.
- Five concurrent captures per provider, with the two providers running in parallel.
  Judging and offline audio processing run alongside captures with separate limits.
- Dependencies: Pipecat 1.10.0, OpenAI 3.14.0, google-genai 2.23.0. The original
  pilot matches the current OpenAI/Pipecat versions and adapter source hash.
- `protocol.json`, source hashes, frozen model/judge prompt files, and dependency
  lockfiles record reproducibility. Source recordings and first-pass judgments of
  imported attempts remain untouched.

## Measurements and publication

Strict pass requires tool use, instruction following, and KB grounding together.
All 750 planned turns remain in each model's denominator. Turn-taking and reliability
are separate from semantic accuracy. Report explicit API/rate-limit evidence
separately from no-response timeouts or judge-service failures.

Voice-to-voice latency uses actual speech boundaries in recorded stereo WAVs and
includes acknowledgments. Gemini retains the original tag-based metrics and receives an independent input-window audit; GPT-Live uses
input/commit windows because its output stream contains continuous silence. Both
use the same Silero detector and thresholds. Report measurement coverage and missing
or nonpositive latencies explicitly. GPT-Live also receives the waveform alignment
check used for the original pilot. Audio-analysis failures do not remove captures.

The prior Gemini low-arm latency analysis has known measurement gaps and a pending
turn-association audit. New results must expose analogous gaps rather than treating
missing measurements as instant responses. Model/frontend prompts remain frozen
regardless of analysis findings.

No text-model prompt, judge, score, or README text-model section is changed.

## Audio association audit

The independent audit confirmed that a missing bot marker can shift the legacy
analyzer's ordinal turn labels. The reported medium latencies therefore use actual
speech matched to logged input windows, verified against the user waveform at three
speech landmarks per utterance. The same audit is applied to all 25 preserved low
recordings for a comparable low/medium latency comparison. The detector and its
thresholds remain identical. Original metrics and semantic judgments are preserved.

In the low cohort, seven closing turns call `end_session` without speech. Attempt
24 additionally has no logged turn-0 user input and records a startup greeting in
that transcript position. This is documented as a prior harness/startup limitation,
not an API failure. All new medium captures must pass a complete input-sequence
check before being described as full benchmark conversations.

## Imported pilot load context

The five imported GPT-Live pilot sessions overlapped the tail of the earlier OpenAI
Realtime campaign, as recorded in
`../gpt-live-readiness-2026-09-15/concurrency-pilot/background-load.json`. The 20 new
GPT-Live captures use five simultaneous OpenAI sessions. Their model, prompts,
adapter, and OpenAI/Pipecat versions match the pilot, but the original pilot had
additional background provider load and occurred earlier in the day.
