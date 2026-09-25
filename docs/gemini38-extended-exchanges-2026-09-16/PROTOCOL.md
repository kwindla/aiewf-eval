# Gemini 3.8 Extended Thinking: exchange-aware validation

Model `gemini-3.8-live-extended-thinking`, thinking **medium**, as in the most recent
Gemini benchmark arm. Shared full speech-v2 prompt, KB, tools, scripted questions and
audio are unchanged. Judge: Opus 5, speech-v2-policy-compliance-1.

## Why this differs from GPT-Live

Google supplies an authoritative overall interaction status. `turnComplete` can
close an intermediate utterance while `interaction_status` remains `IN_PROGRESS`.
We already deferred terminal pipeline frames in that case. This experiment adds
raw-message provenance, status-only IDLE handling and explicit NON_BLOCKING tool
declarations. It does not add an LLM turn-completion classifier or change prompts.

Source: [Google Thinking in the Live API](https://ai.google.dev/gemini-api/docs/live-api/thinking).

## Capture and evaluation

- Isolated adapter in this directory; existing benchmark source and results retained.
- Capture full stereo `conversation.wav`, raw server events (audio payload lengths
  only; actual audio is in the WAV), setup config, phase boundaries and exchange commits.
- Continue receiving after intermediate `turnComplete`; hold the next question until
  IDLE, input completion and playback drain. Handle status-only IDLE as completion.
- All functions explicitly NON_BLOCKING. No function scheduling override.
- Reconstruct each entire spoken exchange from output-transcription fragments; keep
  interim speech. Do not use thought text as speech or move answers across user turns.
- Preserve actual tools/results. Compare reconstruction to original transcript, with
  only whitespace normalization for the integrity check. Mismatches block judging
  pending review. Retain a per-fragment mapping and original hashes.
- Validate every input index and IDLE-before-commit. Input logs round to milliseconds;
  allow 2ms rounding tolerance when checking against full-precision commit timestamps.
- Judge only reconstructed complete exchanges with the unchanged quality rubric.
  Assess audio interaction separately; transcript judge default turn-taking values
  are not audio validation results.
- Audio: canonical Silero and input-window association, independently verified at
  three waveform landmarks per user utterance. Retain signed overlap and missing
  measurements. Silent `end_session` tool-only turns are counted explicitly.
- Report first-speech V2V (acknowledgments included) and speech-completion timing.
  Do not call either “time to substantive answer.” That requires verified semantic
  audio alignment and remains a separate measurement, not guessed from receipt time.

## Stages

1. Full 30-turn smoke capture, provider-event audit, Opus 5 judge and quantitative
   audio analysis. Exercise intermediate phases in real traffic and unit tests.
2. Only if capture/measurement checks pass, 25 fresh attempted conversations at
   concurrency five, 750 planned turns. Smoke excluded. No replacement of failures.
   Pilot quality score is not an acceptance threshold.
3. Per-run processing and a separate cohort report with reliability and coverage.
   Do not overwrite old medium/low scores or the shared prompt.

Other provider traffic is recorded as background load: the GPT-Live experimental
launcher was stopped after its external coordinator misclassified valid replies;
previously started captures were allowed to finish as diagnostics. Gemini uses
provider state, not that classifier.

Recovery note: RealtimePipeline inherits supports_recovery=False, so its global
recovery switch does not inject synthetic turns. The cohort also sets that switch
explicitly off, and reconstruction rejects any unexpected recovery rows.

## Closing-turn defect caught by smoke v1

The shared end_session handler returned success then canceled capture immediately,
writing an empty final transcript despite a spoken farewell fragment. The full
exchange audit rejected this: no IDLE, no normal exchange commit, and lost spoken
text. First smoke artifacts remain under smoke-v1-results and smoke-v1-closing-defect.log.

The isolated adapter now returns the same successful tool result, then waits for
provider IDLE and playback drain before closing. A genuinely silent close has an
IDLE-and-drain fallback; an early end_session remains an incomplete model outcome.
Smoke v2 repeats all 30 turns with this fix before the cohort can launch.

## Launch status

Smoke v2 completed all 30 exchanges, preserved 10 intermediate phases and the full
spoken closure, passed waveform alignment and transcript reconstruction, and scored
27/30 on the unchanged Opus 5 rubric. All 30 V2V observations were measurable: mean
1,733ms and median 1,552ms to first speech (including interim acknowledgments).
The 25-attempt cohort launched at concurrency five with frozen source, prompt and
judge hashes in [cohort/protocol.json](cohort/protocol.json). Both smoke attempts
are excluded. See the [validated smoke report](smoke-results/REPORT.md).
