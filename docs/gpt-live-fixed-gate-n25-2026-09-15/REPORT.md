# GPT-Live corrected-gate standard cohort: 25-attempt results

| Model | Complete | Strict pass | Non-tool V2V median | Tool V2V mean | Overall V2V mean | Overall p90 | Measured positive responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-Live + GPT-5.6-Terra (medium), corrected gate | 25/25 | 707/750 (94.3%) | 1696ms | 2169ms | 1870ms | 2400ms | 748/750 |

**Validity:** all recorded turn commits passed the post-input quiet-interval audit.

Every planned turn remains in the accuracy denominator; incomplete attempts are retained. Strict pass combines tool use, instruction following, and factual grounding. Audio turn-taking is separate. V2V measures actual user-speech end to first actual assistant speech, including acknowledgments. It is not time to the substantive answer or tool completion. Missing measurements are not treated as zero.

## Reliability and measurement coverage

- **live:** 25/25 complete; failure reasons none; explicit API-error evidence in 0 attempts; rate-limit evidence in 0 attempts. 748 positive V2V measurements, 1 missing/unrepresented planned turns, 1 nonpositive measurements. Strict 95% conversation-bootstrap interval 92.7–95.7%.
  Confirmed closed sessions: 25/25. Waveform alignment checks passed for 24/25 attempts.

## Judge audit

0 explicit score/explanation conflict flags; raw outputs retained without manual overrides. See `aggregate.json` for every flag and failed turn. Retries are limited to judge execution/parse failures.

## Cohort and provenance

This cohort contains 25 fresh attempts with the corrected input-end gate, concurrency five, and no imported pilot conversations. All previous captures remain diagnostic and are excluded. The exact shared full speech-v2 prompt is used in both frontend and backend; the revised Opus 5 rubric is unchanged. Every capture freezes its implementation and prompt hashes.

See [protocol](PROTOCOL.md), [frozen settings](protocol.json), [all attempts](capture/attempts.jsonl), [gate audit](live-gate-fix/audit.json), and [aggregate](aggregate.json). V2V uses canonical Silero speech detection and input/commit windows; waveform alignment is independently audited. Missing and signed measurements are retained explicitly. The text benchmark and prompt-optimized experiment are untouched.
