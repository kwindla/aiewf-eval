# GPT-Live and Gemini 3.8 medium: 25-attempt results

| Model | Complete | Strict pass | Non-tool V2V median | Tool V2V mean | Overall V2V mean | Overall p90 | Measured positive responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-Live + GPT-5.6-Terra (medium), diagnostic cohort | 25/25 | 710/750 (94.7%) | 1728ms | 2105ms | 1828ms | 2368ms | 746/750 |
| Gemini 3.8 Extended Thinking (medium) | 25/25 | 698/750 (93.1%) | 1472ms | 1554ms | 1499ms | 1920ms | 743/750 |

**GPT-Live validity hold:** these 25 attempts are diagnostic, not a canonical result. The input-end gate can advance after an early backchannel before allowing a post-input response interval; see the [evidence and isolated fix](live-gate-fix/README.md). Gemini medium is unaffected.

Every planned turn remains in the accuracy denominator; incomplete attempts are retained. Strict pass combines tool use, instruction following, and factual grounding. Audio turn-taking is separate. V2V measures actual user-speech end to first actual assistant speech, including acknowledgments. It is not time to the substantive answer or tool completion. Missing measurements are not treated as zero.

## Reliability and measurement coverage

- **live:** 25/25 complete; failure reasons none; explicit API-error evidence in 0 attempts; rate-limit evidence in 0 attempts. 746 positive V2V measurements, 3 missing/unrepresented planned turns, 1 nonpositive measurements. Strict 95% conversation-bootstrap interval 93.3–96.0%.
  Confirmed closed sessions: 25/25. Waveform alignment checks passed for 25/25 attempts.
- **gemini38medium:** 25/25 complete; failure reasons none; explicit API-error evidence in 0 attempts; rate-limit evidence in 0 attempts. 743 positive V2V measurements, 7 missing/unrepresented planned turns, 0 nonpositive measurements. Strict 95% conversation-bootstrap interval 91.7–94.3%.

## Judge audit

0 explicit score/explanation conflict flags; raw outputs retained without manual overrides. See `aggregate.json` for every flag and failed turn. Retries are limited to judge execution/parse failures.

## Cohort and provenance

GPT-Live consists of the five original concurrent pilot attempts plus exactly 20 new attempts. The separate single-run validation is excluded. The imported pilot overlapped earlier OpenAI Realtime traffic; the 20 new Live captures use five OpenAI sessions at a time (see the protocol load-context note). Gemini medium consists of 25 new attempts. The same shared full speech-v2 prompt and revised Opus 5 rubric apply to both arms and the existing comparison table. Imported pilot judgments are their preserved uniform rejudgments.

See [protocol](PROTOCOL.md), [frozen settings](protocol.json), [all attempts](capture/attempts.jsonl), and [aggregate](aggregate.json). Gemini medium latency uses input-window association validated against the recorded user waveform, avoiding the one-bot-marker-per-turn assumption in legacy analysis. Original tag-based metrics remain preserved. All source recordings and earlier scores remain preserved. Text-model prompts, judging, and README scores are unchanged.
