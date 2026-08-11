# Gemini 3.1 Flash Live Preview minimal-thinking top-up

Date: 2026-08-05

The combined cohort contains the frozen March 10-run cohort and five new complete conversations.

| Metric | New five | Combined fifteen |
|---|---:|---:|
| Judged turns | 150 | 450 |
| Strict pass | 96.0% | 93.1% (90.9–95.1%) |
| Tool use | 96.0% | 93.3% |
| Instruction following | 96.7% | 93.8% |
| Knowledge grounding | 100.0% | 100.0% |

## No-response, no-replay reliability

The same policy is applied to both models: a logged 15-second no-response event terminates the run and the utterance is not replayed. A replay-enabled August 4 Gemini control is therefore classified as failed at its first logged timeout, even though replay eventually completed the conversation. WebSocket reconnects without a no-response timeout are tracked separately.

| Cohort | No-response-free completion | First failure turns |
|---|---:|---:|
| Historical recent runs | 11/12 (91.7%) | [25] |
| Current top-up attempts | 7/8 (87.5%) | [12] |
| Combined | 18/20 (90.0%; Wilson 95% CI 69.9–97.2%) | [25, 12] |
