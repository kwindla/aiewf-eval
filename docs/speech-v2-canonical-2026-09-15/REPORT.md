**Superseded first-pass scores:** see the [uniform revised-judge results](rejudge-policy-compliance-1/REPORT.md) and [before/after audit](rejudge-policy-compliance-1/AUDIT.md). Original figures below are retained.

# Speech-to-speech benchmark: voice policy v2 — first-pass scores

**Scoring review pending:** all 125 captures and first-pass judgments are complete. A cross-model audit found inconsistent application of the existing no-counting policy on turn 4. These are the unmodified frozen-judge scores, not corrected final reference scores. See [audit data](session-count-audit.json) and the [judge-only clarification proposal](judge-clarification-proposal.md), which has not been applied. Model prompts are unchanged.

| Model | Complete | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grok-voice-think-fast-1.0 | 25/25 | 98.8% | 742/750 | 742/750 | 750/750 | 730/750 | 2496ms | 5984ms | 2689ms | 88ms |
| gpt-realtime-2.1 (low) | 25/25 | 95.2% | 733/750 | 729/750 | 747/750 | 750/750 | 1568ms | 9568ms | 1499ms | 75ms |
| gpt-realtime-1.5 | 25/25 | 94.5% | 739/750 | 720/750 | 736/750 | 737/750 | 1120ms | 3424ms | 1890ms | 160ms |
| gemini-3.8-live-extended-thinking (low) | 25/25 | 93.5% | 747/750 | 706/750 | 739/750 | 748/750 | 1440ms | 8896ms | 1504ms | 60ms |
| gemini-3.1-flash-live-preview (minimal)* | 24/25 | 91.7% | 711/750 | 702/750 | 732/750 | 744/750 | 1328ms | 3936ms | 1646ms | 98ms |

25 attempted conversations per model; every row uses the fixed 750-turn denominator. Missing future turns count as failures. Strict pass is the conjunction of tool use, instruction following, and KB grounding. Audio turn-taking is reported separately. V2V is conditional on measurable responses and includes the first scripted response, matching the historical speech table convention.

## Uncertainty and latency coverage

- gpt-realtime-2.1: strict 95% conversation-bootstrap interval 94.3–96.1%; V2V observations 615 non-tool / 135 tool.
- grok-voice-think-fast-1.0: strict 95% conversation-bootstrap interval 98.1–99.3%; V2V observations 592 non-tool / 138 tool.
- gpt-realtime-1.5: strict 95% conversation-bootstrap interval 93.5–95.5%; V2V observations 603 non-tool / 122 tool.
- gemini-3.1-flash-live-preview: strict 95% conversation-bootstrap interval 89.2–94.1%; V2V observations 588 non-tool / 126 tool.
- gemini-3.8-live-extended-thinking: strict 95% conversation-bootstrap interval 91.7–95.1%; V2V observations 598 non-tool / 143 tool.

## Reliability and observed-turn accuracy

* marks a model with incomplete conversations. Failures in this corrected cohort are counted for every model under the same policy. The interrupted pre-fix cohort is excluded in full; its tool-delivery timeouts were harness failures, not API errors.

- gpt-realtime-2.1: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 714/750 (95.2%).
- grok-voice-think-fast-1.0: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 741/750 (98.8%).
- gpt-realtime-1.5: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 709/750 (94.5%).
- gemini-3.1-flash-live-preview: 24/25 complete; failure reasons {'no_audio_timeout': 1}; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 688/744 (92.5%).
- gemini-3.8-live-extended-thinking: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 701/750 (93.5%).

Historical reliability note: the [August investigation](../async-gemini-live-comparison-2026-08-04/root-cause-probes/README.md) reported 18/20 no-response-free Gemini 3.1 conversations and documented a WebSocket 1006 reconnect in an additional current attempt. Those historical observations are separate from the corrected cohort counts above.

## Provenance

See [protocol](PROTOCOL.md), [frozen settings](protocol.json), [aggregate with all attempts](aggregate.json), and [capture ledger](capture/attempts.jsonl). These results supersede the speech-to-speech reference table only. The legacy text prompt, shared benchmark data, and README text-model scores were preserved.
