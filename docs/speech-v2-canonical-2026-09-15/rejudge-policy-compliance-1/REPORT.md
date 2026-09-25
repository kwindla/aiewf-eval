# Canonical speech-to-speech benchmark: voice policy v2

| Model | Complete | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grok-voice-think-fast-1.0 | 25/25 | 97.7% | 742/750 | 734/750 | 750/750 | 730/750 | 2496ms | 5984ms | 2689ms | 88ms |
| gpt-realtime-2.1 (low) | 25/25 | 94.4% | 733/750 | 712/750 | 731/750 | 750/750 | 1568ms | 9568ms | 1499ms | 75ms |
| gpt-realtime-1.5 | 25/25 | 94.0% | 739/750 | 708/750 | 725/750 | 737/750 | 1120ms | 3424ms | 1890ms | 160ms |
| gemini-3.8-live-extended-thinking (low) | 25/25 | 92.9% | 747/750 | 697/750 | 730/750 | 748/750 | 1440ms | 8896ms | 1504ms | 60ms |
| gemini-3.1-flash-live-preview (minimal)* | 24/25 | 86.5% | 710/750 | 656/750 | 723/750 | 744/750 | 1328ms | 3936ms | 1646ms | 98ms |

25 attempted conversations per model; every row uses the fixed 750-turn denominator. Missing future turns count as failures. Strict pass is the conjunction of tool use, instruction following, and KB grounding. Audio turn-taking is reported separately. V2V is conditional on measurable responses and includes the first scripted response, matching the historical speech table convention.

## Uncertainty and latency coverage

- gpt-realtime-2.1: strict 95% conversation-bootstrap interval 93.5–95.3%; V2V observations 615 non-tool / 135 tool.
- grok-voice-think-fast-1.0: strict 95% conversation-bootstrap interval 96.8–98.5%; V2V observations 592 non-tool / 138 tool.
- gpt-realtime-1.5: strict 95% conversation-bootstrap interval 92.9–95.1%; V2V observations 603 non-tool / 122 tool.
- gemini-3.1-flash-live-preview: strict 95% conversation-bootstrap interval 83.1–89.7%; V2V observations 588 non-tool / 126 tool.
- gemini-3.8-live-extended-thinking: strict 95% conversation-bootstrap interval 91.1–94.7%; V2V observations 598 non-tool / 143 tool.

## Reliability and observed-turn accuracy

* marks a model with incomplete conversations. Failures in this corrected cohort are counted for every model under the same policy. The interrupted pre-fix cohort is excluded in full; its tool-delivery timeouts were harness failures, not API errors.

- gpt-realtime-2.1: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 708/750 (94.4%).
- grok-voice-think-fast-1.0: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 733/750 (97.7%).
- gpt-realtime-1.5: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 705/750 (94.0%).
- gemini-3.1-flash-live-preview: 24/25 complete; failure reasons {'no_audio_timeout': 1}; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 649/744 (87.2%).
- gemini-3.8-live-extended-thinking: 25/25 complete; failure reasons none; API-error evidence in 0 attempts; quota-error evidence in 0 attempts. Observed-turn strict accuracy: 697/750 (92.9%).

Historical reliability note: the [August investigation](../../async-gemini-live-comparison-2026-08-04/root-cause-probes/README.md) reported 18/20 no-response-free Gemini 3.1 conversations and documented a WebSocket 1006 reconnect in an additional current attempt. Those historical observations are separate from the corrected cohort counts above.

## Provenance

See [protocol](../PROTOCOL.md), [frozen settings](../protocol.json), [aggregate with all attempts](aggregate.json), and [capture ledger](../capture/attempts.jsonl). These results supersede the speech-to-speech reference table only. The legacy text prompt, shared benchmark data, and README text-model scores were preserved.

## Revised judging

All 131 saved conversations were rejudged uniformly with Opus 5 using `speech-v2-policy-compliance-1`. Model prompts, source judgments, recordings, audio metrics, and text scoring are unchanged. See [revision protocol](protocol.json), [before/after audit](AUDIT.md), and [original first-pass scores](../aggregate.json).

### Interpretation and remaining judge limitations

The audit flags three explicit score/explanation contradictions: two Gemini 3.1
turns and one GPT-Realtime-1.5 turn. The published numbers retain the Boolean
judge outputs without selective retries or manual overrides. Following those
explanations instead would give 86.3% and 93.9%, respectively, without changing the
ordering. Qualified subsection counts also remain a judgment boundary; the audit
exposes every answer rather than claiming perfect semantic consistency.

Gemini 3.1's larger decline is not just the MCP question: 11 newly failed turns
recommended sessions before establishing attendance details, and eight newly
failed turns asked again for an already-known name. These are existing model-policy
requirements. The broader scoring changes and normal judge variability are visible
in the full before/after audit; they should not be attributed solely to counting.
