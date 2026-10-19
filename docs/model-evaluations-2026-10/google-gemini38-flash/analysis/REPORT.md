# Gemini 3.8 Flash (low) — AIEWF medium context

- Cohort: 30 canonical conversations; 30 reached all 30 scripted turns.
- Strict pass rate: 848/900 (94.2%), conversation-cluster bootstrap 95% CI 93.7–94.9%.
- Errors: tool 0.0%, instruction 5.8%, KB 0.0%.
- Warm TTFAT: 1411ms P50, 1968ms P95, 7588ms max across 870 later scripted and recovery responses; scripted turn 0 is excluded.
- Warm raw TTFT: 1411ms P50, 1967ms P95. Median reasoning-to-answer gap: 0ms.
- Request policy: low thinking, no filler, provider-default sampling, explicit omission of the output-token cap. Software: Pipecat 1.12.0 and google-genai 2.31.0. All visible responses must have positive TTFAT and raw TTFT to pass analysis validation.
- Introductory-price estimate: $0.0091 per observed turn ($8.18 total), using $0.75/M uncached input, $0.075/M cached input, and $3.75/M output including thinking through 2026-12-31.

See [the frozen protocol](../PROTOCOL.md) for eligibility and [the canonical manifest (local archive)](../../README.md) for cohort membership. Source hashes are retained in [source-sha256.txt (local archive)](../../README.md). Smoke conversations and the initial captures with missing TTFAT are harness diagnostics and excluded from this cohort.

Comparison provenance: all 30 historical Gemini 3.7 Flash summaries identify `claude-opus-4-5`; this campaign uses `claude-opus-5` with judge-prompt SHA-256 `2ffdd0b3c5072e197b95cafba49dda5731e59eda7f6b73bfbdbc43e246e06f26`. The older summaries do not record a prompt hash. Their 100.0% accuracy and this campaign's 94.2% are not a controlled model comparison: a nearly identical attendance-day clarification passed the historical judge and failed the current one. Rejudge the historical transcripts with the current judge and rubric before attributing the accuracy difference to the model. Latency was collected on different dates and SDK versions as well.

[Evidence storage and local audit instructions](../../README.md).
