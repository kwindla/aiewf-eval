# Recent-model benchmark results

| Configuration | Complete / assigned | Fixed-denominator pass | Completed-only pass | Completed TTFAT P50 | P95 |
|---|---|---|---|---|---|
| gpt-6-luna (none) | 0/30 | 24.1% | — | — | — |
| gpt-6-luna (low) | 0/30 | 80.9% | — | — | — |
| claude-sonnet-5-5 (no up-front thinking, low) | 30/30 | 99.9% | 99.9% | 1180ms | 2195ms |
| claude-sonnet-5-5 (low) | 30/30 | 99.9% | 99.9% | 838ms | 2140ms |
| claude-haiku-5-5 (thinking off, low) | 29/30 | 90.4% | 91.1% | 556ms | 819ms |
| claude-haiku-5-5 (low) | 30/30 | 96.9% | 96.9% | 651ms | 1563ms |
| muse-spark-1.3 (minimal) | 30/30 | 99.4% | 99.4% | 1752ms | 5638ms |
| muse-spark-1.3 (low) | 30/30 | 100.0% | 100.0% | 2965ms | 9143ms |
| muse-spark-open (low, NVFP4) | 30/30 | 95.1% | 95.1% | 626ms | 1299ms |

The fixed-denominator column retains early-ending conversations and counts missing future turns as failures. Completed-only quality and latency match the current README convention. Smoke runs are excluded; all attempts and original captures remain available. See [protocol](PROTOCOL.md) and [aggregate](summary.json).

The [mode settings](MODE-SETTINGS.md) document the explicit controls and the Sonnet 5.5 SDK-default caveat.

Capture-time sources are verified against their unchanged manifests. When current library code evolves after capture, the original bytes must match the pre-capture runtime snapshot; see [source verification (local archive)](../README.md).

Observed timing across all assigned conversations, including incomplete prefixes. These timings use the observed responses only; they do not imply the missing future turns were answered. As in the main table, scripted turn zero is excluded and later recovery responses are included.

| Configuration | Observed scripted turns | Missing future turns | Observed TTFAT P50 | P95 | Timing N |
|---|---|---|---|---|---|
| gpt-6-luna (none) | 273 | 627 | 550ms | 940ms | 243 |
| gpt-6-luna (low) | 834 | 66 | 831ms | 1300ms | 847 |
| claude-sonnet-5-5 (no up-front thinking, low) | 900 | 0 | 1180ms | 2195ms | 870 |
| claude-sonnet-5-5 (low) | 900 | 0 | 838ms | 2140ms | 870 |
| claude-haiku-5-5 (thinking off, low) | 896 | 4 | 556ms | 814ms | 900 |
| claude-haiku-5-5 (low) | 900 | 0 | 651ms | 1563ms | 878 |
| muse-spark-1.3 (minimal) | 900 | 0 | 1752ms | 5638ms | 882 |
| muse-spark-1.3 (low) | 900 | 0 | 2965ms | 9143ms | 871 |
| muse-spark-open (low, NVFP4) | 900 | 0 | 626ms | 1299ms | 905 |

The [Luna completion audit (local archive)](../README.md) inventories every premature end_session and the excluded streaming check. No tool-name translation defect was found in that diagnostic.

Grok Voice 3.0 off: 25 complete conversations; 1/26 unsuccessful attempts; 89.7% strict pass. First-speech P50 2048ms, P95 2624ms.

See the [incomplete-session review (local archive)](../README.md) and [all voice attempt records (local archive)](../README.md) for operational failures excluded from the completed-conversation quality cohort.

All grades use Opus 5.5 Medium, the existing text or speech rubric, and unchanged task prompts. Successful judge request cost: $50.181265. Provider-blocked cases: 0.

Credit reload retries: 0 during generation; 0 judging cooldown events. Every credit retry waits at least 90 seconds. Generation credit waits remain included in affected latency observations.

[Evidence storage and local audit instructions](../README.md).
