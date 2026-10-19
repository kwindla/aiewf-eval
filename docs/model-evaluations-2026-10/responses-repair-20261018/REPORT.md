# Responses adapter repair and prospective refresh

| Configuration | Complete / assigned | Fixed-900 pass | Completed-only pass | Completed TTFAT P50 | P95 | Effective tier |
|---|---|---|---|---|---|---|
| gpt-4.1-mini | 13/30 | 76.0% | 82.3% | 638ms | 1010ms | priority |
| gpt-4.1 | 26/30 | 94.1% | 94.6% | 743ms | 1086ms | priority |
| gpt-5.4 (low) | 30/30 | 96.9% | 96.9% | 932ms | 1458ms | priority |
| gpt-5.4 (none, +96 dots) | 30/30 | 96.2% | 96.2% | 813ms | 1234ms | priority |
| gpt-5.4 (none) | 30/30 | 91.7% | 91.7% | 810ms | 1257ms | priority |
| gpt-5.5 (none) | 29/30 | 96.8% | 97.2% | 780ms | 1173ms | priority |
| gpt-5.6-luna (none) | 27/30 | 82.0% | 84.7% | 708ms | 1198ms | priority |
| gpt-5.6-sol (none) | 28/30 | 91.7% | 96.1% | 765ms | 1298ms | priority |
| gpt-5.6-terra (medium) | 26/30 | 99.0% | 99.9% | 1004ms | 1569ms | priority |
| gpt-5.6-terra (none) | 26/30 | 85.4% | 90.8% | 826ms | 1257ms | priority |
| gpt-6-luna (none) | 0/30 | 25.0% | — | — | — | fast |
| gpt-6-luna (low) | 29/30 | 91.2% | 91.4% | 917ms | 1439ms | fast |

Terminal unanswered requests canceled at the registered idle timeout remain incomplete assigned conversations; they are not retried toward completion. Their earlier native histories remain valid.

The fixed denominator retains early endings and counts missing future turns as failures. The README reports conditional accuracy and latency among complete conversations; this report also records completion reliability. Each row received 30 fixed assignments, without topping up incomplete conversations. Latency excludes scripted turn zero and includes recovery responses, matching the existing analysis. Conversation-cluster bootstrap intervals and all component counts are in [aggregate.json](summary.json).

The adapter now preserves native phases, output boundaries, empty final answers and reasoning. Ordinary requests link previous_response_id; the dots arm replays native outputs so historical user messages remain filler-free. A request-boundary cursor preserves every parallel tool result. Explicit incomplete/error streams do not execute pending tools. This repair does not guarantee completion: the excluded none smoke still ended early, and exact native replay sometimes reproduced the early ending. A retrieval audit confirmed that the failed cloud request retained every prior native output, including commentary and the empty final-answer boundary. See [protocol](PROTOCOL.md), [smokes (local archive)](../README.md), and [excluded native/cloud diagnostic (local archive)](../README.md).

Operational integrity: 1 assignments were retried after positive evidence of provider stream interruption. Original raw captures, original portable canonical manifests and original COLLECTED manifests remain unchanged in the audit archive. Replacements use the identical frozen runtime/settings and retain early model endings. See [amendment (local archive)](../README.md) and [supersession ledger (local archive)](../README.md).

Public-text fidelity: 34 ending records had public commentary omitted by the existing cancellation recorder. 1 belong to complete conversations (the valid Terra medium closing response); the others are early endings. A derived judge view restores only output_text from stored responses whose public message and native function-call hashes exactly match the capture-time audit. Persisted-response retrieval omits encrypted reasoning; any such private-item hash remains explicitly unverified in the manifest. Originals are unchanged; private reasoning is excluded. See [restoration policy](PUBLIC-TEXT-AMENDMENT.md), [manifest (local archive)](../README.md), and [Luna review (local archive)](../README.md).

After generation, a defensive adapter guard rejects changed contexts after an unfinished request; identical-context retries remain allowed. It also rejects non-completed nonstreaming results before history commit and requests encrypted reasoning for unstored replay. All accepted captures used store=true, and this hardening does not alter their completed-response request paths; exact generation bytes remain in the runtime snapshot. See [post-capture hardening (local archive)](../README.md).

Twelve configurations were conservatively refreshed based on actual historical adapter logs, not model names. Luna history loss was directly demonstrated; the other ten rows used the same adapter but their historical score impact is not established. These fresh cohorts are independent, and all now explicitly request Priority; some older tiers were mixed or unlogged. Therefore changes in scores or latency are not a controlled estimate of the adapter fix alone. Older raw captures and grades remain available.

| Configuration | Observed scripted turns | Missing future turns | Observed timing P50 | P95 | N |
|---|---|---|---|---|---|
| gpt-4.1-mini | 864 | 36 | 627ms | 1001ms | 950 |
| gpt-4.1 | 896 | 4 | 750ms | 1076ms | 897 |
| gpt-5.4 (low) | 900 | 0 | 932ms | 1458ms | 886 |
| gpt-5.4 (none, +96 dots) | 900 | 0 | 813ms | 1234ms | 897 |
| gpt-5.4 (none) | 900 | 0 | 810ms | 1257ms | 929 |
| gpt-5.5 (none) | 897 | 3 | 779ms | 1163ms | 884 |
| gpt-5.6-luna (none) | 873 | 27 | 707ms | 1197ms | 938 |
| gpt-5.6-sol (none) | 860 | 40 | 763ms | 1299ms | 857 |
| gpt-5.6-terra (medium) | 896 | 4 | 1006ms | 1581ms | 866 |
| gpt-5.6-terra (none) | 850 | 50 | 825ms | 1238ms | 883 |
| gpt-6-luna (none) | 283 | 617 | 604ms | 1031ms | 254 |
| gpt-6-luna (low) | 898 | 2 | 918ms | 1423ms | 906 |

All 360 grades use Opus 5.5 Medium and the unchanged text rubric. Successful judge cost: $59.708960. Anthropic credit cooldowns: 0; each waits at least 90 seconds before retrying. Private reasoning is excluded from grading evidence.

[Frozen judge inputs (local archive)](../README.md) · [Generation sources (local archive)](../README.md) · [Source verification (local archive)](../README.md) · [Impact inventory (local archive)](../README.md)

[Evidence storage and local audit instructions](../README.md).
