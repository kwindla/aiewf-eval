# BaseTen Kimi K2.6 thinking comparison

Both arms use 30 complete, valid conversations and the same fixed 900 scripted
turn denominator. Recovery turns are excluded from accuracy and TTFAT.

| Measure | Thinking on | Thinking off |
|---|---:|---:|
| Strict pass rate | 98.3% | 93.9% |
| Conversation-cluster bootstrap 95% CI | 97.7–99.0% | 92.3–95.3% |
| Any error | 1.7% | 6.1% |
| Tool error | 1.4% | 6.0% |
| Instruction error | 1.7% | 3.9% |
| KB error | 0.0% | 0.0% |
| TTFAT P50 | 1596 ms | 480 ms |
| TTFAT P95 | 5252 ms | 854 ms |
| TTFAT max | 13622 ms | 4458 ms |
| `end_session` on scripted turn | 30/30 | 9/30 |
| `end_session` on recovery turn | 0/30 | 16/30 |
| Missing `end_session` | 0/30 | 5/30 |
| Complete-conversation yield | 100.0% | 73.2% |

Thinking on minus off is +4.4 percentage points in strict pass rate;
the independent conversation-cluster bootstrap 95% interval for that difference
is +2.9 to +6.1 points. Median TTFAT is
3.33× the thinking-off value.

The thinking-on timing is content/tool TTFAT, not time to the first reasoning
token. Its raw first-chunk P50 is 666 ms and its
median measured reasoning delay is 854 ms;
900/900 scripted
rows report positive thinking tokens.

Important protocol caveat: this is a reproduction of the two vendor-mode
signatures, not a pure one-setting causal experiment. Thinking on uses
temperature 1.0 and top-p 0.95; thinking off used temperature 0.6 and the
provider-default top-p. Thinking on explicitly sends
`chat_template_args.enable_thinking=true`; thinking off omitted that argument,
leaving BaseTen's default off behavior. Although the off request transmitted
`reasoning_effort=none`, Kimi K2.6 is not a BaseTen `reasoning_effort` model and
that field is ignored. The off cohort's zero reported thinking tokens on all
900 scripted turns confirms its effective state.

The completion-yield row is operational campaign provenance, not a clean model
reliability effect. The earlier thinking-off campaign's 41 recorded attempts
include its initial concurrency-2 429 failures, provider stream/502 failures,
and two out-of-cohort duplicate/interrupted attempts; the thinking-on campaign
started with the stabilized serial 30-second-cooldown protocol.
