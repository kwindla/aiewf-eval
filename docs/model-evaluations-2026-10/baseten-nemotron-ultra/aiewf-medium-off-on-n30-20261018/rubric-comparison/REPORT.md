# Same Ultra transcripts, two rubric packages

The same 30 fresh Baseten thinking-on conversations were judged twice with
Opus 5.5 Medium. No new Ultra conversations were generated. The configured
difference is the system rubric: the legacy v4 rubric from git `5dfcd96` versus
the current policy-grounded rubric, which also supplies the full assistant policy
and knowledge base. The formatted conversation prompt and transcript hashes match
exactly for every pair; tools and project settings are disabled in both arms.

| Rubric | Strict passes | Pass rate |
|---|---:|---:|
| Legacy v4 | 869/900 | 96.6% |
| Current policy + full KB | 827/900 | 91.9% |

Legacy minus current: 4.7 percentage
points; 95% paired-conversation bootstrap interval
3.2–6.2
points (20,000 resamples). Legacy passes/current failures:
58; legacy failures/current passes:
16.

This tests rubric/reference sensitivity on identical fresh outputs. One grade
per rubric/input means individual differences also include judge variability.
The old near-perfect published Ultra rows used a different cohort, serving/budget
configuration, and historical judge version. This is not a reconstruction of
that exact historical experiment or proof of a model regression. The reference
KB/policy addition is part of the rubric package; this does not separate its
effect from the wording changes.

Only first valid grades are retained; credit failures wait at least 90 seconds.
Original current-rubric grades remain unchanged. Legacy SDK-reported cost:
**$5.95**. See [frozen inputs (local archive)](../../../README.md), [paired-result summary](summary.json),
and [legacy grades (local archive)](../../../README.md).

[Evidence storage and local audit instructions](../../../README.md).
