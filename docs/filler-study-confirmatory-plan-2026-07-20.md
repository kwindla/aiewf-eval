# Filler-token confirmatory replication plan — 2026-07-20

## Decision

The 18-model study remains an exploratory screen with unadjusted, model-specific
estimates. It will not use a global Bonferroni comparison as a headline result.

Run one confirmatory study first: **GPT-5.4, no filler versus 96 trailing dashes,
41 fresh conversations per arm (82 total)**. This prospectively validates the
selected operational configuration and provides cross-pattern confirmation of the
broader late-filler result; it is not an exact replication of the primary dot cell.
Existing runs remain pilot evidence and do not count toward the confirmatory sample.
Do not add a dot arm or otherwise change the design after results begin arriving.

## Why 41 per arm

The experimental unit is a complete 30-turn conversation, not an individual turn.
The primary dot pilot effect was +6.0 percentage points. The selected dash cell had
a larger exploratory estimate, but the study is deliberately not resized downward
from that selected result. Planning assumes:

- a +4.5-point effect (75% of the selected pilot estimate);
- a 6.2-point conversation-level SD, which accounts for uncertainty in the small
  pilot variance rather than plugging in its 4.3-point estimate;
- a prespecified positive direction and one-sided alpha of 0.025;
- fixed equal allocation and no interim significance checks.

Under a noncentral two-sample t approximation, 31 conversations per arm gives 80%
power and 41 per arm gives 90% power. The checked-in calculator is
`docs/filler-study-data/replication_power.py`.

## Frozen design

- Model: dated snapshot `gpt-5.4-2026-03-05` through the same Responses API route
  used in the pilot; never an OpenAI `*-pro` model. Pinning prevents alias drift.
- Reasoning: `none`.
- Arms: no filler and exactly 96 space-separated trailing dashes appended after the final user
  message; history remains clean.
- Allocation: randomize in short balanced time blocks or pairs. Do not alternate
  deterministically.
- Sample: 41 valid conversations per arm, fixed before the first request.
- Primary outcome: each conversation's strict pass proportion over a fixed
  30-turn denominator.
- Model-caused aborts: count the aborting turn and every forfeited future turn as
  failures. Never discard them as missing.
- Infrastructure failures: replace only failures for which no valid model response
  was captured, using a rule frozen before launch. Preserve every attempt and the
  replacement reason.
- Primary analysis: difference in mean conversation pass proportion, with a
  studentized whole-conversation randomization test or cluster-level Welch test and
  a confidence interval.
- Secondary outcomes: strict completion, error-free-conversation rate, TTFAT, and
  per-turn differences. Per-turn analyses remain descriptive.
- Stopping: no peeking, no stopping for significance, and no reallocating attempts
  toward the arm with fewer completions.

Report the effect estimate and confidence interval even if the primary p-value is
not significant.

## Other model claims

The table below gives fresh per-arm counts for standalone model-specific decisions,
using the same 75%-of-pilot effect, 6.2-point planning SD, and one-sided alpha 0.025.
These are not recommendations to run every cell.

| model-specific claim | 80% power | 90% power | recommendation |
|---|---:|---:|---|
| GPT-5.4 selected-dash benefit | 31 | **41** | run first; retain conservative dot-anchored sizing |
| GPT-5.5 benefit | 78 | 104 | retain as pilot unless the lineage claim justifies the cost |
| GPT-5.6-sol benefit | 98 | 131 | retain as pilot; zero pilot variance cannot justify a tiny n |
| GLM-5.2 harm | 59 | 78 | operationally avoid filler; replicate only for a formal harm claim |

The GLM design confirms directional harm; it is not a safety test. Demonstrating
that filler is no worse than control requires a separate noninferiority margin and
substantially larger samples.

## Separate mini completion trial

The effort-medium GPT-5.4-mini result used adaptive allocation and is screening
evidence only. If strict completion is a priority, run a separate fixed trial with
**24 fresh conversations per arm** (no filler versus 96 dashes). With conservative
planning rates of 25% versus 75% strict completion, 23 per arm gives about 90% power
under a two-sided Fisher exact test; 24 provides a balanced operational target.
Strict completion at scripted turn 29 is the sole primary endpoint. Accuracy among
completers remains secondary and survivor-selected.

## Scope of the confirmation

Repeating the same benchmark conversation estimates stochastic performance on this
one scripted workload. It does not establish generalization across different
conversation structures. A broader claim requires several new benchmark scripts and
an analysis that treats script as a higher-level cluster; that study needs its own
pilot variance before it can be sized responsibly.
