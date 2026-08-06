# Frozen protocol

## Primary comparison

The campaign contains 30 `reasoning_effort=none` and 30
`reasoning_effort=low` conversations. Thirty temporal pairs are executed
strictly sequentially in the order recorded by `frozen-order.tsv`. The pair
order is balanced within five consecutive six-pair blocks.

Every arm estimate uses a fixed denominator of 900 scheduled turns. Missing
future turns after a model-caused early exit count as failures. Latency and token
summaries use observed model responses only. Arm uncertainty is bootstrapped by
whole conversation; the `low − none` comparison resamples temporal pairs.

Both complete arm rows may be published in the README. The `none` arm is the
prespecified primary voice and filler configuration because it is directly
comparable to the existing full-Inkling `none` row. `low` remains a parallel,
fully powered comparison rather than a model-selection pilot.

## Eligibility and stopping

Any attempt with at least one valid model response is canonical, including an
early `end_session`, a recovery-driven termination, a later malformed response,
or a later idle timeout. Only an objective provider or transport failure with
zero valid responses may be replaced. Four failed zero-response attempts at one
slot constitute an operational blocker.

Accuracy, completion, reasoning frequency, and latency are outcomes and cannot
stop the campaign. The campaign may stop early only for persistent API or
protocol incompatibility that prevents collecting valid responses.

## Follow-up filler phase

After the primary campaign is judged, test the `none` configuration with 96
space-separated dots appended to the current user message. Start with six dot
conversations. Extend to ten if the absolute fixed-denominator difference from
the frozen `none` control is at least two percentage points or strict-completion
rates differ. At ten, extend the dot arm to 30 if any of the following holds:

- the whole-conversation bootstrap interval excludes zero;
- the absolute effect is at least three points and an aggregate-aligned
  same-turn failure direction recurs in at least three conversations;
- strict-completion rates still differ.

The historical full-Inkling dot effect is contextual evidence only and does not
alter these stopping rules.
