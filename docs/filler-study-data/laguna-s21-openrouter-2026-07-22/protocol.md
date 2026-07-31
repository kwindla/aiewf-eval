# Laguna S 2.1 OpenRouter no-thinking and 96-dot screen

Protocol frozen on 2026-07-22 before any full-conversation campaign request.
The excluded instrumentation runs are:

- `runs/aiwf_medium_context/20260722T152435_poolside_laguna-s-2.1_bd0df069`
  (one-turn connectivity check);
- `runs/aiwf_medium_context/20260722T152455_poolside_laguna-s-2.1_2bf6f92a`
  (five selected scripted turns plus the closing tool call).

## Objective and configuration

Screen `poolside/laguna-s-2.1` on `aiwf_medium_context` through OpenRouter's
paid, Poolside-hosted BF16 endpoint under two contemporaneous conditions:

1. no filler;
2. 96 space-separated dots appended to the current user turn.

Both arms explicitly disable the model's default max-thinking mode with
`MTE_OPENROUTER_REASONING_OFF=1`, which sends
`reasoning: {"enabled": false}`. They set `max_tokens=8192`, retain Poolside's
default sampling, enable recovery and tool-call deduplication, do not request
an extra LLM response after tool results, and use a 45-second text idle
timeout. TTFAT is the first content or tool-call output; reasoning-only chunks
do not stop it.

The free FP8 route is excluded. It differs in quantization, context length,
rate limits, reliability, and data-use terms, and therefore is not pooled with
the paid BF16 route.

## Fixed stage 1

- Ten fresh no-filler conversation assignments.
- Six fresh 96-dot conversation assignments.
- The order in `schedule.tsv` is frozen and interleaves the arms.
- Unit: one assigned 30-turn conversation attempt.
- Model-caused early exits, malformed outputs, missing turns, and missing
  terminal `end_session` remain outcomes on the fixed 30-turn denominator.
- Only explicit provider/transport failures are replacement-eligible, with at
  most four attempts per assignment.
- The repository Claude judge scores every observed non-recovery turn. Judge
  retries reuse the immutable transcript and never replace a model attempt.

## Prospective adaptive rule

At dots n=6, extend dots to n=10 if either the absolute fixed-denominator
pass-rate difference is at least 2.0 points or strict-completion rates differ.

At dots n=10, stop unless one of these holds:

- the whole-conversation bootstrap 95% interval excludes zero;
- the absolute difference is at least 3.0 points and an aggregate-aligned
  same-turn failure direction recurs in at least three conversations;
- strict-completion rates still differ.

If triggered, extend both arms to 30 using a separately frozen schedule.
Thresholds determine sample size only; all reached-stage estimates remain
visible. The standard no-filler README row may later be extended to 30 for
precision without reopening a stopped filler arm.

## Publication rule

Do not update README or Section 3 until all scheduled runs are judged and a
fixed-30-turn-denominator aggregate is verified. The README Provider value is
`OpenRouter`; provenance must identify Poolside/BF16 and explicit
reasoning-off. A 10/6 or 30/6 filler contrast remains exploratory and receives
no focused n=30 effect whisker.

## Frozen stage-1 artifacts

| file | SHA-256 |
|---|---|
| `schedule.tsv` | `ece7b3e83708f018627c78343c74db97642683f1adc77a4d77526ce80970886e` |
| `run_campaign.py` | `34d2e516a5b23e9da27235ab74ac69cbdf2929419ed98c54fe5248361c2a3583` |
| `tests/test_openrouter_routing.py` | `7a650fd76a27435f7299e92a996abae97280c1e94938fc60a6698e4c681f6981` |

## Pre-analysis classifier amendment

After LS21-01 attempt 1 completed, the driver incorrectly classified its
missing `end_session` as infrastructure because the broad bare-429 regular
expression matched the ordinary processing-time string `0.429s` in
`run.log`. The driver was stopped during the automatically launched retry.
Before any judgment or aggregate was inspected, the classifier was narrowed
to explicit HTTP/status-429 forms and to the concise CLI output, which does
not embed the full prompt or debug timing lines.

LS21-01 attempt 1 remains the assigned model outcome as
`incomplete_no_end_session`. The operator-interrupted retry at
`runs/aiwf_medium_context/20260722T153054_poolside_laguna-s-2.1_d224bd59`
is excluded and never enters the manifest. No schedule, replacement rule, or
adaptive threshold changed.

Amended `run_campaign.py` SHA-256:
`cece63ecc18e15316ea3ed95d6b7418d18d05e8b625549e2e5be1fd3eef056d6`.

## Stage-2 dots top-up decision

After all 16 stage-1 transcripts were fixed but before accuracy aggregates
were available, strict completion was 0/10 for no filler and 1/6 for dots.
The prespecified unequal-completion trigger therefore fired. Four additional
dot assignments were frozen in `schedule-dots-topup.tsv`, bringing the dots
arm to n=10. The no-filler arm remains at n=10 at this decision point.

| stage-2 file | SHA-256 |
|---|---|
| `schedule-dots-topup.tsv` | `6521d0be0ab91bc3f64a631b4635e17de2e38dcfcec536cccb1a50aab0da6491` |
| `run_dots_topup.py` | `964dbd4166fc54c4d6807f2ec1a5da3c49a7e068102c932e0f1443b0512c2220` |
| amended `run_campaign.py` | `2a40a63624d92c11cf730a8b53d6356eeda780d71faea5426a51c59590bdab4b` |

## Stage-2 n=10 outcome and final expansion

All 20 stage-2 conversations were judged on a fixed 300-turn denominator per
arm. No filler scored 91.0%; dots scored 85.3%, a -5.7-point difference with
a whole-conversation bootstrap 95% interval of -9.3 to -2.0 points. The
aggregate-aligned harm direction recurred on turns 11, 15, 17, 19, and 24.
Strict completion also remained different: 0/10 for no filler and 2/10 for
dots.

All three n=10 promotion conditions fired: the interval excluded zero, the
absolute effect exceeded three points with recurring same-turn direction, and
strict-completion rates still differed. The frozen action is therefore to
extend both arms to 30 conversations. These n=10 values are intermediate and
must not be published as the final README/report estimates.

| n=10 artifact | SHA-256 |
|---|---|
| `aggregates-n10.json` | `465b5096085848203c58bece6a8d66f5f7fdcaa39701652c17c8052925679c8a` |
| `judge_campaign.py` (n=10) | `40d6548d164c3af8a733cb99a2a1e6ebb6089828af8a56dbb3dd1ecc337b186b` |
| `analyze.py` (n=10) | `8ec7faf910bc808e6a0b5037608f706756d27ef00dd73da83f5af6cd6364e8d6` |

## Frozen n=30 expansion

The final expansion adds 20 no-filler and 20 dot assignments, alternating in
one lane as `LS21-N11`/`LS21-D11` through `LS21-N30`/`LS21-D30`. Combined
with the completed n=10 pool, the final target is 30 conversations and 900
fixed scripted turns per arm. The assignment order and driver were frozen
before any expansion request.

| n=30 expansion file | SHA-256 |
|---|---|
| `schedule-n30-topup.tsv` | `7ea9b6e3dfc53d104aca9d91eafdb4487623a8862a62c2aca3ae78b836d259e7` |
| `run_n30_topup.py` | `24221347dd4ee6ce711dbcc354eaf3137b9ff3ea59e62f416d569cc19ff6ffd9` |

## Final n=30 outcome

All 60 assigned conversations were retained and judged. On the fixed
900-turn denominator per arm, no filler scored 85.6% and +96 dots scored
83.3%, a -2.2-point estimate with a whole-conversation bootstrap 95%
interval of -8.3 to +5.1 points. The interval spans zero, so the final result
does not establish a filler effect for this model and route. Strict completion
was 4/30 in each arm.

The no-filler row used for the public benchmark has 295ms median TTFAT, 620ms
P95, and a 21,032ms maximum. All included transcripts report zero thinking
tokens and use the paid Poolside-hosted BF16 route through OpenRouter with
`reasoning.enabled=false`.

The final aggregate is `aggregates.json` (SHA-256
`5bcfaf8b9d534a44cfc84674e21f9e8c3c9a23eda617c654e629506561002505`).
