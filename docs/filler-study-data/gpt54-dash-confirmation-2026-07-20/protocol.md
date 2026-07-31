# GPT-5.4 selected-dash prospective confirmation protocol

Frozen at 2026-07-20 17:23 PDT, before the first confirmatory API request.

## Claim and scope

The confirmatory claim is directional and configuration-specific: on the fixed
`aiwf_medium_context` 30-turn benchmark, GPT-5.4 with reasoning effort `none` has a
higher mean strict pass proportion when 96 space-separated trailing dashes are
appended to the current final user message than when no filler is added.

This is a prospective confirmation of the selected dash configuration and a
cross-pattern confirmation of the broader late-filler finding. It is not an exact
replication of the exploratory 96-dot comparison, and it does not establish
generalization to other scripts, models, reasoning levels, or deployments.

## Frozen configuration

- Model/API route: `gpt-5.4-2026-03-05`, OpenAI Responses API through service
  `openai`. This is the dated GPT-5.4 snapshot documented at freeze time, pinned to
  prevent alias drift during the run.
- Service tier: `priority`, as configured by the checked harness.
- Reasoning: `MTE_OPENAI_RESPONSES_REASONING_EFFORT=none`.
- Control: filler variables explicitly unset.
- Treatment: `MTE_FILLER_DOTS=96`, `MTE_FILLER_TOKEN=-`, and
  `MTE_FILLER_POSITION=suffix`. The implementation constructs 96 dashes separated by
  single spaces and applies them to a copy of the final user message; persisted
  history stays clean.
- Benchmark and prompt: the checked `aiwf_medium_context` files listed under
  Integrity below.
- Judge: the checked-in Claude judge using its fixed `claude-opus-4-5` route. The
  judge sees recorded, unmodified user text and therefore does not see the filler.
- OpenAI `*-pro` variants are prohibited by both protocol and the checked harness.

## Sample size and allocation

There are 41 fresh conversations per arm, 82 counted conversations total. All
exploratory runs are excluded. This retains the previously recommended conservative
90%-power target based on a +4.5 percentage-point planning effect, 6.2-point
conversation-level SD, and directional alpha 0.025. The study is deliberately not
resized downward using the larger, post-screen-selected dash pilot estimate.

Allocation is frozen in `schedule.tsv` as 41 two-conversation time blocks. Each block
contains one control and one dash conversation, with order determined before launch
by the low bit of `SHA256(seed + ":" + pair_number)`. Seed:

`a66476ee30461f6d79507f44fddff22bfeddde7049435d59b8d46b8927863e3a`

This gives exactly 41 assignments per arm and no run of more than two consecutive
assignments to the same arm. The order will not be changed in response to outcomes.
The frozen `schedule.tsv` SHA-256 is
`d851c4fd3906492118d775ddfecb7f2e95cd7963687cb153cfb7cec44429a624`.

## Outcome and analysis

- Experimental unit: one conversation.
- Primary outcome: strict pass proportion over a fixed denominator of 30 scripted
  turns. A turn passes only when every Boolean judge dimension passes. Missing or
  forfeited scripted turns are failures.
- Primary estimate: mean conversation pass proportion for dash minus control.
- Primary test: block-respecting paired, studentized sign-flip randomization test in
  the positive direction, alpha 0.025. A two-sided 95% paired-t confidence interval
  and the corresponding two-sided randomization p-value will also be reported.
- Secondary outcomes: strict completion (`end_session` at scripted turn 29),
  error-free-conversation rate, tool-call behavior, and per-conversation median TTFAT.
  These are descriptive and do not change the primary decision.

The checked-in `analyze.py` is frozen before launch and refuses to analyze unless all
82 scheduled slots are counted and judged.

## Failures, replacements, and stopping

- Model-caused aborts, malformed/empty replies recorded as model turns, and their
  forfeited future turns count as failures. They are never replaced.
- A scheduled slot is replaced only when the attempt records zero transcript rows
  and the log contains objective provider, transport, or harness failure evidence.
  Zero-row failures without such evidence stop the driver for manual arm-blind
  classification. Every attempt and replacement reason is retained.
- At most three objective infrastructure replacements are attempted for one slot in
  one driver invocation. Exhaustion stops the driver; resumption continues the same
  frozen schedule.
- Judge failure does not replace a model attempt. Judging is retried later against
  the same transcript.
- No interim outcome analysis, significance stopping, sample-size re-estimation, arm
  reallocation, or post-launch third arm is permitted. Operational progress and
  infrastructure logs may be monitored without aggregating scores.

## Integrity

Git HEAD at freeze: `3e9f805a86fb556a53724a1c83444d8d0de897d7`.

The runner checks these SHA-256 values before every model attempt:

| file | SHA-256 |
|---|---|
| `benchmarks/aiwf_medium_context/config.py` | `ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164` |
| `benchmarks/aiwf_medium_context/prompts/system.py` | `6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6` |
| `benchmarks/_shared/turns.py` | `c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b` |
| `src/multi_turn_eval/services/filler.py` | `d0d3ea02d69797c56e7d1395b752b04d132f003b3148b3d8e847f69067bf0d15` |
| `src/multi_turn_eval/services/openai_responses.py` | `863b58d390fefb84d237f4382039f89ad77af12ab70f006274925a32d8cdfb80` |
| `src/multi_turn_eval/pipelines/base.py` | `2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c` |
| `src/multi_turn_eval/judging/claude_judge.py` | `3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18` |

Runtime artifacts are append-only: `attempts.tsv`, `counted.tsv`, `manifest.tsv`,
per-attempt logs, judge logs, and the driver log. The resumable runner uses a lock to
prevent duplicate concurrent launches.
