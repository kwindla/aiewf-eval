# Prospective protocol: Inkling Small +96 dots

Frozen on 2026-07-31 before any scored dot-treated request.

## Objective

Estimate the effect of appending 96 space-separated dots to the current final
user message for `thinkingmachines/inkling-small` at
`reasoning_effort=none` on `aiwf_medium_context`.

The comparison is intentionally additive. Its control is exactly the 30
none-arm conversations from
`ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/`. The control is
not rerun, topped up, or sampled from the primary low arm. Before the first dot
request, all 30 control run paths and transcript hashes are frozen locally.

## Fixed request configuration

- Provider: BaseTen shared Model API, `https://inference.baseten.co/v1`.
- Model: `thinkingmachines/inkling-small`.
- Thinking: top-level `reasoning_effort=none`; no vLLM thinking-template knob.
- Sampling: temperature 1.0 and maximum completion tokens 16,384.
- Treatment: `MTE_FILLER_DOTS=96`, `MTE_FILLER_TOKEN=.`, and
  `MTE_FILLER_POSITION=suffix`.
- Filler is added only to a copy of the outgoing current user message; persisted
  transcripts and conversation history remain filler-free.
- Collection concurrency: one. Dot runs never overlap one another.
- Recovery, tool-call deduplication, tool-result handling, and the 45-second
  text idle timeout exactly match the primary campaign.

## Eligibility and denominator

The unit is one assigned 30-turn conversation. Any attempt with at least one
valid model response becomes canonical. Model-caused early termination,
malformed later output, missing future turns, and later idle timeouts remain
outcomes on the fixed 30-turn denominator. Only an objective provider or
transport failure with zero valid responses may be replaced. Each dot slot has
a durable ceiling of four attempts.

## Adaptive sample size

The dot schedule is frozen through 30 but exposed only in caps 6, 10, and 30.

At dots n=6, extend to n=10 if either:

- the absolute fixed-denominator pass-rate difference is at least 2.0 points;
- strict-completion rates differ.

At dots n=10, extend to n=30 if at least one condition holds:

- the whole-conversation bootstrap 95% interval excludes zero;
- the absolute difference is at least 3.0 points and an aggregate-aligned
  same-turn failure direction recurs in at least three conversations;
- strict-completion rates differ.

Each extension requires a completed-stage analysis artifact and an explicit
`extend` row in `stage-decisions.tsv`. The artifact's hash is frozen. These
thresholds govern sample size only; every reached-stage estimate remains
reportable. The 30-run reused control stays fixed at every stage.

## Analysis

Missing or forfeited future turns fail all displayed criteria. Confidence
intervals use 100,000 independent-arm whole-conversation bootstrap resamples.
The analysis reports strict pass, any/tool/instruction/KB errors, strict
completion, serving classifications, TTFAT, and per-turn error concentrations.
It automatically evaluates the reached stage's prespecified extension rule and
writes a recommendation, but it cannot write `stage-decisions.tsv` or launch
the next stage. Because treatment runs occur after the primary controls rather
than in a contemporaneously randomized pair, any observed filler contrast must
retain a deployment-time/provider-drift caveat.

Only canonical dot transcripts are judged by this bundle. Control judgments
must already be complete under the same pinned `claude-opus-4-5` /
`claude-agent-sdk-v4-turn-taking` identity. Judge retries reuse immutable
transcripts. The dots judge has a two-worker ceiling and its child environment
contains only `ANTHROPIC_API_KEY` among provider credentials; BaseTen variables
and dotenv auto-loading are disabled.
