# Qwen3.6 thinking-off 96-dot exploratory screen

Protocol frozen on 2026-07-28 before the first Qwen3.6-35B-A3B AIEWF
conversation and before either dot-treatment arm.

## Objective and scope

Test whether appending 96 space-separated literal periods to a request-local
copy of the current final user message materially changes performance on the
30-turn `aiwf_medium_context` benchmark for:

1. `Qwen/Qwen3.6-27B`, official BF16 checkpoint, BaseTen/vLLM 0.26;
2. `Qwen/Qwen3.6-35B-A3B-FP8`, official FP8 checkpoint,
   BaseTen/vLLM 0.26.

Both deployments use one H100, automatic prefix caching,
`mamba_cache_mode=align`, and two-token MTP. Native thinking is disabled.
Persisted conversation history remains filler-free. These comparisons are
exploratory because the reusable controls and later dot arms are not
contemporaneous or interleaved. They never enter the report's focused-effect
whisker set, even if a treatment arm reaches 30.

## Controls and commitments

- Reuse the 27B thinking-off N=30 arm from
  `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z`
  after its configuration, hashes, eligibility, and judgment coverage pass
  audit. Its decision subset was frozen before any judged outcome was
  inspected: source slots `2,3,6,7,9,11,13,15,18,19`.
- Collect 35B N=30 no-filler controls using `schedule-qwen35-control.tsv`.
  Its decision subset is control slots `C35-01` through `C35-10`, frozen
  before any 35B AIEWF outcome exists.
- Final descriptive rows use every valid N=30 control. Adaptive decisions use
  only the frozen first-ten subset.

## Fixed request and attempt contract

- Pipeline: text.
- Scheduled turns: 30.
- Sampling: temperature 0.6, top-p 0.95, maximum output 8,192 tokens.
- Reasoning: `MTE_VLLM_THINKING=0`; no native thinking budget.
- Recovery enabled; tool calls deduplicated; no extra model turn after tool
  results; 45-second text idle timeout.
- Treatment: `MTE_FILLER_DOTS=96`, `MTE_FILLER_TOKEN=.`,
  `MTE_FILLER_POSITION=suffix`.
- Experimental unit: one assigned conversation. A model-caused early exit,
  malformed response after a valid response, missing terminal call, or
  harness idle timeout after a valid response remains an outcome.
- Replace only objective provider/transport failure with no valid model
  response. Preserve every attempt and cap at four attempts per assignment.
- Judge each observed non-recovery scripted turn from the immutable
  transcript with the repository Claude judge. A judge failure retries only
  the judgment.
- Scoring uses a fixed 30-turn denominator per conversation. Missing and
  forfeited future turns fail every displayed dimension.

## Prospective adaptive treatment rule

The complete dot schedules are frozen in assignment order, but only reached
stages may execute.

Stage 1 runs dot assignments 1–6 per model and compares them with that model's
frozen ten-control decision subset.

- Stop at six if strict-completion rates are identical and the absolute
  fixed-denominator pass-rate difference is below 2.0 points.
- Otherwise run dot assignments 7–10.

At ten dot assignments, promote dots to 30 if any condition holds:

1. the whole-conversation bootstrap 95% interval excludes zero;
2. the absolute effect is at least 3.0 points and an
   aggregate-direction-aligned same-turn failure direction recurs in at
   least three conversations;
3. strict-completion rates differ.

Otherwise stop at ten. Promotion adds assignments 11–30; the N=30 control is
not recollected. Thresholds choose sample size only. Every reached-stage
estimate and the unequal sample sizes remain visible.

## Reporting

Report fixed-denominator pass rate, whole-conversation interval where
applicable, strict completion, error dimensions, and no-filler P50 TTFAT.
Disclose checkpoint precision, BaseTen execution route, reused controls,
non-interleaved timing, and stopped-stage N. Both Markdown and product HTML
must be regenerated regardless of stopping stage.
