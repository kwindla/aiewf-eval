# Gemma 4 26B A4B contemporaneous 96-dot experiment

Protocol frozen on 2026-07-31 before any request in this experiment. The
earlier 30-run no-filler campaign under the parent directory remains immutable
provenance and is not used as this experiment's causal control.

## Objective

Estimate the effect of appending 96 space-separated literal periods to the
request-local final user message for `google/gemma-4-26B-A4B-it`. Persisted
conversation history and recorded user text remain filler-free.

The experiment collects fresh contemporaneous `nofiller` and `dots96` arms on
the exact earlier dedicated BaseTen deployment:

- endpoint: `https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1`
- vLLM: `0.26.1rc1.dev77+g6f91edf96`
- automatic prefix caching enabled
- one-token `google/gemma-4-26B-A4B-it-assistant` MTP
- Gemma thinking disabled
- temperature 1.0, top-p 0.95, top-k 64, maximum output 8,192

## Frozen allocation

`frozen-order.tsv` contains 30 temporal pairs. Each pair has one control and
one treatment request, in randomized order. Pairs 1–10 are the initial stage;
pairs 11–30 may run only after a reviewed promotion decision. Each of the
three ten-pair blocks has five control-first and five treatment-first pairs.
All model requests run strictly sequentially on the one-replica deployment.

## Eligibility and replacement

The experimental unit is one assigned full-conversation attempt. Once an
attempt records at least one valid model response, it is canonical. Premature
terminal calls, later malformed responses, later idle timeouts, and missing
future turns are model outcomes and are never replaced. Only objective
provider or transport failures with zero valid model responses are replaced.
Every attempt remains in `attempts.tsv`; replacement stops after four attempts
for a slot.

## Adaptive rule

After 10 canonical conversations per arm have been judged on the fixed
300-turn denominator, promote both arms to 30 if any condition holds:

1. the paired whole-conversation bootstrap 95% interval excludes zero;
2. the absolute pass-rate effect is at least 3.0 points and an
   aggregate-direction-aligned same-turn failure recurs in at least three
   conversations; or
3. strict-completion rates differ.

The thresholds choose sample size only. A continuation requires a durable
promotion decision naming at least one trigger and hashing the exact aggregate
and included-run artifacts. No unplanned arm reallocation or optional stopping
is allowed.

## Resource lifecycle

Each live invocation first sets the dedicated deployment to `min_replica=1`,
`max_replica=1` and waits for an active replica. Collection is strictly
sequential. A `finally` block always requests `min_replica=0`, `max_replica=1`
and waits for `SCALED_TO_ZERO`; SIGINT, SIGTERM, and SIGHUP are converted to a
controlled interruption so teardown still runs. SIGKILL and host failure
cannot be handled in-process and require the documented read-only status check
and teardown rerun.

## Scoring handoff

Judging and analysis are deliberately separate from collection. `judge_stage.py`
pins `claude-opus-4-5` / `claude-agent-sdk-v4-turn-taking`, judges every
observed non-recovery scheduled turn, and has a hard two-worker ceiling. The
judge freezes a stage-specific input manifest containing every transcript hash,
uses a durable retry ledger and exclusive lock, and gives its child process only
`ANTHROPIC_API_KEY` among provider credentials. It never calls BaseTen.

Each conversation contributes a fixed 30-turn denominator; missing future
turns fail strict pass and all three displayed accuracy dimensions. Arm
intervals resample whole conversations. Effect intervals resample the frozen
temporal pairs, preserving each pair's contemporaneous no-filler/dots contrast.
Analysis reports strict completion, serving classifications, TTFAT on observed
responses, per-turn error counts, and top-three/top-five error concentration.

At the initial stage, `analyze_stage.py` evaluates exactly the three
prespecified promotion conditions above. It writes a collector-compatible
promotion decision only if at least one condition fires and the operator names
an explicit reviewer. The decision hashes the frozen aggregate and included-run
artifacts. Analysis cannot invoke collection; continuation remains a separate,
reviewed `collect.py --stage full --decision-file ...` action. The full stage is
terminal and produces no promotion decision.
