# Gemini Flash minimal-thinking and 96-dot screen

Protocol frozen at 2026-07-21T17:27:00-07:00, after a one-turn excluded live
instrumentation smoke and before any scored campaign request.

## Objective

Add current Gemini Flash-family latency/accuracy rows to the text leaderboard and
run a prospective exploratory 96-dot filler screen for the report. The requested
models, in fixed order, are:

1. `gemini-3.5-flash`
2. `gemini-3.5-flash-lite`
3. `gemini-3.6-flash`

All requests use the Google Gemini Developer API through `--service google` and
the text pipeline.

## Reasoning setting and labels

Google's current Gemini 3 API does not guarantee complete thinking-off. The
supported low-latency floor is `thinking_level=minimal`, which Google describes
as matching no thinking for most queries while allowing very small amounts of
reasoning. Therefore this campaign explicitly sets
`MTE_GOOGLE_THINKING_MODE=minimal`, and every public row is labeled `(minimal)`;
it is not labeled `thinking off`.

The clean historical ten-conversation `gemini-3.5-flash (minimal)` cohort is
retained as a descriptive audit reference in `existing-included.tsv`, but is not
used in the primary comparison: it predates the dot arm and used an older Pipecat
client/service implementation. Gemini 3.5 Flash therefore receives ten fresh,
contemporaneous no-filler controls. The legacy `thinking_budget=0` runs are also
excluded because that is not the recommended Gemini 3 control for the new
releases and cannot honestly be generalized as full thinking-off.

## Fixed stage 1

- No-filler target: 10 new, contemporaneous eligible attempts per model.
- Dot screen: six new attempts per model.
- Treatment: `MTE_FILLER_DOTS=96`, `MTE_FILLER_TOKEN=.`, and
  `MTE_FILLER_POSITION=suffix` on a request-local copy of the current user turn.
  Persisted history remains filler-free.
- Common settings: recovery enabled, tool-call deduplication enabled, no extra
  LLM turn after tool results, 45-second text idle timeout, provider-default
  sampling, and 4096 maximum output tokens from the Google adapter default.
- Experimental unit: one assigned 30-turn conversation attempt. Model-caused
  early exits, malformed responses, missing turns, and missing `end_session`
  remain outcomes. Objective provider/transport failures without `end_session`
  are replacement-eligible and remain in the audit ledger. A bare harness idle
  timeout without provider/transport error evidence is an outcome, not an
  infrastructure replacement.
- Judge: the repository's Claude judge. A judge failure retries the same
  transcript and never replaces a model attempt.

The stage-1 schedules were fixed before launch. Reported accuracy uses a
30-turn fixed denominator: absent and post-abort turns fail each applicable
displayed dimension.

Gemini 3.5 Flash's fresh control and dot schedules run in parallel and are
contemporaneous, but assignment is not randomized or interleaved within one
lane. Its filler estimate therefore remains exploratory and may retain a small
within-day load/order confound.

## Prospective adaptive dot rule

The initial six-dot result is compared with the frozen ten-conversation control.
No dot top-up is run when strict-completion rates are identical and the absolute
pass-rate difference is below 2.0 percentage points.

Otherwise, the dot arm is extended to ten attempts. After ten dot attempts:

- stop at ten unless the whole-conversation bootstrap 95% interval excludes zero,
  or the absolute difference is at least 3.0 points and the same error-site
  direction recurs in at least three conversations;
- if either stronger-signal condition holds, or strict-completion rates still
  differ at n=10, promote the model to a focused follow-up and extend both arms
  to 30 attempts under a separately frozen stage-3 schedule.

For this rule, a recurring same-turn direction means at least three conversations
in one arm fail the same scripted turn's joint pass criterion, and that arm's
failure proportion at that turn is higher than the other arm's. This is a
turn-level joint-pass rule, not a post-hoc choice among scoring dimensions.

These thresholds govern sample size only; all estimates and uncertainty remain
visible regardless of the stopping branch. Stage-1 rows remain exploratory.

## Stage-1 mechanical decision and frozen top-up schedules

At 2026-07-21T18:38:00-07:00, after all stage-1 transcripts had complete
judgment coverage, `analyze.py` and `decide_stage.py` mechanically applied the
rule above. Every model advanced from six to ten dot attempts:

- `gemini-3.5-flash`: interim dot-minus-control pass difference +6.44 points;
  strict completion 90.0% vs 100.0%;
- `gemini-3.5-flash-lite`: interim difference -0.67 points; strict completion
  60.0% vs 50.0%;
- `gemini-3.6-flash`: interim difference -9.78 points; strict completion 80.0%
  vs 83.3%.

Exactly four additional dot assignments per model were then frozen in
`schedule-g35topup.tsv`, `schedule-g35litetopup.tsv`, and
`schedule-g36topup.tsv`. Their SHA-256 hashes are respectively
`2eba63aff34d9426afdc9a436325d145bdf30e61cc39fb1ada52d8f248482b15`,
`2a936ef1f5f3deb52a5f7d87f2d955cd4591ea353d15d5887372b8ad53436c69`,
and `bdc055458eaa90e9a739f1d9d21783786ef85a32c884b4770e3c4ce8c145ea97`.
No n=10 result was available when these top-up schedules were frozen.

At 2026-07-21T18:54:00-07:00, the same mechanical rule was applied after all
three dot arms reached ten attempts. It promoted all three models to 30 attempts
per arm:

- `gemini-3.5-flash`: +6.33 points, 95% conversation-bootstrap interval
  [-2.00, +20.67], strict completion 90.0% vs 100.0%;
- `gemini-3.5-flash-lite`: +0.67 points, interval [-24.00, +26.33], strict
  completion 60.0% vs 50.0%;
- `gemini-3.6-flash`: -5.33 points, interval [-16.00, +3.67], strict completion
  80.0% vs 90.0%.

The first mechanical implementation flagged recurring-turn signals for the 3.5
Flash and 3.6 Flash cells. A pre-follow-up independent audit found that 3.5
Flash's sole recurring signal pointed opposite its aggregate benefit. Before any
n=30 outcome existed, the implementation was narrowed to require alignment with
the aggregate effect direction. This removes that trigger from 3.5 Flash, but
does not change any promotion: all three cells independently advanced because
strict completion remained unequal at n=10. Flash Lite advanced only on that
completion trigger; 3.6 Flash retained an aligned recurring-turn trigger.
Exactly 20 further no-filler and 20 further dot assignments per model were frozen,
alternating arms within each model lane, in `schedule-g35focused.tsv`,
`schedule-g35litefocused.tsv`, and `schedule-g36focused.tsv`. Their SHA-256
hashes are respectively
`6123f3679e5de4697cb521bb9fd72138db8aa99f57ef10e556a167db323ba839`,
`c43a8d76b744d67572c081fd813a55095cc5a6b0a6c9535a0ed32e54b175bc99`,
and `ca570799585424aa79f48d32d5ec1966f4c374ea493bf12bed85519d8f5213fa`.
No n=30 result was available when these focused schedules were frozen.

## Judge-output recovery amendment

At 2026-07-21T21:09:00-07:00, focused slot `G36-N27` had an intact frozen
model transcript but exhausted the runner's three judge attempts; each attempt
failed before producing a parseable judgment JSON document. No score from the
transcript existed to inspect. The model attempt is not replacement-eligible and
remains the assigned experimental unit. To finish schema-valid coverage, further
judge-output recovery attempts may be made against this same immutable transcript,
under the same shared judge lock and judge implementation. They are logged
separately, never regenerate the conversation, and the lane may resume only after
the normal `valid_judgment` check passes. This amendment changes operational
recovery only, not sample membership, scoring, or the fixed denominator.

After one additional full-transcript recovery attempt failed in the same way,
recovery switched to the judge's supported turn-filter path. Three overlapping
15-turn windows retain judgments for turns 0–9, 10–19, and 20–29 respectively;
the overlap supplies five turns of adjacent context around each merge boundary.
The recovery script requires exact coverage and boolean score schemas in every
window, mechanically merges each retained ten-turn block, and writes one normal
30-turn judgment artifact. It never changes or regenerates the model transcript.

The first chunked run confirmed the failure mode: most judge responses were
Python-literal mappings with single-quoted keys rather than strict JSON. The
recovery script therefore gained a recovery-local parser fallback: strict
`json.loads` first, then Python's non-executing `ast.literal_eval`. The fallback
changes only serialization acceptance; it does not infer, repair, or alter any
score value, and the production judge source remains unchanged. Exact turn
coverage and boolean-schema validation still gate the merged artifact.

Some recovery responses also placed brace-bearing explanatory prose before the
final mapping. The recovery parser therefore tries brace-starting suffixes from
last to first and accepts a candidate only if it parses as JSON/literal data and
the normal exact window-coverage checks pass. Each validated window is persisted
before the next begins so no accepted judgment window is silently recomputed.

## Excluded validation

- Direct one-prompt API probes established that both new model IDs accept
  `thinking_level=minimal`; neither reported thought tokens for the trivial
  probe.
- `runs/aiwf_medium_context/20260721T172636_gemini-3.6-flash_ae75b944`
  is a one-turn, 96-dot live instrumentation smoke and is excluded from every
  aggregate.

## Pre-outcome audit amendment

At 2026-07-21T17:38:00-07:00, after generation/completion metadata but before
inspecting any judged scores, an independent implementation audit found three
correctable risks. Runners were stopped and the following changes were frozen:

1. add ten fresh Gemini 3.5 Flash no-filler controls and demote the May cohort to
   descriptive-only, avoiding a date/client confound in the filler comparison;
2. count the first Flash-Lite no-filler attempt, which reached eight turns and
   then hit only the harness idle watchdog, rather than replacing it as provider
   infrastructure; its already-generated replacement remains excluded as an
   out-of-protocol extra;
3. require complete judgments for every observed non-recovery turn, validate
   model/thinking/filler signatures per run, and recheck source integrity before
   each request.

The operator stop interrupted one Flash-Lite process before its first response;
that run is recorded in `excluded-operator-interrupt.tsv` and is not an assigned
attempt. No scored outcomes were examined in making these amendments.

## Initial frozen integrity snapshot

Git HEAD at freeze: `3e9f805a86fb556a53724a1c83444d8d0de897d7`.

| file | SHA-256 |
|---|---|
| `benchmarks/aiwf_medium_context/config.py` | `ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164` |
| `benchmarks/aiwf_medium_context/prompts/system.py` | `6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6` |
| `benchmarks/_shared/turns.py` | `c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b` |
| `src/multi_turn_eval/pipelines/base.py` | `2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c` |
| `src/multi_turn_eval/services/google_logged.py` | `97294f5a086d9516ff501c638aa14d525e67cceb11e8df692f50c8f0d1c227c3` |
| `src/multi_turn_eval/judging/claude_judge.py` | `3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18` |
| `tests/test_google_filler.py` | `cd1980c3b912406fe3e197ef80087b5d185e20f3d566f588575b32ec87993a78` |
| `existing-included.tsv` | `558d94971784fd7e6243a4bf5b2c190aaa695dcd1b6d4791b11dc4c4cb44cd01` |
| `schedule-g35.tsv` | `b7ee9062fb28b4bbddb2ce716774040ead57293fb15beb555005cc709e7156d7` |
| `schedule-g35control.tsv` | `8a1173fb531f01a0206652310bc54282a59ab7f4f69c71c8e1672b7394268d68` |
| `schedule-g35lite.tsv` | `054277eb9fbee1c054836f71aed88a099c3d4c59bd89442628e698246bb0f9c5` |
| `schedule-g36.tsv` | `ecb1d145ce270beecaafba364670c753997854e37f4137289a9851f0f8765b66` |
| `run_lane.sh` | `444a43ef4dc58bf9f8a5d4cb2b8f14f5eba70b1035d1556d4ecf9bf1625f7064` |
| `analyze.py` | `6ef2d5f2b0a72fa787c5e8b98153ce086928e9dec92a90f015be532c53fc77df` |
| `decide_stage.py` | `c24654dff8020caded7e283c1341ef53b54ec028caba1856899b73c1a7c68e9c` |
| `update_readme.py` | `9dfcaf52d365166823b6e1e327acb2dde7320b98ebe67daf0600d290fbaca9af` |

This table records the initial freeze. The operational amendments above required
audited changes to several campaign scripts; the schedules themselves remained
immutable after their respective stage freezes. The completed-campaign hashes
below supersede the initial script hashes and identify the published artifacts.

## Completed-campaign integrity

| file | SHA-256 |
|---|---|
| `run_lane.sh` | `6a0883d66946f57d8dcfd54816f4fe93c8975dabb2bed7f6ec9abf95cfd1ac07` |
| `analyze.py` | `aa7b2ed23cb5cb5ec612626f3aa788f85d6f8b5af286c8eed991ae165ab8d2ee` |
| `decide_stage.py` | `c24654dff8020caded7e283c1341ef53b54ec028caba1856899b73c1a7c68e9c` |
| `update_readme.py` | `375acded4490b46a2cb276fbf0dfdf82fdbfeb93967dbed417ecde72b947ad2a` |
| `analyze_idle_sensitivity.py` | `12ac91fe3eb8b0fbfd62ca77a5dc717a5cb8bbfaba4c3bfdfe3eabb25def5a37` |
| `verify_outputs.py` | `32c1f259c27762a4ce2db3baabc3cbf139056778c652498e8dfce156ac17f466` |
| `chunked_judge_recovery.py` | `537f1d6c9e155ea67be07f7f9000afa23bccdbb93af7782eef6cbe29deafe836` |
| `schedule-g35.tsv` | `b7ee9062fb28b4bbddb2ce716774040ead57293fb15beb555005cc709e7156d7` |
| `schedule-g35control.tsv` | `8a1173fb531f01a0206652310bc54282a59ab7f4f69c71c8e1672b7394268d68` |
| `schedule-g35lite.tsv` | `054277eb9fbee1c054836f71aed88a099c3d4c59bd89442628e698246bb0f9c5` |
| `schedule-g36.tsv` | `ecb1d145ce270beecaafba364670c753997854e37f4137289a9851f0f8765b66` |
| `schedule-g35topup.tsv` | `2eba63aff34d9426afdc9a436325d145bdf30e61cc39fb1ada52d8f248482b15` |
| `schedule-g35litetopup.tsv` | `2a936ef1f5f3deb52a5f7d87f2d955cd4591ea353d15d5887372b8ad53436c69` |
| `schedule-g36topup.tsv` | `bdc055458eaa90e9a739f1d9d21783786ef85a32c884b4770e3c4ce8c145ea97` |
| `schedule-g35focused.tsv` | `6123f3679e5de4697cb521bb9fd72138db8aa99f57ef10e556a167db323ba839` |
| `schedule-g35litefocused.tsv` | `c43a8d76b744d67572c081fd813a55095cc5a6b0a6c9535a0ed32e54b175bc99` |
| `schedule-g36focused.tsv` | `ca570799585424aa79f48d32d5ec1966f4c374ea493bf12bed85519d8f5213fa` |
| `aggregates.json` | `41be324032aaecffd03b3e43ffa35242a3e9b19c82c404093138f5905a7ecff2` |
| `adaptive-decision.json` | `991dfcc80dd38559fdc3f5db00d23a395bdeeb9d3e59d2eb591d996d0035d62c` |
| `idle-timeout-sensitivity.json` | `3aad4b7321caf20bdb43cd89de08c4f73299d97392b5f0791e247f76b9b98e03` |
| `scripts/build_filler_report.py` | `bdcf276d3ff18e78a0ff869eb45c4938cacf6138e3513182e401de81852d5aa4` |
| `README.md` | `e755568713e34aeeffd73f1aaa65b7c30d674218b5a966d1dd2c6f4fe92f1392` |
| `docs/filler-token-latent-scratchpad-study.md` | `8cdc2d4180b800b6d474772f9eae0087d98fc28ec942bbc4e7d3e0e95d1c2ae5` |
| `docs/filler-token-latent-scratchpad-study.html` | `60e0029b83fe2121d2af6fdbf8fa40fc53ec9e32aacfaea9d2319461ce1ce0dc` |
