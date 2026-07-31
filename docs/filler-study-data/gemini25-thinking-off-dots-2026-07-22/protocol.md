# Gemini 2.5 Flash thinking-off and 96-dot screen

Protocol frozen at 2026-07-22T12:25:40-07:00, before any scored campaign
request. The analyst had seen the historical unlabeled Gemini 2.5 Flash
README row and the completed Gemini 3 filler results. This is a prospective
extension of the report screen, not an outcome-naive first test of filler.

## Objective and configuration

Measure `gemini-2.5-flash` on `aiwf_medium_context` through the Google Gemini
Developer API (`--service google`, text pipeline) under two contemporaneous
conditions:

1. no filler;
2. 96 space-separated dots appended to the current user turn, suffix position.

Both arms explicitly set `thinking_budget=0` through
`MTE_GOOGLE_THINKING_MODE=disabled`. The Google API defaults Gemini 2.5 Flash
to dynamic thinking when the budget is omitted, but Pipecat's Google service
has supplied its own low-latency `thinking_budget=0` default for this model
since June 2025. Google documents zero as the supported way to disable
thinking. The campaign pins the value explicitly so the run logs and protocol
are unambiguous; the public row is labeled thinking off, not `minimal`.

Common settings match the July 21 Gemini screen: recovery enabled, tool-call
deduplication enabled, no extra LLM turn after tool results, a 45-second text
idle timeout, provider-default sampling, and the Google adapter's default
maximum output tokens. TTFAT ends at the first user-visible content or tool
call; thought-only chunks do not stop it.

## Fixed stage 1 and attempt policy

- Fresh no-filler target: 10 eligible conversation attempts.
- Initial dot target: 6 eligible conversation attempts.
- Unit: one assigned 30-turn conversation attempt.
- Model-caused early exits, malformed outputs, missing turns, and missing
  terminal `end_session` remain outcomes on the fixed 30-turn denominator.
- Only objective provider/transport failures are replacement-eligible. A bare
  harness idle timeout without provider/transport evidence remains an outcome.
- The repository Claude judge scores every observed non-recovery turn. Judge
  retries reuse the immutable transcript and never replace a model attempt.

The schedules were frozen before launch. Control and dot lanes may run in
parallel; they are contemporaneous but not randomized within one interleaved
lane, so the filler comparison remains exploratory.

## Prospective adaptive rule

At dots n=6, extend dots to n=10 if either:

- the absolute fixed-denominator pass-rate difference is at least 2.0 points;
- strict-completion rates differ.

At dots n=10, stop unless at least one condition holds:

- the whole-conversation bootstrap 95% interval excludes zero;
- the absolute difference is at least 3.0 points and an aggregate-aligned
  same-turn failure direction recurs in at least three conversations;
- strict-completion rates still differ.

If triggered, extend both arms to 30 attempts using a separately frozen
alternating-arm schedule. Thresholds govern sample size only; estimates remain
visible on whichever branch is reached. Bootstrap intervals use 100,000
independent-arm whole-conversation resamples with seed 20260722.

## Planned publication

The no-filler arm replaces the old unlabeled Gemini 2.5 Flash README row
with a labeled thinking-off row. The filler comparison is appended to Section 3
of the HTML/Markdown report in the original extension order. It enters the
focused confidence-interval and turn-family views only if the prospective rule
promotes both arms to 30.

## Frozen stage-1 inputs

Git HEAD: `3e9f805a86fb556a53724a1c83444d8d0de897d7`

| file | SHA-256 |
|---|---|
| `benchmarks/aiwf_medium_context/config.py` | `ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164` |
| `benchmarks/aiwf_medium_context/prompts/system.py` | `6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6` |
| `benchmarks/_shared/turns.py` | `c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b` |
| `src/multi_turn_eval/pipelines/base.py` | `eaa2b36ce5efd591d0657b37e904f64c339cd8feb7102754e670c01e0bd53d35` |
| `src/multi_turn_eval/services/google_logged.py` | `97294f5a086d9516ff501c638aa14d525e67cceb11e8df692f50c8f0d1c227c3` |
| `src/multi_turn_eval/judging/claude_judge.py` | `3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18` |

Stage-1 schedule hashes and completed artifact hashes are appended after the
files are created and verified, without changing the sampling rule.

The excluded one-turn instrumentation smoke is
`runs/aiwf_medium_context/20260722T122741_gemini-2.5-flash_2fab87f1`.
Its run log contains the required `thinking_budget=0 (disabled)` signature,
its transcript reports zero thought tokens, and it is not present in either
stage-1 schedule or any analysis manifest.

| stage-1 file | SHA-256 |
|---|---|
| `schedule-control.tsv` | `de13ddb7039f196eac4a144618dd936f28e688b65d188d0cec35990a916306f0` |
| `schedule-dots.tsv` | `1f5e855e3f39e80b22bd5ed7f8ac80f5fb5caaf29e9aa360b139bc8437e1ff14` |
| `run_lane.py` | `a8dfdbcd8bc5197ec1c64e9bbf3128a0899d1a3bcbc02262a066119b968512b2` |
| `tests/test_google_filler.py` | `f7eb1859cb10c8264b7a1f9897666d878e79b73e80a16a42135997f46015ba44` |

## Final adaptive decision

All 16 scheduled stage-1 conversations were valid, strictly complete, and
judged: 10 no-filler and six dot-treated. No replacement attempt entered the
analysis. The no-filler pass rate was 91.7%; the dot-treated pass rate was
90.6%, a −1.1-point exploratory difference. Strict completion was 100% in both
arms. Neither stage-1 trigger fired, so the prespecified action was
`stop_at_6`; no top-up or n=30 promotion was run.

The Pipecat-default clarification above was added after the scored calls, when
the service's request-building override was audited. It corrects the provenance
description but does not change the frozen model requests: both arms already
carried explicit `thinking_budget=0`.

| stage-1 artifact | SHA-256 |
|---|---|
| `aggregates.json` | `05864b1638a1f38ae9ef27fec6fbfb3f35e9da626bbe8b0d7259b1366a1060dc` |
| `source-manifest.tsv` | `293923f8a6e55befe8182537bf1b285bbd48b87377a6e2a707e57d540198e691` |

## Post-screen no-filler precision extension

Frozen at `2026-07-22T13:00:30-07:00`, before any extension request, after the
user requested a 30-conversation confidence standard for the public no-filler
benchmark row. Twenty fresh no-filler assignments (`G25-N11` through
`G25-N30`) are added to the original ten controls. They use the identical
explicit `thinking_budget=0` configuration and attempt policy.

This extension estimates the no-filler pass rate and TTFAT more precisely for
the README and the open-circle control point in Section 3. It does not reopen
the prespecified dot-arm stopping decision: dots remain at six. The 30/6
filler contrast stays exploratory and receives no focused n=30 effect-interval
whisker. The source-only changes between stage 1 and this extension are the
corrected Pipecat-default log wording and its test name; explicit-disabled
request behavior is unchanged.

| precision-extension input | SHA-256 |
|---|---|
| `schedule-control-topup.tsv` | `5911a44345638fa322fd19d1d01d63e8fdf5378aea0b6389d2022bbf3c7763cc` |
| `run_lane.py` | `51f820a658764c3c2cf74d5f754021f34c1191c6f00d2655337c470bf9b62a4f` |
| `analyze.py` | `c563510d952cdd7f740bf23cc0f428a8b50c7a8f45060ec20f7048ff070ad529` |
| `build_source_manifest.py` | `54d3421db4f8dd272e8a44bbf4ce1916821938ca88df4f4bb9beff994d39a28c` |
| `verify_outputs.py` | `7a2630e5d9d3199f3252b0c6dd521f57023928ca62664875bd1a528b18747907` |
| `src/multi_turn_eval/pipelines/base.py` | `70b77c51da6dd6232d4aa44aa2b1c95922e21200cabbb65ee5abf76cbbb06a98` |
| `tests/test_google_filler.py` | `0cda16dbefc4c48da5beb6bc5e0b14f1140e596399bd757b55e1d814bddc21fb` |

## Precision-extension outcome

All 20 extension assignments were valid, judged, and strictly complete. There
were no provider replacements. Combined with stage 1, the final no-filler pool
contains 30 conversations and 900 fixed scripted turns. It scored 89.9% pass
rate with 100% strict completion and a 550.5 ms pooled turn-level P50 TTFAT
(displayed as 550 ms under the report's existing integer-rounding convention).
The unchanged six-conversation dot arm scored 90.6%, so the displayed 30/6
exploratory contrast is +0.7 points and has no focused effect whisker.

The first post-extension analysis invocation exposed a stale sample-size guard
that rejected 30/6 before reading outcomes into the aggregate. The guard was
mechanically corrected to admit the already-prespecified control-only design;
the inclusion, scoring, resampling, and publication rules were unchanged. The
analyzer was then run twice with byte-identical output.

The README table now includes an audited serving-provider column for all 48
text-model rows. Gemini text API rows use the requested `AI Studio` label;
Inkling and this study's GLM-5.2/Qwen3-8B rows use `BaseTen`; the current README
Nemotron cohorts use their actual `Modal` route.

| completed output | SHA-256 |
|---|---|
| `analyze.py` | `413b58cabc7605ee4fba653004b9402c9d96e78b396e3427875b08d5ec77b30b` |
| `build_source_manifest.py` | `54d3421db4f8dd272e8a44bbf4ce1916821938ca88df4f4bb9beff994d39a28c` |
| `update_readme.py` | `8538dedf005975774a6c26a191b0311351da40c2993c82619f2542a5f18038b8` |
| `verify_outputs.py` | `ffd388132729cde716eb76261c0710829389f98800c5bbcbead1dced6d803757` |
| `aggregates.json` | `50e79622b895f9a2ccba3e5f286f4108dd62b72a4b065987cfa61245098cd0d5` |
| `source-manifest.tsv` | `b87f9af174b147a3c198ae30c2f59bb51455f92612486054d544938a1e94395f` |
| `README.md` | `44dc4e0d9fda724aa9b0ee0f0495c00b6896f9a61c86002363ddb28f65d3ae8d` |
| `scripts/build_filler_report.py` | `a1e8460d20bfa77aec1b1f988be8f506784a98fd93a7740a16c74613c40b26a7` |
| `docs/filler-token-latent-scratchpad-study.md` | `f3b5210723ef27e44c391eaef49c4a2f4f1a6d9f34347bc2a21f556807bf6681` |
| `docs/filler-token-latent-scratchpad-study.html` | `196c85633387c4aceff381e18f96d91ad2aa6b39191e84213087b3369d45d71c` |
