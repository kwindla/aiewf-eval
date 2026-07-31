# Focused eight-model 96-dot stability campaign

Target and outcome rules frozen at 2026-07-20T20:57:23-07:00, before any new
campaign request and before inspecting the completed GPT-5.4 dash-confirmation
outcomes.

## Objective and scope

Estimate reproducible thinking-off pass and error rates for eight prespecified
model configurations on the 30-turn `aiwf_medium_context` benchmark. Compare no
filler with 96 space-separated literal dots appended to the current final user
message. Persisted conversation history remains filler-free.

The prespecified models, in the order inherited from the historical dot screen,
are:

1. `gpt-5.4`
2. `gpt-5.6-terra`
3. `gpt-5.5`
4. `gpt-5.6-sol`
5. `gemma-4-31b-it`
6. `inkling`
7. `qwen3-8b`
8. `glm-5.2`

OpenAI `*-pro` variants are prohibited. The checked harness rejects them at the
CLI and pipeline layers.

## Fixed sample and historical reuse

- Target: exactly 30 eligible conversation attempts per model and arm.
- Arms: `nofiller` and `dots96`.
- Historical attempts are eligible only when their run logs establish the same
  model route, thinking/reasoning setting, benchmark behavior knobs, and filler
  treatment. Canonical run directories are deduplicated.
- Eligibility is attempt-based, never survivor-based. Premature `end_session`,
  missing `end_session`, malformed or empty model turns recorded in a transcript,
  and forfeited future turns remain outcomes.
- When more than 30 historical attempts are eligible for an arm, take the first
  30 by attempt start time, without reference to scores or completion.
- Historical cohorts without full source hashes are marked runtime-signature
  matches in the inclusion ledger rather than represented as byte-identical.
- The new-run deficit for each arm is `30 - included historical attempts`.

The frozen inclusion ledger contains 190 historical attempts and the five deficit
schedules contain 290 new assigned conversations.

## Frozen configurations

| label | requested model | service | thinking/reasoning | other request settings |
|---|---|---|---|---|
| gpt54 | `gpt-5.4` | OpenAI Responses | effort `none` | priority tier |
| terra | `gpt-5.6-terra` | OpenAI Responses | effort `none` | priority tier |
| gpt55 | `gpt-5.5` | OpenAI Responses | effort `none` | priority tier |
| sol | `gpt-5.6-sol` | OpenAI Responses | effort `none` | priority tier |
| gemma431 | `lilac/gemma-4-31b-it` | Lilac | thinking disabled | provider default sampling |
| inkling | `thinkingmachines/inkling` | BaseTen Model API | effort `none` | max tokens 8192, temperature 1 |
| qwen3_8b | `qwen/qwen3-8b` | OpenRouter | reasoning disabled | max tokens 8192 |
| glm52 | `zai-org/GLM-5.2` | BaseTen Model API | effort `none`, thinking unset | max tokens 8192, temperature 1 |

All new attempts use recovery nudges, tool-call deduplication, no extra LLM turn
after tool results, and a 45-second text idle timeout, matching the historical
screen. Treatment sets `MTE_FILLER_DOTS=96`, `MTE_FILLER_TOKEN=.`, and
`MTE_FILLER_POSITION=suffix`; control explicitly unsets all filler variables.

## Allocation, failures, and stopping

- Deficit schedules are generated and frozen before launch using a recorded seed.
  Where both arms need observations, assignment is randomized in short balanced
  blocks; unavoidable surplus assignments are distributed through the schedule.
- Schedules are not changed in response to outcomes. There is no optional stopping,
  arm reallocation, or sample-size re-estimation.
- A model-caused failure is counted and never replaced.
- An attempt is replacement-eligible only when its log contains objective provider,
  transport, or harness failure evidence and the model did not call `end_session`.
  This rule applies whether the failure happened before the first response or after
  a partial transcript. Model-generated malformed/empty turns, premature
  `end_session`, ordinary missing `end_session`, and other model behavior remain
  counted outcomes. Classification never inspects judged scores, and every
  replacement attempt remains in the audit ledger.
- Judge failure never replaces a model attempt. The same transcript is re-judged.
- Each provider lane is resumable and locked against concurrent duplicate drivers.

## Outcomes and reporting

- Experimental unit: one assigned conversation.
- Every arm has a fixed denominator of 900 scheduled turns.
- A turn passes only when tool use, instruction following, and KB grounding all
  pass. Missing or forfeited turns fail every applicable displayed criterion.
- Report: pass rate, any-error rate, tool-error rate, instruction-error rate,
  KB-error rate, strict-completion rate, and one no-filler row-level P50 TTFAT.
- Confidence intervals resample whole conversations, not turns. Section 3 displays
  confidence intervals without p-values or a Bonferroni headline.
- The completed 41-per-arm GPT-5.4 dash confirmation remains a separate frozen
  analysis and is not pooled into the dot treatment.

The historical GPT-5.4 dot screen used the rolling `gpt-5.4` alias. New dot-campaign
attempts continue that exact requested ID. Neither the dated-snapshot dash arm nor
its dated-snapshot no-filler control is included in the primary dot aggregate.

## Integrity snapshot

Git HEAD at freeze: `3e9f805a86fb556a53724a1c83444d8d0de897d7`.

| file | SHA-256 |
|---|---|
| `benchmarks/aiwf_medium_context/config.py` | `ebab4cc8f8465a844adc0c9542ef60e16400fdd5528b91eceed042486560e164` |
| `benchmarks/aiwf_medium_context/prompts/system.py` | `6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6` |
| `benchmarks/_shared/turns.py` | `c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b` |
| `src/multi_turn_eval/services/filler.py` | `d0d3ea02d69797c56e7d1395b752b04d132f003b3148b3d8e847f69067bf0d15` |
| `src/multi_turn_eval/services/openai_responses.py` | `863b58d390fefb84d237f4382039f89ad77af12ab70f006274925a32d8cdfb80` |
| `src/multi_turn_eval/services/lilac_logged.py` | `dd79b7227cc2c3578eb113b879e4efbd2b08af1283e93d5ce3226a55007a936d` |
| `src/multi_turn_eval/pipelines/base.py` | `2afe1c3d531e4201b5f43c9fc1e3d0235667524ab94cead9a68639058f51be8c` |
| `src/multi_turn_eval/judging/claude_judge.py` | `3f5d2372959d7e9a30f6e426742cc9cb8ff662227c4769c3c8090bd5e5776e18` |
| `src/multi_turn_eval/model_policy.py` | `e3bbb32a1a9fc279c928006fe614627cedd70e59f8defbbf2cddd303906f349f` |

Campaign artifacts frozen before launch:

| file | SHA-256 |
|---|---|
| `existing-included.tsv` | `5ea200716b997e6c9f6c7582ea324d53bf3097aa088aa59aafd840c523c4217f` |
| `deficits.tsv` | `40ecde0dfd343ed1a8d7a077dd8f1330a90f92b8b6d4d4ffc9bd7b69ed26bfda` |
| `schedule-openai-a.tsv` | `a08c3a6294b8c453da5425ccc1c409b058af576a7d50474c41953e211fff3951` |
| `schedule-openai-b.tsv` | `7c02257f044440d476c62697cea4fee7b55bc77fd2d5ec040fc993f33211d92a` |
| `schedule-lilac.tsv` | `694a058d26460860e2b9f06b07558f2ee0382f25a8e64a071a8f4b67e54ee4b5` |
| `schedule-baseten.tsv` | `e451a1c07bb55439adf22396e17a98efe7eb6665305ef912bd66d8212a831eaf` |
| `schedule-openrouter.tsv` | `1752b284a5afaeb2570b8466ecaeee3f1f56bea7a072a9dac49b52e01b10d42b` |
| `run_lane.sh` | `adfc214382c7a79f47cdf8c4e2e97e5d1901add387810cf75050042d1f9a1ecd` |
| `judge_existing.sh` | `19271287ed2837912ace08014a7e3990fdd07a5ac36d1cbfce327c9f366b8d7d` |

## Pre-outcome operational amendment, 2026-07-20T21:24-07:00

The first Lilac requests exposed an ambiguity in the original zero-row-only
replacement wording: Lilac returned several valid turns and then logged an explicit
provider `EngineCore` failure. The historical inclusion ledger already excluded
equivalent partial provider failures, so counting the live ones would have made the
new and historical validity rules inconsistent. Before inspecting any judged scores
or aggregate outcomes, the rule above was amended arm-blind to replace an objective
provider/transport/harness failure even after a partial transcript, provided the
model did not call `end_session`. An explicit OpenRouter upstream content-filter
error is treated the same way. This does not replace ordinary model aborts or
missing-`end_session` outcomes.

The initially counted `lilac-001` attempt 1 and the uncommitted `lilac-002` attempt
2 are invalidated in the append-only `invalidated.tsv` audit ledger, as is the
uncommitted OpenRouter provider-error attempt. Target cells, schedules, and stopping
rules are unchanged. The driver was also made crash-resumable: a completed model
attempt awaiting judgment is adopted on restart instead of rerun, and replacement
counts survive process restarts. The amended driver hash is recorded below after
syntax validation.

| amended file | SHA-256 |
|---|---|
| `run_lane.sh` | `4b4e50b0e16c1bbbec365100b57512221178267dfffc30d39fd53f66d0185298` |
| `invalidated.tsv` | `d3bf0ac14b8be704c9e735007c51046bc1159ec262f30d4413bac8d7e76d213d` |

## Arm-blind provider-outage pause, 2026-07-20T21:31-07:00

Lilac returned a third consecutive explicit `EngineCore` failure while filling
`lilac-002`, so the driver stopped at its three-replacement safety threshold. The
threshold is an operational pause, not an analysis outcome or a license to count an
invalid provider response. Without inspecting judged scores, the logs were reviewed
to confirm that all three stopped for the same provider error and that the earlier
no-filler slot had also experienced an `EngineCore` failure. After a cooldown, the
same frozen schedule may resume with a second bounded batch, raising the per-slot
ceiling from three to six objective infrastructure replacements. Eligibility,
denominators, and the model-versus-infrastructure rule do not change. The driver now
accepts an explicitly bounded `MTE_N30_MAX_INFRA_REPLACEMENTS` value (3–12) so this
manual outage recovery remains logged and cannot become an unbounded retry loop.

## Arm-blind OpenRouter filter pause, 2026-07-21T02:57-07:00

The patched OpenRouter refill reached the three-replacement safety threshold on
`openrouter-002`. Attempts 1 through 3 all stopped before `end_session` with the
same explicit Alibaba upstream inappropriate-content filter error. No judged scores
or aggregate outcomes were inspected. As with the earlier Lilac outage pause, the
same frozen schedule may resume with a second bounded batch by raising the per-slot
ceiling from three to six objective infrastructure replacements. Eligibility,
denominators, assignments, and the model-versus-infrastructure rule remain
unchanged; every replacement stays in the append-only attempt ledger.

## Second arm-blind OpenRouter filter pause, 2026-07-21T04:50-07:00

`openrouter-032` exhausted the six-replacement ceiling. Attempts 1 through 6 all
stopped before `end_session` with objective Alibaba upstream filter evidence; the
later attempts failed earlier in the conversation, including one zero-response
attempt. No judged scores or aggregate outcomes were inspected. To finish the
frozen slot without counting a provider-censored transcript as model behavior, the
same schedule may resume for one final bounded batch with the per-slot ceiling set
to twelve objective infrastructure replacements. No further ceiling increase is
authorized. Eligibility, assignments, denominators, and analysis remain unchanged.

## Qwen route-replacement amendment, 2026-07-21T08:14-07:00

The final bounded OpenRouter refill did not produce complete 30-attempt cells.
The route accumulated 45 objective Alibaba `DataInspectionFailed` content-filter
failures and stopped with 27 valid no-filler attempts and 28 valid dot attempts.
OpenRouter's live route inventory exposed only the Alibaba endpoint, and neither
OpenRouter nor Alibaba documents a supported moderation bypass for this error.
Continuing to raise the retry ceiling would therefore select the subset of
benchmark conversations that survive provider censorship rather than estimate
the requested model's behavior.

The Qwen primary comparison is replaced in full by a new provider-homogeneous
BaseTen cohort. Historical and campaign OpenRouter/Alibaba attempts remain in the
audit trail but are excluded from the primary Qwen aggregate; they are not pooled
with BaseTen. The replacement has 30 assigned conversations per arm rather than
topping up the two incomplete OpenRouter cells. Its balanced schedule was frozen
before the first primary request using seed `20260721`: each block of four has two
no-filler and two dot assignments, with no run of four identical arms.

The replacement route is a dedicated custom-vLLM deployment with these fixed
inputs:

- provider: BaseTen deployment `wgvnndv` for model `wnp6rky3`;
- checkpoint: official BF16 `Qwen/Qwen3-8B` at immutable revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- serving: one H100, TP1, vLLM 0.25.1 pinned by image digest, BF16 weights and
  unquantized KV cache, 32,768-token context, chunked prefill, and automatic
  prefix caching;
- tools/reasoning: Qwen3 reasoning parser, Hermes automatic tool parser, and
  `chat_template_kwargs.enable_thinking=false` at both server and request level;
- sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0, and max tokens 8192;
- harness behavior: the same recovery, deduplication, no-post-tool-LLM, timeout,
  filler, and attempt/replacement rules as the frozen campaign.

The no-filler row's TTFAT is computed only from this BaseTen cohort and is labeled
as BaseTen-specific. Automatic prefix caching is part of the fixed serving
configuration, not an arm-varying intervention.

Before schedule generation, excluded smoke run
`runs/aiwf_medium_context/20260721T081123_qwen_qwen3-8b_f350e2ea` exercised all 30
benchmark turns against this endpoint. It produced 30/30 KB grounding, 30/30
turn-taking, 27/30 strict turn passes, valid automatic tool calls, no reasoning
stream, and a 556 ms row-level P50 TTFAT. This was an operational quality gate,
not a primary observation, and is not eligible for either arm.

| replacement artifact | SHA-256 |
|---|---|
| `ops/baseten-qwen3-8b-vllm/config.yaml` | `7a8eeb0b14c43f7a0d770fd7019d40d255b39718d97c6a1a4071815230c9597d` |
| `src/multi_turn_eval/services/vllm_openai.py` | `6b605a2c065ba35f561cf57621e5d2d2dd7f6df24d78cc82e37a1993a1e7fb08` |
| `prepare_qwen_baseten.py` | `f74cf97a600e9ffaa56d209cb3de979874712cdb93dfd17986b9ca7f5e40d249` |
| `schedule-baseten-qwen.tsv` | `8bb882b7ea4af415fbfd94ffaf46152cb33122df84081aabd91065be3c381e99` |
| amended `run_lane.sh` | `a6e58b7c9b4ac9181f3f8b5822670c3022de50479b2cd8106ac3ec758ad75861` |

## Qwen replacement completion, 2026-07-21T10:51-07:00

The BaseTen replacement completed all 60 frozen assignments, 30 per arm, with
zero inference failures or replacements. Three initial Claude judge subprocess
failures affected slots 17, 18, and 23; the model transcripts were already
complete and were not rerun. A non-adaptive judge-only recovery pass succeeded
for slots 17 and 18. Slot 23 returned malformed judge JSON once more and then
succeeded on the next judge-only attempt. The strict campaign audit subsequently
verified all 480 primary conversations: 174 historical conversations and 306
valid top-ups/replacements.

The Qwen fixed-denominator results are 81.3% no-filler and 82.2% +96 dots, a
+0.9-point difference with a 95% conversation-bootstrap interval of −4.0 to
+6.7 points. Strict completion was 10% no-filler and 0% dots. The BaseTen
no-filler TTFAT was 564 ms P50, 678 ms P95, and 1,563 ms maximum. The result does
not establish a reliable filler effect for Qwen3-8B, while the dedicated route's
latency is substantially lower than the superseded exploratory route.

The final analyzer independently reproduced the seven previously completed model
rows byte-for-byte from the isolated provisional aggregate. The README and both
report formats were then rebuilt and passed the provenance/structure verifier and
the full 24-test suite. After public-report verification, the exact BaseTen
deployment was retained but scaled to zero active replicas.

| final artifact | SHA-256 |
|---|---|
| `aggregates.json` | `573e53779774f61c8cc9641d553c02c2368c56a2785fddc87071cdb5c22a1d99` |
| `aggregates.tsv` | `30bcd1e771cd15cff1780b96991d35473881d3fff35f0fb3d3e7a2452cc7e6f9` |
| `README.md` | `a9d82e8c9dc5966ab74bda6895867af773c2a24a736d6ae10f45d5e1f7fa6b0a` |
| `docs/filler-token-latent-scratchpad-study.md` | `091f1e98f49b3eb8062ec5d73d1d4727de696912733ed7e7b3c5081002becabf` |
| `docs/filler-token-latent-scratchpad-study.html` | `b2de1dff61e7249475e764eb43c9c9ac2692d551d67da875fcd37a0a765a24f7` |

## Descriptive reasoning-effort reporting extension, 2026-07-21

The report now places the GPT-5.4 reasoning-effort comparison in its own section.
The final n=30-per-arm `none` comparison and the separate contemporaneous
n=8-per-arm `low` comparison are displayed as parallel filler contrasts with
conversation-bootstrap intervals and arm-specific median TTFAT. Their −1.7-point
effect difference is labeled descriptive only: the collections have unequal sample
sizes and were not randomized together as a joint factorial interaction experiment.

The comparison artifact records every low-effort run and hashes its transcript and
judgment sources. A deterministic rebuild reproduced it byte-for-byte.

| reasoning-comparison artifact | SHA-256 |
|---|---|
| `docs/filler-study-data/analyze_gpt54_reasoning_comparison.py` | `1dbab8376f616dc90cf5e9d17853a25ba27f5a93343a9119b0779941e2f44f85` |
| `docs/filler-study-data/gpt54-reasoning-comparison.json` | `1e9877e3cc173ace0089b78de23f913fd0a1cba8e43e15115c6f4ef7ad591cdb` |
| `scripts/build_filler_report.py` | `1eb7624ee0a107e68c79f65a083ab3fd5d15634a65470a41e0057e556a8ded56` |
