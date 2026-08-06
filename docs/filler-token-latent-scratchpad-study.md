# Filler-Token Latent Scratchpad — Cross-Model Study

**Question:** Can appending content-free filler positions to the current user
message improve a reasoning model's accuracy when visible reasoning is disabled,
without a large change in time to first answer token (TTFAT)?

**Scope:** Twenty-six standard filler comparisons plus one separate Nemotron
thinking-toggle contrast across six serving categories on the
30-turn `aiwf_medium_context` conversational tool-use benchmark. The screen is
exploratory. “Latent compute” is a mechanism hypothesis motivated by arXiv
2607.03502 and Pfau et al. (2024), not something these API-level experiments observe
directly.

## Method

- `MTE_FILLER_DOTS=<n>` appends `n` space-separated dots to a copy of the final
  user message while building each request. Persisted conversation history stays
  filler-free. Pattern and placement follow-ups use the corresponding filler knobs.
- The primary screen compares no filler with 96 trailing dots under each model's
  low-latency reasoning configuration. Eight prespecified rows use 30 eligible
  conversations per arm and a fixed denominator of 900 scripted turns; missing,
  malformed, and post-abort future turns count as failures. Three appended Gemini 3
  rows use Google's `minimal` reasoning floor, which does not guarantee complete
  thinking-off. A separate Gemini 2.5 Flash extension explicitly sets
  `thinking_budget=0`. Each extension starts with ten fresh no-filler attempts and a
  prospectively staged dot arm, with possible 30-per-arm promotion. After its dot screen
  stopped at 10/6, Gemini 2.5 Flash received a separate control-only precision extension
  to 30 no-filler conversations; its dot arm remained frozen at six. Laguna S 2.1 uses a
  separate frozen prospective 30-per-arm campaign on the paid OpenRouter route to
  Poolside-hosted BF16 weights, with reasoning explicitly disabled. Two exploratory
  Qwen3.6 extensions disable native thinking and reuse frozen 30-conversation
  no-filler cohorts; their later, non-interleaved dot arms follow a prespecified
  6→10→30 stopping rule. The other nine standard
  rows retain their original exploratory pools, generally 6–10 available judged
  conversations per arm. Nemotron Super is
  discussed separately because its available result is a thinking-toggle contrast,
  not a filler-effect comparison.
- The intervention counts are nominal glyph counts. Token counts and token IDs were
  not verified for every provider tokenizer, so “96 tokens” should not be read as a
  tokenizer-normalized intervention.
- TTFAT is content-aware: reasoning deltas are excluded and the metric stops at the
  first visible answer token or tool-call output. Similar observed medians are not
  equivalence tests and do not establish equal end-to-end latency, cost, or tail
  behavior.
- The rubric judge is filler-blind because transcripts contain the unmodified user
  text. A 90-turn re-judge reproduced 90/90 labels; that establishes repeatability,
  not validity against human adjudication.
- For focused rows, confidence intervals use 100,000 whole-conversation
  bootstrap samples; strict-completion intervals use Wilson's method. Section 3
  reports estimates and these intervals without p-values or a multiple-testing
  headline. Original dose, pattern, placement, and retained historical-screen cells
  keep their exploratory conversation-cluster analyses and are labeled as such.
- The separate prospective GPT-5.4 validation is complete: 41 fresh conversations
  per arm on the pinned snapshot scored 92.44% without filler and 95.37% with 96
  trailing dashes, a +2.93-point effect (95% CI +1.09 to +4.77). Strict completion
  was 41/41 and median TTFAT was 678 ms in both arms. It confirms that selected dash
  configuration, not the dot cell or general glyph equivalence.

<!-- N30_PRIMARY_START -->
## Primary screen: 96 trailing dots vs no filler, thinking off or provider-minimal

| model | endpoint | no filler | +96 dots | Δ pt [95% CI, focused] | strict completion | P50 TTFAT ms (row config) | included runs no/dots | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gpt-5.4 | OpenAI | 90.2 | 95.2 | +5.0 [+3.0, +6.9] | 100% → 100% | 689 | 30 / 30 | increase |
| gpt-5.6-terra | OpenAI | 91.3 | 55.8 | −35.6 [−43.3, −26.9] | 100% → 23% | 621 | 30 / 30 | decrease |
| gpt-5.5 | OpenAI | 97.4 | 99.7 | +2.2 [+1.1, +3.4] | 100% → 100% | 875 | 30 / 30 | increase |
| gpt-5.6-sol | OpenAI | 96.6 | 100.0 | +3.4 [+3.3, +3.7] | 100% → 100% | 1098 | 30 / 30 | increase |
| gemma-4-31b-it | Lilac | 97.7 | 99.4 | +1.8 [+0.9, +2.7] | 100% → 100% | 672 | 30 / 30 | increase |
| gpt-oss-120b | BaseTen | 84.9 | 85.3 | +0.4 | — | 405 | 6 / 6 | no detectable effect |
| qwen3-32b | OpenRouter | 90.7 | 91.0 | +0.3 | — | 832 | 10 / 10 | no detectable effect |
| gpt-4.1 | OpenAI | 96.6 | 96.7 | +0.0 | — | 610 | 6 / 5 | no detectable effect |
| kimi-k2.6 | BaseTen | 94.4 | 94.4 | +0.0 | — | 463 | 6 / 6 | no detectable effect |
| nemotron-3-ultra | BaseTen | 98.9 | 98.9 | +0.0 | — | 447 | 6 / 6 | no detectable effect |
| qwen3-14b | OpenRouter | 90.7 | 90.3 | −0.3 | — | 707 | 10 / 10 | no detectable effect |
| gpt-5.6-luna | OpenAI | 89.2 | 88.5 | −0.6 | — | 583 | 8 / 8 † | no detectable effect |
| deepseek-chat-v3.1 | OpenRouter | 95.3 | 94.7 | −0.7 | — | 1406 | 10 / 10 | no detectable effect |
| claude-haiku-4-5 | Anthropic | 99.3 | 98.7 | −0.7 | — | 777 | 10 / 10 | no detectable effect |
| inkling | BaseTen | 94.8 | 82.7 | −12.1 [−19.9, −4.8] | 90% → 73% | 447 | 30 / 30 | decrease |
| qwen3-8b | BaseTen | 81.3 | 82.2 | +0.9 [−4.0, +6.7] | 10% → 0% | 564 | 30 / 30 | uncertain |
| glm-5.2 | BaseTen | 99.7 | 97.2 | −2.4 [−3.8, −1.2] | 100% → 100% | 936 | 30 / 30 | decrease |
| gemini-3.5-flash | Google | 93.3 | 96.6 | +3.2 [−2.3, +10.7] | 93% → 97% | 892 | 30 / 30 | suggestive |
| gemini-3.5-flash-lite | Google | 68.6 | 68.2 | −0.3 [−14.7, +14.4] | 60% → 40% | 591 | 30 / 30 | no detectable effect |
| gemini-3.6-flash | Google | 97.1 | 89.3 | −7.8 [−13.3, −2.9] | 90% → 90% | 798 | 30 / 30 | decrease |
| gemini-2.5-flash | Google | 89.9 | 90.6 | +0.7 | 100% → 100% | 550 | 30 / 6 | no detectable effect |
| laguna-s-2.1 | OpenRouter | 85.6 | 83.3 | −2.2 [−8.3, +5.1] | 13% → 13% | 295 | 30 / 30 | uncertain |
| qwen3.6-27b | BaseTen | 97.3 | 97.2 | −0.1 | 100% → 100% | 668 | 30 / 6 | no detectable effect |
| qwen3.6-35b-a3b (FP8) | BaseTen | 91.6 | 78.0 | −13.6 | 90% → 73% | 765 | 30 / 30 | decrease |
| inkling-small | BaseTen | 75.1 | 76.9 | +1.8 [−9.0, +13.1] | 57% → 33% | 279 | 30 / 30 | no detectable effect |
| gemma-4-26b-a4b | BaseTen | 80.7 | 79.8 | −0.9 [−4.6, +1.8] | 97% → 97% | 580 | 30 / 30 | no detectable effect |

The 14 focused rows use exactly 30 eligible conversations and 900 fixed scripted turns per arm. Missing, malformed, or forfeited future turns fail all displayed criteria. Their intervals resample whole conversations. The three appended Gemini 3 rows use prospective fixed-denominator pools at Google's `minimal` reasoning floor; Gemini 3 does not guarantee complete thinking-off. Gemini 2.5 Flash is a separate prospective fixed-denominator extension with thinking explicitly disabled via `thinking_budget=0`. Its prespecified dot screen stopped at 10/6; a later control-only precision extension expanded no filler to 30 for the public benchmark estimate without reopening dot sampling. Laguna S 2.1 is a separate frozen prospective 30/30 campaign using the paid OpenRouter route to Poolside-hosted BF16 weights, with reasoning explicitly disabled. The two Qwen3.6 rows are separate exploratory fixed-denominator comparisons with native thinking disabled. Each reuses a frozen 30-conversation no-filler cohort and applies the prespecified 6→10→30 stopping rule only to the later dot arm; the arms are not contemporaneous or interleaved. Inkling Small adds a separate fixed-denominator BaseTen comparison: its 30-run `none` control is frozen from the none/low campaign and its later adaptive dot arm stopped at 30; the two arms are not interleaved. Gemma 4 26B A4B adds a separate fixed-denominator, temporally paired BaseTen comparison with 30 fresh contemporaneous conversations per arm and native thinking disabled. The other nine standard rows retain their original exploratory available-outcome pools and show no new confidence interval; `†` marks a selected historical estimate.

The original 17 rows retain their exploratory-screen order rather than being resorted after the n=30 refresh; the three Gemini 3 extensions are appended in their prespecified requested order, followed by Gemini 2.5 Flash, Laguna S 2.1, and the two Qwen3.6 configurations from their separate campaigns. The original eight focused rows and the corresponding thinking-off rows in `README.md` share the same frozen no-filler aggregates. The Gemini 3 rows and their `(minimal)` README rows likewise share one campaign aggregate. The appended Gemini 2.5 Flash row and its `(thinking off)` README row share a separate campaign aggregate; the chart's open control point uses all 30 no-filler conversations while its exploratory dot point remains at six. The appended Laguna S 2.1 row comes from its separate 30/30 campaign aggregate; both arms use `reasoning.enabled=false`, and its TTFAT is specific to the paid OpenRouter/Poolside BF16 route. The appended Qwen3.6 rows use BaseTen single-H100 vLLM 0.26 APC+MTP deployments: official BF16 weights for 27B and the official FP8 checkpoint for 35B-A3B. Their open control points use all 30 reusable no-filler conversations, while each dot point uses its mechanically selected stopped-stage sample. The Inkling Small row uses BaseTen for both arms, the frozen `none` arm's TTFAT, and the highest mechanically reached dot-stage artifact. In Inkling Small's primary 30-pair `none`/`low` campaign, 22/60 retained attempts ended short after a BaseTen HTTP 429 followed by the harness idle timeout (12 `none`, 10 `low`); these were serving failures rather than generated terminal calls, and fixed-denominator scoring retains them with missing future turns counted as failures. A post-hoc sensitivity check changing the 4 disputed `tool_use_correct` labels shifted any arm-level published rate by no more than 0.5 percentage points; official judgments remain unchanged. The Gemma 4 26B row and its README row share the fresh BaseTen no-filler arm; the screen TTFAT is that row configuration's observed-response P50. The Qwen3-8B primary row uses only its dedicated 30-per-arm BaseTen replacement cohort, including its no-filler TTFAT; no OpenRouter attempt is pooled into it. That endpoint serves official BF16 weights with vLLM automatic prefix caching, so its latency is specific to that configuration.

Flash Lite attempt-policy sensitivity: one no-filler attempt reached the harness idle timeout after eight turns. Under the frozen attempt-based rule, it remains in the primary pool and its missing future turns fail. Replacing it only for sensitivity analysis with the already-generated complete extra attempt moves the no-filler pass rate from 68.6% to 71.0% (+2.4 points) and strict completion from 60.0% to 63.3%. The primary estimates remain attempt-based and unchanged.

Nemotron Super is excluded from this screen because its available comparison holds +96 dots fixed and changes thinking mode, rather than estimating a filler effect. The thinking-off mode was not repaired: all 24 dot-treated attempts called `end_session` at turn 0. The separate thinking-on result was 91.7% over four judged conversations; a 6/6 BaseTen gate established completion only. Three Modal cells without automatic prefix caching (APC) also completed 6/6, while APC+MTP caused a distinct tool-execution collapse.

Each row reports one pooled turn-level P50 TTFAT from that row's no-filler configuration—not an arm-to-arm timing comparison.
<!-- N30_PRIMARY_END -->

## gpt-5.4 dose and placement follow-ups

| dots | conversations | pass rate | Δ vs no filler | cluster p | median TTFAT |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 90.3 | — | — | 677 ms |
| 24 | 8 | 95.4 | +5.1 | 0.0098 | 657 ms |
| 48 | 8 | 91.2 | +0.9 | 0.70 | 649 ms |
| 96 | 10 | 96.3 | +6.0 | 0.0072 | 658 ms |
| 192 | 8 | 97.5 | +7.2 | 0.0046 | 641 ms |

The observed curve is not monotone: the 48-dot cell returns near baseline. Its
cluster test does not detect a difference from zero dots, but that does not prove the
dip is noise. Sample medians remain in a 36 ms band; the 192-dot P95 is higher
(1990 vs 1546 ms), so “free” is too strong once tail latency, input cost, and context
consumption are considered.

Pattern and placement cells used eight conversations per new treatment and reused the
10-conversation no-filler baseline:

| cell | Δ vs no filler | cluster p | interpretation |
|---|---:|---:|---|
| 96 dots, suffix | +6.0 | 0.0072 | positive |
| 96 dashes, suffix | +7.2 | 0.0007 | positive; no direct dots-vs-dashes equivalence test |
| `the` ×96, suffix | +3.8 | 0.0546 | suggestive |
| 96 dots, prefix before current question | +7.2 | 0.0007 | positive |
| 96 dots, system prompt | +0.9 | 0.7370 | no detectable change |

Late current-user placements were positive while the system-prompt cell was not. This
argues against a position-independent prompt-lengthening account, but role, recency,
available preceding context, and cache treatment remain confounded with position.

## Where effects occur: exploratory turn and task-family analysis

About three quarters of the net +6.0-point gain comes from turns 12 and 15: 13 of
18 avoided failures. Both are tool-commitment moments where the baseline often asks
for information already established in history instead of calling the tool. Smaller
changes occur elsewhere, including one new failure at turn 25.

This pattern is consistent with filler affecting decisiveness over conversational
state, but does not demonstrate hidden computation. Tokenization, punctuation
pragmatics, recency, API routing, and decoding effects remain alternatives.

<!-- TURN_FAMILY_INSERT -->

<!-- TURN_FAMILY_START -->
### Cross-model descriptive decomposition

The HTML report shows two aligned 11×30 heatmaps: the strict-pass change at every scripted turn and the benefit-aligned contribution from missing turns. It also shows each task family's additive contribution to the 30-turn overall effect. No turn was selected because of its observed result.

A separate reviewer assigned every scripted turn to one behavioral family using only the benchmark specification—not the 11-model filler outcomes. The mapping was frozen before family-level computation, but after the pilot and primary overall results were known. This is retrospective and exploratory, not an outcome-naive preregistration.

| model | Grounded information (12) | Recommendation (4) | Tool preparation (6) | Tool commitment (5) | Boundary / closing (3) |
|---|---:|---:|---:|---:|---:|
| gpt-5.4 | −0.3 [−1.1, +0.6] | +0.0 [+0.0, +0.0] | −1.7 [−3.9, +0.6] | +32.0 [+21.3, +42.7] | +1.1 [+0.0, +3.3] |
| gpt-5.6-terra | −42.8 [−51.1, −33.6] | −1.7 [−5.0, +0.0] | −49.4 [−58.3, −40.0] | −4.0 [−14.7, +6.7] | −76.7 [−90.0, −60.0] |
| gpt-5.5 | +0.6 [+0.0, +1.4] | +0.0 [+0.0, +0.0] | +1.7 [+0.0, +3.9] | +10.0 [+5.3, +14.7] | +0.0 [+0.0, +0.0] |
| gpt-5.6-sol | +0.3 [+0.0, +0.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +20.0 [+20.0, +20.0] | +0.0 [+0.0, +0.0] |
| gemma-4-31b-it | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | −1.1 [−2.8, +0.0] | +12.0 [+6.7, +17.3] | +0.0 [+0.0, +0.0] |
| inkling | −11.9 [−20.6, −3.3] | +0.0 [+0.0, +0.0] | −22.2 [−31.1, −13.3] | −4.7 [−12.7, +3.3] | −21.1 [−38.9, −4.4] |
| qwen3-8b | −3.9 [−7.8, +1.1] | +5.8 [−1.7, +12.5] | +26.1 [+15.0, +37.2] | −24.0 [−35.3, −12.7] | +4.4 [−4.4, +13.3] |
| glm-5.2 | −1.4 [−2.5, −0.3] | +0.0 [+0.0, +0.0] | +1.1 [+0.0, +2.8] | −12.0 [−17.3, −6.7] | −1.1 [−3.3, +0.0] |
| gemini-3.5-flash | +3.1 [−1.9, +9.2] | +10.0 [+4.2, +16.7] | −5.0 [−13.3, +5.6] | +6.7 [−1.3, +17.3] | +5.6 [−2.2, +15.6] |
| gemini-3.5-flash-lite | −3.9 [−16.7, +9.2] | +13.3 [+4.2, +24.2] | −10.6 [−29.4, +8.3] | +20.0 [+3.3, +36.7] | −17.8 [−42.2, +6.7] |
| gemini-3.6-flash | −4.7 [−10.8, +0.3] | +0.0 [+0.0, +0.0] | −33.3 [−38.9, −27.8] | +6.7 [−4.0, +18.7] | −3.3 [−16.7, +8.9] |

Cells are fixed-denominator within-family dot-minus-control pass-rate points with pointwise, unadjusted 95% whole-conversation bootstrap intervals. The number in each header is the family turn count. No interval has simultaneous 95% coverage across the 55 model-family cells. In the HTML contribution matrix, each cell is this value multiplied by the family turn count / 30, and the five cells sum exactly to the primary overall point estimate.

Effects do not follow a universal family rule: positive and negative tool-commitment estimates both occur, and some near-zero overall effects contain offsetting family contributions.

Frozen zero-based turn mapping — Grounded information: 0, 1, 2, 3, 4, 13, 20, 21, 25, 26, 27, 28; Recommendation: 5, 6, 7, 8; Tool preparation: 9, 10, 14, 16, 22, 23; Tool commitment: 11, 12, 15, 17, 24; Boundary / closing: 18, 19, 29.

The companion artifact partitions every failure into a missing/post-abort turn or an observed judged failure. Long suffix bands can therefore reflect one early exit propagated through later fixed-denominator turns, not many independent semantic failures. These 330 turn cells and 55 family cells are descriptive decompositions, not treatment-by-turn or treatment-by-family interaction tests. Families differ in size and position. The whole-conversation intervals in fig 1 remain the primary inferential summary.
<!-- TURN_FAMILY_END -->

## Filler effects at two reasoning-effort settings

“Low” is the OpenAI API's internal reasoning-budget setting, not a description of
run quality. Holding reasoning effort fixed within each comparison produced:

| reasoning effort | no filler | +96 dots | dot-minus-control [95% conversation-bootstrap interval] | conversations per arm | P50 TTFAT, no filler → dots |
|---|---:|---:|---:|---:|---:|
| `none` | 90.2% | 95.2% | +5.0 [+3.0, +6.9] | 30 | 689 → 694 ms |
| `low` | 96.2% | 99.6% | +3.3 [+1.2, +5.0] | 8 | 1,091 → 1,131 ms |

Dots improved pass rate at both measured reasoning settings. The low-effort gain
was driven by the same turn-15 tool-commitment behavior identified in the original
pilot.

The descriptive difference between the effects is −1.7 points. It is **not an
interaction estimate**: the reasoning-off and low-effort slices were collected
separately, have unequal sample sizes, and did not randomize reasoning effort and
filler together in a joint 2×2 experiment. These results therefore do not establish
whether filler substitutes for or complements reasoning. That question requires a
new balanced factorial experiment.

## Failure modes and deployment implications

- The refreshed fixed-denominator GLM-5.2 and Terra estimates are reported in the
  primary table above; missing and post-abort turns are failures rather than omitted
  survivor outcomes.
- In the earlier hazard study, `gpt-5.6-terra` aborted 25/33 dot-treated attempts.
  Dashes reduced the observed abort rate to 3/12, but that small pattern comparison
  remains exploratory.
- `nemotron-super` with thinking disabled aborts at turn 0 on all 24 dot-treated
  attempts. Small thinking-enabled filler cells did not reproduce the abort, but
  were too small and negative to support a broad recommendation.
- `gpt-5.4-mini` is excluded from the master accuracy screen because strict
  completion at effort none is sparse: 0/17 without filler, 1/17 with dots, and
  3/14 with dashes. At effort medium, an adaptively stopped screen observed strict
  completion of 4/20 without filler versus 8/10 with dashes (fixed-table Fisher
  arithmetic `p=0.004111`, not calibrated for the stopping rule), while
  the 86.7→93.3 accuracy comparison is survivor-selected; see
  `docs/gpt-5.4-mini-abort-investigation-2026-07-20.md`.

The selected GPT-5.4 dash configuration has now been confirmed in a fixed,
prospective run on the pinned `gpt-5.4-2026-03-05` snapshot: **41 fresh
conversations per arm** (82 total), 92.44% without filler versus 95.37% with 96
trailing dashes, +2.93 points (95% CI +1.09 to +4.77). Strict completion was
41/41 and median TTFAT was 678 ms in both arms. The exploratory pilot was excluded.
This result validates that configuration; it does not establish dots–dashes
equivalence or transfer to other models.

The practical rule is model-specific validation with strict completion and aborts as
primary outcomes. If testing a responsive model, include dashes as a candidate pattern
because they performed at least as well as dots in the measured gpt-5.4 cells and had
fewer observed terra aborts. Do not infer equivalence or generalize the pattern without
a model-specific factorial screen.

## What this screen does and does not support

- Baseline score did not show a simple monotone relationship with the treatment
  estimate in the original screen, and no formal cross-model correlation or causal
  model was fit. The refreshed focused estimates should be read directly from the
  primary table rather than used to retrofit a new cross-model rule.
- The Qwen3 8B/14B/32B ladder does not show a consistent size gradient at `n=10`.
  This does not rule out depth effects in other controlled families.
- Positive, negative, and null estimates occur across dense and MoE rows. The
  heterogeneous provider and serving-stack comparison cannot isolate architecture,
  scale, training, or quantization.
- DeepSeek's positive arithmetic result in the source paper does not transfer to the
  different DeepSeek-chat-v3.1 checkpoint and conversational task here. The task and
  checkpoint both differ, so this is not a direct replication failure.

## Provenance

The original focused refresh contains exactly 480 eligible conversations: 30 per
arm for eight prespecified models. Three Gemini 3 extensions use prospective
fixed-denominator pools at Google's `minimal` reasoning floor; the separate Gemini
2.5 Flash extension explicitly disables thinking with `thinking_budget=0`. Dot-arm
top-ups and possible 30-per-arm promotion are governed by prespecified rules. Nine other standard
rows retain their original exploratory pools; the separate GPT-5.4 dash confirmation
contains 82 fresh conversations. Config-to-run manifests and the fixed-denominator
analyzers are in
`docs/filler-study-data/`. The self-contained rendered report is
`docs/filler-token-latent-scratchpad-study.html`, generated by
`scripts/build_filler_report.py`.

The harness was commit `3e9f805` plus the filler and Responses-routing work in the
current tree. Judge: `claude-opus-4-5` via Claude Agent SDK. Focused rates use fixed
900-turn denominators and whole-conversation bootstrap intervals; original follow-up
cells retain their explicitly exploratory analyses.
