# Turn-12 redundant-confirmation analysis

Date: 2026-08-09

## Bottom line

Yes: the probability of redundant confirmation at Turn 12 is sensitive to the
tested KV representation.  On the balanced paired replay, unit-scaled FP8 E4M3
KV produced 1,332 redundant confirmations in 4,800 completions (27.75%), versus
944/4,800 (19.67%) with BF16 KV.  The FP8-minus-BF16 difference is **+8.08
percentage points**; a post-hoc history-cluster bootstrap interval is **+5.56
to +10.60 points**.

This is a narrow deployment-level causal claim: same Gemma 4 NVFP4 checkpoint,
same frozen token prefix, same seed, same serving image and settings, with only
`--kv-cache-dtype` changed.  It is not yet a universal claim about FP8, other
quantizers, or general model intelligence.

The cross-model census supports a broader behavioral interpretation.  Turn 12
is a tool-commitment boundary: the user has already supplied the name and a new
suggestion, so the correct next response is a tool call.  Many otherwise strong
configurations instead ask again for the name, ownership, content, or permission.
That behavior can change sharply with KV dtype, reasoning mode, or filler while
the task facts remain unchanged.

## The direct KV result

| Frozen history origin | BF16 redundant | FP8 redundant | FP8 - BF16 |
|---|---:|---:|---:|
| Local BF16 | 469/2,400 (19.54%) | 688/2,400 (28.67%) | +9.13 pp |
| Local FP8 | 475/2,400 (19.79%) | 644/2,400 (26.83%) | +7.04 pp |
| Balanced mixture | 944/4,800 (19.67%) | 1,332/4,800 (27.75%) | **+8.08 pp** |

The history-origin-specific post-hoc cluster intervals are +5.50 to +12.75
points on BF16-origin histories and +3.54 to +10.54 on FP8-origin histories.
The direction therefore does not depend on which deployment produced the
history.  The preregistered success analysis likewise found a BF16 advantage
in both strata and did not establish a history-origin interaction.

FP8's 388 extra redundant confirmations account numerically for 88.8% of its
437-call success deficit.  Other broad categories barely move: false-completion
claims are 24.9% under BF16 and 24.6% under FP8; `no_tool_other` is 2.25% and
3.79%, respectively.

### What kind of confirmation changes?

| Mechanical subtype | BF16 | FP8 | FP8 excess |
|---|---:|---:|---:|
| Identity or ownership reconfirmation | 572 | 843 | **+271** |
| Content or intent reconfirmation | 287 | 345 | +58 |
| Authorization reconfirmation | 35 | 92 | +57 |
| Other question | 21 | 30 | +9 |
| Claimed action, then follow-up question | 29 | 22 | -7 |

Identity/ownership questions explain 69.8% of the 388-case increase.  The
dominant failure is therefore not forgetting the requested topic.  It is
behaving as if a previously supplied slot—usually Jennifer Smith or ownership
of the second suggestion—needs to be collected again before acting.

## Version-mixing and other confounds

The specific concern that one replay arm received history from a different
model version is ruled out as an explanation for the BF16/FP8 contrast:

- Both inference arms loaded `RedHatAI/gemma-4-31B-it-NVFP4` at revision
  `edafdf3dcaef23ff76f75b91edd6a4a975a399cf`.
- Both used SGLang image
  `sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2`.
- A mechanical comparison of the retained server plans finds only one launch
  command difference: `--kv-cache-dtype fp8_e4m3` versus `bfloat16`.
- All 300 frozen prefixes had identical token IDs across arms, and all 9,600
  planned completion cells passed the combined integrity audit.
- Source transcript and log hashes are frozen in the snapshot manifest.  No
  generated response from one arm is used by the other.
- The 150 BF16-origin and 150 FP8-origin histories are analyzed separately.
  FP8 raises redundant confirmation in both, so history provenance cannot
  produce the observed sign.
- Separate containers were destroyed at dtype transitions; no live KV state
  was converted, transferred, or reused across arms.

Even a hypothetical foreign-model frozen prefix would not confound the
within-prefix BF16/FP8 contrast, because both arms receive that exact same
prefix.  It would affect which history population the result generalizes to.
Here, the population is explicitly the balanced census of the two local
150-run campaigns, and the origin-stratified result is stable.

Remaining limitations are real but different:

1. The FP8 arm uses unit K/V scales because this checkpoint has no calibrated
   KV-cache scaling scheme.  The result applies to this FP8 implementation.
2. Changing KV dtype can select dtype-specific numerical code paths.  The
   experiment identifies the deployed KV-path effect, not a decomposition into
   rounding error versus kernel implementation.
3. Histories are sampled from one benchmark and one turn.  History-level
   treatment effects are heterogeneous, although the 300-history average is
   positive and both origin strata agree.
4. The confirmation subtypes are regex-based descriptions.  The broader
   success/failure scorer passed a 99.67% validation gate on 900 historical
   Gemma target turns; subtype labels should still be treated as descriptive.

## Cross-model census

The repository contains 4,597 raw observations with the canonical Turn-12
prompt.  The strict descriptive view retains 3,177 complete 30-turn,
fully-judged, no-filler conversations spanning 80 exact reported model names.
It does not merge aliases, providers, model revisions, or reasoning settings.
The Turn-11-eligible sensitivity subset contains 2,412 conversations where the
first suggestion was correctly submitted before Turn 12.

This census is frozen through August 9. The subsequently completed Muse
Glimmer campaign is reported separately below so its selected N=30 cohort is
not mixed with earlier tuning and serving probes.

Across the strict view, Turn-12 outcomes are:

| Outcome | Count | Rate |
|---|---:|---:|
| Correct tool and semantically correct arguments | 2,296 | 72.3% |
| Redundant confirmation/question | 543 | 17.1% |
| False claim of completion | 220 | 6.9% |
| Other no-tool response | 94 | 3.0% |
| Other tool failure | 24 | 0.8% |

The aggregate is only an inventory: the repository intentionally contains
many repeated experiments, so it must not be read as a population-weighted
model leaderboard.

### Representative standard-prompt configurations

| Configuration | N | Turn-12 correct | Redundant confirmation | False completion |
|---|---:|---:|---:|---:|
| Kimi K2.6, BaseTen, reasoning none | 36 | 100% | 0% | 0% |
| GLM-5.2, BaseTen, reasoning none | 30 | 100% | 0% | 0% |
| Qwen3.6 27B, thinking off | 30 | 100% | 0% | 0% |
| Gemma 4 31B, thinking on | 19 | 100% | 0% | 0% |
| GPT-5.4 snapshot, low reasoning | 10 | 100% | 0% | 0% |
| GPT-5.6 Terra, medium reasoning | 7 | 100% | 0% | 0% |
| GPT-5.5, reasoning none | 30 | 83.3% | 16.7% | 0% |
| Gemma 4 31B, Lilac, thinking off | 40 | 40.0% | 32.5% | 27.5% |
| Gemma 4 31B exact-name aggregate, vLLM, thinking off (mixed deployments) | 452 | 39.8% | 25.9% | 32.7% |
| GPT-5.4 snapshot, reasoning none | 41 | 4.9% | 80.5% | 0% |
| GPT-5.6 Terra, reasoning none | 30 | 3.3% | 83.3% | 0% |
| GPT-5.6 Sol, reasoning none | 30 | 0% | 100% | 0% |
| Gemma 4 26B A4B, thinking off | 59 | 0% | 100% | 0% |

The mixed Gemma exact-name row includes BaseTen and both local KV deployments;
it is intentionally not used to estimate the KV effect.  The paired replay
above does that.  It is included here only to show the raw on-policy error mix.

## Patterns

### 1. “Good-model failure” is directionally right, but not universal

Using at least 10 conversations and at least 90% mean overall judged accuracy
as a transparent definition of a good reported-model row, redundant
confirmation accounts for 315/572 (55.1%) of Turn-12 failures.  It is tied for
or is the largest failure category in 17 of the 26 such rows that have any
Turn-12 failures.

At a 95% quality threshold, it is 181/390 (46.4%), almost tied with false
completion at 177/390; that false-completion count is dominated by local Gemma.
At 97%, most configurations are perfect on Turn 12 and only 31 failures remain,
so percentages become unstable.  Thus redundant confirmation is a common
residual error of capable models, not an inevitable signature of capability.

There is no simple monotone quality relationship.  Across 68 exact-name rows
with N>=10, the model-level Spearman correlation between overall judged quality
and raw redundant-confirmation rate is -0.34.  This statistic is descriptive
and confounded by modality and configuration.  The useful pattern is
concentration: many strong configurations have zero errors, while a distinct
subset has a sharp Turn-12 confirmation mode.

### 2. Reasoning mode is the strongest recurring modifier

Within several exact model names, low or medium reasoning removes the mode:

- GPT-5.4 snapshot: reasoning none has 33/41 redundant confirmations; low and
  medium have 0/10 each.
- GPT-5.6 Luna: none has 6/18; low and medium have 0/11 and 0/10.
- GPT-5.6 Terra: none has 25/30; low and medium have 0/6 and 0/7.
- Lilac Gemma 4 31B: thinking off has 13/40 redundant and 11/40 false
  completions; thinking on has 0/19 of either and 19/19 correct calls.

These are not all contemporaneous randomized comparisons, but the repeated
direction across families is strong.  The model usually has the necessary
information; added inference computation changes whether it asks or acts.

### 3. Content-free filler can move the same boundary—either way

Selected historical arms show large reductions: GPT-5.4 reasoning-none moves
from 28/30 redundant without filler to 3/10 with 96 dots; GPT-5.6 Sol moves
from 30/30 to 0/10; GPT-5.6 Terra moves from 25/30 to 0/17 among complete dot
runs; and Lilac Gemma 4 thinking-off moves from 13/40 to 0/10.

This is not a universal remedy.  Examples in the opposite direction include
Kimi K2.6 (0/36 to 1/6), GLM-5.2 (0/30 to 2/10), and one Qwen3 8B route (0/39
to 3/6).  Terra's dot arm also has a severe early-abort hazard, so conditioning
on complete conversations makes its semantic improvement look safer than the
deployment actually is.

The filler and reasoning patterns reinforce the same interpretation as the KV
result: Turn 12 can sit near a next-token action/question boundary.  Small
changes to numerical representation or available inference computation can
move probability mass between a structured tool call and cautious prose.

### 4. The mode is family- and deployment-specific

- Kimi K2.6, GLM-5.1/5.2, Qwen3.6 27B, and several Claude configurations are
  essentially clean at Turn 12 in the retained standard runs.
- Thinking-off Gemma 4 is repeatedly vulnerable, split between redundant
  questions and false claims of completion.  Thinking-on Gemma is clean.
- Reasoning-none GPT-5.4 and GPT-5.6 configurations often fail almost purely by
  redundant confirmation; small reasoning budgets remove it.
- Realtime/audio models are mixed and frequently arrive at Turn 12 with a
  different prior-history success rate.  They should not be pooled with text
  models when making causal claims.
- Comparing a BF16 model to a different FP8 checkpoint across families is not a
  quantization experiment.  The local crossed Gemma replay is the only clean
  KV-dtype contrast in this analysis.

### 5. Muse Glimmer is an extreme new example

The selected `muse-glimmer-30b` N=30 campaign (thinking high, GGUF, Q8_0 KV,
DFlash draft length 15) produced only 3/30 correct Turn-12 calls. The other
27/30 responses were redundant confirmations—90% of all conversations and
100% of its Turn-12 failures. Mechanical subtyping gives 14 identity/ownership
reconfirmations, nine other confirmation questions, three content/intent
reconfirmations, and one authorization reconfirmation.

Only 21/30 Glimmer conversations correctly completed the first suggestion at
Turn 11, but this does not explain the Turn-12 classification: every observed
Turn-12 failure explicitly asked the user to repeat or reconfirm information
instead of issuing the required tool call. Glimmer therefore strengthens the
cross-model conclusion that redundant confirmation can be the dominant local
failure of an otherwise coherent model, while its 84.9% overall score places
it below the report's >=90% operational “good-model” threshold.

## Interpretation and next test

The best current statement is:

> In Gemma 4 31B NVFP4 on this benchmark, unit-scaled FP8 E4M3 KV causally
> increases the probability of a redundant Turn-12 confirmation relative to
> BF16 KV on identical frozen prefixes.  The increase is primarily redundant
> recollection of already-known identity/ownership information.

The finding is interesting precisely because the same error is a common
residual failure in a subset of otherwise good models and is strongly altered
by reasoning/filler settings.  A mechanistic follow-up should record the
teacher-forced logit margin between the first tool-call token and prose/question
alternatives under BF16, unit-scaled FP8, and a calibrated-FP8 implementation,
then test additional tool-commitment turns.  That would separate this specific
serving path from a broader claim about KV quantization.

## Artifacts

- `analyze.py`: reproducible repository-wide census.
- `analyze_kv_redundant.py`: reproducible post-hoc paired-replay analysis.
- `analyze_glimmer.py`: selected Muse Glimmer N=30 classification.
- `results/all-turn12-rows.csv`: run-level classifications and text.
- `results/model-summary-standard.csv`: exact reported-name summaries.
- `results/eligible-model-summary-standard.csv`: Turn-11-eligible sensitivity.
- `results/configuration-summary-standard.csv`: reasoning/service splits.
- `results/configuration-summary-with-fillers.csv`: descriptive filler splits.
- `results/redundant-subtypes-standard.csv`: cross-model subtype counts.
- `results/analysis.json`: scope and aggregate checks.
- `results/kv-redundant-analysis.json`: direct KV counts, subtype shifts, and
  history-cluster intervals.
- `results/glimmer-canonical.json`: selected Glimmer counts and subtypes.
