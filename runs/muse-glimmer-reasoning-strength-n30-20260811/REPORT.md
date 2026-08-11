# Muse Glimmer reasoning-strength sweep

## Result

`low` is the best operating point on this benchmark. It has the highest observed strict pass rate, but the accuracy differences are unresolved: low minus high is +1.00% (conversation-cluster bootstrap 95% CI -0.67% to +2.67%). `low` nevertheless uses 34.7% fewer mean completion tokens than `high` and cuts P95 answer latency by 59.8%.

The exploratory redundant-confirmation result points in the same operational direction but is not conclusive. Across the two consecutive suggestion turns, low minus xhigh is -15.00% (conversation-cluster bootstrap 95% CI -31.67% to +1.67%). Turn 12 alone remains highly failure-prone at every strength and shows no monotonic strength effect.

## Accuracy and latency

| Strength | Strict pass | 95% CI | Completion tok mean | Scripted TTFAT P50 / P95 | Reasoning delay P50 / P95 |
|---|---:|---:|---:|---:|---:|
| low | 775/900 (86.1%) | 84.9%–87.3% | 145.3 | 231/3105 ms | 30/1906 ms |
| medium | 769/900 (85.4%) | 84.6%–86.2% | 192.3 | 236/5958 ms | 30/4456 ms |
| high | 766/900 (85.1%) | 83.9%–86.2% | 222.6 | 236/7719 ms | 30/6481 ms |
| xhigh | 764/900 (84.9%) | 83.4%–86.2% | 220.2 | 234/7767 ms | 30/6147 ms |

## Pairwise strict-rate differences

- `low_minus_medium`: +0.67% (conversation-cluster bootstrap 95% CI -0.78% to +2.11%)
- `low_minus_high`: +1.00% (conversation-cluster bootstrap 95% CI -0.67% to +2.67%)
- `low_minus_xhigh`: +1.22% (conversation-cluster bootstrap 95% CI -0.56% to +3.11%)
- `medium_minus_high`: +0.33% (conversation-cluster bootstrap 95% CI -1.11% to +1.78%)
- `medium_minus_xhigh`: +0.56% (conversation-cluster bootstrap 95% CI -1.00% to +2.22%)
- `high_minus_xhigh`: +0.22% (conversation-cluster bootstrap 95% CI -1.56% to +2.00%)

## Redundant confirmation on the two suggestion turns

This exploratory composite counts redundant confirmations on scripted Turns 11 and 12. Its interval resamples whole conversations, preserving dependence between the two turns.

| Strength | Turn 11 redundant | Turn 12 redundant | Combined | Combined 95% CI |
|---|---:|---:|---:|---:|
| low | 6/30 (20.0%) | 21/30 (70.0%) | 27/60 (45.0%) | 35.0%–55.0% |
| medium | 9/30 (30.0%) | 24/30 (80.0%) | 33/60 (55.0%) | 43.3%–66.7% |
| high | 10/30 (33.3%) | 24/30 (80.0%) | 34/60 (56.7%) | 45.0%–68.3% |
| xhigh | 13/30 (43.3%) | 23/30 (76.7%) | 36/60 (60.0%) | 46.7%–71.7% |

Pairwise combined-rate differences:

- `low_minus_medium`: -10.00% (conversation-cluster bootstrap 95% CI -25.00% to +5.00%)
- `low_minus_high`: -11.67% (conversation-cluster bootstrap 95% CI -28.33% to +5.00%)
- `low_minus_xhigh`: -15.00% (conversation-cluster bootstrap 95% CI -31.67% to +1.67%)
- `medium_minus_high`: -1.67% (conversation-cluster bootstrap 95% CI -18.33% to +15.00%)
- `medium_minus_xhigh`: -5.00% (conversation-cluster bootstrap 95% CI -21.67% to +11.67%)
- `high_minus_xhigh`: -3.33% (conversation-cluster bootstrap 95% CI -20.00% to +13.33%)

## Turn 12 redundant confirmation

These are on-policy outcomes: each strength generated its own preceding history. They measure the production configuration, not a same-prefix direct effect.

| Strength | Turn 11 direct tool success | Correct call | Redundant confirmation | Other outcome |
|---|---:|---:|---:|---:|
| low | 24/30 | 9/30 | 21/30 (70.0%) | 0/30 |
| medium | 21/30 | 6/30 | 24/30 (80.0%) | 0/30 |
| high | 20/30 | 6/30 | 24/30 (80.0%) | 0/30 |
| xhigh | 17/30 | 6/30 | 23/30 (76.7%) | 1/30 |

Turn 12 stratified by whether the scripted Turn 11 response directly made its expected tool call:

| Strength | Turn 11 direct call? | N | Turn 12 correct | Turn 12 redundant | Other |
|---|---:|---:|---:|---:|---:|
| low | true | 24 | 7/24 | 17/24 | 0/24 |
| low | false | 6 | 2/6 | 4/6 | 0/6 |
| medium | true | 21 | 5/21 | 16/21 | 0/21 |
| medium | false | 9 | 1/9 | 8/9 | 0/9 |
| high | true | 20 | 5/20 | 15/20 | 0/20 |
| high | false | 10 | 1/10 | 9/10 | 0/10 |
| xhigh | true | 17 | 4/17 | 12/17 | 1/17 |
| xhigh | false | 13 | 2/13 | 11/13 | 0/13 |

Pairwise redundant-rate differences:

- `low_minus_medium`: -10.00% (bootstrap 95% CI -33.33% to +10.00%)
- `low_minus_high`: -10.00% (bootstrap 95% CI -33.33% to +10.00%)
- `low_minus_xhigh`: -6.67% (bootstrap 95% CI -30.00% to +16.67%)
- `medium_minus_high`: +0.00% (bootstrap 95% CI -20.00% to +20.00%)
- `medium_minus_xhigh`: +3.33% (bootstrap 95% CI -16.67% to +23.33%)
- `high_minus_xhigh`: +3.33% (bootstrap 95% CI -16.67% to +23.33%)

Redundant-confirmation subtypes:

| Subtype | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| authorization_reconfirmation | 5 | 5 | 1 | 2 |
| content_or_intent_reconfirmation | 1 | 4 | 3 | 5 |
| identity_or_ownership_reconfirmation | 11 | 7 | 12 | 9 |
| other_question | 4 | 8 | 8 | 7 |

## Controls and interpretation

- The official supported values are exactly `low`, `medium`, `high`, and `xhigh`; the model card says reasoning cannot be disabled.
- The live embedded-template audit proves absent/default equals `high`, and that top-level `reasoning_effort=none` and `enable_thinking=false` are render no-ops. `none` and `minimal` merely render unsupported literal labels, so they were not promoted to experimental arms.
- The benchmark system instruction is unchanged and appears exactly once in every audited render. The only intended arm difference is `chat_template_kwargs.reasoning_strength`.
- These are independent on-policy trajectories, balanced and interleaved by arm. Pairwise intervals are descriptive fixed-sample comparisons with no multiplicity adjustment; a same-prefix replay would answer a different question.
- Completion tokens include the model's hidden reasoning and answer output; the local backend does not expose a separate thinking-token count. TTFAT is time to the first answer/tool token, while raw TTFT is time to the first reasoning token.

## Per-turn strict passes

| Turn | low | medium | high | xhigh |
|---:|---:|---:|---:|---:|
| 0 | 30/30 | 30/30 | 30/30 | 30/30 |
| 1 | 30/30 | 30/30 | 30/30 | 30/30 |
| 2 | 30/30 | 30/30 | 30/30 | 30/30 |
| 3 | 30/30 | 30/30 | 30/30 | 30/30 |
| 4 | 30/30 | 30/30 | 30/30 | 30/30 |
| 5 | 30/30 | 30/30 | 30/30 | 30/30 |
| 6 | 30/30 | 30/30 | 30/30 | 30/30 |
| 7 | 30/30 | 30/30 | 30/30 | 30/30 |
| 8 | 30/30 | 30/30 | 30/30 | 29/30 |
| 9 | 30/30 | 30/30 | 30/30 | 30/30 |
| 10 | 30/30 | 30/30 | 30/30 | 30/30 |
| 11 | 24/30 | 21/30 | 20/30 | 17/30 |
| 12 | 9/30 | 6/30 | 6/30 | 6/30 |
| 13 | 30/30 | 30/30 | 29/30 | 30/30 |
| 14 | 30/30 | 30/30 | 30/30 | 30/30 |
| 15 | 2/30 | 1/30 | 2/30 | 2/30 |
| 16 | 30/30 | 30/30 | 30/30 | 28/30 |
| 17 | 1/30 | 1/30 | 1/30 | 3/30 |
| 18 | 30/30 | 30/30 | 30/30 | 30/30 |
| 19 | 27/30 | 30/30 | 28/30 | 29/30 |
| 20 | 30/30 | 30/30 | 30/30 | 30/30 |
| 21 | 27/30 | 29/30 | 27/30 | 28/30 |
| 22 | 30/30 | 30/30 | 30/30 | 30/30 |
| 23 | 30/30 | 30/30 | 30/30 | 30/30 |
| 24 | 0/30 | 0/30 | 0/30 | 0/30 |
| 25 | 29/30 | 29/30 | 30/30 | 28/30 |
| 26 | 29/30 | 29/30 | 30/30 | 30/30 |
| 27 | 30/30 | 29/30 | 30/30 | 30/30 |
| 28 | 30/30 | 29/30 | 30/30 | 30/30 |
| 29 | 27/30 | 25/30 | 23/30 | 24/30 |
