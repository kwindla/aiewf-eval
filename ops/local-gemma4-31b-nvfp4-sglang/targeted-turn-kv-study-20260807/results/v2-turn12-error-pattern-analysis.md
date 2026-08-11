# Turn 12 post-analysis: no-tool failure patterns

This is a descriptive analysis performed after the sealed confirmatory result. It does not alter the primary effect estimate or interval.

## Aggregate categories

| Outcome | BF16 | FP8 | FP8 - BF16 |
|---|---:|---:|---:|
| `correct_tool_and_arguments` | 3943/6144 (64.18%) | 2967/6144 (48.29%) | -15.89 pp |
| `no_tool_redundant_confirmation_or_question` | 814/6144 (13.25%) | 1552/6144 (25.26%) | +12.01 pp |
| `no_tool_false_claim_of_completion` | 1326/6144 (21.58%) | 1473/6144 (23.97%) | +2.39 pp |
| `no_tool_other` | 54/6144 (0.88%) | 149/6144 (2.43%) | +1.55 pp |

## Prefix ranking

| Prefix | Source | BF16 success | FP8 success | BF16-FP8 success | FP8-BF16 redundant | FP8-BF16 other |
|---|---|---:|---:|---:|---:|---:|
| `turn12-local_fp8-02` | local_fp8 | 89.3% | 28.9% | +60.4 pp | +40.6 pp | +3.9 pp |
| `turn12-local_bf16-03` | local_bf16 | 100.0% | 42.4% | +57.6 pp | +33.4 pp | +3.3 pp |
| `turn12-baseten_bf16-01` | baseten_bf16 | 99.8% | 32.2% | +67.6 pp | +29.9 pp | +5.5 pp |
| `turn12-baseten_bf16-02` | baseten_bf16 | 100.0% | 38.9% | +61.1 pp | +26.2 pp | +2.3 pp |
| `turn12-baseten_bf16-03` | baseten_bf16 | 50.6% | 26.2% | +24.4 pp | +22.5 pp | +1.0 pp |
| `turn12-local_bf16-02` | local_bf16 | 95.9% | 38.1% | +57.8 pp | +17.8 pp | +4.7 pp |
| `turn12-local_bf16-01` | local_bf16 | 30.7% | 28.9% | +1.8 pp | +13.7 pp | +0.2 pp |
| `turn12-local_fp8-04` | local_fp8 | 27.5% | 27.5% | +0.0 pp | +11.1 pp | -1.4 pp |
| `turn12-local_bf16-04` | local_bf16 | 35.2% | 40.6% | -5.5 pp | +9.8 pp | +2.3 pp |
| `turn12-baseten_bf16-04` | baseten_bf16 | 64.3% | 99.8% | -35.5 pp | -14.5 pp | -1.6 pp |
| `turn12-local_fp8-01` | local_fp8 | 29.1% | 76.0% | -46.9 pp | -23.0 pp | -1.2 pp |
| `turn12-local_fp8-03` | local_fp8 | 47.9% | 100.0% | -52.1 pp | -23.2 pp | -0.6 pp |

## Prefix provenance

| Source | Prefixes | BF16 success | FP8 success | BF16-FP8 success | BF16 redundant | FP8 redundant |
|---|---:|---:|---:|---:|---:|---:|
| baseten_bf16 | 4 | 78.7% | 49.3% | +29.4 pp | 6.1% | 22.1% |
| local_bf16 | 4 | 65.4% | 37.5% | +27.9 pp | 9.8% | 28.4% |
| local_fp8 | 4 | 48.4% | 58.1% | -9.7 pp | 23.9% | 25.2% |

The provenance interaction is large and reverses sign. Across the eight BF16-based prefixes, BF16 success exceeds FP8 success by 28.7 points. Across the four local-FP8 prefixes, FP8 success exceeds BF16 success by 9.7 points. The preregistered +15.9-point bank average is therefore conditional on a bank containing eight BF16-based and four FP8-origin histories; it is not an on-policy deployment estimate. The BaseTen histories also came from BF16 weights/KV plus MTP, not merely the local model with a different KV dtype.

## Concentration and observable prefix qualities

Four prefixes contribute 666 of the net 738 additional redundant-question labels (90.2%). Nevertheless, nine of twelve prefixes have a positive redundant-label shift, and the aggregate change is stable across the four 128-seed quartiles: +12.2 pp, +12.5 pp, +11.5 pp, +11.8 pp.

All twelve prefixes have the same user messages, operational state, immediately preceding successful tool call, tool result, and target request. Their prompt lengths span only 13,956–14,051 tokens. The variable content is earlier assistant wording and generated call IDs.

One post-hoc wording association is visible: the four histories whose name acknowledgement begins `Nice to meet you, Jennifer!` shift from 4.3% BF16 redundant to 34.1% FP8 redundant (+29.8 pp), versus +3.1 pp for the other eight. This phrase does not isolate a cause: one non-`Nice to meet you` history also has a large positive shift, and the twelve histories differ at many earlier assistant turns.

The source-origin reversal is the more consequential prefix property. It is compatible with an on-policy or history-manifold effect: each KV path can be more stable on histories it helped generate. With only four selected prefixes per source, source, wording, deployment, and selection are confounded, so this is a diagnosis and a reason to narrow the claim—not proof of that mechanism.

## Paired category transitions

The dominant paired movement is correct BF16 tool call to FP8 redundant question: 1031 cases, versus 472 in the reverse direction (net +559). False-completion to redundant-question transitions add a net +176. Thus most of the redundant-question increase is a loss of correct calls, although some is relabeling among no-tool failure styles.

## Redundant-confirmation subtypes

| Descriptive subtype | BF16 | FP8 | FP8 - BF16 |
|---|---:|---:|---:|
| `identity_or_ownership_reconfirmation` | 424 | 888 | +464 |
| `content_or_intent_reconfirmation` | 263 | 429 | +166 |
| `authorization_reconfirmation` | 42 | 68 | +26 |
| `post_action_claim_then_followup` | 44 | 82 | +38 |
| `other_question` | 41 | 85 | +44 |

## Other no-tool subtypes

`no_tool_other` is a residual mechanical category, not one coherent behavior.

| Descriptive subtype | BF16 | FP8 | FP8 - BF16 |
|---|---:|---:|---:|
| `future_action_promise` | 31 | 77 | +46 |
| `textual_pseudo_tool_call` | 13 | 52 | +39 |
| `uncaught_false_completion` | 10 | 20 | +10 |
| `other_acknowledgement` | 0 | 0 | +0 |

The residual category rises from 54 to 149 cases (2.76x). Future-action promises and textual pseudo-calls contribute 85 of the net 95 additional cases (89.5%). Four prefixes contribute 89 of that net change. Correct-BF16 to FP8-other transitions number 116, versus 31 in reverse (net +85).

## Decision-boundary diagnostic

All twelve prefixes have one unique normalized four-message suffix after generated call IDs are removed. Across prefixes, the BF16-FP8 change in the teacher-forced first `<|tool_call>` margin against the best ordinary-assistant alternative has Pearson r=0.931 and Spearman rho=0.909 with the behavioral success difference. Its post-hoc Pearson association with the redundant-confirmation difference is r=0.810.

The no-tool answers often preserve the correct name, topic, function name, and arguments. The likely failure boundary is therefore choosing structured tool-call syntax versus ordinary assistant prose, not simply forgetting required state. Once prose wins, the model falls into familiar confirmation, promise, narration, or completion templates.

These associations are descriptive mechanism diagnostics, not additional confirmatory tests. The subtype rules are a post-hoc mechanical audit, not independently human-validated semantic labels.
