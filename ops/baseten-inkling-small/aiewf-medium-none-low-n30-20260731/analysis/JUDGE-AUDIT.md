# Inkling Small post-hoc judge sensitivity audit

This audit leaves the official judgments and aggregates unchanged. It applies one pinned alternate policy to four `tool_use_correct` labels, all in the `none` arm.

## Counterfactual policy

A generic request_tech_support call at turn 16, before the user identifies the location-map problem, is not credited as the expected specific support action. The premature generic call at turn 16 and the absent call after the specific turn-17 report are both tool-use errors.

## Official to counterfactual results

| arm | metric | official | counterfactual | delta |
|---|---|---:|---:|---:|
| none | Strict pass | 676/900 (75.111%) | 673/900 (74.778%) | -3 (-0.333 pp) |
| none | Any error | 224/900 (24.889%) | 227/900 (25.222%) | +3 (+0.333 pp) |
| none | Tool error | 214/900 (23.778%) | 218/900 (24.222%) | +4 (+0.444 pp) |
| low | Strict pass | 465/900 (51.667%) | 465/900 (51.667%) | +0 (+0.000 pp) |
| low | Any error | 435/900 (48.333%) | 435/900 (48.333%) | +0 (+0.000 pp) |
| low | Tool error | 432/900 (48.000%) | 432/900 (48.000%) | +0 (+0.000 pp) |

The exact fixed-denominator sensitivity is:

- `none`: strict pass -3/900 (-0.333 pp), any error +3/900 (+0.333 pp), tool error +4/900 (+0.444 pp).
- `low`: unchanged.

## Changed labels

| slot | arm | turn | official tool | counterfactual tool | strict change |
|---|---|---:|---:|---:|---|
| IS-18 | none | 16 | true | false | true → false |
| IS-18 | none | 17 | true | false | true → false |
| IS-47 | none | 16 | true | false | false → false |
| IS-47 | none | 17 | true | false | true → false |

## Scope and reproducibility

`JUDGE-AUDIT.json` pins the official policy identity, final aggregate and completion-marker hashes, exact transcript/judgment file hashes, and exact semantic row hashes. Any drift fails closed. The calculation does not recompute uncertainty intervals and must be read as a post-hoc sensitivity check, not an official relabeling.
