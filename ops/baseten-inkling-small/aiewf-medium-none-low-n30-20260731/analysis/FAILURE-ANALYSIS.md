# Inkling Small raw completion and failure attribution

This additive analysis uses only the 60 runs named by `canonical.tsv` and their raw transcripts/run logs. It has no Claude-judge dependency.

## Conversation outcomes

| arm | strict complete | model abort | recovery end | BaseTen 429 + idle | observed / fixed turns |
|---|---:|---:|---:|---:|---:|
| none | 17/30 (56.7%) | 1/30 (3.3%) | 0/30 (0.0%) | 12/30 (40.0%) | 723 / 900 |
| low | 3/30 (10.0%) | 9/30 (30.0%) | 8/30 (26.7%) | 10/30 (33.3%) | 508 / 900 |

## Missing scheduled turns by immediate cause

| arm | model abort | recovery end | BaseTen 429 + idle | total missing |
|---|---:|---:|---:|---:|
| none | 1 | 0 | 176 | 177 |
| low | 126 | 110 | 156 | 392 |

## Focus-turn response patterns

### Turn 13

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 25 | 5 | `missing` 5; `text_only` 25 |
| low | 26 | 4 | `missing` 4; `text_only` 26 |

### Turn 14

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 25 | 5 | `missing` 5; `text_only` 18; `tool_only:submit_dietary_request` 7 |
| low | 26 | 4 | `missing` 4; `text_only` 9; `tool_only:submit_dietary_request` 17 |

### Turn 15

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 25 | 5 | `missing` 5; `text_only` 10; `tool_only:submit_dietary_request` 15 |
| low | 26 | 4 | `missing` 4; `text_only` 9; `tool_only:end_session` 9; `tool_only:submit_dietary_request` 8 |

### Turn 16

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 24 | 6 | `missing` 6; `text_only` 2; `tool_only:request_tech_support` 22 |
| low | 10 | 20 | `missing` 20; `tool_only:request_tech_support` 10 |

### Turn 17

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 22 | 8 | `missing` 8; `text_only` 8; `tool_only:request_tech_support` 14 |
| low | 9 | 21 | `missing` 21; `text_only` 6; `tool_only:request_tech_support` 3 |

### Turn 28

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 18 | 12 | `missing` 12; `text_only` 17; `tool_only:end_session` 1 |
| low | 3 | 27 | `missing` 27; `text_only` 3 |

### Turn 29

| arm | observed | missing | exact response patterns |
|---|---:|---:|---|
| none | 17 | 13 | `missing` 13; `tool_only:end_session` 17 |
| low | 3 | 27 | `missing` 27; `tool_only:end_session` 3 |

## Transition cross-tabs

### t14 to t15

- `none`: `t14=missing -> t15=missing` 5; `t14=text_only -> t15=text_only` 3; `t14=text_only -> t15=tool_only:submit_dietary_request` 15; `t14=tool_only:submit_dietary_request -> t15=text_only` 7.
- `low`: `t14=missing -> t15=missing` 4; `t14=text_only -> t15=text_only` 1; `t14=text_only -> t15=tool_only:submit_dietary_request` 8; `t14=tool_only:submit_dietary_request -> t15=text_only` 8; `t14=tool_only:submit_dietary_request -> t15=tool_only:end_session` 9.

### t16 to t17

- `none`: `t16=missing -> t17=missing` 6; `t16=text_only -> t17=tool_only:request_tech_support` 2; `t16=tool_only:request_tech_support -> t17=missing` 2; `t16=tool_only:request_tech_support -> t17=text_only` 8; `t16=tool_only:request_tech_support -> t17=tool_only:request_tech_support` 12.
- `low`: `t16=missing -> t17=missing` 20; `t16=tool_only:request_tech_support -> t17=missing` 1; `t16=tool_only:request_tech_support -> t17=text_only` 6; `t16=tool_only:request_tech_support -> t17=tool_only:request_tech_support` 3.

## Interpretation

`none` strictly completed 17/30 conversations; `low` completed 3/30. All 22 unended short runs were BaseTen 429-plus-idle serving failures, not generated terminal calls.

At turn 15, `low` generated 9 direct `end_session` calls. It also produced 8 recovery-terminal conversations, while `none` produced none. This localizes the low-effort completion collapse to generated closing behavior around required tool/recovery boundaries rather than the intended turn-29 close.

Missing future turns remain fixed-denominator benchmark failures, but this cause table should be used whenever those failures are attributed to model behavior versus serving. Synthetic recovery turn 30 is not a scored scheduled turn.

## Reproducibility

`FAILURE-ANALYSIS.json` records the SHA-256 of `canonical.tsv`, this analyzer, and every included transcript and run log. All paths are repository-relative.
