# Clever Chatter realtime comparison

Date: 2026-08-04

This is a standalone comparison. Clever Chatter is intentionally not added to the
public README table or the filler-effect report.

## Quality and reliability

Strict pass requires tool use, instruction following, and knowledge-base grounding;
turn-taking is reported separately. Confidence intervals resample whole conversations
(100,000 bootstrap samples), preserving within-conversation error clustering.

| Model | Configuration | N | Strict pass (95% CI) | Tool | Instruction | KB | Turn-taking | Retry turns | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gemini 2.5 Flash Native Audio | provider default | 10 | 88.7% (86.7–90.7%) | 268/300 | 267/300 | 300/300 | 298/300 | 2/300 (0.7%) | AI Studio |
| Gemini 3.1 Flash Live Preview | minimal thinking | 10 | 91.7% (89.0–94.0%) | 276/300 | 277/300 | 300/300 | 300/300 | 0/300 (0.0%) | AI Studio |
| Clever Chatter | minimal thinking | 15 | 59.1% (54.9–63.1%) | 365/450 | 306/450 | 439/450 | 218/450 | 124/450 (27.6%) | AI Studio |
| GPT-Realtime-2.1 | low reasoning | 30 | 97.2% (95.8–98.4%) | 876/900 | 876/900 | 900/900 | 899/900 | 0/900 (0.0%) | OpenAI |

A retry turn is a distinct benchmark turn with at least one logged empty-response,
no-response, or reconnection recovery event. It is not an additional denominator turn.

## Latency

| Model | Non-tool V2V P50 | Non-tool V2V max | Tool-turn V2V mean | Silence padding mean |
|---|---:|---:|---:|---:|
| Gemini 2.5 Flash Native Audio | 1504ms | 2944ms | 1761ms | 51ms |
| Gemini 3.1 Flash Live Preview | 1632ms | 5664ms | 3172ms | 100ms |
| Clever Chatter | 7104ms | 365179ms | 121813ms | 112ms |
| GPT-Realtime-2.1 | 1504ms | 4288ms | 1537ms | 75ms |

Latency is conditional on responses the timing analyzer could align. Recovery-heavy
runs can therefore have fewer latency observations than judged turns.

## Clever Chatter detail

README-compatible row (shown for comparison only; not added to the public leaderboard):

| Model | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clever-chatter (minimal thinking) | 59.1% | 365/450 | 306/450 | 439/450 | 218/450 | 7104ms | 365179ms | 121813ms | 112ms |

| Metric | Result | Error rate / interval |
|---|---:|---:|
| Strict pass | 266/450 (59.1%) | 54.9–63.1% whole-conversation bootstrap CI |
| Tool use | 365/450 | 18.9% error |
| Instruction following | 306/450 | 32.0% error |
| KB grounding | 439/450 | 2.4% error |
| Turn-taking | 218/450 | 51.6% error |
| Turns requiring recovery | 124/450 (27.6%) | no response: 217, reconnection: 1 events |
| Per-conversation strict score | median 18/30 | range 13–22/30 |
| Full-conversation completion | 14/25 (56.0%) | 11 invalid attempts |

Error hotspots below use zero-based benchmark turn indices.

| Turn | Strict failures | Tool failures | Instruction failures | KB failures | Turn-taking failures | Conversations needing recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 15/15 | 15/15 | 7/15 | 0/15 | 12/15 | 2/15 |
| 12 | 15/15 | 15/15 | 4/15 | 0/15 | 14/15 | 8/15 |
| 15 | 15/15 | 15/15 | 5/15 | 0/15 | 14/15 | 0/15 |
| 17 | 15/15 | 15/15 | 5/15 | 0/15 | 13/15 | 2/15 |
| 24 | 15/15 | 15/15 | 15/15 | 0/15 | 0/15 | 8/15 |
| 23 | 12/15 | 0/15 | 12/15 | 0/15 | 0/15 | 2/15 |
| 21 | 11/15 | 0/15 | 11/15 | 1/15 | 5/15 | 8/15 |
| 26 | 11/15 | 0/15 | 11/15 | 1/15 | 0/15 | 0/15 |
| 28 | 11/15 | 0/15 | 11/15 | 1/15 | 0/15 | 1/15 |
| 29 | 10/15 | 10/15 | 10/15 | 0/15 | 0/15 | 10/15 |

## Campaign completion

| Campaign | Full conversations | Finished attempts | Full-conversation completion | Invalid-attempt causes |
|---|---:|---:|---:|---|
| Clever Chatter additions | 14 | 25 | 56.0% | end_session at turn 10: 3, end_session at turn 13: 6, end_session at turn 5: 1, terminated stalled attempt after 15 turns: 1 |
| GPT-Realtime-2.1 | 30 | 30 | 100.0% | none |

The Clever Chatter quality cohort contains the original full smoke run plus the 14
campaign additions. Diagnostic runs before the frozen campaign are excluded.
The Gemini 2.5 row uses the newer frozen March retest rather than the older 86.0%
aggregate still shown in the public README.

## Conclusions

- Strict-pass ranking was: GPT-Realtime-2.1 97.2%, Gemini 3.1 Flash Live Preview 91.7%, Gemini 2.5 Flash Native Audio 88.7%, Clever Chatter 59.1%.
- Clever Chatter passed 266/450 turns (59.1%); its whole-conversation interval was 54.9–63.1%.
- Its largest recurring strict-error turn indices were 11 (15/15), 12 (15/15), 15 (15/15), 17 (15/15), 24 (15/15).
- Clever Chatter needed recovery on 124/450 turns (27.6%), compared with 0/900 for GPT-Realtime-2.1.
- The completion failures and frequent recovery turns make Clever Chatter materially
  less reliable in this harness than the three comparison models, independent of its
  strict content score.

## Thought-text probe

A standalone text-input probe used `send_realtime_input(text=...)` with
`include_thoughts=True`. Minimal and high thinking reported 69 and 209 reasoning
tokens respectively, but both returned zero thought-text parts. Clever Chatter is
therefore reasoning internally in these probes without exposing reasoning traces.
The text field itself triggered the model turn; no separate `turn_complete` argument
was required on the Gemini 3.1 realtime-input path.
Google's [Live API capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)
documents this Gemini 3.1-specific `send_realtime_input(text=...)` path.

## Provenance

- Gemini 2.5 and Gemini 3.1 use the frozen 2026-03-28 ten-run allowlists and aggregates.
- Clever Chatter and GPT-Realtime-2.1 were run on 2026-08-04 and judged with the
  repository's Claude v4 turn-taking judge.
- GPT-Realtime-2.1 is the current snapshot documented on OpenAI's
  [model page](https://developers.openai.com/api/docs/models/gpt-realtime-2.1).
- The JSON source for every number above is `comparison.json`.
- Rebuild with `./.venv/bin/python docs/clever-chatter-comparison-2026-08-04/analyze.py`.
  Add `--refresh-latency` to recompute the new audio timing aggregates.
