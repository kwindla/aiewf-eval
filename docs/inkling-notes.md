# Inkling (Thinking Machines) — evaluation notes

Findings from evaluating `thinkingmachines/inkling` on BaseTen against
`aiwf_medium_context` (2026-07-15/16). Integration/wiring details are in
`docs/inkling-baseten-integration.md`; this doc is the "what we found."

## The model

- 975B-param MoE (~41B active), 1M context, reasons over text/image/audio.
  Released 2026-07-15. Controllable thinking effort via `reasoning_effort`.
- Runs serverless on **BaseTen** (`thinkingmachines/inkling`). NOT serverless on
  Together (needs a paid dedicated endpoint); Modal endpoint here serves GLM-5.
- **Inkling Small** (276B total / 12B active) was released on 2026-07-30 and is
  available through BaseTen's shared Model API as
  `thinkingmachines/inkling-small`. It is evaluated separately in the frozen
  30-`none`/30-`low` campaign under
  `ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/`.

## Headline

A strong, accurate model in the **wrong latency envelope** — **not voice-viable at
any effort**, like Claude Fable 5. On this benchmark, **thinking effort is a pure
latency cost with no accuracy gain.**

## Effort sweep (aiwf_medium_context)

`none`/`low`/`max` = 10 runs each; `minimal`/`medium`/`high`/`xhigh` = 2 runs.

| effort | runs | pass% (mean, range) | TTFT P50 | TTFT P95 | TTFT max | reasoning tok (med) |
|---|---|---|---|---|---|---|
| none | 10 | 94.7% (87–100) | 917ms | 1674ms | 3453ms | 0 (folded into answer) |
| minimal | 2 | 93.3% | 1688ms | — | — | 76 |
| low | 10 | 96.3% (93–100) | 1559ms | 3464ms | 4717ms | 66 |
| medium | 2 | 93.3% | 2026ms | — | — | 118 |
| high | 2 | 95.0% | 2231ms | — | — | 143 |
| xhigh | 2 | 93.3% | 2326ms | — | — | 176 |
| max | 10 | 93.7% (90–97) | 2486ms | 5954ms | 12779ms | 162 |

- Median TTFT climbs ~920ms → ~2490ms none→max; reasoning tokens 0 → ~160.
  **Pass rate is flat ~94–96% with no effort trend.** KB grounding is perfect
  throughout; every failure is tool-use/instruction.
- **The tail is the killer:** TTFT P95 1.7s → 3.5s → 6.0s (none→low→max), worst
  turn **12.8s at max**. Higher effort mainly fattens the latency tail.
- **`low` is the best config** (best mean pass, tightest floor, best tail among
  thinking levels) — but even `low` (~1.5s P50 / ~3.5s P95) is ~2× the ~700ms
  voice bar.

## Failure patterns (the interesting part)

Inkling's signature failure is **over-EAGER tool calls** — it fires functions
*prematurely*, before the golden flow's clarifying question. This is the **mirror
image of Claude Sonnet 5**, whose failures were over-*confirmation* (re-asking for
info it already had). KB grounding never fails.

- **Turn 16 ("I'm having trouble with the mobile app")** is the dominant failure
  and **worsens monotonically with effort: 5/10 → 8/10 → 10/10** (none/low/max).
  The model immediately fires `request_tech_support` with a vague description
  instead of asking "what's the problem?". **More thinking → more premature action.**
- **U-shape in total failures: none 16, low 11, max 19** — `none` is impulsive
  (extra premature *placeholder* calls at turns 10/14 that a little thinking fixes),
  `low` is the minimum, `max` over-thinks its way back into premature calls.

## Judging fairness (investigated and cleared)

We checked whether we unfairly *cascade*-penalize the turn after an eager call
(e.g. turn 17 penalized because `request_tech_support` already fired at 16).

**Verdict: not unfair — no change needed.** Quantified across all 30 runs:
**17/17 clean early calls (semantically-equivalent args) were correctly realigned**
(the expected turn was credited), **0 unfair cascades.** The only expected-turn
penalties (11) were on *wrong-args* early calls — the early call captured a
different/vaguer description and the required specific issue was **never recorded**
(model said "already submitted"). Those are legitimate: the required outcome
genuinely wasn't achieved. The judge already credits `instruction_following` on the
downstream turn and only docks `tool_use` — it is *not* blindly cascading.

Note on terminology: turn 17's failure is a **causal** consequence of the turn-16
over-eagerness, but it is **not a double-penalty** — turn 16 (premature action) and
turn 17 (specific issue never captured) are two distinct real failures with one
common cause. A special case to credit the downstream turn would *wrongly* reward
premature, low-quality tool calls, so we did not add one. The existing early/late
realignment (a judge-prompt rule, `claude_judge.py`) handles the clean case at 100%.

## Bottom line

Accurate but latency-disqualified for voice. If added to the leaderboard, a `low`
row (like Fable's `(low)`/`(default)`) is the fair representation. Revisit with
Inkling Small is now the direct latency-oriented follow-up. Its campaign uses
the same first-answer-token timing convention and separately measures `none`
and `low`, so the result should not be inferred from the full-size model's
effort sweep.
