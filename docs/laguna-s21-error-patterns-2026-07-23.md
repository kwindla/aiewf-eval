# Laguna S 2.1 error-pattern summary

**Scope:** 30 no-filler and 30 `+96 dots` conversations on
`aiwf_medium_context`, using the paid OpenRouter route to Poolside-hosted BF16
weights with reasoning disabled. Each conversation has 30 scripted turns.
Missing turns count as failures in the primary fixed-denominator result.

## Main finding

Laguna's dominant failure is **action gating and conversational state reuse**,
not malformed tool calls. The model commonly identifies the right action and
even the correct arguments, but asks for the user's name, preference, or
confirmation again instead of calling the tool. It also usually produces an
appropriate spoken goodbye without calling `end_session`.

| required action | turn | no filler: tool-correct | +96 dots: tool-correct | recurring failure |
|---|---:|---:|---:|---|
| submit first session suggestion | 11 | 23/30 | 11/30 | asks for more detail or confirmation |
| submit second session suggestion | 12 | 25/30 | 23/30 | delays the first call and misses the second |
| submit vegan dietary request | 15 | 13/30 | 7/30 | asks again for the known name or preference |
| request mobile-app support | 17 | 18/30 | 9/30 | asks again for the known name |
| vote for the vibe-coding session | 24 | 7/30 | 1/30 | finds session ID `936902`, then asks for the known name |
| end the conversation | 29 | 4/30 | 4/30 | gives a friendly farewell but omits `end_session` |

Only 5/30 no-filler conversations executed all five non-closing task actions
correctly; none of the dot-treated conversations did. No conversation in
either arm executed all five actions and the closing tool correctly.

The tool failures were overwhelmingly **omissions**, not invalid payloads. All
79 observed no-filler tool failures contained no tool call. In the dots arm,
119 of 122 observed tool failures contained no call; the other three were
parseable but mistimed calls. There was no recurring malformed-JSON or
tool-schema failure.

## Where the errors cluster

The six required-action turns above account for 90 of 130 fixed-denominator
no-filler errors (69%) and 126 of 150 dot-arm errors (84%). Every conversation
had at least one error, with a median of 3.5 failed turns without filler and
4.5 with dots.

| arm | fixed any-error turns | observed judged failures | missing scripted turns | observed tool failures | observed instruction failures | observed KB failures |
|---|---:|---:|---:|---:|---:|---:|
| no filler | 130/900 | 86 | 44 | 79 | 57 | 7 |
| +96 dots | 150/900 | 134 | 16 | 122 | 112 | 0 |

Dimension counts overlap. A missing scripted turn contributes one any-error
failure and fails every displayed dimension.

Grounding was otherwise strong. The seven observed no-filler KB failures were
localized to session recommendations and follow-up references: wrong session
dates on turns 7–8, one incorrect speaker attribution on turn 20, and two
wrong/ambiguous “second one” resolutions on turn 21. The dots arm had no
observed KB-grounding failures. Its published KB error rate is nonzero only
because missing turns fail under the fixed-denominator policy.

## Serving stalls and completion labels

Four conversations—two per arm—hit the harness's 45-second idle timeout with
no explicit provider error. They stopped after scripted turns 3 and 11 without
filler, and after turns 16 and 26 with dots. The frozen protocol retains these
as outcomes. The earlier no-filler stalls create 44 missing turns versus 16 in
the dots arm and materially offset the dots arm's larger number of observed
tool/instruction failures.

The manifest label `model_abort` should not be read as premature conversational
closure here. Its 15 no-filler instances are `end_session` calls on recovery
turns 30–34 after the model missed the required turn-29 call. Strict completion
is 4/30 in both arms; 11 no-filler and 26 dots conversations never called
`end_session` at all.

Median successful-response TTFAT was fast at about 295 ms, but the serving tail
was not: successful responses reached 21.0 seconds without filler and 40.4
seconds with dots, in addition to the four 45-second idle timeouts.

## Interpretation

The final aggregate is 85.6% without filler versus 83.3% with dots, a
-2.2-point estimate with a whole-conversation 95% interval of -8.3 to +5.1.
That interval spans zero, so this campaign does **not** establish an overall
filler effect.

The turn pattern is still useful descriptively. The five harm-direction turns
that triggered the prospective n=30 expansion—11, 15, 17, 19, and 24—retained
that direction and together had 37 more dot-arm failures. Adding turns 12 and
16 brings the dot-arm excess at those seven workflow turns to 44 failures.
Twenty-four fewer failures elsewhere, largely from the timing and severity of
the four serving stalls and the no-filler grounding mistakes, reduce the net
difference to 20 turns. This is consistent with dots amplifying Laguna's
hesitation around tool execution, but it is not a causal conclusion.

## Conclusions

- Laguna S 2.1 is fast at the median but is not reliable enough in this
  reasoning-off configuration for a long-running voice agent.
- The best remediation target is explicit state/action handling: never ask
  again for a field already collected, and call the tool immediately once all
  required arguments are available. Application-side state and deterministic
  dispatch may be more dependable than another prompt instruction.
- Closing should be enforced by the application or a dedicated termination
  policy; a pleasant textual farewell is not a reliable signal that Laguna
  will invoke `end_session`.
- Do not add dots for Laguna based on this result. They provided no confirmed
  aggregate benefit and were descriptively worse on the important workflow
  turns.
- Any follow-up should test the remediation directly on turns 11, 15, 17, 24,
  and 29 while continuing to measure full-conversation stalls and tail
  latency.

Source artifacts:
[`aggregates.json`](filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json),
[`protocol.md`](filler-study-data/laguna-s21-openrouter-2026-07-22/protocol.md),
and the frozen run directories listed in the aggregate.
