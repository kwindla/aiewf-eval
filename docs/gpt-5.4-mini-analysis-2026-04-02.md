# gpt-5.4-mini Benchmark Analysis (2026-04-02)

## Overview

10-run sweeps of `gpt-5.4-mini` on `aiwf_medium_context` at four Responses API reasoning effort levels.

| Level | Pass Rate | Turns Scored | Full Completions | TTFT Med | TTFT P95 | TTFT Max |
|-------|-----------|-------------|------------------|----------|----------|----------|
| none | 83.1% | 148/300 | 1/10 | 459ms | 731ms | 2369ms |
| low | 90.1% | 172/300 | 2/10 | 608ms | 1527ms | 2736ms |
| medium | 91.1% | 203/300 | 4/10 | 808ms | 2120ms | 2786ms |
| high | 89.8% | 236/300 | 6/10 | 1263ms | 3235ms | 5748ms |

`medium` has the highest pass rate. `high` completes more turns but its per-turn accuracy drops slightly, landing at 89.8%. KB grounding is near-perfect at all levels. Turn-taking never fails.

## Dominant Failure: Premature end_session at Turn 13

The single largest factor in the sub-300 turn counts. After the user says "Thanks for submitting both session suggestions. Is there food at the conference?", the model calls `end_session` instead of answering the food question, terminating the conversation and forfeiting ~16 remaining turns.

| Level | Premature turn-13 exits |
|-------|------------------------|
| none | 9/10 |
| low | 8/10 |
| medium | 6/10 |
| high | 4/10 |

**Root cause:** The `end_session` call appears to be generated in the same API response as the turn-12 `submit_session_suggestion`. Evidence: every premature `end_session` has `thinking_tokens=0` and null token counts, suggesting the model never processes the turn-13 user message before emitting it. When reasoning is active and the model does think at turn 13 (`thinking_tokens > 0`), it never calls `end_session`. The model interprets "Thanks for submitting both session suggestions" as a conversation-ending signal and preemptively schedules the termination alongside the turn-12 tool call.

Higher reasoning reduces this multi-tool-call pattern but does not eliminate it (still 40% at high).

## Other Failure Modes

### Forgotten name (turns 15, 17, 24)

The user provides their name ("Jennifer Smith") at turn 10. The model uses it correctly at turns 11-12 (for `submit_session_suggestion`), but by turn 15+ it asks "What name should I use?" again. This is a long-range context retrieval problem. Reasoning level does not help -- even at high, turn 15 fails ~83% of the time and turn 24 fails ~50%.

### Premature tool call at turn 16

When the user mentions "I'm having trouble with the mobile app," the model calls `request_tech_support` immediately with a vague description instead of waiting for the specific issue ("I can't access the location maps") at turn 17. This is actually *more* common at higher reasoning levels -- the model becomes more action-oriented but less patient about gathering complete information.

### Words without action at turn 12 (none-specific)

At `none` reasoning, 7/9 runs fail to actually call `submit_session_suggestion` at turn 12. The model says "I can submit that too" but emits no tool call. This pattern virtually disappears once any reasoning is enabled, suggesting even minimal reasoning helps the model commit to executing a tool call alongside text.

## Key Takeaways

1. **Reasoning primarily reduces premature exits, not per-turn accuracy.** Among runs that complete all 30 turns, pass rates are comparable across levels (79-90%).

2. **Higher reasoning introduces different failures, not just fewer.** The premature `request_tech_support` call at turn 16 is more common at high reasoning than low.

3. **Long-range name recall is a systematic weakness.** The model loses track of the user's name after ~5 turns regardless of reasoning level.

4. **The turn-13 exit is a multi-tool-call artifact.** It is not a reasoning failure per se -- it is the model batching `end_session` with a prior tool call before seeing the next user message.

5. **For voice agent use cases**, `low` is likely the best choice: fast TTFT (608ms median), 90.1% pass rate, and the per-turn accuracy on completed turns is comparable to `medium`/`high`.

## Configuration

```bash
MTE_OPENAI_RESPONSES_REASONING_EFFORT=<level> \
  uv run multi-turn-eval run aiwf_medium_context \
    --model gpt-5.4-mini --service openai
```

Reasoning effort is routed through the Responses API via `OpenAIResponsesLLMService`. The allowed levels for gpt-5.4-mini are `none`, `low`, `medium`, `high` (no `xhigh`).
