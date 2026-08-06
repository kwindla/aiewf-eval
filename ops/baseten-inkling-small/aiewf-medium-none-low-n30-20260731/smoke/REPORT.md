# Excluded compatibility smokes

These observations were completed before the 60-slot campaign schedule was
executed. They are excluded from every campaign aggregate.

## Raw reasoning-effort probe

Both requests used BaseTen's OpenAI-compatible endpoint and the requested model
`thinkingmachines/inkling-small`. The prompt was a small combinatorics problem;
temperature was 1.0 and the completion cap was 1,024 tokens.

| Effort | Returned model | First chunk | First answer | Reasoning chars | Reasoning tokens | Content chars |
|---|---|---:|---:|---:|---:|---:|
| `none` | `thinkingmachines/Inkling-Small` | 572ms | 572ms | 0 | 0 | 2 |
| `low` | `thinkingmachines/Inkling-Small` | 181ms | 415ms | 197 | 63 | 2 |

This verifies that BaseTen honors the two effort settings and that `low`
reasoning arrives separately before visible content.

## Full-conversation Pipecat smokes

- `none`: `runs/aiwf_medium_context/20260731T094929_thinkingmachines_inkling-small_c411201a`
  completed all 30 scheduled turns. It reported zero reasoning tokens and
  exercised all required tool types. A recovery turn followed a premature tech
  support call at turn 16.
- `low`: `runs/aiwf_medium_context/20260731T095023_thinkingmachines_inkling-small_0e06c755`
  exercised ordinary responses, consecutive session-suggestion tools, dietary
  tool use, and a tool result continuation. It ended after 16 scheduled turns
  when a recovery response called `end_session`. This is a model behavior, not
  an API compatibility failure.

The smokes established stable streaming, tool-call parsing, tool-result history,
content-aware TTFAT recording, and response usage. The campaign therefore
proceeded without changing its outcome-retention rule: a short conversation
after any valid model response remains a measured outcome.
