# Supernova EAP

## Results (10 runs, 300 turns)

| Model | Pass Rate | Turn Pass | Tool Use | Instruction | KB Ground | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-sonnet-4-6 | 100.0% | 300/300 | 300/300 | 300/300 | 300/300 | 850ms | 4126ms | 9396ms |
| gemini-3-flash-preview | 100.0% | 300/300 | 300/300 | 300/300 | 300/300 | 1107ms | 1599ms | 2781ms |
| claude-haiku-4-5 | 98.0% | 294/300 | 298/300 | 294/300 | 300/300 | 637ms | 1615ms | 3152ms |
| gpt-5.1 | 98.0% | 294/300 | 294/300 | 294/300 | 300/300 | 739ms | 1492ms | 4244ms |
| **Supernova (minimal)** | **97.7%** | **293/300** | **295/300** | **293/300** | **300/300** | **773ms** | **918ms** | **1100ms** |
| gpt-4.1 | 96.3% | 289/300 | 289/300 | 290/300 | 300/300 | 536ms | 1771ms | 5056ms |
| **Supernova (high)** | **95.7%** | **286/299** | **286/299** | **286/299** | **299/299** | **1374ms** | **1678ms** | **1892ms** |
| gpt-4o | 94.7% | 284/300 | 291/300 | 285/300 | 299/300 | 546ms | 1369ms | 4897ms |
| nemotron-3-nano-30b | 92.3% | 277/300 | 287/300 | 281/300 | 295/300 | 745ms | 920ms | 6679ms |
| gemini-2.5-flash | 89.7% | 269/300 | 274/300 | 269/300 | 300/300 | 597ms | 1137ms | 2313ms |
| gpt-5.2 | 89.3% | 268/300 | 270/300 | 268/300 | 298/300 | 624ms | 1171ms | 2509ms |
| gpt-oss-120b (groq) | 86.3% | 259/300 | 272/300 | 261/300 | 298/300 | 98ms | 217ms | 2117ms |
| gpt-5-mini | 83.7% | 251/300 | 258/300 | 251/300 | 297/300 | 682ms | 1132ms | 1904ms |
| gpt-4o-mini | 82.7% | 248/300 | 269/300 | 259/300 | 293/300 | 553ms | 1947ms | 6497ms |

## Hallucinated Tool Calls

A hallucinated tool call is defined as the model generating text that claims an action was completed when no actual function call was made.

### Minimal thinking budget (`thinking_level=MINIMAL`)

Across 300 turns, 2 hallucinated tool calls were found (0.7%).

**Run 3b825eaa, Turn 12 (Minimal)** — User asked to submit a second session suggestion ("state machine abstractions for complex workflows"). The model responded "I have successfully submitted your suggestion" but `submit_session_suggestion` was never called. No tool call at all.

**Run 050dd802, Turn 15 (Minimal)** — User confirmed "Yes" to submit a vegan dietary request. The model wrote out the tool call in its text as markdown (`*Action: Calling submit_dietary_request(name="Jennifer Smith", dietary_preference="vegan")*`) and then said "I've submitted your request" — but no actual `submit_dietary_request` function call was generated. The model literally wrote the tool call syntax as text instead of emitting it as a function call.

### High thinking budget (`thinking_level=HIGH`)

Across 299 turns (filtered + backfill 10-run set), 2 hallucinated action-completion claims were found (0.7%).

**Run d9f88c38, Turn 17 (High)** — User provided the specific tech-support issue ("I can't access the location maps"). The model said it had "noted" and would "add" those details to the tech-support request, but no tool call was made on that turn.

**Run 85dfbac2, Turn 17 (High)** — User provided the same specific issue. The model said it had "added this detail" to the tech-support request, but no tool call was made on that turn.

**Additional note (early `end_session`)** — Run `f69eeee3` called `end_session` at turn 28 (expected turn 29). The proximate user input was: "One last detail: when is continental breakfast on June 4th?"
