# Nemotron 3 Super Progress

## 2026-02-19: B200 endpoint validation after tool parsing reconfigure

### Scope
- Benchmark: `aiwf_medium_context` (text pipeline)
- Service: `nemotron` (OpenAI-compatible)
- Endpoints:
  - Nano B200: `https://kwindla--nemotron-nano-b200-serve.modal.run/v1`
  - Super B200: `https://kwindla--nemotron-super-b200-serve.modal.run/v1`

Parameter note:
- Treat `thinking on/off` as the only control knob for Nemotron behavior on these endpoints.
- `reasoning_effort` may be accepted by the API shape but is not a reliable control for evaluation behavior; harness support for it was removed.

### Run 1: Nano B200
- Run dir: `runs/aiwf_medium_context/20260219T085415_nemotron-3-nano-30b_65ce78b3`
- Transcript rows: `31` (`30` scripted + `1` recovery)
- Recovery: `1` (synthetic turn `30`, `recovery_for_turn=15`)

Judged:
- Turn-taking: `30/30`
- Tool use: `28/30`
- Instruction following: `27/30`
- KB grounding: `29/30`
- Strict turn pass: `27/30` (`90.0%`)

TTFT (scripted turns):
- P50: `970ms`
- P95: `2805ms`

Key misses:
- Turn `15`: missed required `submit_dietary_request` (asked for known name again).
- Turn `16`: premature `request_tech_support` call with partial issue.
- Turn `21`: wrong location/time for Charles Frye follow-up (instruction + KB miss).

### Run 2: Super B200
- Run dir: `runs/aiwf_medium_context/20260219T090236_nemotron-3-super-120b_cbc5eb30`
- Transcript rows: `30`
- Recovery: `0`
- Sanity check: expected tool-call turns present (`11, 12, 15, 17, 24, 29`), clean turn IDs `0..29`.

Judged:
- Turn-taking: `30/30`
- Tool use: `30/30`
- Instruction following: `30/30`
- KB grounding: `30/30`
- Strict turn pass: `30/30` (`100.0%`)

TTFT:
- P50: `996ms`
- P95: `4952ms`
- Max: `18438ms`

## 2026-02-19: 10-run benchmark + judging snapshot

### Scope
- Model: `nemotron-3-super-120b`
- Benchmark: `aiwf_medium_context` (text pipeline)
- Evaluation set: 10 most recent complete runs (30 judged turns per run)

Run set:
- `runs/aiwf_medium_context/20260218T203833_nemotron-3-super-120b_b1466975`
- `runs/aiwf_medium_context/20260218T214401_nemotron-3-super-120b_8741645c`
- `runs/aiwf_medium_context/20260218T214523_nemotron-3-super-120b_2b7c2623`
- `runs/aiwf_medium_context/20260218T214623_nemotron-3-super-120b_41252cee`
- `runs/aiwf_medium_context/20260218T214723_nemotron-3-super-120b_ee114aee`
- `runs/aiwf_medium_context/20260218T214909_nemotron-3-super-120b_96d9f9a7`
- `runs/aiwf_medium_context/20260218T215006_nemotron-3-super-120b_d5a3da0b`
- `runs/aiwf_medium_context/20260218T215105_nemotron-3-super-120b_302b3180`
- `runs/aiwf_medium_context/20260218T215203_nemotron-3-super-120b_48c108e2`
- `runs/aiwf_medium_context/20260218T215253_nemotron-3-super-120b_b8bc7296`

### Turn-based aggregate metrics (strict)

Strict turn pass definition: a turn passes only if all three are true on that turn:
- `tool_use_correct`
- `instruction_following`
- `kb_grounding`

Aggregate across 300 turns:
- Strict turn pass: `248/300` = `82.7%`
- Median strict pass rate across runs: `83.3%` (`25/30`)
- Best run: `27/30` (`90.0%`)
- Worst run: `23/30` (`76.7%`)

Dimension totals across the same 300 turns:
- Tool use: `252/300`
- Instruction following: `258/300`
- KB grounding: `298/300`

Per-run strict:
- `20260218T203833_nemotron-3-super-120b_b1466975`: `27/30`
- `20260218T214401_nemotron-3-super-120b_8741645c`: `25/30`
- `20260218T214523_nemotron-3-super-120b_2b7c2623`: `25/30`
- `20260218T214623_nemotron-3-super-120b_41252cee`: `24/30`
- `20260218T214723_nemotron-3-super-120b_ee114aee`: `23/30`
- `20260218T214909_nemotron-3-super-120b_96d9f9a7`: `26/30`
- `20260218T215006_nemotron-3-super-120b_d5a3da0b`: `25/30`
- `20260218T215105_nemotron-3-super-120b_302b3180`: `24/30`
- `20260218T215203_nemotron-3-super-120b_48c108e2`: `24/30`
- `20260218T215253_nemotron-3-super-120b_b8bc7296`: `25/30`

### Failure pattern notes

Total strict failures: `52` turns.

Primary pattern: required tool call not executed (`44/52`)
- Common behavior: assistant says it is submitting/ending, but no actual function call is emitted.
- Common behavior: asks unnecessary confirmation even when required args are already known.
- Most affected scripted turns:
  - Turn `24` (`vote_for_session`): failed in `10/10` runs
  - Turn `29` (`end_session`): failed in `10/10` runs
  - Turn `17` (`request_tech_support`): failed in `8/10` runs
  - Turn `15` (`submit_dietary_request`): failed in `7/10` runs
  - Turn `12` (second `submit_session_suggestion`): failed in `7/10` runs
  - Turn `11` (first `submit_session_suggestion`): failed in `2/10` runs

Secondary pattern: unexpected tool calls on non-tool turns (`4/52`)
- Seen as premature/fabricated `submit_session_suggestion` calls on non-tool turns in one run.

Secondary pattern: instruction-only failures (`2/52`)
- Over-deflection on event-relevant questions (e.g., location-related conference questions that should be answered from KB).

Secondary pattern: KB-only failures (`2/52`)
- One run included off-topic/hallucinated content and a false claim that both suggestions were submitted when one tool call was missing.

### Summary
- KB grounding is generally strong (`298/300`), with a single outlier run causing both KB misses.
- Reliability is limited by tool execution discipline, not by core factual QA.
- The dominant risk is mismatch between conversational intent text and actual function emission.

## 2026-02-19: Other Models (Turn-Based Batch)

### Scope and method
- Models:
  - `gpt-5.1`
  - `gemini-3-flash-preview`
  - `claude-sonnet-4-6`
  - `gpt-4.1`
  - `gemini-2.5-flash`
  - `gpt-5-mini`
  - `gpt-4o-mini`
  - `gpt-4o`
  - `gpt-5.2`
  - `claude-haiku-4-5`
- Run target reached: 10 complete runs per model.
- Run selection for reporting: 10 most recent complete runs per model (complete = non-empty transcript and `runtime.json` turns >= 30).
- Turn-based pass definition (strict): a turn passes only if `tool_use_correct && instruction_following && kb_grounding`.
- TTFT metrics use scripted turns `0..29` from `transcript.jsonl` (recovery turns excluded).

### README-style table (turn-based)

Ordered by pass rate (desc), then TTFT median (asc).

| Model | Tool Use | Instruction | KB Ground | Turn Pass | Pass Rate | Median Rate | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini-3-flash-preview | 300/300 | 300/300 | 300/300 | 300/300 | 100.0% | 100.0% | 1107ms | 1599ms | 2781ms |
| claude-sonnet-4-6 | 299/300 | 299/300 | 300/300 | 299/300 | 99.7% | 100.0% | 850ms | 4126ms | 9396ms |
| claude-haiku-4-5 | 298/300 | 294/300 | 300/300 | 294/300 | 98.0% | 100.0% | 637ms | 1615ms | 3152ms |
| gpt-5.1 | 294/300 | 294/300 | 300/300 | 294/300 | 98.0% | 100.0% | 739ms | 1492ms | 4244ms |
| nemotron-3-super-120b (FP8 endpoint, thinking_budget=default/unlimited) | 299/300 | 290/300 | 300/300 | 290/300 | 96.7% | 96.7% | 1220ms | 1368ms | 16572ms |
| gpt-4.1 | 289/300 | 290/300 | 300/300 | 289/300 | 96.3% | 96.7% | 536ms | 1771ms | 5056ms |
| nemotron-3-super-120b (full thinking, budget endpoint) | 296/300 | 289/300 | 299/300 | 289/300 | 96.3% | 96.7% | 922ms | 1262ms | 167321ms |
| gpt-4o | 291/300 | 285/300 | 299/300 | 284/300 | 94.7% | 93.3% | 546ms | 1369ms | 4897ms |
| nemotron-3-super-120b (thinking_budget=20, budget endpoint) | 293/300 | 288/300 | 298/300 | 284/300 | 94.7% | 96.7% | 1005ms | 1087ms | 9283ms |
| nemotron-3-super-120b (FP8 endpoint, thinking_budget=20) | 288/300 | 285/300 | 300/300 | 280/300 | 93.3% | 93.3% | 1276ms | 1398ms | 16529ms |
| nemotron-3-nano-30b | 287/300 | 281/300 | 295/300 | 277/300 | 92.3% | 93.3% | 745ms | 920ms | 6679ms |
| gemini-2.5-flash | 274/300 | 269/300 | 300/300 | 269/300 | 89.7% | 90.0% | 597ms | 1137ms | 2313ms |
| gpt-5.2 | 270/300 | 268/300 | 298/300 | 268/300 | 89.3% | 88.3% | 624ms | 1171ms | 2509ms |
| gpt-oss-120b (groq) | 272/300 | 261/300 | 298/300 | 259/300 | 86.3% | 86.7% | 98ms | 217ms | 2117ms |
| gpt-5-mini | 258/300 | 251/300 | 297/300 | 251/300 | 83.7% | 83.3% | 682ms | 1132ms | 1904ms |
| gpt-4o-mini | 269/300 | 259/300 | 293/300 | 248/300 | 82.7% | 83.3% | 553ms | 1947ms | 6497ms |


### Failure pattern notes

`gpt-4.1`
- Primary misses are tool-action turns, especially vote submission.
- Hotspots: turn 24 (vote), turn 15 (dietary), rare carryover at turn 25.
- Category mix: mostly tool+instruction combined misses (assistant asks again instead of calling).

`gemini-2.5-flash`
- Repeated misses on required tool turns after confirmation.
- Hotspots: turns 15 (dietary), 17 (tech support), 24 (vote).
- Secondary issues: occasional premature tool call (turn 16) and occasional instruction-only misses around session-suggestion handling.

`gpt-5-mini`
- Strong recurring pattern of re-asking known fields and not executing required functions.
- Hotspots: turns 12, 15, 24; additional misses at 11 and 17.
- Category mix: dominated by tool+instruction combined failures.

`gpt-4o-mini`
- Mixed failure profile across tool use, instruction, and some KB errors.
- Hotspots: turns 15, 21, 24, 29.
- Notable pattern: over-deflection on location-related event questions (instruction-only at turns 19/21), plus frequent missing vote/end_session calls.

`gpt-4o`
- Generally strong; failures are concentrated and mostly recoverable.
- Hotspots: turns 15, 21, 24.
- Pattern: occasional missed tool calls and instruction slips on location/room follow-up.

`gpt-5.2`
- Primary issue is missing required tool calls after enough user context is already present.
- Hotspots: turns 24, 12, 15, 17.
- Secondary issue: occasional instruction/KB miss around follow-up location question (turn 21).

`claude-haiku-4-5`
- Best turn-based result in this batch.
- Most runs are perfect; residual misses are clustered early.
- Hotspots: turns 9-11 (hallway-session suggestion flow/context continuation).

## 2026-02-19: Nemotron Nano B200 Thinking-Off Streaming

### Harness change
- Updated `src/multi_turn_eval/services/nemotron.py` so `MTE_NEMOTRON_THINKING_OFF=1` no longer forces non-streaming mode.
- `enable_thinking=false` is still sent via `extra_body.chat_template_kwargs`.
- Updated `src/multi_turn_eval/pipelines/text.py` so Nemotron idle-timeout auto-extension (180s) applies only when `MTE_NEMOTRON_NON_STREAMING=1`.

### Smoke checks (streaming ON, thinking OFF)
- Run `20260219T101741_nemotron-3-nano-30b_6dd944f7`
  - Config logged: `non_streaming=False, thinking_off=True`
  - Turn 0 made an unexpected `end_session` call (single-turn output).
  - TTFT: `1209ms`
- Run `20260219T101809_nemotron-3-nano-30b_04ac4c88`
  - Config logged: `non_streaming=False, thinking_off=True`
  - 3/3 turns completed normally with no tool calls.
  - TTFT by turn: `825ms`, `768ms`, `783ms`

### Full-run sanity checks (streaming ON, thinking OFF)
- Run `20260219T101826_nemotron-3-nano-30b_eacc7fed`
  - Ended early at turn `18/30` due premature `end_session` tool call.
- Run `20260219T101923_nemotron-3-nano-30b_47ebca25`
  - Also ended at turn `18/30` due premature `end_session`.
  - Transcript includes one recovery record at `turn=30` (`recovery_for_turn=15`) before resuming scripted turn indices.

### Current read
- Thinking-off is now working with streaming transport in the harness.
- Behavior quality under thinking-off remains unstable on full conversations (premature `end_session` around turn 18 in 2/2 full runs).

## 2026-02-19: Budget Endpoints (`thinking_budget`)

### Harness support added
- `src/multi_turn_eval/services/nemotron.py` now supports `MTE_NEMOTRON_THINKING_BUDGET=<int>`.
- This injects `extra_body.vllm_xargs.thinking_budget` on each request.
- Precedence: if `MTE_NEMOTRON_THINKING_OFF=1` is also set, the budget is ignored and a warning is logged.

### Endpoint smoke checks (`thinking_budget=20`)
- Nano budget endpoint:
  - URL: `https://kwindla--nemotron-nano-b200-budget-serve.modal.run/v1`
  - Warm smoke run: `runs/aiwf_medium_context/20260219T103602_nemotron-3-nano-30b_5a0316e6`
  - TTFT: turn0 `887ms`, turn1 `816ms`, turn2 `773ms`
- Super budget endpoint:
  - URL: `https://kwindla--nemotron-super-b200-budget-serve.modal.run/v1`
  - Warm smoke run: `runs/aiwf_medium_context/20260219T103748_nemotron-3-super-120b_e0b9e31a`
  - TTFT: turn0 `913ms`, turn1 `829ms`, turn2 `821ms`

Note:
- Initial cold-ish Nano smoke `runs/aiwf_medium_context/20260219T103123_nemotron-3-nano-30b_e5a252be` hit idle-timeout with no recorded turns.
- For budget endpoint testing, `MTE_TEXT_IDLE_TIMEOUT_SECS=180` was used to avoid startup/cold variance.

### Targeted turn-18 tool-discipline probe (weather turn)
- Exact turn-18 context replayed with tools enabled, `temp=0.6`, `top_p=0.95`, stream mode.
- Nano budget endpoint (`nemotron-3-nano-30b`, budget=20): `12/12` text-only, `0/12` premature `end_session`.
- Super budget endpoint (`nemotron-3-super-120b`, budget=20): `12/12` text-only, `0/12` premature `end_session`.

### Full-run checks (`thinking_budget=20`)
- Nano full run: `runs/aiwf_medium_context/20260219T103808_nemotron-3-nano-30b_efcef76a`
  - Transcript rows: `34` (contains recovery turns)
  - Judge: turn-taking `30/30`, tool `25/30`, instruction `27/30`, KB `29/30`
  - Recovery/tool flow observed at suggestion, dietary, vote, and final acknowledgement turns.
- Super full run: `runs/aiwf_medium_context/20260219T103926_nemotron-3-super-120b_626f2493`
  - Transcript rows: `30` (no recovery turns)
  - Judge: turn-taking `30/30`, tool `30/30`, instruction `29/30`, KB `30/30`

### Current read
- Budgeted thinking (`thinking_budget=20`) resolves the specific premature-`end_session` failure mode seen with strict `thinking_off`.
- Super budget endpoint looks immediately usable for full-run evaluation.
- Nano budget endpoint shows mixed reliability in full-run benchmark behavior despite strong targeted turn-18 discipline.

### Budget endpoint table lines (10-run aggregate)
- Aggregation basis:
  - 1 prior full run per model config (already judged)
  - plus 9 new full runs per model config from this batch
  - 18 new runs judged in this update

| Model Config | Tool Use | Instruction | KB Ground | Turn Pass | Pass Rate | Median Rate | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nemotron-3-super-120b thinking_budget=20 (budget endpoint) | 293/300 | 288/300 | 298/300 | 284/300 | 94.7% | 96.7% | 1005ms | 1087ms | 9283ms |
| nemotron-3-nano-30b thinking_budget=20 (budget endpoint) | 260/300 | 263/300 | 282/300 | 242/300 | 80.7% | 80.0% | 967ms | 1190ms | 11584ms |

## 2026-02-19: Full Thinking Super + Nano Budget Sweep (Final Judged)

### Scope
- Super default/full thinking on budget endpoint:
  - `nemotron-3-super-120b`
  - 10 complete runs (all judged)
- Nano budget sweep on budget endpoint:
  - `nemotron-3-nano-30b`
  - 3 complete runs each for `default`, `30`, `40`, `60`, `80` (all judged)
- Judge execution note:
  - one transient judge JSON-parse failure occurred mid-batch; rerun with per-run retry completed successfully.

### Aggregated results

Ordered by pass rate (desc), then TTFT median (asc).

| Model Config | Tool Use | Instruction | KB Ground | Turn Pass | Pass Rate | Median Rate | TTFT Med | TTFT P95 | TTFT Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nemotron-3-super-120b (full thinking, budget endpoint) | 296/300 | 289/300 | 299/300 | 289/300 | 96.3% | 96.7% | 922ms | 1262ms | 167321ms |
| nemotron-3-nano-30b (full thinking, budget endpoint default) | 85/90 | 83/90 | 90/90 | 83/90 | 92.2% | 90.0% | 796ms | 950ms | 1062ms |
| nemotron-3-nano-30b (thinking_budget=80, budget endpoint) | 81/90 | 82/90 | 87/90 | 77/90 | 85.6% | 86.7% | 916ms | 1031ms | 3706ms |
| nemotron-3-nano-30b (thinking_budget=30, budget endpoint) | 78/90 | 78/90 | 88/90 | 76/90 | 84.4% | 83.3% | 973ms | 1229ms | 3048ms |
| nemotron-3-nano-30b (thinking_budget=40, budget endpoint) | 76/90 | 76/90 | 83/90 | 72/90 | 80.0% | 80.0% | 901ms | 1021ms | 1084ms |
| nemotron-3-nano-30b (thinking_budget=60, budget endpoint) | 79/90 | 77/90 | 79/90 | 70/90 | 77.8% | 80.0% | 967ms | 1012ms | 1118ms |

### Failure-pattern highlights

- Super full thinking:
  - strongest overall quality in this set (`96.3%` strict turn pass),
  - dominant misses are instruction-only on turn `19` (location answer style/constraint handling),
  - tool misses are rare (`4` total across `300` turns).
- Nano default full thinking:
  - second-best quality (`92.2%` strict turn pass),
  - recurrent failures cluster around turn `16` (tech-support tool turn).
- Nano with budget caps:
  - quality degrades vs default at all tested caps in this sweep,
  - best capped setting observed here is `thinking_budget=80` (`85.6%`),
  - `40` and `60` show broader failure spread including KB misses on turns `20/21/23/24/29`.

### Latency note

- The `167321ms` TTFT max on Super full-thinking is a single outlier (turn `0` in run `20260219T113935_nemotron-3-super-120b_bf1259f4`).
- Central tendency remains stable (`922ms` median, `1262ms` P95).
