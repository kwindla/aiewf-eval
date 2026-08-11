# Anthropic Direct Probe Progress

## Goal
Isolate whether Sonnet 4.6 benchmark misses are model behavior or runtime context/tool-loop artifacts by replaying problematic turns directly with the Anthropic SDK and capturing exact request/response payloads.

## Completed Work (Requested 4-Step Plan)

### 1) Benchmark-side exact Anthropic payload logging
- Added service wrapper: `src/multi_turn_eval/services/anthropic_logged.py`
- Switched CLI `anthropic` alias to wrapper in `src/multi_turn_eval/cli.py`
- Wrapper logs exact post-adapter request payload (including `cache_control` markers, `betas`, tools, full messages) when:
  - `MTE_LOG_ANTHROPIC_PAYLOADS=1`
- Verification run:
  - `runs/aiwf_medium_context/20260217T212219_claude-sonnet-4-6_cbba1028`
  - `run.log` contains `Anthropic exact request payload ...`

### 2) Benchmark A/B: caching on vs off
- Added runtime toggle in pipeline:
  - `MTE_ANTHROPIC_PROMPT_CACHING=1|0`
  - Implemented in `src/multi_turn_eval/pipelines/base.py`
- Matrix runs (full 30-turn + judge):
  - Baseline (cache on, recovery on):
    - `runs/aiwf_medium_context/20260217T212321_claude-sonnet-4-6_43806d81`
    - `runs/aiwf_medium_context/20260217T213301_claude-sonnet-4-6_d82655a5`
  - Cache off (recovery on):
    - `runs/aiwf_medium_context/20260217T212628_claude-sonnet-4-6_1d7b3005`
    - `runs/aiwf_medium_context/20260217T213602_claude-sonnet-4-6_d8527d86`
- Aggregate (2 runs each):
  - Baseline: tool-use avg `26.0/30`, instruction avg `17.5/30`
  - Cache off: tool-use avg `27.0/30`, instruction avg `17.0/30`
- Result: disabling caching did not remove stale/late behavior (turn-13 stale repeats still occurred).

### 3) Benchmark A/B: recovery nudges on vs off
- Added runtime toggle in pipeline:
  - `MTE_ENABLE_RECOVERY=1|0`
  - Implemented in `src/multi_turn_eval/pipelines/base.py`
- Recovery off runs (cache on):
  - `runs/aiwf_medium_context/20260217T212948_claude-sonnet-4-6_7e5d3ad3`
  - `runs/aiwf_medium_context/20260217T213914_claude-sonnet-4-6_d7c2728c`
- Aggregate (2 runs):
  - Tool-use avg `25.5/30`, instruction avg `16.5/30`
- Key pattern shift:
  - `request_tech_support` at turn 17 regressed in both recovery-off reps.
  - `end_session` remained missing in both reps.
  - Recovery affects failure shape, but does not eliminate stale drift.

### 4) Service-parity standalone probe mode
- Extended `scripts/anthropic_direct_probe.py`:
  - `--api-mode messages|beta_stream`
  - `--enable-prompt-caching`
  - `--dedupe-tool-calls`
  - `--service-parity` shortcut:
    - `beta_stream`
    - prompt caching markers
    - duplicate tool call suppression in context updates
- Existing context modes preserved:
  - `golden` scaffold
  - `observed` (`run.log` payload seed)
- Service-parity observed replay run:
  - `runs/direct_probes/20260217T214252_claude-sonnet-4-6`
  - Results:
    - Reproduced `T25` late vote call (unexpected tool call on target turn)
    - Reproduced `T17` extra stale tool call alongside expected one
    - `T29` in this run did include expected `end_session`

## Additional Probe Runs
- Sonnet 4.6 golden problematic set:
  - `runs/direct_probes/20260217T201425_claude-sonnet-4-6`
- Sonnet 4.5 control:
  - `runs/direct_probes/20260217T201545_claude-sonnet-4-5`
- Sonnet 4.6 observed replay (original mode):
  - `runs/direct_probes/20260217T202559_claude-sonnet-4-6`
- Sonnet 4.6 observed ambiguity check (`T13`, occurrence 2):
  - `runs/direct_probes/20260217T202832_claude-sonnet-4-6`
- Focused observed check (`T12,T13`):
  - `runs/direct_probes/20260217T205521_claude-sonnet-4-6`
- Fresh benchmark payload+judge run:
  - `runs/aiwf_medium_context/20260217T215632_claude-sonnet-4-6_75be2299`
  - Scores: tool-use `25/30`, instruction `16/30`

## New Checks (After 4-Step Plan)

### Anthropic SDK version parity
- Confirmed benchmark/runtime and standalone probe are using the same `uv` environment package:
  - `anthropic==0.49.0` (from `uv.lock`)
  - Pipecat Anthropic service imports the same installed package in `.venv`
- System Python outside `uv` does not have `anthropic` installed.
- Conclusion: SDK version mismatch is not the driver of the observed behavior in current runs.

### Exact payload diff at first concrete divergence
- Run inspected: `runs/aiwf_medium_context/20260217T215632_claude-sonnet-4-6_75be2299`
- Earliest divergence around turn 12/13 segment:
  - Benchmark emitted two consecutive payloads with the same last user text
    (`"Oh, one more suggestion. How about a session on state machine abstractions for complex workflows?."`)
  - Both had `message_count=25` before the next user turn text appeared.
- Direct service-parity probe (`runs/direct_probes/20260217T214252_claude-sonnet-4-6/case_13.json`) behavior:
  - Turn 12 request: `message_count=25` (expected)
  - Turn 13 request: `message_count=27` with next user text
    (`"Thanks for submitting both session suggestions. Is there food at the conference?"`)
  - No duplicate same-payload request for turn 12.
- Interpretation:
  - Benchmark path is issuing an additional inference on the same context state in this segment.
  - This is consistent with the stale/one-turn-late symptoms and not explained by cache/recovery toggles alone.

### A/B test: duplicate tool-call suppression on vs off
- Added runtime toggle:
  - `MTE_DEDUPE_TOOL_CALLS=1|0` in `src/multi_turn_eval/pipelines/base.py`
- Matrix manifest:
  - `runs/dedupe_ab_20260217T220552_results.tsv`
- Runs:
  - `dedupe_on` rep1: `runs/aiwf_medium_context/20260217T220216_claude-sonnet-4-6_a75d1b26` → tool `26`, instruction `18`
  - `dedupe_off` rep1: `runs/aiwf_medium_context/20260217T220554_claude-sonnet-4-6_06af709e` → tool `25`, instruction `19`
  - `dedupe_on` rep2: `runs/aiwf_medium_context/20260217T220909_claude-sonnet-4-6_c512dbdf` → tool `25`, instruction `20`
  - `dedupe_off` rep2: `runs/aiwf_medium_context/20260217T221227_claude-sonnet-4-6_4862ce71` → tool `26`, instruction `21`
- Aggregate:
  - `dedupe_on`: tool avg `25.5`, instruction avg `19.0`
  - `dedupe_off`: tool avg `25.5`, instruction avg `20.0`
- Function-tracking pattern remained essentially the same in all four runs:
  - `submit_dietary_request` late
  - `request_tech_support` late
  - `vote_for_session` late
  - `end_session` missing/never-called
- Conclusion: duplicate suppression is not the primary cause of stale/late tool behavior.

### Instrumentation: where duplicate requests originate
- Added queue/finalization debug logs:
  - `queue_turn: ...` in `TextPipeline`
  - `on_turn_end start/decision` in `BasePipeline`
- Instrumented run:
  - `runs/aiwf_medium_context/20260217T221755_claude-sonnet-4-6_ff45ef44`
- Key finding around turn 12:
  - `queue_turn: reason=next turn_idx=12 ...` appears once
  - But two Anthropic payloads are sent before turn 12 finalization:
    - `PAYLOAD[12]` and `PAYLOAD[13]` with same last-user text
  - `on_turn_end start: turn_idx=12 ...` occurs only after `PAYLOAD[13]`
- Conclusion:
  - The extra same-context request is not caused by duplicate turn queueing in our pipeline.
  - It is produced within the service/aggregator function-call loop for a single queued turn.

### Experiment: disable automatic post-tool LLM rerun
- Added runtime toggle:
  - `MTE_TOOL_RESULT_RUN_LLM=1|0` in `BasePipeline`
  - Tool callback now sets `FunctionCallResultProperties(run_llm=...)`
- Also normalized recorded tool-result properties in:
  - `src/multi_turn_eval/processors/tool_call_recorder.py`
- Observed behavior with `MTE_TOOL_RESULT_RUN_LLM=0`:
  - Run: `runs/aiwf_medium_context/20260217T222420_claude-sonnet-4-6_cd58e712`
  - Payload sequence around turn 12/13 became clean (no duplicate same-context payload):
    - `... turn12 user ...` then `... turn13 user ...` (expected progression)
  - However the run later idled out (no assistant timestamp after a tool-only response path).
  - Second run with additional turn-end handling attempt also idled:
    - `runs/aiwf_medium_context/20260217T222744_claude-sonnet-4-6_692e87f4`
- Interpretation:
  - Disabling automatic post-tool rerun reduces stale duplicate behavior early.
  - But current turn-end detection relies on assistant timestamp semantics that can fail on tool-only responses when rerun is disabled.
  - This is promising but requires a deliberate turn-finalization strategy change before using `run_llm=0` broadly.

## Interpretation (Current)
- We now have exact benchmark payload logging, so request-level comparison is directly available.
- Caching state changes output rates but is not the root-cause explanation for stale behavior.
- Recovery nudges materially change error distribution but do not fully resolve late/stale tool behavior.
- Service-parity standalone replay can reproduce key problematic behavior when seeded from observed context, supporting the view that trajectory/context state drives failures.
- SDK-version mismatch has been ruled out for current runs.
- Dedupe suppression toggle does not materially improve tool-use accuracy, so it is unlikely to be the root cause.
- The strongest new signal is an extra benchmark-side inference call on an unchanged context state (first observed in the turn 12/13 segment).
- That extra call is now traced to service/aggregator behavior during tool-call handling rather than duplicate benchmark turn queueing.

## Notes
- Matrix manifest: `runs/ab_matrix_20260217T212319_results.tsv`
- Matrix manifest (dedupe on/off): `runs/dedupe_ab_20260217T220552_results.tsv`
- Primary benchmark run originally analyzed: `runs/aiwf_medium_context/20260217T194732_claude-sonnet-4-6_60c311d5`

## Context-Flow Cleanup (Current)

### Code changes to align benchmark turn flow with standalone "good path"
- Updated text pipeline default behavior so tool results do **not** trigger immediate rerun:
  - `TextPipeline.default_tool_result_run_llm = False`
  - File: `src/multi_turn_eval/pipelines/text.py`
- Hardened tool-only turn finalization to avoid stale/late double-finalize:
  - Added full turn signature to `ToolResultTurnCompleteFrame`:
    - `turn_index`, `turn_start_monotonic`
  - Emitted signature from recorder in `ToolCallRecorder`
  - In `NextTurn`, ignore stale tool-complete signals unless signature matches current turn
  - Added duplicate-finalize guard per turn signature
  - Files:
    - `src/multi_turn_eval/frames.py`
    - `src/multi_turn_eval/processors/tool_call_recorder.py`
    - `src/multi_turn_eval/pipelines/text.py`
- Aligned recovery gating with judge semantics for tool-arg matching:
  - `_has_required_call()` now uses semantic arg matching (same heuristic family as judge)
  - Prevents unnecessary recovery turns when args are equivalent but phrased differently
  - File: `src/multi_turn_eval/pipelines/base.py`

### Validation runs
- Focused turns 10-17 with payload logging:
  - `runs/aiwf_medium_context/20260217T231500_claude-sonnet-4-6_25455cb1`
  - Outcomes:
    - No duplicate same-context payloads in this window
    - No double-recording of turn 17
    - No recovery injected at turn 17 (`should_recover=False`)
- Full 30-turn run with payload logging:
  - `runs/aiwf_medium_context/20260217T231610_claude-sonnet-4-6_17ebcbe1`
  - Outcomes:
    - 30 payloads for 30 turns
    - 0 consecutive same-context payload duplicates
    - Clean completion with `end_session` at turn 29
- Full judge on that run:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260217T231610_claude-sonnet-4-6_17ebcbe1`
  - Scores:
    - Turn-taking `30/30`
    - Tool use `30/30`
    - Instruction following `30/30`
    - KB grounding `30/30`
