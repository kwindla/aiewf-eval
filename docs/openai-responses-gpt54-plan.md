# OpenAI Responses API Plan for GPT-5.4 Benchmarking

Date: 2026-03-05

## Goal

Enable `aiwf_medium_context` benchmark runs for `gpt-5.4-2026-03-05` using OpenAI Responses API (`/v1/responses`) while preserving existing Pipecat frame semantics and benchmark scoring behavior.

This plan also includes upgrading the OpenAI Python SDK to the latest version.

## Why this is needed

- Current text-mode OpenAI path is `chat.completions` in Pipecat:
  - `pipecat/src/pipecat/services/openai/base_llm.py:245`
  - `pipecat/src/pipecat/services/openai/base_llm.py:251`
  - `pipecat/src/pipecat/services/openai/base_llm.py:254`
- For `gpt-5.4-2026-03-05`, `reasoning_effort` + function tools on `chat.completions` fails (400 requiring `/v1/responses`).
- Pipecat `0.0.104` does not include an `OpenAIResponsesLLMService` class (verified by inspecting the wheel contents).

## Current frame contract we must preserve

The benchmark pipeline and assistant aggregator depend on specific frame behavior.

### Required upstream inputs to the LLM service

- `LLMContextFrame` / `OpenAILLMContextFrame` / `LLMMessagesFrame` (context trigger)
- `LLMUpdateSettingsFrame` (dynamic settings)
- `InterruptionFrame` (function cancellation flow handled in base `LLMService`)

Reference:
- `pipecat/src/pipecat/services/openai/base_llm.py:517`
- `pipecat/src/pipecat/services/llm_service.py:351`

### Required downstream outputs from the LLM service

- `LLMFullResponseStartFrame` before model output
- `LLMTextFrame` for assistant content chunks
- `LLMFullResponseEndFrame` after completion
- Function lifecycle via `run_function_calls(...)`, which emits:
  - `FunctionCallsStartedFrame`
  - `FunctionCallInProgressFrame`
  - `FunctionCallResultFrame`

Reference:
- `pipecat/src/pipecat/services/openai/base_llm.py:546`
- `pipecat/src/pipecat/services/openai/base_llm.py:481`
- `pipecat/src/pipecat/services/openai/base_llm.py:556`
- `pipecat/src/pipecat/services/openai/base_llm.py:515`
- `pipecat/src/pipecat/services/llm_service.py:649`

### Why this matters for assistant context aggregation

The assistant aggregator finalizes turns from `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` and consumes `TextFrame` + function call frames to update context.

Reference:
- `pipecat/src/pipecat/processors/aggregators/llm_response_universal.py:923`
- `pipecat/src/pipecat/processors/aggregators/llm_response_universal.py:925`
- `pipecat/src/pipecat/processors/aggregators/llm_response_universal.py:947`
- `pipecat/src/pipecat/processors/aggregators/llm_response_universal.py:950`

## OpenAI SDK state

- Current locked version: `openai==2.13.0` (`uv.lock`).
- Latest on PyPI (checked 2026-03-05): `openai==2.25.0`.

## Proposed implementation

## 1) Add a new Responses-based service class in this repo

Add:

- `src/multi_turn_eval/services/openai_responses.py`
- Class: `OpenAIResponsesLLMService`

Recommended base class:

- Inherit from `BaseOpenAILLMService` to reuse:
  - API client creation (`AsyncOpenAI`)
  - metrics hooks
  - `process_frame` wrapper behavior (start/end frames, error handling)

Override:

- `_process_context(...)` to call `client.responses.stream(...)` and map events to frames
- `run_inference(...)` to use `responses.create(stream=False)` for summarization/one-shot paths

## 2) Responses request mapping (context -> input/tools)

Convert `LLMContext` / `OpenAILLMContext` messages to `responses` `input` items:

- `system`/`developer`/`user`/`assistant` text messages -> `{"type":"message", ...}`
- assistant tool calls from chat-style context (`tool_calls`) -> `{"type":"function_call", "call_id": ..., "name": ..., "arguments": ...}`
- tool results (`role: "tool"`, `tool_call_id`) -> `{"type":"function_call_output", "call_id": ..., "output": ...}`

Tool schema conversion:

- Current adapter builds Chat Completions tools (`{"type":"function","function":{...}}`)
- Responses expects function tools in Responses shape (`{"type":"function","name":...,"parameters":...}`)
- Implement explicit converter for `ToolsSchema` in the new service (or a new adapter dedicated to Responses).

Tool choice conversion:

- Preserve `"none" | "auto" | "required"` directly
- Convert chat style forced function:
  - chat: `{"type":"function","function":{"name":"x"}}`
  - responses: `{"type":"function","name":"x"}`

## 3) Responses event mapping (stream -> Pipecat frames)

Use `async with client.responses.stream(...) as stream` and iterate events:

- `response.output_text.delta`
  - emit `LLMTextFrame(delta)`
  - stop TTFB at first content delta
- `response.output_item.added` (if `item.type == "function_call"`)
  - cache mapping: `item_id -> call_id` and `item_id -> function name`
- `response.function_call_arguments.done`
  - find `call_id` via `item_id` map (fallback to `item_id`)
  - parse JSON args
  - call `run_function_calls([FunctionCallFromLLM(...)])`
  - stop TTFB if first actionable output is a tool call
- `response.completed`
  - extract usage:
    - prompt/input tokens
    - completion/output tokens
    - total tokens
    - cached input tokens
    - reasoning tokens
  - emit usage metrics via `start_llm_usage_metrics(...)`
  - record full model name
- `response.failed` / `response.error`
  - push non-fatal error frame with API message

Important detail:

- `response.function_call_arguments.done` does not include `call_id` in SDK models; it includes `item_id`.
- We must map `item_id -> call_id` from `response.output_item.added` or `response.output_item.done`.

## 4) Service wiring in benchmark app

Add service alias:

- In `src/multi_turn_eval/cli.py`:
  - `openai-responses: multi_turn_eval.services.openai_responses.OpenAIResponsesLLMService`

Update pipeline model config logic:

- In `src/multi_turn_eval/pipelines/base.py`, branch behavior by service class:
  - Chat-completions OpenAI service keeps current behavior
  - Responses service should send reasoning as:
    - `extra={"reasoning": {"effort": "none"}}` for `gpt-5.4-*` benchmark requirement
  - Keep `service_tier="priority"` behavior where desired

## 5) OpenAI SDK upgrade plan

Upgrade to `openai==2.25.0`.

Recommended repo steps:

1. Add explicit direct dependency in `pyproject.toml`:
   - `openai==2.25.0`
2. Refresh lock:
   - `uv lock --upgrade-package openai`
3. Sync env:
   - `uv sync`
4. Smoke checks:
   - import and signature checks for `client.responses.stream` and `client.responses.create`
   - run a 3-turn benchmark smoke test with judge

Rationale:

- Today `openai` is transitive via `pipecat-ai`; pinning explicitly stabilizes SDK behavior for Responses code.

## Validation plan

## Phase A: Service-level smoke

- Run:
  - `uv run multi-turn-eval run aiwf_medium_context --model gpt-5.4-2026-03-05 --service openai-responses --only-turns 0,1,2`
- Judge:
  - `uv run multi-turn-eval judge <run_dir> --only-turns 0,1,2`
- Check:
  - tool calls execute
  - no context-frame warnings around missing tool call lifecycle
  - transcript includes assistant text and tool call/result records

## Phase B: Frame-contract verification

- Confirm `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame` fire exactly once per LLM pass.
- Confirm tool-only paths still advance turn logic:
  - `ToolCallRecorder` sees `FunctionCallInProgressFrame` + `FunctionCallResultFrame`.
- Confirm `LLMContextAssistantTimestampFrame` appears for content turns.

## Phase C: Benchmark run

- Full 10-run sequential matrix for `gpt-5.4-2026-03-05` using `openai-responses`.
- Judge each run.
- Aggregate in README table format.

## Risks and mitigations

- Risk: incorrect call ID mapping (`item_id` vs `call_id`) breaks tool-output continuity.
  - Mitigation: explicit `item_id -> call_id` map and fallback logging.
- Risk: message conversion drops non-text content.
  - Mitigation: start with text benchmark path; add strict conversion logs for unsupported message types.
- Risk: regressions in existing `openai` chat-completions flow.
  - Mitigation: keep new service under new alias; do not mutate existing OpenAI service code paths.
- Risk: SDK upgrade side effects.
  - Mitigation: pin exact version and run both `openai` and `openai-responses` smoke tests.

## Deliverables

- New service class implementing Responses streaming + function calling
- New CLI alias (`openai-responses`)
- Updated model config handling for Responses reasoning settings
- OpenAI SDK pin upgrade to latest
- 3-turn and full benchmark validation artifacts
