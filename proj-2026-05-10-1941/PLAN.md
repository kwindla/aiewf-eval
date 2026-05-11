# Plan: Nemotron Audio-In Service and Pipeline

Project directory: `./proj-2026-05-10-1941`

## Context

Add audio-input / text-output LLM support to `aiewf-eval`, targeting Nemotron 3 Nano Omni served via an OpenAI-compatible chat-completions endpoint at `http://192.168.7.228:8000/v1`. User turns carry WAV audio (`input_audio` → `audio_url` data URLs); the model streams text back. Implementation follows `docs/nemotron-audio-in-implementation-plan.md`, which has been adversarially reviewed three times. v1 keeps conversation cache disabled by default and pushes the suffix-only optimization to a later step.

## Reference implementations

- **Upstream service to adapt**: `../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py` — full reference for SSE streaming, payload shape, `_convert_context_message`/`_convert_context_content_part`, `_merge_tool_call_delta`, `_finalize_tool_calls`, `_trace_json_value` (audio redaction). Divergences from upstream:
  - **No service-owned `_canonical_messages`** — Pipecat `LLMContext` is the single source of truth in v1. Defer `_canonical_messages_from_context`, `_commit_canonical_messages`, `_payload_after_tool_calls` to v2.
  - **No `run_bash` tool or policy layer.**
  - **`LLMRunFrame` is ignored, not used as trigger** — `LLMContextFrame` triggers inference (this matches upstream's normal flow once the aggregator converts).
- **Upstream regression tests to port (v1)**: `../nemotron-nano-omni/tests/test_nemotron_omni_conversation_cache.py:334,376,405`.
- **Local pipeline reference**: `src/multi_turn_eval/pipelines/text.py` — `AudioInPipeline` subclasses `TextPipeline` (text.py:273). Realtime audio precedent for `benchmark.get_audio_path()` usage: `src/multi_turn_eval/pipelines/realtime.py:518-539`.
- **Local service reference**: `src/multi_turn_eval/services/nemotron.py` is the OpenAILLMService-based text path; do not reuse its tool-call merge code (it operates on OpenAI SDK objects, not raw JSON deltas).
- **Pipecat tool-schema conversion**: `OpenAILLMAdapter.from_standard_tools()` is an instance method (not classmethod) in pipecat 0.0.101.
- **Raw LLMService caveat**: unlike `BaseOpenAILLMService`, a bare `LLMService` does not trigger inference on `LLMContextFrame` or opt into metrics automatically. The new service must implement that frame handling and `can_generate_metrics()` itself.

## Current state

Files that will be modified or created:

- `src/multi_turn_eval/services/nemotron_audio_in.py` — **NEW**. Local `LLMService` subclass; raw `aiohttp` + SSE.
- `src/multi_turn_eval/pipelines/audio_in.py` — **NEW**. `TextPipeline` subclass.
- `src/multi_turn_eval/cli.py` — modify `SERVICE_ALIASES` (line 31), `PIPELINE_CLASSES` (line 52), `infer_pipeline()` (line 108), and the call site at line 217.
- `src/multi_turn_eval/pipelines/base.py:170` — extend `_sanitize_openai_context_tool_ids` allowlist to include `"nemotron-audio-in"`.
- `pyproject.toml` — update only if pytest dependency/config is needed so `uv run pytest tests` works without collecting `pipecat/tests`.
- `tests/test_audio_in_pipeline.py` — **NEW** (tests directory does not exist yet).
- `tests/test_nemotron_audio_in_service.py` — **NEW**.

Key existing-code references:

- `BasePipeline._create_llm()` — `src/multi_turn_eval/pipelines/base.py:249`. Provider-specific; new pipeline must override.
- `BasePipeline._get_actual_turn_index()` — `base.py:150`.
- `BasePipeline._sanitize_openai_context_tool_ids()` — `base.py:160`. Allowlist at line 170: `{"openai", "openrouter", "nemotron", "modal"}`.
- `TextPipeline._setup_context` — `text.py:293-310`. Builds `[system, user]` initial messages, sets `self.context`, `self.context_aggregator`, `self.last_msg_idx`.
- `TextPipeline._queue_next_turn` — `text.py:439-451`. Appends user message, sanitizes, updates `last_msg_idx`, queues `LLMRunFrame`.
- `TextPipeline._queue_first_turn` — `text.py:428-437`. Inherits as-is; only queues `LLMRunFrame`.
- `LLMUserContextAggregator._handle_llm_run` — `.venv/.../llm_response_universal.py:571`. Consumes `LLMRunFrame`, emits `LLMContextFrame`. This is why the service sees `LLMContextFrame`, not `LLMRunFrame`.
- `BenchmarkConfig.get_audio_path` — `benchmarks/aiwf_medium_context/config.py:26`. Returns `Path` to `benchmarks/_shared/audio/turn_NNN.wav`.
- `aiohttp 3.13.x` is already in `uv.lock` via pipecat. No new dependency required.

## Rules

### Plan adherence

- **The implementation plan in `docs/nemotron-audio-in-implementation-plan.md` is authoritative for behavior.** This project plan is authoritative for execution order where it is more specific; in particular, tracing is pulled before the first live smokes so those smokes have inspectable payload JSON. If a behavior is ambiguous, defer to the source plan's prose for that specific section, not invention.
- **v1 must not maintain service-owned canonical messages.** Pipecat `LLMContext` is the single source of truth.
- **v1 must not enable conversation cache by default.** `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE` and `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` default to `0`.
- **Never expose `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_ID`** — the service generates a fresh `uuid.uuid4()` per run when cache is on.

### Safety and sequencing

- Each step ends in a working state: code compiles, existing benchmarks for unrelated models continue to run.
- Smoke tests are validation gates between code steps, run manually against the live endpoint. Do not delegate smoke-test execution to Codex.
- Do not enable tools in payloads until streaming tool-call parsing is implemented (step 6). Before then, send no `tools` field or `tool_choice: "none"`.
- After any tool-call turn handled locally (results not sent back to vLLM), force `_conversation_cache_committed = False` for the next turn.

### Frame and context handling

- The service handles `LLMContextFrame` as the inference trigger. `LLMRunFrame` is ignored with a debug log; never push it downstream from the service.
- Convert `LLMContext.get_messages()` per request — no service-side canonical state.

### Tool-schema and ID hygiene

- Use `OpenAILLMAdapter().from_standard_tools(context.tools)` — instance method, not classmethod.
- Omit `NOT_GIVEN` / unset values from outgoing JSON.
- Preserve tool-call IDs; synthesize stable IDs (`call_0`, `call_1`) when the server omits them.

### Tracing

- Default to redacted audio payloads (sha256 + char count) when `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR` is set; never write full base64 audio to trace files by default.

### Tests

- Unit tests use mocked aiohttp transport — do not require the live endpoint.
- Skip upstream `run_bash` / bash-policy tests entirely.

## Steps

- [x] **1. Service skeleton: transport, message conversion, full-context streaming**
  Create `src/multi_turn_eval/services/nemotron_audio_in.py`. Define `NemotronAudioInLLMService` as a `pipecat.services.llm_service.LLMService` subclass. Constructor accepts `model`, `api_key` (optional bearer), `base_url`, `temperature`, `max_tokens`, `top_p`, `top_k`, `chat_template_kwargs`, `request_timeout_secs`, `conversation_cache_enabled`, `suffix_only_conversation`. Defaults per `docs/nemotron-audio-in-implementation-plan.md` Constructor Defaults section. Constructor must call `self.set_model_name(model)` so metrics carry the model, and the class must implement `can_generate_metrics() -> bool` returning `True`; otherwise Pipecat's `start_ttfb_metrics()` and `start_llm_usage_metrics()` helpers are no-ops. Frame handling must be explicit because bare `LLMService` does not implement OpenAI-style context processing: override `process_frame`, call `await super().process_frame(frame, direction)`, handle `LLMContextFrame` by triggering inference, ignore `LLMRunFrame` with debug logging (no downstream push), cancel in-flight generation on `InterruptionFrame`, pass through unhandled frames, create `aiohttp.ClientSession` on `StartFrame`, and close session plus cancel in-flight generation on `EndFrame`/`CancelFrame`. Track the current inference as a cancellable `_generation_task` created with `self.create_task(...)`, mirroring upstream. Implement `_convert_context_message` / `_convert_context_content_part` mirroring upstream `nemotron_omni.py:425-495`: roles `developer→system`, `system`/`user`/`assistant`/`tool` pass through; `input_audio→audio_url` with `data:audio/wav;base64,` URL, `audio_url`/`text`/`image_url` pass through. Build payload per plan's Payload Shape section but with **no `tools`** in v1 (force `tool_choice: "none"` if any tool surface is touched). SSE streaming: ignore malformed events, stop on `[DONE]`, call `await self.start_llm_usage_metrics(LLMTokenUsage(...))` on usage chunks, stop TTFB on first visible text/tool-call delta, push streamed text with `_push_llm_text()` or `LLMTextFrame`, wrap the request in `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame`, and call `start_processing_metrics()` / `stop_processing_metrics()` around the request. Define a local `ConversationCacheMissError(RuntimeError)` class but do not exercise it in v1. Cache state on service instance: `_conversation_cache_enabled`, `_conversation_id`, `_suffix_only_conversation`, `_conversation_cache_committed`. With cache off (v1 default), payload omits `conversation_id` and `conversation_require_cache`. No canonical messages list.
  Key files: `src/multi_turn_eval/services/nemotron_audio_in.py`

- [x] **2. Pipeline subclass with audio loading helper**
  Create `src/multi_turn_eval/pipelines/audio_in.py`. `AudioInPipeline(TextPipeline)` overrides exactly four hooks. Implement `_build_audio_user_message(turn_index)` per plan's User Audio Message section: resolve actual benchmark turn via `_get_actual_turn_index`, call `benchmark.get_audio_path`, raise `FileNotFoundError` if missing, base64-encode WAV bytes, optionally prepend a text part from `MTE_AUDIO_IN_PROMPT` (default `"Listen to the audio and respond to the spoken instruction."`, disable via empty/`0`/`false`/`none`). Implement `_audio_prompt()` helper that reads the env var and returns the prompt string or `None`. Override `_setup_context` to build `[{role: system, ...}, _build_audio_user_message(0)]`, then mirror `TextPipeline._setup_context` for `tools = benchmark.tools_schema`, `LLMContext`, `LLMContextAggregatorPair`, `self.last_msg_idx = len(messages)`. Override `_queue_next_turn` to append `_build_audio_user_message(self.turn_idx)`, call `_sanitize_openai_context_tool_ids` (inherited), update `self.last_msg_idx`, queue `LLMRunFrame`. Override `_queue_recovery_turn` to keep `{"role": "user", "content": "Please go ahead."}` (text, not audio). Override `_create_llm` to assert `service_class` is `NemotronAudioInLLMService` (or subclass) and instantiate with env-derived: `api_key` (`NEMOTRON_AUDIO_IN_API_KEY` or `OPENAI_API_KEY` or `None`), `base_url` (`MTE_NEMOTRON_AUDIO_IN_BASE_URL`), `request_timeout_secs` (`MTE_NEMOTRON_AUDIO_IN_TIMEOUT_SECS`, default `180`), `temperature` (`MTE_NEMOTRON_AUDIO_IN_TEMPERATURE`, default `0.2`), `max_tokens` (`MTE_NEMOTRON_AUDIO_IN_MAX_TOKENS`, default `1024`), `top_k` (`MTE_NEMOTRON_AUDIO_IN_TOP_K`, default `1`), `chat_template_kwargs` derived from `MTE_NEMOTRON_AUDIO_IN_THINKING` (default `0` → `{"enable_thinking": False}`; any truthy value → `{"enable_thinking": True}`), and cache flags `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE` / `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` both defaulting to `0`. Pipeline-supplied values must override the service constructor defaults; keep both default sets numerically aligned.
  Key files: `src/multi_turn_eval/pipelines/audio_in.py`

- [x] **3. CLI wiring + sanitizer allowlist**
  Update `src/multi_turn_eval/cli.py`: add `"nemotron-audio-in"` to `SERVICE_ALIASES` (line 31) pointing at `multi_turn_eval.services.nemotron_audio_in.NemotronAudioInLLMService`; add `"audio-in"` to `PIPELINE_CLASSES` (line 52) pointing at `multi_turn_eval.pipelines.audio_in.AudioInPipeline`. Rewrite `infer_pipeline` (line 108) to accept an optional `service: str | None = None` parameter and short-circuit: explicit `--service nemotron-audio-in` → `audio-in`; model containing `nemotron_3_nano_omni` AND `service != "nemotron"` → `audio-in`; remaining rules unchanged. Update the caller at line 217 to pass `service`. Update `src/multi_turn_eval/pipelines/base.py:170` to add `"nemotron-audio-in"` to the sanitizer allowlist set. Update CLI help text to mention `audio-in` where the other pipelines are listed.
  Key files: `src/multi_turn_eval/cli.py`, `src/multi_turn_eval/pipelines/base.py`

- [x] **4. Tracing with audio-payload redaction**
  Pulled earlier in the sequence so smoke tests have observable trace JSON to validate against. Add `_write_trace_file`, `_trace_json_value`, and `_trace_request_id` to the service, mirroring `../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py:568-625`. Behind `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR`. Audio data URLs (`data:audio/...,...`) get replaced with `{"url": "<data-audio-base64 sha256=... chars=...>"}` by default. Record per-request: trace ID (`nemotron-<conv-or-no-conversation>-turn-NNN-attempt-MM`), conversation ID, suffix-only flag, cache-required flag, role summary, audio part count, response status/error. The fields written should include the actual outgoing JSON body (with audio redaction) so smoke tests can inspect payload presence/absence of `tools`, `conversation_id`, `conversation_require_cache`, and the full message list.
  Key files: `src/multi_turn_eval/services/nemotron_audio_in.py`

- [x] **5. Smoke tests 1 and 2 (manual, validates steps 1-4)**
  Manual verification — do not delegate to Codex. Set `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d)` and run the Smoke 1 and Smoke 2 commands from `docs/nemotron-audio-in-implementation-plan.md`. Confirm via trace JSON: requests reach `http://192.168.7.228:8000/v1/chat/completions`, audio data URL present (redacted form), no callable tools are exposed (`tools` omitted, or `tool_choice: "none"` if any tool surface is touched), no `conversation_id`, text streams back, transcript records both sides, second turn sends full context. If smoke fails, file findings and pause before step 6.
  Key files: (validation only — no code changes)

- [x] **6. Tool-schema conversion + streaming tool-call parsing**
  In `src/multi_turn_eval/services/nemotron_audio_in.py`: import `pipecat.adapters.services.open_ai_adapter.OpenAILLMAdapter` and instantiate `self._tools_adapter = OpenAILLMAdapter()` in the constructor. Before payload build, call `tools_payload = self._tools_adapter.from_standard_tools(context.tools)`; include `tools` in payload only when the result is non-empty and is not `NOT_GIVEN`; pass through `tool_choice` from `context.tool_choice` only when set and not `NOT_GIVEN`. Port `_merge_tool_call_delta` and `_finalize_tool_calls` from `../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py:902,933`. SSE handling: accumulate `delta.tool_calls` per index; on stream end, finalize, parse JSON `function.arguments`, synthesize `call_N` IDs for any missing IDs. Build `FunctionCallFromLLM` objects and call `await self.run_function_calls(...)` so Pipecat + `ToolCallRecorder` handle execution. Do NOT copy upstream `run_bash` or the bash policy layer. Verify outgoing JSON omits sentinel values (no `NOT_GIVEN` in payload).
  Key files: `src/multi_turn_eval/services/nemotron_audio_in.py`

- [x] **7. Smoke test 3 (manual, validates step 6)**
  Manual verification. The benchmark's tool-requiring turns at `benchmarks/_shared/turns.py:92,96` depend on prior context (turn 11's `submit_session_suggestion` needs the name supplied in turn 10). Single-turn smokes against turn 11 will likely fail for context reasons, not tool-call reasons. Run a contiguous range: `--only-turns 9,10,11,12`, with `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0` and `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d)`. Confirm via trace/logs/transcript: request includes `tools` (converted to OpenAI-compatible function definitions), tool-call deltas merge to a complete call on turn 11, `run_function_calls()` produces `FunctionCallsStartedFrame` plus downstream `FunctionCallInProgressFrame` / `FunctionCallResultFrame`, `ToolCallRecorder` records the call with `submit_session_suggestion`, turn 12 sends full context, no `conversation_require_cache` on any request. If anything fails, pause before step 8.
  Key files: (validation only)

- [x] **8. Conversation cache: full-context mode**
  Wire `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE` plumbing through `AudioInPipeline._create_llm` (already partially wired in step 2; this step makes it functional). When enabled and `suffix_only_conversation` is False: generate `_conversation_id = uuid.uuid4().hex` in `__init__` (or on first inference if simpler), include `conversation_id` in every payload, never send `conversation_require_cache`. On successful response set `_conversation_cache_committed = True`. Run Smoke 4 (manual) after Codex completes this step: same conversation ID across two requests, no `conversation_require_cache` in either, full context in both. Then re-run Smoke 2 and Smoke 3 with cache off to confirm no regressions in the cache-off paths (per the source plan's Implementation Sequence step 10, which mandates re-running smokes 2 and 3). Pause if any smoke fails.
  Key files: `src/multi_turn_eval/services/nemotron_audio_in.py`, `src/multi_turn_eval/pipelines/audio_in.py`

- [ ] **9. Suffix-only mode + tool-call disable guard**
  Wire `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` (only honored when cache is on). State machine per plan's Conversation Cache State / Behavior with both on: first turn sends full context with `conversation_id` and no `conversation_require_cache`; on success set `_conversation_cache_committed = True`; later plain turns send only the latest user message and set `conversation_require_cache = True`; on HTTP 409 with `error.type == "ConversationCacheMissError"` raise local `ConversationCacheMissError`, retry once with full context and no `conversation_require_cache`, then set `_conversation_cache_committed = False`; after any turn that contained locally-handled tool calls (results not sent back to vLLM), force `_conversation_cache_committed = False` for the next turn. The latest-user-message helper extracts the last user message from the converted context (no service-side canonical state). Run Smoke 5 (manual) after Codex completes: turn 0 full context, turn 1 single-message suffix with `conversation_require_cache: true`, same conversation ID; then re-run Smoke 3 (contiguous range `9,10,11,12`) with suffix-only on and confirm the post-tool-call turn (turn 12) falls back to full context.
  Key files: `src/multi_turn_eval/services/nemotron_audio_in.py`

- [ ] **10. Unit tests (pipeline + service)**
  Create `tests/` directory (currently absent) and add `tests/test_audio_in_pipeline.py` and `tests/test_nemotron_audio_in_service.py`. If pytest is not already available in the active environment, add the minimal test dependency/config needed for local execution; avoid a plain repository-wide `pytest` command because this checkout also contains `pipecat/tests`. Preferred command: `uv run pytest tests`. If adding pytest config, set `testpaths = ["tests"]`. Pipeline tests: first turn creates `input_audio` message; next turn uses actual benchmark index for audio file lookup; `--only-turns` does not shift audio filenames; transcript text never included in user audio messages by default; optional audio prompt included only when env set; recovery turn remains text; missing audio file raises `FileNotFoundError`. Service tests (mocked aiohttp transport — no live endpoint): `input_audio→audio_url` conversion; existing `audio_url` passes through; top-level `chat_template_kwargs` emitted; raw service implements `can_generate_metrics()` and calls `set_model_name(model)`; usage chunks call `start_llm_usage_metrics(LLMTokenUsage(...))`; cache-off requests omit `conversation_id` and `conversation_require_cache`; cache-on/full-context requests include `conversation_id` but omit `conversation_require_cache`; suffix-only request sends only the latest user message with `conversation_require_cache: true`; cache miss retries once with full context and disables suffix-only for next turn; benchmark `ToolsSchemaForTest` converts to OpenAI-compatible `tools`; `NOT_GIVEN` / unset `tools` / `tool_choice` omitted from JSON; malformed SSE events ignored; streaming tool-call deltas merge correctly; missing tool-call IDs synthesized; tool-call turns disable suffix-only when results not sent to vLLM; `conversation_logical_checkpoint_token_count` never sent. Port from `../nemotron-nano-omni/tests/test_nemotron_omni_conversation_cache.py`: `test_suffix_only_uses_latest_text_turn_after_audio_turn` (line 334), `test_successful_turn_marks_suffix_only_ready` (line 376), `test_cache_miss_recovery_disables_suffix_only_for_next_turn` (line 405). Skip the two `test_payload_after_tool_calls_*` tests (v2 — see Defer list in the plan) and skip upstream bash-policy tests entirely.
  Key files: `tests/test_audio_in_pipeline.py`, `tests/test_nemotron_audio_in_service.py`

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Service skeleton: transport, message conversion, full-context streaming | done | c266c66 | Import + payload checks pass |
| 2 | Pipeline subclass with audio loading helper | done | 6906ede | All verification checks pass |
| 3 | CLI wiring + sanitizer allowlist | done | 9767c18 | All dispatch + alias checks pass |
| 4 | Tracing with audio-payload redaction | done | cceaff2 | Redaction + lazy dir + 3 phases verified |
| 5 | Smoke tests 1 and 2 (manual) | done | 9f09573 | Both pass: cache-off, no tools, full context turn 2 |
| 6 | Tool-schema conversion + streaming tool-call parsing | done | 8e9c9ac | 5 tools converted, merge+finalize verified |
| 7 | Smoke test 3 (manual, contiguous range 9-12) | done | 67f8701 | submit_session_suggestion fired; cache off |
| 8 | Conversation cache: full-context mode | done | 7eded45 | Smoke 4 + re-runs of 2 and 3 pass |
| 9 | Suffix-only mode + tool-call disable guard | pending | — | Includes Smoke 5 + re-run Smoke 3 |
| 10 | Unit tests (pipeline + service) | pending | — | |
