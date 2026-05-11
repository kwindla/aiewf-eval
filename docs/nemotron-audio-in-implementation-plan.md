# Nemotron Audio-In Implementation Plan

Date: 2026-05-10

## Goal

Add support for an audio-input, text-output LLM class to `aiewf-eval`.

The first target is Nemotron 3 Nano Omni, served through an OpenAI-compatible
chat-completions endpoint:

- Base URL: `http://192.168.7.228:8000/v1`
- Tailscale base URL: `http://100.117.133.113:8000/v1`
- Endpoint: `/chat/completions`
- Model: `nemotron_3_nano_omni`

This is not a realtime websocket model. The request shape is chat completions,
but user turns contain audio instead of transcript text.

## Upstream Source To Reuse

The most useful implementation is in:

- `../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py`
- `../nemotron-nano-omni/tests/test_nemotron_omni_conversation_cache.py`

Useful pieces to adapt:

- `LLMService`-based chat-completions streaming over `aiohttp`.
- Conversion from Pipecat universal `input_audio` parts to vLLM-compatible
  `audio_url` data URLs.
- SSE parsing for streaming text and tool-call deltas.
- `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` framing.
- TTFB and token usage metrics.
- Conversation-cache state machine:
  - full request before cache is committed
  - suffix-only request after successful cache commit
  - one full-context retry on `ConversationCacheMissError`
  - keep suffix-only disabled for the next turn after cache-miss recovery
- The current upstream regression tests around suffix-only payloads, cache miss
  recovery, and mixed audio/text turns.

Pieces not to copy directly:

- VAD and bot audio collection.
- RTVI client handling.
- The local `run_bash` tool implementation and policy layer.
- Bot-specific system prompts.

## Local Design

Add this as two local components:

1. `AudioInPipeline`
2. `NemotronAudioInLLMService`

The pipeline should own benchmark audio loading and turn sequencing. The service
should own Nemotron's request conversion, streaming response handling, and
conversation-cache behavior.

This avoids making the existing `TextPipeline` carry audio-specific conditionals
and keeps the service reusable for other audio-input benchmark flows.

### Pipeline class strategy: subclass, not copy

`AudioInPipeline` should subclass `TextPipeline`, not copy it. The only required
pipeline-level differences are how the first and subsequent user messages are
built. Subclassing keeps `NextTurn`, tool recording, assistant aggregation,
metrics forwarding, recovery hooks, and end-of-turn logic identical to the
text pipeline.

Override exactly four hooks:

- `_create_llm()` - instantiate `NemotronAudioInLLMService`.
- `_setup_context()` - build the initial system + first-user-audio message list.
- `_queue_next_turn()` - append a user-audio message instead of text.
- `_queue_recovery_turn()` - keep the synthetic recovery nudge as text.

Inherit `_setup_llm()`, `_build_task()`, and `_queue_first_turn()` unchanged.
`TextPipeline._queue_first_turn()` only queues `LLMRunFrame`; the first user
message must already be in `LLMContext`, which the overridden `_setup_context()`
must guarantee.

## Pipeline

Create:

- `src/multi_turn_eval/pipelines/audio_in.py`

Start from the current `TextPipeline` structure because it already has the right
multi-turn mechanics:

- universal `LLMContext`
- `LLMContextAggregatorPair`
- tool recorder
- assistant aggregation
- `NextTurn`
- metrics forwarding
- recovery turn support
- transcript writing through `BasePipeline`

The audio pipeline should replace only the user-message construction.

### User Audio Message

All user-audio-message construction must go through one helper so the first turn
and every subsequent turn behave identically:

```python
def _build_audio_user_message(self, turn_index: int) -> dict:
    actual_index = self._get_actual_turn_index(turn_index)
    path = self.benchmark.get_audio_path(actual_index)
    if not path.exists():
        raise FileNotFoundError(
            f"Audio file missing for turn {actual_index}: {path}"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    content = []
    prompt = self._audio_prompt()  # reads MTE_AUDIO_IN_PROMPT
    if prompt:
        content.append({"type": "text", "text": prompt})
    content.append({
        "type": "input_audio",
        "input_audio": {"data": data, "format": "wav"},
    })
    return {"role": "user", "content": content}
```

Call sites:

- `_setup_context()` builds initial `[system, audio_user_message_for_turn_0]`.
- `_queue_next_turn()` appends `_build_audio_user_message(self.turn_idx)` via
  `self.context.add_messages([...])`, then calls
  `_sanitize_openai_context_tool_ids()` (inherited), updates
  `self.last_msg_idx`, and queues `LLMRunFrame`.
- `_queue_recovery_turn()` does not call this helper. Recovery stays as text
  `"Please go ahead."`.

The `_setup_context()` override must also preserve the non-message setup that
`TextPipeline` does: read `benchmark.tools_schema`, construct `LLMContext`,
construct `LLMContextAggregatorPair`, and set `self.last_msg_idx =
len(messages)`.

The optional audio prompt is controlled by `MTE_AUDIO_IN_PROMPT`, defaults to
`Listen to the audio and respond to the spoken instruction.`, and can be disabled
with an empty string, `0`, `false`, or `none`.

Do not send the benchmark transcript as user text. The transcript is the
expected spoken content and stays available for local recording and judging.

### Turn Indexing

When `--only-turns` is used, preserve the actual benchmark turn index for audio
file selection. For example, `--only-turns 5` must load `turn_005.wav`, not the
first effective turn's `turn_000.wav`.

Use `BasePipeline._get_actual_turn_index(self.turn_idx)` when resolving audio.

### Recovery Turns

For synthetic recovery turns, keep text content:

```python
{"role": "user", "content": "Please go ahead."}
```

Rationale: only `TextPipeline` enables `supports_recovery` today. We inherit that
flag through subclassing but emit text rather than synthesizing audio for the
nudge. This is a v1 simplification, not a correctness claim. Risks to revisit:

- The model may respond differently to a mid-conversation modality switch than
  to a pure audio follow-up.
- Some audio-only models may reject mixed-modality user messages.

If the model rejects text recovery, either disable recovery for this pipeline
(`supports_recovery = False`) or add a pre-rendered "please go ahead" WAV and
send that for recovery turns. Do not block v1 on this; measure first.

### Service Construction

`BasePipeline._create_llm()` currently has OpenAI/Gemini/etc. provider logic and
requires `OPENAI_API_KEY` for `service_name == "nemotron"`. A new service alias
like `nemotron-audio-in` will not be configured correctly by that generic path.

Therefore `AudioInPipeline` should override `_create_llm()` and instantiate
`NemotronAudioInLLMService` with audio-specific defaults:

- `model`
- `api_key=os.getenv("NEMOTRON_AUDIO_IN_API_KEY") or os.getenv("OPENAI_API_KEY") or None`
- `base_url=os.getenv("MTE_NEMOTRON_AUDIO_IN_BASE_URL", "http://192.168.7.228:8000/v1")`
- request timeout from env
- conversation cache disabled by default
- generated `conversation_id` only when conversation cache is enabled
- suffix-only disabled by default and honored only when conversation cache is on

Defensive contract: `_create_llm()` should assert that the passed
`service_class` is `NemotronAudioInLLMService` (or a subclass) and raise a clear
error otherwise. This prevents `--service openai --pipeline audio-in` from
silently instantiating the wrong service. `--service` cannot be omitted because
`BasePipeline.run()` enforces `requires_service = True`, and `TextPipeline` keeps
that default.

Conversation IDs must be unique per benchmark run. Reusing a fixed conversation
ID across independent runs can violate the server's append-only cache contract.

Recommended env vars:

- `MTE_NEMOTRON_AUDIO_IN_BASE_URL` (default `http://192.168.7.228:8000/v1`)
- `MTE_NEMOTRON_AUDIO_IN_TIMEOUT_SECS` (default `180`)
- `MTE_NEMOTRON_AUDIO_IN_THINKING` (default `0`)
- `MTE_NEMOTRON_AUDIO_IN_TOP_K` (default `1`)
- `MTE_NEMOTRON_AUDIO_IN_TEMPERATURE` (default `0.2`)
- `MTE_NEMOTRON_AUDIO_IN_MAX_TOKENS` (default `1024`)
- `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE` (default `0`; enable only after
  full-context transport smokes pass)
- `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` (default `0`; only honored when
  conversation cache is on and the tool-call guard has been tested)

Do not expose `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_ID`. The service generates a
fresh `uuid.uuid4()` per run when cache mode is enabled. Hard-coding an ID via
env across multiple runs violates the server's append-only cache contract. If a
debug override is genuinely needed later, add it as a CLI flag
(`--conversation-id`) rather than an env var so it cannot survive across
invocations by accident.

## Service

Create:

- `src/multi_turn_eval/services/nemotron_audio_in.py`

Make it a local `LLMService` subclass rather than an `OpenAILLMService`
subclass. The standard OpenAI service does not give enough control over:

- `input_audio` to `audio_url` conversion
- top-level `conversation_id`
- top-level `conversation_require_cache`
- exact cache-miss retry behavior
- top-level `chat_template_kwargs`
- raw SSE parsing and tool-call delta handling without relying on OpenAI SDK
  objects

Dependency note: `aiohttp` 3.13.x is already in `uv.lock` via Pipecat. No new
dependency needs to be added to `pyproject.toml`.

### Constructor Defaults

Use Nemotron Omni defaults:

```python
model = "nemotron_3_nano_omni"
base_url = "http://192.168.7.228:8000/v1"
temperature = 0.2
max_tokens = 1024
top_p = None
top_k = 1
chat_template_kwargs = {"enable_thinking": False}
request_timeout_secs = 180.0
```

Do not require an API key. If one is supplied, send it as a bearer token.

These are constructor defaults. `AudioInPipeline._create_llm()` supplies
env-derived overrides such as `MTE_NEMOTRON_AUDIO_IN_BASE_URL`, so keep this
block and the pipeline construction defaults aligned.

### Frame Handling

The service must handle at least:

- `StartFrame`: create `aiohttp.ClientSession`
- `EndFrame` / `CancelFrame`: close session and cancel in-flight generation
- `LLMContextFrame`: trigger inference against `frame.context`
- `LLMRunFrame`: ignore with debug logging. Do not push downstream — the
  user aggregator consumes `LLMRunFrame` and emits `LLMContextFrame`, so the
  service should never see one in the inherited flow; re-emitting could loop
- `InterruptionFrame`: cancel in-flight generation if needed

This is compatible with the inherited `TextPipeline` flow. `TextPipeline` queues
`LLMRunFrame`, but `LLMContextAggregatorPair.user()` consumes that frame and
emits an `LLMContextFrame` containing the current `LLMContext`. The service
therefore normally sees `LLMContextFrame`, not the original `LLMRunFrame`.

Do not implement a service-side "store context on `LLMContextFrame`, run later
on `LLMRunFrame`" gate for v1. That would invert Pipecat's current universal
context flow and can leave the service waiting for a frame that never arrives.
If a future pipeline needs direct `LLMRunFrame` support, make that an explicit
new path with a clearly stored context reference.

### Message Conversion

Convert Pipecat universal messages into OpenAI/vLLM chat messages.

Handle roles:

- `developer` -> `system`
- `system`
- `user`
- `assistant`
- `tool`

Handle content parts:

- `input_audio` -> `audio_url`
- `audio_url` -> pass through
- `text` -> pass through
- `image_url` -> pass through for forward compatibility

The important conversion:

```python
{
    "type": "input_audio",
    "input_audio": {"data": "...", "format": "wav"}
}
```

becomes:

```python
{
    "type": "audio_url",
    "audio_url": {
        "url": "data:audio/wav;base64,..."
    }
}
```

### Tool Schema Conversion

`TextPipeline._setup_context()` stores benchmark tools on `LLMContext.tools`.
Because this service is a raw `LLMService` subclass, it will not automatically
get `OpenAILLMService`'s adapter conversion.

Before building the payload, convert `context.tools` to OpenAI chat-completions
tool definitions. The simplest local approach is to reuse Pipecat's
`OpenAILLMAdapter.from_standard_tools()`, which is an instance method (not a
classmethod). Instantiate the adapter on the service and call it per request:

```python
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter

self._tools_adapter = OpenAILLMAdapter()
# ... later, per request:
tools_payload = self._tools_adapter.from_standard_tools(context.tools)
```

Alternatively, write a local helper that maps each `FunctionSchema` to:

```python
{"type": "function", "function": function_schema.to_default_dict()}
```

Payload rules:

- Include `tools` only when converted tools are present.
- Include `tool_choice` when `context.tool_choice` is set.
- Omit Pipecat/OpenAI sentinel values such as `NOT_GIVEN`; do not pass them to
  `aiohttp` JSON serialization.
- Keep tool definitions stable across any same-turn tool follow-up request.
  vLLM feeds `tools` into the chat template, so dropping them can invalidate
  exact cache attachment.
- Do not expose callable tools in live smoke runs until the service can parse
  streamed tool-call deltas and emit `FunctionCallFromLLM` frames. During the
  transport-only phase, either omit `tools` from the payload or force
  `tool_choice = "none"`.

Add service tests proving benchmark `ToolsSchemaForTest` appears in the outgoing
payload as OpenAI-compatible function tools. Without this, tool-call parsing can
be correct while the model never receives tool definitions.

### Payload Shape

Build requests like:

```python
{
    "model": "nemotron_3_nano_omni",
    "messages": [...],
    "stream": true,
    "stream_options": {"include_usage": true},
    "temperature": 0.2,
    "max_tokens": 1024,
    "top_k": 1,
    "chat_template_kwargs": {"enable_thinking": false},
    "tools": [...],  # only after tool-call support is enabled
}
```

When conversation cache is enabled, add a generated per-run `conversation_id`.
Add:

```python
"conversation_require_cache": true
```

only when sending a strict suffix-only request after the service has confirmed
that the conversation cache is committed.

Do not send `conversation_logical_checkpoint_token_count`.

### Conversation Cache State

Conversation-cache state lives in `NemotronAudioInLLMService`, not in
`AudioInPipeline`. The pipeline only reads env vars and passes constructor
settings. The service owns:

```python
_conversation_cache_enabled: bool
_conversation_id: str | None
_suffix_only_conversation: bool
_conversation_cache_committed: bool
```

These fields are per service instance, so the generated conversation ID is
per benchmark run. The service also owns cache-state transitions after
successful responses, cache misses, and tool-call turns.

Behavior with `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0` (v1 default):

- Always send full context.
- Do not send `conversation_id`.
- Do not send `conversation_require_cache`.
- The cache state machine is inactive.

Behavior with conversation cache on and suffix-only off:

- Send full context every turn, but include a generated `conversation_id`.
- The server can cache, but the client never sends suffix-only payloads.
- This is safe with locally handled tool calls.

Behavior with both conversation cache and suffix-only on (staged after tool
support):

1. First turn: send full context with `conversation_id`. Do not send
   `conversation_require_cache`.
2. After a successful response, mark `_conversation_cache_committed = True`.
3. On later plain audio/text turns, send only the latest user message and set
   `conversation_require_cache = True`.
4. On `ConversationCacheMissError` (HTTP 409, `error.type ==
   "ConversationCacheMissError"`): retry once with full context and no
   `conversation_require_cache`. After recovery, set
   `_conversation_cache_committed = False` so the next turn also sends full
   context.
5. After any turn that contained tool calls handled locally (i.e. results not
   sent back to vLLM), force `_conversation_cache_committed = False` for the
   next turn.

### Canonical Messages

v1 must not maintain a parallel `_canonical_messages` list in the service.
Pipecat `LLMContext` is the single source of truth. Each turn, the service:

1. Reads `LLMContext.get_messages()` from the context-frame payload.
2. Converts each message through `_convert_context_message()` /
   `_convert_context_content_part()` (`input_audio` -> `audio_url`).
3. Builds either a full-context payload or a latest-user-only payload, depending
   on the cache flags above.

There is no service-internal append step. Do not add `_canonical_messages`,
`_commit_canonical_messages()`, or `_canonical_messages_from_context()` in v1.

The upstream service's canonical-messages optimization can be revisited in v2
once suffix-only mode is empirically validated for plain turns and tool-result
suffixing is implemented end-to-end. Until then, converting the full Pipecat
context each turn is acceptable.

### Tool Calls

Reuse upstream `_merge_tool_call_delta()` and `_finalize_tool_calls()` directly.
They operate on raw JSON delta dicts, which matches the new aiohttp/SSE
implementation. Do not adapt the OpenAI-SDK-object merge code from
`src/multi_turn_eval/services/nemotron.py`; that code assumes parsed SDK objects
and would have to be rewritten anyway.

Do not copy upstream `run_bash` or its policy layer.

After finalizing tool calls, parse JSON function arguments, build
`FunctionCallFromLLM` objects, and call `await self.run_function_calls(...)` so
Pipecat's `run_function_calls()` plus this repo's `ToolCallRecorder` handle
execution and recording.

The service should preserve tool-call IDs. If the server omits an ID, synthesize
a stable non-empty ID like `call_0`, `call_1`.

Because `BasePipeline._sanitize_openai_context_tool_ids()` currently only
special-cases service names such as `openai`, `openrouter`, `nemotron`, and
`modal`, add `nemotron-audio-in` to that sanitizer allowlist. The sanitizer
patches missing tool IDs and drops empty assistant stubs on later turns; without
the allowlist entry, those bugs can leak into subsequent requests.

### Streaming

Use SSE parsing compatible with vLLM chat completions:

- Ignore malformed events with debug logging.
- Stop on `[DONE]`.
- On usage chunks, emit `LLMTokenUsage`. Note that for audio-input models,
  `prompt_tokens` may include audio tokens/codes rather than text BPEs; record
  the values as-is in v1 and log that the run is `modality=audio_in`.
- Stop TTFB on first visible text token or first tool-call delta.
- Push streamed text through `LLMTextFrame` or the service helper used by
  `LLMService`.
- Emit `LLMFullResponseStartFrame` before streaming.
- Emit `LLMFullResponseEndFrame` in `finally`.

Handle error responses:

- HTTP 409 with `error.type == "ConversationCacheMissError"`:
  raise a local `ConversationCacheMissError`.
- Other non-200 responses:
  raise a clear runtime error and emit `ErrorFrame`.

### Tracing

Add optional request tracing behind an env var:

- `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR`

Trace files should redact or summarize large base64 audio by default. Full WAV
base64 payloads will make traces huge and hard to review.

Record at least:

- request sequence
- attempt number
- conversation ID
- whether suffix-only was used
- whether cache was required
- role summary
- audio part count
- response status/error

## CLI Wiring

Update `src/multi_turn_eval/cli.py`.

Add service alias:

```python
"nemotron-audio-in": "multi_turn_eval.services.nemotron_audio_in.NemotronAudioInLLMService"
```

Add pipeline alias:

```python
"audio-in": "multi_turn_eval.pipelines.audio_in.AudioInPipeline"
```

Update `infer_pipeline()` so the selected service alias participates in
dispatch:

```python
def infer_pipeline(model: str, service: str | None = None) -> str:
    m = model.lower()
    s = (service or "").lower()
    if s == "nemotron-audio-in":
        return "audio-in"
    if "nemotron_3_nano_omni" in m and s != "nemotron":
        return "audio-in"
    # ... existing rules unchanged ...
    return "text"
```

Then update the caller in `_run()` to pass `service`:

```python
pipeline_type = infer_pipeline(model, service)
```

Intent:

- Explicit `--service nemotron-audio-in` always selects `audio-in`.
- Explicit `--service nemotron` opts out and can still use text-mode behavior if
  someone deliberately tests it.
- Model-name-only inference selects `audio-in` for `nemotron_3_nano_omni`.
- Users can still pass `--pipeline audio-in` to override.

Keep the matching conservative. Do not route every model containing `omni` to
this path; unrelated providers may use that word for other APIs.

Update CLI help text to mention `audio-in`.

## Tests

There is no committed `tests/` directory in this checkout, but upstream has
useful regression tests. Add a local `tests/` directory if that is the chosen
test convention for this branch.

Suggested files:

- `tests/test_audio_in_pipeline.py`
- `tests/test_nemotron_audio_in_service.py`

### Pipeline Tests

Cover:

- first turn creates a user message with `input_audio`
- next turn uses the actual benchmark turn index for audio file lookup
- `--only-turns` does not shift audio filenames
- transcript text is not included as user text by default
- optional generic audio prompt is included only when configured
- recovery turn remains text
- missing audio file fails with a useful error

### Service Tests

Cover:

- `input_audio` converts to `audio_url`
- existing `audio_url` passes through unchanged
- top-level `chat_template_kwargs` is emitted
- cache-off requests include neither `conversation_id` nor
  `conversation_require_cache`
- cache-on/full-context requests include `conversation_id` but not
  `conversation_require_cache`
- suffix-only request sends only the latest user message
- cache miss retries once with full context and disables suffix-only for next
  turn
- benchmark `ToolsSchemaForTest` converts to OpenAI-compatible `tools`
- `NOT_GIVEN` / unset `tools` / unset `tool_choice` are omitted from JSON
- malformed SSE events are ignored
- usage chunks emit token metrics
- streaming tool-call deltas merge correctly
- missing tool-call IDs are synthesized
- tool-call turns disable suffix-only unless tool results are known to have been
  sent to vLLM
- `conversation_logical_checkpoint_token_count` is never sent

### Upstream Tests To Port

Port or adapt these assertions for v1:

- `test_suffix_only_uses_latest_text_turn_after_audio_turn`
- `test_successful_turn_marks_suffix_only_ready`
- `test_cache_miss_recovery_disables_suffix_only_for_next_turn`

Defer these upstream tests until `_payload_after_tool_calls()` suffix optimization
is implemented:

- `test_payload_after_tool_calls_uses_tool_suffix_with_conversation_cache`
- `test_payload_after_tool_calls_keeps_full_history_without_conversation_id`

Skip upstream bash-policy tests because this repo should not copy that tool
implementation.

## Smoke Test

All smoke tests assume `MTE_NEMOTRON_AUDIO_IN_TRACE_DIR` is set to a writable
directory (e.g. `export MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d)`). Most
"Expected checks" below are observable only via the trace JSON files; CLI
output alone does not verify cache semantics.

Each smoke command also corresponds to a specific point in the implementation
sequence — do not run a smoke against code that has not yet reached its step.

### Smoke 1: audio transport (after sequence step 4, before step 7)

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0 \
MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 0 \
  --verbose
```

Expected checks (inspect trace JSON unless stated otherwise):

- request reaches `http://192.168.7.228:8000/v1/chat/completions`
- request includes audio data URL after conversion
- before step 7: request payload has `tool_choice: "none"` or omits `tools`
  entirely; no callable tools are exposed
- request does not include `conversation_id`
- first response streams text (visible in CLI / transcript)
- transcript records user text from benchmark metadata and assistant text from
  model output

### Smoke 2: full-context multi-turn, cache off (after step 6)

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0 \
MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 0,1 \
  --verbose
```

Expected checks (inspect trace JSON):

- second request sends full context (both messages present)
- no suffix-only payload is used
- neither request includes `conversation_id` or `conversation_require_cache`

### Smoke 3: tool-call turn, cache off (after step 8)

Pick a benchmark turn whose `required_function_call` is set. Run with cache
off; tracing on. Expected checks (trace JSON):

- request payload includes benchmark tool definitions as OpenAI-compatible
  `tools`
- tool-call deltas are merged and parsed; `FunctionCallFromLLM` frames are
  emitted; `ToolCallRecorder` records the call
- the next scripted turn sends full context
- no `conversation_require_cache` is sent on any request

### Smoke 4: full-context cache mode (after step 10)

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1 \
MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY=0 \
MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 0,1 \
  --verbose
```

Expected checks (inspect trace JSON):

- both requests use the same generated conversation ID
- neither request sends `conversation_require_cache`
- both requests carry full context

### Smoke 5: suffix-only mode (after step 11)

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1 \
MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY=1 \
MTE_NEMOTRON_AUDIO_IN_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 0,1 \
  --verbose
```

Expected checks (inspect trace JSON):

- turn 0 sends full context with `conversation_id` and no
  `conversation_require_cache`
- turn 1 sends a single-message suffix payload (just the latest user message)
  with `conversation_id` and `conversation_require_cache: true`
- both requests share the same conversation ID

Then re-run Smoke 3 with suffix-only on and confirm that, after a tool-call
turn, the next scripted turn falls back to full context (the disable-after-
tool-call guard). Do not rely on suffix-only across tool branches until
tool-result suffixing is implemented and tested.

## Implementation Sequence

v1 transport first; cache machinery last.

1. Add `NemotronAudioInLLMService` with message conversion (`input_audio` ->
   `audio_url`), full-context payload building, SSE streaming, and basic
   metrics. No conversation cache, no callable tools, no canonical state.
2. Add `AudioInPipeline` as a `TextPipeline` subclass that overrides
   `_create_llm()`, `_setup_context()`, `_queue_next_turn()`, and
   `_queue_recovery_turn()`. Centralize audio loading in
   `_build_audio_user_message()`.
3. Wire `nemotron-audio-in` and `audio-in` aliases in the CLI. Update
   `infer_pipeline()` so the chosen `--service` participates in dispatch.
4. Add `nemotron-audio-in` to the sanitizer allowlist in
   `BasePipeline._sanitize_openai_context_tool_ids()`.
5. Smoke test 1: `--only-turns 0` audio transport check with conversation cache
   off.
6. Smoke test 2: `--only-turns 0,1` with conversation cache off, verifying
   full-context multi-turn behavior.
7. Add OpenAI-compatible tool-schema conversion and streaming tool-call parsing
   using upstream `_merge_tool_call_delta()` / `_finalize_tool_calls()`, wired to
   `self.run_function_calls()`.
8. Smoke test 3: a benchmark turn that requires a tool call, still with
   conversation cache off.
9. Add tracing with audio-payload redaction (for example, sha256 + char count,
   mirroring upstream `_trace_json_value()`).
10. Enable conversation cache full-context mode behind
    `MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1`. Re-run smoke tests 2 and 3.
11. Enable suffix-only mode behind `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY=1`.
    Implement the disable-after-tool-call guard. Re-run smoke test 3.
12. Add unit tests covering payload conversion, tool-schema conversion, sentinel
    omission, cache state transitions, cache-miss recovery, and the tool-call
    disable guard.
13. Defer service-owned canonical messages and `_payload_after_tool_calls()`
    suffix optimization to v2.
