# Plan: Vendor `nemotron_omni.py` and upgrade pipecat to 1.1.0

Date: 2026-05-13 (replaces the earlier "consume as dependency" version)

## Goal

Replace our local `src/multi_turn_eval/services/nemotron_audio_in.py` with a
direct vendored copy of upstream's `nemotron_omni.py`. Treat the vendored file
as a pinned snapshot — copy once, refresh when upstream stabilizes a tagged
version. Shrink `AudioInPipeline` to: audio-message construction, four
`TextPipeline` overrides, and swapping the assistant aggregator for the
upstream `NemotronAssistantAggregator`. Carry no parallel reimplementation of
the cache state machine.

## Why vendoring beats path-dependency

The earlier draft of this plan proposed consuming upstream as a path-based
editable uv dependency. Investigation in the same session revealed two things
that make vendoring the cleaner path:

1. **The relevant upstream code is one file.** `nemotron_omni.py` (2405 lines)
   has zero self-imports of `nemotron_voice.*` (verified via grep) — its only
   non-stdlib imports are `aiohttp`, `loguru`, and `pipecat.*`. Vendoring a
   single file is much simpler than introducing a cross-repo editable install.
2. **Upstream's pipecat is PyPI pipecat 1.1.0.** The `pipecat-core-code/`
   directory in upstream is a working copy of pipecat 1.1.0 with minimal
   delta. The four pipecat files `nemotron_omni.py` imports from
   (`services/llm_service.py`, `services/settings.py`,
   `processors/aggregators/llm_context.py`,
   `processors/aggregators/llm_response_universal.py`) are **byte-identical**
   between upstream's vendored pipecat and PyPI `pipecat-ai==1.1.0` (verified
   via `diff -r`). So we can install pipecat 1.1.0 from PyPI and the vendored
   `nemotron_omni.py` will work unchanged.

## What we're vendoring

Exactly one file, copied verbatim:

| Source | Destination |
|---|---|
| `../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py` | `src/multi_turn_eval/vendor/nemotron_omni.py` |

Plus a one-line `src/multi_turn_eval/vendor/__init__.py` and a sibling
`README.md` documenting the upstream commit hash and refresh procedure.

We will **not** vendor:
- bash-tool internals (the constants live in `nemotron_omni.py` but are
  inactive when `enable_bash_tool=False`)
- voice-bot aggregators (`AudioOnlyLLMUserAggregator`,
  `UserAudioContextCollector`, `AudioOnlySmartTurnStopStrategy`) — they handle
  live mic/WebRTC audio; our pipeline pre-builds audio messages
- the upstream test suite — our integration tests are sufficient
- the upstream `pipecat-core-code/` fork — we use PyPI pipecat 1.1.0 instead

## Pipecat upgrade audit

Pipecat goes from 0.0.101 → 1.1.0. The risk surface and mitigations:

### Verified compatible

- **AudioBufferProcessor**: method signatures unchanged between 0.0.101 and
  1.1.0 (only cosmetic `Optional[bytes]` → `bytes | None` type-hint diff).
  Our `WallClockAlignedAudioBufferProcessor` overrides
  `_sync_buffer_to_position` — that method exists with the same signature in
  1.1.0.
- **All existing service classes import paths** are unchanged:
  `pipecat.services.openai.llm.OpenAILLMService`,
  `pipecat.services.anthropic.llm.AnthropicLLMService`,
  `pipecat.services.google.llm.GoogleLLMService`,
  `pipecat.services.aws.llm.AWSBedrockLLMService`,
  `pipecat.services.groq.llm.GroqLLMService`,
  `pipecat.services.cerebras.llm.CerebrasLLMService`,
  `pipecat.services.ultravox.llm.UltravoxRealtimeLLMService`,
  `pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService`. Verified
  via direct `import` in a clean pipecat 1.1.0 venv.
- **Adapter paths** unchanged: `pipecat.adapters.services.open_ai_adapter`.
- **Frame paths** unchanged for the frames we use.

### Unverified (require runtime smoke)

- Internal frame behavior (especially `LLMContextAggregatorPair.user()` and
  `LLMUserAggregator._handle_llm_run` which our `TextPipeline` relies on for
  the `LLMRunFrame → LLMContextFrame` conversion). Mitigation: smoke test the
  TextPipeline path on a small benchmark before declaring the upgrade safe.
- Anthropic / Google / Bedrock service constructor signatures and internal
  behavior. Mitigation: per-service smoke test for each model family we
  benchmark.
- Optional dependencies: `websockets`, `anthropic`, `google.genai` are
  imported lazily by the service modules and were missing in our pipecat-1.1.0
  test venv. Our project's existing extras should pull them; `uv sync` will
  surface any missing ones.

### Known mitigations

- CLAUDE.md already documents `WallClockAlignedAudioBufferProcessor` as a
  Pipecat-compat hot spot. Re-validate after the upgrade.
- Tests in `tests/` are mostly logic-level (no full pipeline runs), so they
  will catch import-level breakage but not behavioral changes.

## Pipeline wiring change (`AudioInPipeline`)

The minimum change to integrate the vendored `NemotronAssistantAggregator`:

1. **Refactor `TextPipeline._build_task`** to factor the assistant aggregator
   construction into a small overridable hook:

   ```python
   # In TextPipeline._build_task, replace:
   #   self.context_aggregator.assistant(),
   # with:
   #   self._build_assistant_aggregator(),

   def _build_assistant_aggregator(self) -> FrameProcessor:
       """Return the assistant aggregator. Override in subclasses to swap."""
       return self.context_aggregator.assistant()
   ```

   This is a one-method refactor in `src/multi_turn_eval/pipelines/text.py`.
   No behavior change for existing text-mode runs.

2. **`AudioInPipeline._build_assistant_aggregator`** override:

   ```python
   def _build_assistant_aggregator(self) -> FrameProcessor:
       from multi_turn_eval.vendor.nemotron_omni import NemotronAssistantAggregator
       return NemotronAssistantAggregator(
           self.context,
           interrupted_tool_pass_signal=None,
           conversation_commit_boundary_tracker=self.llm.conversation_commit_boundary_tracker,
       )
   ```

   The vanilla `LLMUserAggregator` from `LLMContextAggregatorPair.user()`
   stays — the new service consumes `LLMContextFrame` (which the user
   aggregator emits on `LLMRunFrame`) just like our old service did.

3. **`AudioInPipeline._create_llm`** rewrites to instantiate the vendored
   service:

   ```python
   from multi_turn_eval.vendor.nemotron_omni import (
       NemotronOmniAudioLLMService,
       NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE,
       NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS,
       NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K,
   )

   def _create_llm(self, service_class, model):
       conv_cache = _env_bool("MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE", False)
       conversation_id = f"mte-{uuid.uuid4().hex}" if conv_cache else None
       settings = NemotronOmniAudioLLMService.Settings(
           model=model,
           system_instruction=getattr(self.benchmark, "system_instruction", ""),
           temperature=float(os.getenv("MTE_NEMOTRON_AUDIO_IN_TEMPERATURE",
                                      str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE))),
           max_tokens=int(os.getenv("MTE_NEMOTRON_AUDIO_IN_MAX_TOKENS",
                                    str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS))),
           top_p=None,
           top_k=int(os.getenv("MTE_NEMOTRON_AUDIO_IN_TOP_K",
                                str(NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K))),
           chat_template_kwargs={
               "enable_thinking": _env_bool("MTE_NEMOTRON_AUDIO_IN_THINKING", False),
           },
       )
       return NemotronOmniAudioLLMService(
           api_key=os.getenv("NEMOTRON_AUDIO_IN_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
           base_url=os.getenv("MTE_NEMOTRON_AUDIO_IN_BASE_URL", "http://192.168.7.228:8000/v1"),
           conversation_id=conversation_id,
           request_timeout_secs=float(os.getenv("MTE_NEMOTRON_AUDIO_IN_TIMEOUT_SECS", "180")),
           enable_bash_tool=False,
           settings=settings,
       )
   ```

   **API change**: upstream no longer has a separate `suffix_only_conversation`
   flag — when `conversation_id` is set, suffix-only is implicit (engine
   contract). Drop `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` from our env vars; log
   a deprecation warning if set.

   **DO NOT** set `NEMOTRON_OMNI_STRIP_HISTORICAL_AUDIO_FROM_PAYLOAD=1`. See
   the load-bearing comment at upstream `nemotron_omni.py:654-666`: stripping
   audio breaks the committed-prefix invariant and causes cache rotation every
   turn.

4. **`_handle_audio_user_message` / `_audio_prompt`** — unchanged. We still
   own audio loading from WAV files.

5. **CLI alias** in `src/multi_turn_eval/cli.py:37`: change to
   `"multi_turn_eval.vendor.nemotron_omni.NemotronOmniAudioLLMService"`.

6. **Sanitizer allowlist** in `src/multi_turn_eval/pipelines/base.py:170`:
   keep `"nemotron-audio-in"` (unchanged).

## Steps

Each step ends in a working state with green tests. Smoke tests run against
the live vLLM endpoint between steps where behavior changes meaningfully.

### Step 1 — Pipecat upgrade trial run

Goal: validate that pipecat 1.1.0 doesn't break our existing benchmark
pipelines before touching the audio-in code.

- Update `pyproject.toml`: `pipecat-ai==0.0.101` → `pipecat-ai==1.1.0`. Keep
  all our extras (`anthropic`, `google`, `bedrock`, `groq`, etc.).
- `uv sync` and observe lock changes.
- Run the existing test suite: `uv run pytest tests -v`. Fix import errors
  if any.
- Smoke each pipeline shape we currently use, one model per family:
  - text: `aiwf_medium_context --model gpt-5 --service openai --only-turns 0`
  - text Anthropic: `--model claude-sonnet-4-5 --service anthropic --only-turns 0`
  - realtime: `--model gpt-realtime --service openai-realtime --pipeline realtime --only-turns 0`
  - nova-sonic: skip if no AWS creds; otherwise minimal smoke
- Re-validate `WallClockAlignedAudioBufferProcessor` overrides exist on
  `AudioBufferProcessor` (CLAUDE.md hot spot): check `_sync_buffer_to_position`
  via `inspect.getsource`.

**Acceptance**: each smoked pipeline produces a transcript with assistant
text. Test suite green.

**If this step fails**: STOP. The pipecat upgrade has too much surface area to
fix in the same change as the audio-in rewrite. Either pin pipecat to a
specific intermediate version that satisfies upstream's needs, or accept the
delta and budget separate work.

### Step 2 — Vendor `nemotron_omni.py`

- Create `src/multi_turn_eval/vendor/` with `__init__.py` and a `README.md`
  noting the upstream commit hash (currently `6100582` on branch
  `khk/cache-refactor-5090`) and refresh procedure (`cp upstream/...
  vendor/nemotron_omni.py`).
- Copy the file verbatim:
  ```bash
  cp ../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py \
     src/multi_turn_eval/vendor/nemotron_omni.py
  ```
- Run `uv run python -c "from multi_turn_eval.vendor.nemotron_omni import
  NemotronOmniAudioLLMService, NemotronAssistantAggregator, InterruptedToolPassSignal,
  NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE; print('OK')"`. Must succeed
  without modification.

**Acceptance**: import succeeds; no other code changes.

### Step 3 — Refactor `TextPipeline._build_task` to add an aggregator hook

Pure refactor of `src/multi_turn_eval/pipelines/text.py`. Add
`_build_assistant_aggregator(self) -> FrameProcessor` returning
`self.context_aggregator.assistant()`. Replace the inline use in
`_build_task` with a call to the hook. No behavior change.

**Acceptance**: existing tests pass; existing benchmarks unchanged on quick
smoke.

### Step 4 — Rewrite `AudioInPipeline` to use the vendored service

- Replace `AudioInPipeline._create_llm` per the spec in "Pipeline wiring
  change" above.
- Override `AudioInPipeline._build_assistant_aggregator` to return
  `NemotronAssistantAggregator(...)`.
- Keep `_build_audio_user_message`, `_audio_prompt`, `_setup_context`,
  `_queue_next_turn`, `_queue_recovery_turn` unchanged.
- Update `cli.py` `SERVICE_ALIASES["nemotron-audio-in"]` to point at the
  vendored class.
- Drop `MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` from the env-var contract; log a
  warning on first use of the now-deprecated name.

**Acceptance**: `uv run python -c "from multi_turn_eval.pipelines.audio_in
import AudioInPipeline; print('OK')"`.

### Step 5 — Delete the old service

Remove `src/multi_turn_eval/services/nemotron_audio_in.py`. Any callers go
through the CLI alias path or the pipeline; both updated in step 4.

### Step 6 — Rewrite/trim unit tests

Delete tests targeting the old service internals (mostly in
`tests/test_nemotron_audio_in_service.py` — ~25 tests of `_canonical_messages`,
`_pending_tool_result_messages`, `_should_send_suffix_only`,
`_is_conversation_cache_miss`, etc.). Keep tests that target the integration
boundary (audio message construction, file-missing error, env-var glue) and
add:

- `test_audio_in_pipeline_uses_vendor_service` — assert `_create_llm`
  returns a `NemotronOmniAudioLLMService` instance from the vendor path.
- `test_audio_in_pipeline_uses_custom_assistant_aggregator` — assert
  `_build_assistant_aggregator()` returns a `NemotronAssistantAggregator`.
- `test_conversation_id_generated_when_cache_enabled` — env=1 ⇒ ID is set;
  env=0 ⇒ ID is None.
- `test_deprecated_suffix_only_env_warns` — setting the old env var emits a
  warning and is otherwise a no-op.

**Acceptance**: `uv run pytest tests -v` green.

### Step 7 — Smoke: cache off, single turn

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=0 \
NEMOTRON_OMNI_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 0 --verbose
```

Assertions: request reaches the endpoint, no `conversation_id` in payload,
streamed text, transcript has 1 record. Note: upstream uses
`NEMOTRON_OMNI_TRACE_DIR` (we should adopt that name; was
`MTE_NEMOTRON_AUDIO_IN_TRACE_DIR` in our prior local service).

### Step 8 — Smoke: cache on, tool-call range 9-12

```bash
MTE_NEMOTRON_AUDIO_IN_CONVERSATION_CACHE=1 \
NEMOTRON_OMNI_TRACE_DIR=$(mktemp -d) \
uv run multi-turn-eval run aiwf_medium_context \
  --model nemotron_3_nano_omni \
  --service nemotron-audio-in \
  --pipeline audio-in \
  --only-turns 9,10,11,12 --verbose
```

Assertions:
- Single `conversation_id` across all 4 requests.
- Turn 11 fires `submit_session_suggestion`.
- Turn 12's request carries the tool exchange (visible in trace JSON).
- 0 `client-error` traces, 0 `attempt-02` retries.

### Step 9 — Statistical validation (10 cache-on × full 30 turns)

Re-run the same 10×2 comparison from earlier in this session, now with the
vendored service.

Target: ≥ 8/10 cache-on runs reach turn 30 (cache-off baseline 10/10); 0
cache misses; judge `turn pass strict` rate within 10 percentage points of
the cache-off baseline (26/30 = 87%).

If those targets aren't met, the residual delta is either:
- (a) we're misusing the upstream API (most likely: bad env-var glue,
  wrong aggregator wiring, or accidentally enabled audio stripping), OR
- (b) the user's vLLM server doesn't have the matching engine patches
  (`fb423d7`, `11a8fe9`) applied.

Trace JSON is the diagnostic.

### Step 10 — Documentation

- Update `CLAUDE.md` to note `pipecat-ai` is now 1.1.0 and the audio-in
  service is vendored from `nemotron-nano-omni@<commit>`.
- Update `src/multi_turn_eval/vendor/README.md` with the exact upstream
  commit hash and a one-paragraph refresh procedure.
- Mark `docs/nemotron-audio-in-implementation-plan.md` as superseded by
  this plan (or delete).

## Risks

1. **Pipecat 1.1.0 upgrade breakage** in our non-audio-in pipelines. This is
   the highest-probability failure mode. Step 1 is gated specifically to
   surface this before we touch audio-in. If it fails, abort and treat the
   upgrade as a separate project.

2. **Vendored file goes stale.** Upstream is on an active branch
   (`khk/cache-refactor-5090`). We pin to a commit. Refresh procedure must
   include re-running the full 30-turn statistical test to catch behavioral
   drift.

3. **vLLM server-side patches required.** Upstream's commits `fb423d7`
   (step 6 vLLM serving) and `11a8fe9` (step 6 followup) modify the engine.
   Confirm with the server admin before step 8.

4. **`AudioBufferProcessor` internals.** CLAUDE.md flags this as a hot spot
   across pipecat upgrades. Method signatures are stable between 0.0.101 and
   1.1.0, but the body may have changed in ways that affect our override.
   Step 1 smokes are the gate.

5. **`MTE_NEMOTRON_AUDIO_IN_SUFFIX_ONLY` users.** Anyone with the env var
   set will silently get no behavior change in cache-off mode (already
   default) but a *different* behavior in cache-on mode (suffix-only is now
   implicit, no longer toggled). Mitigation: emit a `logger.warning` on
   pipeline construction if the env var is set.

6. **Recovery turn behavior.** Our `_queue_recovery_turn` appends a text
   `"Please go ahead."` user message. Upstream's tool-batch state machinery
   may or may not handle a synthetic text turn cleanly. Smoke 8 covers this
   indirectly; if it misbehaves, set `supports_recovery = False` on
   `AudioInPipeline` as a fallback.

## Validation matrix

| Metric | Pre-fix cache-on (n=10) | Target (vendored, n=10) |
|---|---|---|
| Complete 30-turn runs | 0/10 | 8/10+ |
| Cache misses | 0 | 0 |
| Mean TTFB (ms) | 703 | < 1500 |
| Mean latency (ms) | 2811 | < 4000 |
| Judge turn-pass strict | 11/22 (50%) | within 10 points of 26/30 (87%) cache-off |

## Estimated effort

- Step 1 (pipecat upgrade trial): 1-2 hours, dominated by smoke-test
  validation of existing pipelines.
- Steps 2-4 (vendor + wire): 1 hour.
- Step 5 (delete old service): 5 min.
- Step 6 (test rewrite): 1 hour.
- Steps 7-8 (smokes): 15 min.
- Step 9 (10×30-turn × 2 modes stat run): 45 min wall.
- Step 10 (docs): 30 min.

**Total: ~5 hours** of focused work, contingent on step 1 not surfacing
deep pipecat-upgrade breakage.

## Refresh procedure (post-merge)

When upstream stabilizes a new commit on `khk/cache-refactor-5090` (or
merges to main):

```bash
cd ../nemotron-nano-omni && git pull
cd ../aiewf-eval
cp ../nemotron-nano-omni/src/nemotron_voice/services/nvidia/nemotron_omni.py \
   src/multi_turn_eval/vendor/nemotron_omni.py
git diff src/multi_turn_eval/vendor/  # review
# Update src/multi_turn_eval/vendor/README.md with new commit hash
# Run the full validation matrix
uv run pytest tests -v
# Run smokes (step 7, 8)
# Run the 10×30-turn statistical test (step 9)
```

No code beyond the vendored file needs to change unless upstream evolves the
public API of `NemotronOmniAudioLLMService` or `NemotronAssistantAggregator`.
