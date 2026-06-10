# CLAUDE.md

## Running Benchmarks

Benchmark runs take 10-20 minutes for a full 30-turn conversation. **ALWAYS** run them as
background tasks to keep the conversation responsive:

```
Bash(run_in_background=true, timeout=600000):
  uv run multi-turn-eval run aiwf_medium_context --model <model> --service <service>
```

**NEVER** block-wait on a benchmark run. Do not call `TaskOutput(block=true)` with a long
timeout. Instead, poll progress periodically:

```
TaskOutput(task_id=<id>, block=false)
```

Or grep the output file for progress markers:
- "Recorded turn N:" — per-turn progress
- "Completed benchmark run" — run finished
- "Transcript:" — final output path

After the run completes, judge and analyze can run in parallel (~1-2 minutes each):

```
uv run multi-turn-eval judge runs/aiwf_medium_context/<run_dir>
uv run python scripts/analyze_turn_metrics.py <run_dir> -v
```

## Service Aliases for Speech-to-Speech Models

| Model | Service | Notes |
|-------|---------|-------|
| gpt-realtime* | openai-realtime | Pipeline auto-detected as `realtime` |
| gemini-*-native-audio-* | gemini-live | Pipeline auto-detected as `realtime` |
| ultravox-* | ultravox-realtime | Pipeline auto-detected as `realtime` |
| grok-realtime | (auto) | Pipeline auto-detected as `realtime` |
| amazon.nova-*-sonic-* | (built-in) | Use `--pipeline nova-sonic` |

## Pipecat Compatibility

Our `WallClockAlignedAudioBufferProcessor` (in `src/multi_turn_eval/processors/audio_buffer.py`)
overrides silence insertion methods in pipecat's `AudioBufferProcessor`. If pipecat is upgraded,
check for new silence/sync methods that need overriding to prevent double-counting with
`NullAudioOutputTransport`. (History: `_compute_silence` ≤0.0.98, `_sync_buffer_to_position`
≥0.0.99, `_fill_buffer_silence_gap` ≥1.3.0 — all three are no-op overridden.)

Current pipecat: 1.3.0. The 1.3.0 `PipelineTask`/`PipelineRunner` → `PipelineWorker`/`WorkerRunner`
renames are deprecation shims for now; migrate our pipelines before the shims are removed.

The `anthropic` package is pinned in `pyproject.toml` above pipecat's `>=0.49` floor for
claude-fable-5 support (adaptive thinking, `output_config.effort`). Re-evaluate the pin after
pipecat upgrades. `LoggedAnthropicLLMService` gates `stop_ttfb_metrics()` and wraps the stream
from `_create_message_stream` so TTFB measures first *visible* token (thinking blocks excluded);
if pipecat's `AnthropicLLMService._process_context` changes its TTFB call sites, re-check the gate.

pipecat 1.x removed the transcript-processor subsystem; the realtime pipeline uses the vendored
copy in `src/multi_turn_eval/vendor/transcript_processor.py`. pipecat 1.x's turn system also
pushes `UserStarted/StoppedSpeakingFrame` where pre-1.x pushed `VADUser*SpeakingFrame` (siblings,
not subclasses) — turn-keyed gates (audio tagging, no-response retry, [VAD] timing logs) accept
both. After any pipecat upgrade, smoke a full realtime run and check for "Bot turn tag" lines in
run.log; their absence silently kills V2V analysis.

## Anthropic Reasoning Runs (claude-fable-5)

- **claude-fable-5 thinks unconditionally**: adaptive thinking is on even with the `thinking`
  param omitted, and an explicit `{"type": "disabled"}` returns a 400. There is no "no-thinking"
  configuration; the omitted-param default is adaptive thinking at default effort (high).
- `MTE_ANTHROPIC_EFFORT=low|medium|high|xhigh|max` sets `output_config.effort`. Unset = the
  model's default config. Non-fable models without effort set get no thinking params at all
  (how the sonnet/haiku rows ran).
- `MTE_ANTHROPIC_MAX_TOKENS` defaults to 16384 for fable/effort runs (thinking counts against it).
- `MTE_ANTHROPIC_THINKING_DISPLAY=summarized|omitted` (default summarized). `omitted` gives the
  lowest time-to-first-text and requires `LoggedAnthropicLLMService` — its
  `PatchedAnthropicLLMAdapter` round-trips empty-text+signature thinking blocks, which pipecat
  1.1.0's stock adapter cannot (next turn crashes with KeyError 'role').
- `MTE_ANTHROPIC_VOICE_STEERING=1` (fable-only) appends a low-latency instruction to the system
  prompt that suppresses deliberation. Steered runs diverge from the fixed benchmark prompt —
  label them in any results.
