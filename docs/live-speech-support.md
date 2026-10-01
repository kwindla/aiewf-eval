# Live speech capture and offline validation

This support layer preserves the September 2026 capture behavior without importing
code from campaign directories. Benchmark prompts and result publication are
separate from these adapters. No runtime semantic classifier is included.

## Capture entry points

Run from the repository root with `uv run python -m scripts.<command>`. Every
command requires `--benchmark NAME`; it loads that benchmark without rewriting
its prompt. The September speech benchmark definitions are supplied separately.

| Command | Preserved configuration |
|---|---|
| `capture_live_standard` | GPT-Live, Terra medium backend, Marin, three-second quiet interval; optional `--turns` and `--label` |
| `capture_live_optimized` | GPT-Live, Terra medium backend, Marin, five-second quiet interval and speech after backend content availability; `--label` required, optional `--turns` |
| `capture_gemini_extended` | Gemini 3.8 extended-thinking medium, explicit activity, NON_BLOCKING tools and provider IDLE completion; `--label` required |
| `capture_gemini_regular` | Gemini 3.8 regular, no thinking config, synchronous completion, startup guard and duplicate-tool acknowledgments |
| `capture_gemini_startup_guard` | Earlier synchronous/startup-guard configuration; explicit `--model` and `--thinking` required |

The profiles remain separate: moving their implementation does not change their
quiet intervals, reasoning settings, prompts, or cohort definitions. The regular
Gemini adapter acknowledges duplicate calls without executing the action twice.
The extended adapter allows the farewell and playback to finish after end_session.
Capture commands write provenance, events, and recordings to their new run directory.
They do not perform cohort selection or silently replace failed conversations.

The GPT-Live adapter has no authoritative answer-finished signal. Its quiet and
playback gate remains a heuristic; backend completion is not proof that the answer
was spoken. See [OpenAI's WebSocket guide](https://developers.openai.com/api/docs/guides/voice-websockets).

## Offline checks

These commands read original events/audio and write derived audit files in the
specified run directory. Use a separate copy when reviewing an immutable capture.
They make no model API requests.

```sh
uv run python -m scripts.reconstruct_speech live RUN_DIR
uv run python -m scripts.audit_live_audio RUN_DIR
uv run python -m scripts.audit_live_alignment RUN_DIR
uv run python -m scripts.audit_live_gate RUN_DIR

uv run python -m scripts.reconstruct_speech gemini RUN_DIR
uv run python -m scripts.audit_gemini_audio RUN_DIR AUDIO_JSON
uv run python -m scripts.audit_gemini_alignment RUN_DIR AUDIO_JSON ALIGNMENT_JSON
```

`reconstruct_speech gemini` validates the extended-thinking **medium** exchange
protocol, including setup and terminal IDLE. It is not the regular-model validator.
`audit_live_gate` validates the **five-second** profile. Do not use it to validate
the standard three-second capture profile.

Reconstruction retains interim and final spoken fragments, never substitutes
backend text, and blocks automatic approval when event ordering needs review.
Waveform alignment uses the existing AI Engineer World's Fair input recordings.
Audio metrics retain the existing Silero detector, windows, thresholds, signed
latencies, and positive-latency summaries. First speech includes acknowledgments;
it does not measure time to the substantive answer. Missing boundaries are not zero.

## Implementation locations

- `pipelines/openai_live.py`: standard Live state and pipeline.
- `pipelines/live_playback.py`: five-second result/playback state extension.
- `pipelines/gemini_exchanges.py`, `gemini_regular.py`, `startup_guard.py`: explicit
  capture variants matching their original campaign implementations.
- `recording/disk_audio.py`: incremental PCM, sample mapping, and stereo recording.
- `auditing/`: lossless transcript reconstruction, gate replay, waveform metrics,
  and alignment checks. Importing these modules does not run an audit.

All module paths above are under `src/multi_turn_eval/`. Tests import these modules
normally. Historical orchestration, classifier prototypes, and campaign snapshots
are not runtime dependencies.
