## Completed uniform rejudge

All six standard-prompt GPT-Live conversations were rejudged with the same clarified
Opus 5 speech rubric as the 125 five-model benchmark attempts. The first full
validation remains 28/30; the N=5 pilot is **141/150 (94.0%)**, and the combined
six-conversation set is 169/180 (93.9%). These are validation results, not a
25-attempt canonical score. All model prompts, recordings, audio measurements, and
original judgments are preserved. See the [revised report](rejudged/VALIDATION.md)
and [shared audit](../speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/AUDIT.md).
Earlier first-pass scores and investigation notes below are historical.

# GPT-Live readiness — 2026-09-15

## Current benchmark constraint

The standard-prompt validation uses the **entire unchanged speech-v2 system prompt, including the knowledge base, at both frontend and backend**, and the exact unchanged speech-v2 Opus 5 rubric. No model-specific frontend wording is permitted for benchmark scores. `standard-prompt-acceptance.json` verifies the API accepts and echoes both copies exactly. Earlier short-frontend smoke runs and `full-validation-1` are diagnostic only; they are not comparable benchmark scores. `full-validation-2` used an unauthorized frontend revision, was interrupted on user correction, and is excluded. The five other models were unaffected: see `canonical-prompt-integrity.json`.

The model-specific prompting advice below describes exploratory probes, not the current benchmark configuration.

## Completed concurrency validation

All five N=5 pilot conversations completed all 30 turns, with zero API/pipeline
errors, zero detected speech-overlap or boundary issues, and five confirmed session
closures. Pooled V2V over 150 turns: **mean 1,855 ms, median 1,856 ms, p90 2,371 ms**.
First-pass Opus 5 strict accuracy is **143/150 (95.3%)**. These accuracy scores are
provisional pending the same MCP-question grading-consistency review as the other
models; no prompt or judge changes have been made for that review.

See [complete validation report](VALIDATION.md). All six standard-prompt captures
(one initial full validation plus the five concurrent attempts) are retained. This
establishes that five concurrent full conversations work under the observed load;
it does not establish a 25-attempt canonical score or guarantee future quotas.

After all 125 other-model captures and processing finished, the main `.venv` was
synced to the updated lock: OpenAI 3.14.0, Pipecat 1.10.0, Google GenAI 2.23.0, and
no LiveKit agents dependency. The old campaign's dependency files are archived.

## Standard-prompt full-conversation result

The first standard-prompt validation completed **30/30 turns**, closed cleanly,
and was judged by **Claude Opus 5 using the exact shared rubric**. Strict accuracy:
**28/30 (93.3%)**. The failures were an early vote before an ambiguous session
reference was clarified and a missing `end_session` call. These are retained model
errors; no prompt changes or recovery nudges were used.

Audio-based V2V over all 30 turns: **mean 1,833 ms, median 1,712 ms, p90 2,464 ms**.
Non-tool median: **1,696 ms** (25 turns); actual-tool mean: **1,882 ms** (5 turns).
First speech includes acknowledgements; these are not time-to-substantive-answer
figures. The audio detector and thresholds are the same as the canonical pipeline.
Live's continuous silence stream requires turn windows from capture events rather
than treating audio-packet arrivals as response starts. No speech overlaps or
response segments crossing committed turn boundaries were detected. All 30 source
utterances matched the recorded waveform, with at most 1.125 ms timing offset.

An initial waveform audit used unnormalized correlation, which incorrectly selected
louder nonmatching audio for three quiet patches. Normalized correlation fixed the
audit; the original failed-check log remains in the run directory. Capture and
latency calculations were unchanged.

Run artifacts (judge output, audio metrics, waveform audit) are local-only under the
gitignored `runs/aiwf_medium_context_speech_live/20260915T150622_gpt-live-1_ba419085/`.

A five-conversation concurrency pilot follows this validation, with all attempts
retained and each receiving Opus 5 judging, V2V analysis, and waveform alignment
checks. It is not a 25-attempt canonical score. Earlier short-frontend probes are
excluded from this standard-prompt cohort in their entirety.

## Initial exploratory configuration and documentation

- Voice model: **`gpt-live-1`**, voice `marin`, PCM16 mono 24 kHz.
- Endpoint: `wss://api.openai.com/v1/live/sessions`.
- Pipecat: **1.10.0**, dedicated `OpenAILiveLLMService`.
- OpenAI SDK: **3.14.0**, latest PyPI release checked September 15.
- Delegated backend: **`gpt-5.6-terra`**; server default resolved to **medium** reasoning, standard mode, all-turn context. This is a two-model system; report both models and backend settings.
- Short conversational frontend instructions; the unchanged speech-v2 policy and complete benchmark knowledge base are backend instructions. All five benchmark tool schemas are available. Tool execution is simulated as in the benchmark; no real conference submissions are made.

Official references:

- [GPT-Live model](https://developers.openai.com/api/docs/models/gpt-live-1)
- [Live guide](https://developers.openai.com/api/docs/guides/live)
- [Delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation)
- [Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live)
- [OpenAI SDK package](https://pypi.org/project/openai/3.14.0/)
- [Pipecat package](https://pypi.org/project/pipecat-ai/1.10.0/)
- [LiveKit dependency constraint](https://pypi.org/pypi/livekit-agents/1.8.2/json)

## Dependency status

The current canonical campaign keeps its original root environment and lockfile: Pipecat 1.10.0 / OpenAI 2.34.0. No runtime dependency was changed underneath active captures.

The clean, independently locked `environment/` project installs Pipecat 1.10.0 and OpenAI 3.14.0 successfully. It supports this smoke test without LiveKit. `environment-install.log` records installation. A direct SDK session start/close also passed; see `sdk-session.json`.

The user confirmed LiveKit is unnecessary. It has been removed from the root requirements, and the root `uv.lock` now resolves OpenAI **3.14.0** successfully. The old dependency conflict is retained only as diagnostic history in `dependency-conflict.txt`; it is no longer a blocker. The active campaign uses direct executables from its unchanged `.venv`, so updating the manifest and lock does not change its runtime. Its original manifest and lock are archived under `../speech-v2-canonical-2026-09-15/frozen-dependencies/`. Avoid syncing that active environment until the campaign and processing finish.

The complete updated root project was also installed into `.venv-gpt-live-project` from the new lock: OpenAI 3.14.0, Pipecat 1.10.0, no LiveKit agents installed. Targeted OpenAI routing, realtime tool delivery, speech prompt, Grok/Pipecat compatibility, and Gemini tests passed: **60 passed** (`compatibility-tests.txt`).

## Smoke procedure

Four existing benchmark audio inputs, in order: 0 (workshop date), 9 (suggestion workflow), 10 (name), 11 (tracing suggestion). Audio is paced in 20 ms chunks, with continuous silence between questions. Each question has a fixed 25-second listening window. The session has an overall 180-second deadline and requests graceful close.

This is a functional integration probe, **not a canonical benchmark or accuracy estimate**. It uses sparse turns and generous fixed listening windows rather than the benchmark's production turn gate. No smoke result enters the five-arm canonical campaign.

The initial overlay-environment probe (`smoke/`) passed all four exchanges, executed exactly one `submit_session_suggestion` with Jennifer Smith and OpenTelemetry tracing, and closed with final session usage and no API/pipeline errors. The spoken final answer was “Okay, I'll pass that along to the organizers. They're all set.” It did not repeat the submitted proposal. Full transcripts, tool arguments, and backend usage are in `smoke/review.json`; raw events and timestamp-aligned received audio are beside it.

The clean locked-environment repeat (`smoke-clean/`) also passed all four exchanges, submitted the correct suggestion exactly once, and closed cleanly without API/pipeline errors. Its final spoken confirmation was “Done. Your suggestion has been recorded.”

To reproduce from the repository root (choose a fresh output directory):

```bash
uv sync --project docs/gpt-live-readiness-2026-09-15/environment --locked
PYTHONPATH=. docs/gpt-live-readiness-2026-09-15/environment/.venv/bin/python docs/gpt-live-readiness-2026-09-15/smoke.py --output smoke-new
python docs/gpt-live-readiness-2026-09-15/summarize_smoke.py docs/gpt-live-readiness-2026-09-15/smoke-new
```

The root `.env` must supply `OPENAI_API_KEY`. The script refuses to overwrite an existing output directory.

## Before a full 25-run GPT-Live benchmark (updated)

1. Wire a dedicated Live service into the existing pipeline components. Live is not a new model name for the Realtime API. It uses continuous full-duplex audio and delegation, without Realtime's audio commit/response trigger cycle.
2. Reuse `NullAudioOutputTransport` for paced playback and audio-based speaking detection; Pipecat 1.10 already handles Live's `SpeechOutputAudioRawFrame` there. Adapt `TurnGate` to track delegation/tool state and a quiet interval after playback drains. Live has no output-audio-done event. Pipecat transcript-gap boundaries and backend `response.completed` do not prove speech has finished. Continue microphone silence while waiting; allow the substantive answer after any acknowledgement and after tool results.
3. Use the complete exact shared speech-v2 prompt at both frontend and backend, and the unchanged shared rubric. Backend reasoning is explicitly fixed to medium. Prompt alternatives belong in model notes, never in comparable benchmark captures.
4. Record voice-session final usage separately from delegated backend token usage. Count API errors, timeouts, early exits, and retries consistently with other models.
5. Use the new SDK 3.14 environment while the current campaign finishes. Validate a complete 30-turn conversation and then test concurrency before committing to 25 attempts.

Access and basic Pipecat/API/tool compatibility are established. Full-conversation turn completion and concurrency remain untested. No voice-to-voice latency figure is claimed from API chunk arrival timestamps in this probe; use the benchmark's audio-based measurement after playback/turn integration is complete.
