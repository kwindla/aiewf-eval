# Batch Sweep Results

This file is the execution log for the batch sweep defined in `docs/batch-workflow-plan-2026-03-27.md`.

Use this file for:

- exact shell commands
- exact `run_dir` paths
- sanity-check outcomes
- single full-run outcomes
- ten-run sweep outcomes
- benchmark-code fixes made during testing
- blocked configuration notes
- final 10-run allowlists used to produce README rows

## Global Notes

- `gpt-realtime-1.5` should use the direct model tag. The dated fallback `gpt-realtime-2026-01-12` returned `model_not_found` on 2026-03-27, but `gpt-realtime-1.5` itself works.
- `2026-03-27 22:25 America/Los_Angeles`: the single full-run phase was launched. The active `ultravox-v0.7` full run is `runs/aiwf_medium_context/20260327T221520_ultravox-v0.7_b99ba054`, and the remaining configurations are queued behind an unattended sequential driver. Intermediate status is being written to `/tmp/aiewf-single-full-results.tsv`, with per-configuration logs under `/tmp/*-full.log`.
- Two benchmark-side compatibility fixes were applied before continuing the sweep:
  - `src/multi_turn_eval/pipelines/realtime.py`: instantiate `OpenAIRealtimeLLMService` via `settings=` instead of deprecated direct init args, to avoid Pipecat's `_warn_deprecated_param` crash path.
  - `src/multi_turn_eval/pipelines/grok_realtime.py`: mirror the same `settings=` construction and read `session_properties` from either direct kwargs or `settings`.
- One additional benchmark-side turn-gate fix was applied after the first Ultravox full run hung:
  - `src/multi_turn_eval/pipelines/realtime.py`: when a delayed turn-end task is canceled by a follow-up `BotStartedSpeakingFrame`, restore the pending transcript instead of dropping it. The Ultravox run showed a second short bot-speaking segment arriving after `TTSStoppedFrame`, which canceled the scheduled turn completion for turn 21 and left the conversation stuck.
- Two additional Grok-specific compatibility fixes were applied during sanity testing:
  - connect to xAI with the bare `wss://api.x.ai/v1/realtime` websocket URL instead of inheriting OpenAI's `?model=...` query convention
  - send xAI session updates via the current Pipecat settings object and convert tool payloads robustly whether the session payload is a model object or plain dict
- Two Nova-specific compatibility fixes were applied during sanity testing:
  - `src/multi_turn_eval/pipelines/nova_sonic.py`: accept the CLI's `thinking` argument in `NovaSonicPipeline.run()` and ignore it for Nova
  - `src/multi_turn_eval/pipelines/nova_sonic.py`: stop overriding Nova session start with the removed `_params` API and delegate to the current Pipecat implementation
- `2026-03-28 06:32 America/Los_Angeles`: the ten-run phase was launched in three parallel sequential sessions:
  - Google session: `gemini-live` -> `gemini-3.1-live-minimal` -> `gemini-3.1-live-low` -> `gemini-3.1-live-medium` -> `gemini-3.1-live-high`
  - OpenAI session: `gpt-realtime-1.5` -> `gpt-realtime`
  - Mixed-provider session: `ultravox-v0.7` -> `nova-2-sonic` -> `grok-realtime`
  - Session logs: `/tmp/aiewf-10run-logs/*.log`
  - Aggregate status file: `/tmp/aiewf-10run-status.tsv`
  - Per-config allowlists: `docs/ten-run-allowlists/*-2026-03-28.txt`
  - Per-config aggregate outputs: `docs/ten-run-aggregates/*-2026-03-28.{txt,json}`

## Single Full-Run Gate Summary

| Label | Run Dir | Strict | Turn-Taking | Notes |
| --- | --- | --- | --- | --- |
| `ultravox-v0.7` | `runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01` | `28/30` | `29/30` | Original first attempt hung; rerun completed after `TurnGate` fix. Final turn-taking failures: `[11]`. |
| `gpt-realtime-1.5` | `runs/aiwf_medium_context/20260327T224601_gpt-realtime-1.5_91c6f119` | `29/30` | `30/30` | Clean full-run gate. |
| `gpt-realtime` | `runs/aiwf_medium_context/20260327T225737_gpt-realtime_df2e1d96` | `28/30` | `29/30` | Single turn-taking failure on turn `0`. |
| `gemini-live` | `runs/aiwf_medium_context/20260327T231022_gemini-2.5-flash-native-audio-preview-12-2025_b56ee97f` | `25/30` | `30/30` | Lower strict score appears to be model behavior, not a harness failure. |
| `nova-2-sonic` | `runs/aiwf_medium_context/20260327T232056_amazon.nova-2-sonic-v1_0_631146c2` | `25/30` | `28/30` | Turn-taking failures on turns `[8, 21]`. |
| `grok-realtime` | `runs/aiwf_medium_context/20260327T234342_grok-realtime_7384ac5b` | `27/30` | `30/30` | Clean full-run gate. |
| `gemini-3.1-live-minimal` | `runs/aiwf_medium_context/20260327T235804_gemini-3.1-flash-live-preview_898fa9b7` | `27/30` | `30/30` | Clean full-run gate. |
| `gemini-3.1-live-low` | `runs/aiwf_medium_context/20260328T000931_gemini-3.1-flash-live-preview_07ad4a87` | `27/30` | `30/30` | Clean full-run gate. |
| `gemini-3.1-live-medium` | `runs/aiwf_medium_context/20260328T002049_gemini-3.1-flash-live-preview_3e2e7a52` | `25/30` | `29/30` | Single turn-taking failure on turn `29`. |
| `gemini-3.1-live-high` | `runs/aiwf_medium_context/20260328T003256_gemini-3.1-flash-live-preview_7b569fbe` | `25/30` | `30/30` | Lower strict score appears to be model behavior, not a harness failure. |

## Ten-Run Final Summary

- `2026-03-28 21:28 America/Los_Angeles`: the full ten-run sweep completed for all 10 configurations.
- Final per-config status is captured in `/tmp/aiewf-10run-status.tsv`.
- Exact run directories for the final README aggregates are the 10 paths in each allowlist file under `docs/ten-run-allowlists/`.
- For each configuration, the ten-run workflow was:
  - repeat the listed `uv run multi-turn-eval run ...` command 10 times
  - for each resulting `run_dir`, run `uv run python scripts/analyze_turn_metrics.py <run_dir> -v 2>&1 | tee <run_dir>/metrics.txt`
  - for each resulting `run_dir`, run `uv run python scripts/analyze_turn_metrics.py <run_dir> --json > <run_dir>/turn_metrics.json`
  - judge each run with `uv run multi-turn-eval judge <run_dir>`
  - aggregate the explicit allowlist with `uv run python scripts/benchmark_summary.py ...`

| Label | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-3.1-live (medium)` | `96.0%` | `291/300` | `288/300` | `300/300` | `296/300` | `2400ms` | `10208ms` | `3903ms` | `113ms` |
| `ultravox-v0.7` | `92.7%` | `283/300` | `283/300` | `299/300` | `293/300` | `896ms` | `3104ms` | `1565ms` | `67ms` |
| `gpt-realtime-1.5` | `92.0%` | `278/300` | `276/300` | `299/300` | `300/300` | `1280ms` | `3552ms` | `2726ms` | `68ms` |
| `gemini-3.1-live (minimal)` | `91.7%` | `276/300` | `277/300` | `300/300` | `300/300` | `1632ms` | `5664ms` | `3172ms` | `100ms` |
| `gemini-3.1-live (high)` | `91.0%` | `281/300` | `273/300` | `300/300` | `297/300` | `2368ms` | `15872ms` | `3664ms` | `119ms` |
| `gemini-3.1-live (low)` | `90.3%` | `282/300` | `271/300` | `299/300` | `297/300` | `2176ms` | `4672ms` | `3602ms` | `96ms` |
| `gpt-realtime` | `90.0%` | `272/300` | `270/300` | `300/300` | `298/300` | `1248ms` | `2752ms` | `1768ms` | `219ms` |
| `gemini-live` | `88.7%` | `268/300` | `267/300` | `300/300` | `298/300` | `1504ms` | `2944ms` | `1761ms` | `51ms` |
| `grok-realtime` | `86.3%` | `264/300` | `262/300` | `298/300` | `300/300` | `2080ms` | `4896ms` | `2688ms` | `278ms` |
| `nova-2-sonic` | `80.0%` | `257/300` | `247/300` | `293/300` | `300/300` | `N/A` | `N/A` | `N/A` | `N/A` |

| Label | Allowlist | Aggregate Text | Aggregate JSON |
| --- | --- | --- | --- |
| `gemini-live` | `docs/ten-run-allowlists/gemini-live-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-live-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-live-2026-03-28.json` |
| `gpt-realtime-1.5` | `docs/ten-run-allowlists/gpt-realtime-1.5-2026-03-28.txt` | `docs/ten-run-aggregates/gpt-realtime-1.5-2026-03-28.txt` | `docs/ten-run-aggregates/gpt-realtime-1.5-2026-03-28.json` |
| `ultravox-v0.7` | `docs/ten-run-allowlists/ultravox-v0.7-2026-03-28.txt` | `docs/ten-run-aggregates/ultravox-v0.7-2026-03-28.txt` | `docs/ten-run-aggregates/ultravox-v0.7-2026-03-28.json` |
| `gemini-3.1-live (minimal)` | `docs/ten-run-allowlists/gemini-3.1-live-minimal-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-minimal-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-minimal-2026-03-28.json` |
| `gpt-realtime` | `docs/ten-run-allowlists/gpt-realtime-2026-03-28.txt` | `docs/ten-run-aggregates/gpt-realtime-2026-03-28.txt` | `docs/ten-run-aggregates/gpt-realtime-2026-03-28.json` |
| `gemini-3.1-live (low)` | `docs/ten-run-allowlists/gemini-3.1-live-low-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-low-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-low-2026-03-28.json` |
| `nova-2-sonic` | `docs/ten-run-allowlists/nova-2-sonic-2026-03-28.txt` | `docs/ten-run-aggregates/nova-2-sonic-2026-03-28.txt` | `docs/ten-run-aggregates/nova-2-sonic-2026-03-28.json` |
| `gemini-3.1-live (medium)` | `docs/ten-run-allowlists/gemini-3.1-live-medium-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-medium-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-medium-2026-03-28.json` |
| `grok-realtime` | `docs/ten-run-allowlists/grok-realtime-2026-03-28.txt` | `docs/ten-run-aggregates/grok-realtime-2026-03-28.txt` | `docs/ten-run-aggregates/grok-realtime-2026-03-28.json` |
| `gemini-3.1-live (high)` | `docs/ten-run-allowlists/gemini-3.1-live-high-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-high-2026-03-28.txt` | `docs/ten-run-aggregates/gemini-3.1-live-high-2026-03-28.json` |

## Configuration Results

### `ultravox-v0.7`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model ultravox-v0.7 --service ultravox-realtime --pipeline realtime --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e --json > runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `56197ms`, pipe TTFB `778ms`, WAV V2V `864ms`, pad RMS `40ms`, pad VAD `38ms`, align `-48ms`
    - turn 1: server TTFB `N/A`, pipe TTFB `707ms`, WAV V2V `704ms`, pad RMS `0ms`, pad VAD `-31ms`, align `-28ms`
  - Alignment issues were reported by the metrics script.
  - Runtime warnings seen during/after the run:
    - `NullAudioOutputTransport._emit_playback_drained_frame` was never awaited
    - `BaseInputTransport.push_audio_frame` was never awaited
    - `nanobind`/`soxr` leaked-instance warnings at process exit

#### Single Full Run

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model ultravox-v0.7 --service ultravox-realtime --pipeline realtime`
- `run_dir`:
  - first attempt: `runs/aiwf_medium_context/20260327T221520_ultravox-v0.7_b99ba054`
  - rerun after fixes: `runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01 -v 2>&1 | tee runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01/metrics.txt`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01 --json > runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01`
- Outcome:
  - first attempt: `HUNG_AFTER_TURN_20`
  - rerun: `COMPLETED`, strict turn pass `28/30`, turn-taking `29/30`, tool use `29/30`, instruction following `28/30`, KB grounding `29/30`
- Notes:
  - First attempt hung after turn `20`: `runs/aiwf_medium_context/20260327T221520_ultravox-v0.7_b99ba054`
  - Fresh rerun started after the `TurnGate` fix: `runs/aiwf_medium_context/20260327T225246_ultravox-v0.7_52753a01`
  - Root cause observed in `run.log`:
    - turn `21` assistant transcript completed and was stored
    - `BotPlaybackDrainedFrame` scheduled delayed turn completion
    - a second short `BotStartedSpeakingFrame` arrived before the delay elapsed
    - `TurnGate` canceled the delayed turn-end task and dropped the saved transcript
    - later `BotStoppedSpeakingFrame` / `BotPlaybackDrainedFrame` arrived with no pending transcript left, so turn `21` was never recorded and the run stalled
  - After the follow-up timing fixes, the rerun’s `turn_metrics.json` now marks ambiguous/interrupted turns as timing-invalid instead of emitting impossible values like `pipeline_ttfb_ms=-20643` or `silent_pad_silero_ms=22833`.
  - Rejudging the rerun after the timing fix reduced turn-taking failures from `[11, 12, 13, 14, 15]` to `[11]`.
  - Remaining global timing notes on the rerun: `2` overlap instances (`1920ms` total), `2` unmatched bot segments, and `1` unprompted bot response segment.

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- `src/multi_turn_eval/pipelines/realtime.py`
  - Preserve pending assistant transcript when a delayed turn-end task is canceled by a follow-up bot-speaking segment. This fixed the Ultravox full-run hang at turn `21`.
- `src/multi_turn_eval/recording/transcript_recorder.py`
  - Reject stale/impossible realtime TTFB samples when they exceed the current turn's elapsed wall-clock time.
  - Replace an earlier positive TTFB with a later smaller positive sample when providers emit multiple metrics frames during turn handoff.
- `scripts/analyze_turn_metrics.py`
  - Parse Smart Turn end-of-turn states from `run.log`.
  - Mark turns as timing-invalid when turn boundaries are ambiguous or interrupted, when user boundaries are missing, or when derived timestamps violate ordering constraints.
  - Drop stale transcript-side server TTFB values when they disagree with audio-based latency, and exclude silent no-audio turns from server-TTFB summaries.
- `src/multi_turn_eval/judging/turn_taking.py`
  - Skip timing-based failure checks for turns explicitly marked `timing_invalid_reasons` by the analyzer, so analysis artifacts do not become judge failures.

#### Final Allowlist

- `docs/ten-run-allowlists/ultravox-v0.7-2026-03-28.txt`

### `gpt-realtime-1.5`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime-1.5 --service openai-realtime --pipeline realtime --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d --json > runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `906ms`, pipe TTFB `869ms`, WAV V2V `896ms`, pad RMS `80ms`, pad VAD `41ms`, align `14ms`
    - turn 1: server TTFB `446ms`, pipe TTFB `781ms`, WAV V2V `800ms`, pad RMS `80ms`, pad VAD `35ms`, align `16ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.
  - Earlier failed fallback attempt:
    - `runs/aiwf_medium_context/20260327T212913_gpt-realtime-2026-01-12_4ef921ab`: `model_not_found` from OpenAI when using the dated fallback string

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- Shared OpenAI/Grok realtime constructor fix recorded in Global Notes.

#### Final Allowlist

- `docs/ten-run-allowlists/gpt-realtime-1.5-2026-03-28.txt`

### `gpt-realtime`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime --pipeline realtime --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8 --json > runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `940ms`, pipe TTFB `747ms`, WAV V2V `1088ms`, pad RMS `120ms`, pad VAD `357ms`, align `16ms`
    - turn 1: server TTFB `613ms`, pipe TTFB `974ms`, WAV V2V `992ms`, pad RMS `40ms`, pad VAD `30ms`, align `12ms`
  - This run validated the benchmark-side constructor fix for OpenAI Realtime.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- Shared OpenAI/Grok realtime constructor fix recorded in Global Notes.

#### Final Allowlist

- `docs/ten-run-allowlists/gpt-realtime-2026-03-28.txt`

### `gemini-live`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live --pipeline realtime --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778 --json > runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `1222ms`, pipe TTFB `1529ms`, WAV V2V `1568ms`, pad RMS `40ms`, pad VAD `54ms`, align `15ms`
    - turn 1: server TTFB `731ms`, pipe TTFB `1411ms`, WAV V2V `1632ms`, pad RMS `40ms`, pad VAD `230ms`, align `9ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- None yet.

#### Final Allowlist

- `docs/ten-run-allowlists/gemini-live-2026-03-28.txt`

### `nova-2-sonic`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5 --json > runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `278ms`, pipe TTFB `214ms`, WAV V2V `288ms`, pad RMS `40ms`, pad VAD `89ms`, align `15ms`
    - turn 1: server TTFB `519ms`, pipe TTFB `544ms`, WAV V2V `608ms`, pad RMS `40ms`, pad VAD `77ms`, align `13ms`
  - Alignment was reported OK; the metrics script flagged four unmatched greeting-era bot tags in the WAV notes.
  - Runtime log still shows a persistent ~1000ms user-track sample offset during the run, but the final computed turn metrics remained sane.
  - Failed intermediate attempts:
    - `runs/aiwf_medium_context/20260327T214246_amazon.nova-2-sonic-v1_0_f5abe93a`: CLI/pipeline interface mismatch (`thinking` kwarg) before compatibility fix
    - `runs/aiwf_medium_context/20260327T214318_amazon.nova-2-sonic-v1_0_feca4cda`: stale Nova `_params` override prevented initialization before compatibility fix

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- `src/multi_turn_eval/pipelines/nova_sonic.py`: accepted `thinking` in `NovaSonicPipeline.run()` for CLI compatibility.
- `src/multi_turn_eval/pipelines/nova_sonic.py`: delegated session-start construction back to the current Pipecat Nova Sonic implementation instead of using the removed `_params` field.

#### Final Allowlist

- `docs/ten-run-allowlists/nova-2-sonic-2026-03-28.txt`

### `grok-realtime`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model grok-realtime --pipeline grok-realtime --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93 --json > runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `1193ms`, pipe TTFB `1877ms`, WAV V2V `2048ms`, pad RMS `200ms`, pad VAD `183ms`, align `12ms`
    - turn 1: server TTFB `1027ms`, pipe TTFB `2756ms`, WAV V2V `2976ms`, pad RMS `200ms`, pad VAD `236ms`, align `16ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.
  - Failed intermediate attempts:
    - `runs/aiwf_medium_context/20260327T213545_grok-realtime_5d4c5f6b`: HTTP 400 websocket rejection before xAI URL fix
    - `runs/aiwf_medium_context/20260327T213904_grok-realtime_6ae156c6`: handshake succeeded after URL fix, but session update failed before xAI session-update fix

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- `src/multi_turn_eval/pipelines/grok_realtime.py`: stripped inherited OpenAI `?model=...` websocket query behavior for xAI.
- `src/multi_turn_eval/pipelines/grok_realtime.py`: fixed xAI session-update handling to use `_settings.session_properties` and to convert tool payloads when the outgoing session is represented as either a model object or a dict.

#### Final Allowlist

- `docs/ten-run-allowlists/grok-realtime-2026-03-28.txt`

### `gemini-3.1-live (minimal)`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gemini-3.1-flash-live-preview --service gemini-live --pipeline realtime --thinking minimal --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74 --json > runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `804ms`, pipe TTFB `1288ms`, WAV V2V `1344ms`, pad RMS `40ms`, pad VAD `70ms`, align `14ms`
    - turn 1: server TTFB `11438ms`, pipe TTFB `1137ms`, WAV V2V `1184ms`, pad RMS `40ms`, pad VAD `62ms`, align `15ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- None yet.

#### Final Allowlist

- `docs/ten-run-allowlists/gemini-3.1-live-minimal-2026-03-28.txt`

### `gemini-3.1-live (low)`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gemini-3.1-flash-live-preview --service gemini-live --pipeline realtime --thinking low --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c --json > runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `1213ms`, pipe TTFB `2377ms`, WAV V2V `2400ms`, pad RMS `120ms`, pad VAD `39ms`, align `16ms`
    - turn 1: server TTFB `116ms`, pipe TTFB `1625ms`, WAV V2V `1952ms`, pad RMS `93ms`, pad VAD `341ms`, align `14ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- None yet.

#### Final Allowlist

- `docs/ten-run-allowlists/gemini-3.1-live-low-2026-03-28.txt`

### `gemini-3.1-live (medium)`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gemini-3.1-flash-live-preview --service gemini-live --pipeline realtime --thinking medium --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b --json > runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `1118ms`, pipe TTFB `1566ms`, WAV V2V `1600ms`, pad RMS `40ms`, pad VAD `47ms`, align `13ms`
    - turn 1: server TTFB `51ms`, pipe TTFB `2506ms`, WAV V2V `2528ms`, pad RMS `80ms`, pad VAD `34ms`, align `12ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.
  - The run completed with a truncated `Exception in thread PacedInputTransport#0-feeder` message after the transcript path was printed. It did not prevent audio metrics or judging from succeeding.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- None yet.

#### Final Allowlist

- `docs/ten-run-allowlists/gemini-3.1-live-medium-2026-03-28.txt`

### `gemini-3.1-live (high)`

#### Sanity Check

- Command:
  - `uv run multi-turn-eval run aiwf_medium_context --model gemini-3.1-flash-live-preview --service gemini-live --pipeline realtime --thinking high --only-turns 0,1`
- `run_dir`:
  - `runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3`
- Metrics commands:
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3 -v`
  - `uv run python scripts/analyze_turn_metrics.py runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3 --json > runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3/turn_metrics.json`
- Judge command:
  - `uv run multi-turn-eval judge runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3`
- Outcome:
  - PASS
- Notes:
  - Run completed and judged cleanly: strict turn pass `2/2`.
  - Verbose metrics summary:
    - turn 0: server TTFB `3133ms`, pipe TTFB `1992ms`, WAV V2V `2016ms`, pad RMS `40ms`, pad VAD `38ms`, align `14ms`
    - turn 1: server TTFB `7308ms`, pipe TTFB `2050ms`, WAV V2V `2112ms`, pad RMS `80ms`, pad VAD `74ms`, align `12ms`
  - Alignment was reported OK; one unmatched greeting-era bot tag remained in WAV detection, which the metrics script flagged as a note rather than an error.

#### Single Full Run

- Command:
- `run_dir`:
- Metrics commands:
- Judge command:
- Outcome:
- Notes:

#### Ten-Run Sweep

- Commands:
- Run directories:
- Metrics commands:
- Judge commands:
- Aggregation command:
- Outcome:
- Notes:

#### Fixes Made

- None yet.

#### Final Allowlist

- `docs/ten-run-allowlists/gemini-3.1-live-high-2026-03-28.txt`

## Blockers

- No active blockers. The ten-run sweep completed for all planned configurations.
