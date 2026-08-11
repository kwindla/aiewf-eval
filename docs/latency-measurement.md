# Latency Measurement: Text vs Speech-to-Speech

This document explains how latency is measured in this repo for:
- **Text-mode (text-in/text-out) models** — simple, fully log-based
- **Speech-to-speech models** — complex, audio-based voice-to-voice (V2V)

It focuses on the **code paths and artifacts** used to compute each metric.

---

## Output Artifacts (Shared)

Each run directory contains the inputs needed for latency analysis:

- `transcript.jsonl` — per-turn metadata and timing (TTFB, latency_ms, tool calls)
- `run.log` — detailed pipeline events (audio tag positions, RMS onset, retries)
- `conversation.wav` — stereo recording (user left, bot right) for V2V

---

## Text-Mode Models (Simple)

**Relevant code:**
- `src/multi_turn_eval/pipelines/text.py`
- `src/multi_turn_eval/pipelines/base.py`
- `src/multi_turn_eval/recording/transcript_recorder.py`

### What gets measured

Text-mode latency is entirely **turn-timing + TTFB**:

- **TTFB (ttfb_ms):** Time to first token/byte as reported by the LLM service
- **Turn latency (latency_ms):** Wall-clock time from turn start to turn end

### How it’s implemented

1. **Turn start**
   - `TranscriptRecorder.start_turn()` stores `turn_start_monotonic = time.monotonic()`.

2. **TTFB capture**
   - In `BasePipeline._handle_metrics()`, `TTFBMetricsData` inside `MetricsFrame` is
     recorded via `TranscriptRecorder.record_ttfb()`.
   - Only the *first* TTFB value per turn is kept.

3. **Turn end + latency**
   - When the end-of-turn marker arrives (`LLMContextAssistantTimestampFrame` in
     `TextPipeline.NextTurn`), `BasePipeline._on_turn_end()` writes the turn.
   - `TranscriptRecorder.write_turn()` computes:
     ```
     latency_ms = (time.monotonic() - turn_start_monotonic) * 1000
     ```

4. **Output**
   - `transcript.jsonl` contains `ttfb_ms` and `latency_ms` per turn.

There is **no audio segmentation** or alignment logic for text pipelines.

---

## Speech-to-Speech Models (Complex)

**Relevant code:**
- `src/multi_turn_eval/pipelines/realtime.py`
- `src/multi_turn_eval/transports/null_audio_output.py`
- `src/multi_turn_eval/processors/audio_buffer.py`
- `scripts/analyze_turn_metrics.py`
- `scripts/analyze_ttfb_silero.py` (simpler VAD-only variant)

Speech-to-speech latency is measured as **true voice-to-voice (V2V)**:

```
V2V = (bot_speech_start_ms) - (user_speech_end_ms)
```

This requires **wall-clock aligned audio recording**, **turn boundary alignment**,
and **VAD-based segmentation**.

### A. Runtime: Recording + Tagging

The realtime pipeline records a synchronized stereo WAV with **user on left** and
**bot on right**. The core pieces are:

1. **Wall-clock aligned recording**
   - `RealtimePipeline` wires:
     - `PacedInputTransport` (user audio pacing)
     - `NullAudioOutputTransport` (bot audio pacing + silence insertion)
     - `WallClockAlignedAudioBufferProcessor` (recording)
   - On pipeline start, `_initialize_recording_and_start_audio()` sets a **shared
     recording baseline** for all three components so time zero is consistent.

2. **Silence insertion (critical for alignment)**
   - `NullAudioOutputTransport` is the **source of truth** for silence insertion.
   - For both user and bot tracks, it:
     - Computes expected sample position from wall-clock time
     - Inserts silence for any gap > 10ms
   - This guarantees `conversation.wav` is continuous and aligned to real time.

3. **Synthetic turn tags (alignment anchors)**
   - At the **first bot audio frame after user speech ends**, the transport mixes
     a **2kHz, 15ms, -12dB sine burst** into the audio frame.
   - The tag is enabled by `VADUserStoppedSpeakingFrame` (or explicitly for
     the initial greeting).
   - Each tag’s position is logged to `run.log` as:
     ```
     Bot turn tag: sample_pos=XXXXms
     ```

4. **Recording output**
   - `WallClockAlignedAudioBufferProcessor` emits stereo data, saved as
     `conversation.wav` (user left, bot right).

### B. Offline: VAD Segmentation + Alignment + V2V

The primary analysis tool is `scripts/analyze_turn_metrics.py`.

It computes per-turn V2V using **three aligned sources**:

| Source | Purpose |
| --- | --- |
| `transcript.jsonl` | Server TTFB, tool-call flags, reconnection info |
| `run.log` | Bot tag positions, RMS onset, retry events, first user end time |
| `conversation.wav` | VAD-based speech boundaries + tag detection |

#### 1) Detect audio tags in WAV

- `detect_tags_in_wav()` scans the bot channel for **2kHz energy spikes**
  using short FFT windows.
- Tags are filtered by minimum RMS level and min gap to remove false positives.

#### 2) Align log tags to WAV tags

- `match_tags_by_proximity()` pairs log tags with WAV tags using a
  proximity window (±150ms).
- `check_alignment()` reports alignment stats and computes:
  ```
  tag_alignment_ms = bot_tag_log_ms - bot_tag_wav_ms
  ```
- This verifies that the WAV timeline matches the log timeline.

#### 3) Segment speech with Silero VAD

- `run_silero_vad()` runs Silero VAD on **both channels**:
  - Resamples to 16kHz
  - `threshold=0.7`, `min_silence_ms=2000`, `min_speech_ms=700`
- Produces `user_segments[]` and `bot_segments[]` in ms.

#### 4) Match segments to turns

For each **bot tag** (one per turn):

- **Bot segment:** choose the Silero bot segment whose **start** is closest
  to the tag (allowing up to 500ms *before* the tag).
- **User segment:** choose the user segment that **ends immediately before**
  the bot starts, but **after the previous bot segment** (prevents backtracking
  when Silero merges segments).

This yields:

```
user_end_ms  = matched_user_seg.end
bot_start_ms = matched_bot_seg.start
```

#### 5) Compute true V2V latency

Once segments are matched:

```
wav_v2v_ms = bot_start_ms - user_end_ms
```

This is the **true audible voice-to-voice latency**.

#### 6) Extra metrics (same pass)

`analyze_turn_metrics.py` also computes:

- **Pipeline TTFB** (first bot audio byte - user end) using log tags
- **Silent padding (RMS)** from `run.log`
- **Silent padding (VAD)** from WAV tag → Silero start
- **Retry-adjusted V2V**: if a turn is retried, it uses the *first*
  user-audio end time from log + recording baseline so latency reflects
  the user’s total wait

### Simpler VAD-only alternative

`scripts/analyze_ttfb_silero.py` is a lighter tool that:
- Runs Silero VAD on the WAV
- Pairs user/bot segments **by index**
- Computes V2V per turn

It skips tag alignment and log-based matching, so it’s less robust but
useful for quick checks.

---

## Summary

- **Text-mode**: latency is computed directly from pipeline timestamps and stored
  in `transcript.jsonl`.
- **Speech-to-speech**: latency is computed offline from **aligned audio** using
  synthetic tone tags + Silero VAD to measure **true voice-to-voice**.

If you want to trace a specific metric, start with:
- `transcript.jsonl` for TTFB + turn latency
- `scripts/analyze_turn_metrics.py` for full V2V calculations
- `src/multi_turn_eval/transports/null_audio_output.py` for tagging + alignment logic
