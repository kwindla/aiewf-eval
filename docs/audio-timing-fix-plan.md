# Audio Timing Fix Plan

## Problem Statement

The `conversation.wav` recording from realtime pipeline runs exhibits timing drift relative to wall-clock timestamps in `run.log`. Analysis of a 30-turn Ultravox run (1280 seconds) shows:

| Channel | Drift Rate | Total Drift | R² |
|---------|-----------|-------------|-----|
| User | -0.18 ms/sec | -230ms | 0.34 |
| Bot | -0.96 ms/sec | -1229ms | 0.94 |

**Target:** Audio segment positions should align with run.log timestamps within ±30ms.

**Current state:** Only 1 of 60 segments (1.7%) meets this tolerance.

## Root Cause Analysis

### User Audio Drift (-0.18 ms/sec)

- `PacedInputTransport` uses cumulative timing with `time.sleep()`
- OS scheduler jitter causes small systematic errors
- The drift is small (0.018%) and may be acceptable
- Current implementation already uses absolute scheduling (`next_chunk_time += interval`)

### Bot Audio Drift (-0.96 ms/sec) - **Primary Issue**

1. **`AudioBufferProcessor._compute_silence()` has a 1-second threshold**
   - Gaps < 1 second get NO silence inserted
   - TTFB gaps (500-700ms) between user end and bot start are ignored
   - This causes ~18 seconds of missing audio over 30 turns

2. **Truncation instead of rounding**
   - `int(quiet_time * sample_rate)` loses fractional samples
   - Systematic loss accumulates over time

3. **`NullAudioOutputTransport` timing resets**
   - `_next_send_time` resets on each new turn
   - No continuity tracking between bot speaking segments

## Solution Design

### Fix 1: NullAudioOutputTransport - Deterministic Silence Frame Insertion

**Approach:** Track sample position based on wall-clock time and insert silence frames to fill gaps before each `OutputAudioRawFrame`.

**Key changes:**
- Add `_recording_start_time` and `_total_output_samples` tracking
- Override `process_frame` to intercept `OutputAudioRawFrame`
- Calculate expected sample position from elapsed wall-clock time
- If actual position < expected, create and emit silence frame first
- Use `round()` instead of `int()` for sample calculations

**Why this works:**
- Silence frames are pushed downstream before the actual frame
- `AudioBufferProcessor` receives continuous frames, so its 1-second threshold never triggers
- Wall-clock alignment is maintained throughout the recording

### Fix 2: PacedInputTransport - Verify Global Clock Reference

**Current state:** Already uses cumulative absolute timing:
```python
next_chunk_time += chunk_interval_sec
```

**Verification needed:** Ensure the timing doesn't drift due to:
- The `min(sleep_for, 0.05)` clamp
- Queue processing delays
- Thread scheduling jitter

**Potential enhancement:** Track total samples sent vs. expected based on elapsed time.

### No Changes to AudioBufferProcessor

With continuous frames from both transports:
- The 1-second threshold in `_compute_silence()` never triggers
- Silence is inserted upstream, not computed retroactively
- Existing pipecat code works correctly

## Implementation Plan

### Phase 1: NullAudioOutputTransport Changes

File: `src/multi_turn_eval/transports/null_audio_output.py`

1. Add instance variables for sample tracking:
   - `_recording_start_time: float`
   - `_total_output_samples: int`
   - `_output_sample_rate: int`
   - `_output_num_channels: int`

2. Add `reset_recording_baseline()` method

3. Override `process_frame()` to intercept `OutputAudioRawFrame`:
   - Calculate expected samples from wall-clock
   - Insert silence frame if behind
   - Call `super().process_frame()` for actual frame

4. Handle edge cases:
   - First frame initialization
   - `InterruptionFrame` resets
   - Sample rate changes (shouldn't happen but be defensive)

### Phase 2: Realtime Pipeline Integration

File: `src/multi_turn_eval/pipelines/realtime.py`

1. In `_queue_first_turn()`, call `output_transport.reset_recording_baseline()` after `audio_buffer.start_recording()`

2. Ensure timing is synchronized between audio buffer and output transport

### Phase 3: Verification

1. Run a full 30-turn Ultravox benchmark
2. Analyze `conversation.wav` with `scripts/analyze_conversation_audio.py`
3. Compare detected segment positions with run.log timestamps
4. Verify all 60 segments are within ±30ms tolerance

## Test Cases

1. **Basic alignment:** First frame of first turn should be at ~0ms
2. **TTFB gaps:** Bot audio should start at correct position after user ends
3. **Long recording:** No cumulative drift over 20+ minutes
4. **Turn boundaries:** Silence correctly inserted between turns
5. **Interruption handling:** Timing resets correctly on interruption

## Rollback Plan

If issues arise:
1. Remove `process_frame` override in NullAudioOutputTransport
2. Revert to timestamp-based silence computation in AudioBufferProcessor
3. Accept current drift levels for evaluation purposes

## Success Criteria

- [ ] Bot audio drift < 0.05 ms/sec (was 0.96 ms/sec)
- [ ] User audio drift < 0.05 ms/sec (was 0.18 ms/sec)
- [ ] 95%+ of segments within ±30ms of expected position
- [ ] No regression in BotStarted/StoppedSpeakingFrame timing
- [ ] Recording duration matches wall-clock elapsed time
