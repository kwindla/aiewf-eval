# Fix: InterruptionFrame Recording Baseline Reset Bug

## Problem

GPT-Realtime audio recordings have massive overlap between user and bot audio tracks, causing invalid TTFB measurements (negative values indicating impossible overlap).

## Root Cause

In `src/multi_turn_eval/transports/null_audio_output.py`, lines 176-182, when an `InterruptionFrame` is received, the code resets both the playback timing AND the recording baseline:

```python
if isinstance(frame, InterruptionFrame):
    self._next_send_time = 0.0
    # Also reset silence tracking
    if self._recording_start_time > 0:
        self._recording_start_time = time.monotonic()  # BUG!
        self._actual_output_samples = 0                 # BUG!
        logger.info("[NullAudioOutput] Silence tracking reset due to interruption")
```

OpenAI Realtime sends `InterruptionFrame` when its server-side VAD detects user speech starting. This causes bot audio to be positioned relative to the interruption time instead of the original recording start, creating overlap with user audio.

Ultravox works correctly because it doesn't send `InterruptionFrame`.

## The Fix

Remove the recording baseline reset (`_recording_start_time` and `_actual_output_samples`), keeping only the playback timing reset (`_next_send_time = 0.0`):

```python
if isinstance(frame, InterruptionFrame):
    self._next_send_time = 0.0
    # Note: Do NOT reset recording baseline (_recording_start_time, _actual_output_samples)
    # Recording is continuous - the AudioBufferProcessor buffer accumulates throughout
    # the session. Resetting counters mid-recording would break wall-clock alignment.
    # Only playback pacing timing needs to reset on interruption.
    logger.debug("[NullAudioOutput] Playback timing reset due to interruption")
```

## Why This Is Safe

1. **Recording is continuous**: The AudioBufferProcessor buffer never resets during a session
2. **Playback vs recording concerns are separate**: `_next_send_time` is for playback pacing (should reset), recording counters are for wall-clock alignment (should NOT reset)
3. **Barge-in scenarios work better**: Without reset, interrupted bot audio and subsequent responses are all correctly positioned
4. **No overflow concerns**: 64-bit integers handle years of audio samples

## Testing

After the fix, re-run TTFB analysis on GPT-Realtime:
```bash
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<new-gpt-realtime-run> -v
```

Expected: All TTFB values should be positive, 30 user segments and 30 bot segments correctly paired.

## Files Changed

- `src/multi_turn_eval/transports/null_audio_output.py`: Remove recording baseline reset on InterruptionFrame
