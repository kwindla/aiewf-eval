# Nova Sonic Pipeline Improvements Plan

## Executive Summary

The recent commit (ff207c2) made significant improvements to the realtime pipeline for OpenAI Realtime and Gemini Live models. These improvements centered around:

1. **Proper audio timing simulation** via NullAudioOutputTransport
2. **Turn gating** based on BotStoppedSpeakingFrame rather than timeouts
3. **Audio recording** of both user and bot audio to a stereo WAV file

This document proposes applying similar improvements to the Nova Sonic pipeline, analyzes the key differences, and outlines a testing strategy.

---

## Part 1: Analysis of Realtime Pipeline Improvements

### 1.1 The Core Problem Solved

LLMs generate audio **faster than real-time**. For example, an LLM might produce 33 seconds of audio in just 10 seconds of wall-clock time. Without proper handling:

- BotStoppedSpeakingFrame fires when the audio queue empties (after ~10s)
- But the audio content represents 33s of speech
- Turn advancement happens too early
- Next user audio overlaps with conceptually-still-playing bot audio

**Note on audio pacing:** Nova Sonic generates responses in multiple audio segments, but there is always enough audio generated that real-time playback sees a continuous stream with no gaps. The NullAudioOutputTransport pacing ensures the audio queue is consumed at real-time speed, so BotStoppedSpeakingFrame fires at the correct time (after all audio has "played").

### 1.2 Key Components Added

#### TurnGate Processor (`realtime.py:53-128`)

```python
class TurnGate(FrameProcessor):
    """Gates turn advancement until bot finishes speaking."""
```

**Behavior:**
1. Stores pending transcript when received (via `set_pending_transcript()`)
2. Waits for `BotStoppedSpeakingFrame` from NullAudioOutputTransport
3. Adds configurable `audio_drain_delay` (default 0.5s)
4. Only then triggers the turn-end callback

**Critical insight:** Decouples transcript availability from turn advancement. The transcript may be complete long before the audio "finishes playing."

#### NullAudioOutputTransport (`transports/null_audio_output.py`)

**Purpose:** Generate BotStoppedSpeakingFrame at the right time by simulating real-time audio playback.

**Key addition - timing simulation:**
```python
async def _simulate_playback_timing(self, duration: float):
    current_time = time.monotonic()
    sleep_duration = max(0, self._next_send_time - current_time)
    if sleep_duration > 0:
        await asyncio.sleep(sleep_duration)
        self._next_send_time += duration
    else:
        self._next_send_time = time.monotonic() + duration
```

**Configuration:**
- `BOT_VAD_STOP_SECS = 2.0` (increased from 0.35s) for reliable turn detection during LLM generation pauses

#### Pipeline Structure

```
paced_input → context_aggregator.user() → transcript.user() →
llm → ToolCallRecorder → assistant_shim → turn_gate →
context_aggregator.assistant() → output_transport → audio_buffer
```

**Key positioning:**
- `turn_gate` after `assistant_shim` - receives transcript, waits for audio completion
- `output_transport` near the end - paces audio frames, generates BotStoppedSpeakingFrame
- `audio_buffer` last - records audio AFTER pacing (accurate timing)

---

## Part 2: Nova Sonic Current State Analysis

### 2.1 Current Pipeline Structure

```
paced_input → context_aggregator.user() → llm →
ToolCallRecorder → turn_detector → context_aggregator.assistant()
```

**Missing components:**
- No `NullAudioOutputTransport` (no BotStoppedSpeakingFrame generation)
- No `AudioBufferProcessor` (no audio recording)
- No TurnGate-style coordination

### 2.2 Current Turn Detection Approach

The `NovaSonicTurnEndDetector` (~350 lines) uses multiple signals:

| Signal | Source | Purpose |
|--------|--------|---------|
| `NovaSonicTextTurnEndFrame` | LLM (AUDIO END_TURN) | Signals end of audio generation (not playout) |
| `NovaSonicCompletionEndFrame` | LLM (completionEnd) | Session completion signal |
| `TTSStoppedFrame` | LLM | Audio generation stopped |
| Text timeout (5s) | Self | Fallback if signals missing |
| Audio silence (2s) | Self | Fallback detection |
| Response timeout (30s) | Self | Max wait for any response |

**Current logic flow:**
1. Track `LLMFullResponseStartFrame` to know response is active
2. Accumulate SPECULATIVE text as it arrives (with audio)
3. On `NovaSonicTextTurnEndFrame` → short delay (1s) → trigger turn end
4. Various timeout fallbacks if signals don't arrive

### 2.3 Nova Sonic Unique Behaviors

| Behavior | Impact |
|----------|--------|
| Text arrives in two phases: SPECULATIVE (with audio) and FINAL (4-6s later) | Must capture SPECULATIVE, ignore FINAL duplicates |
| AUDIO END_TURN signals generation complete | Useful for knowing transcript is ready, but not for turn timing |
| 8-minute connection timeout | Requires reconnection handling |
| Nova v1 needs explicit trigger, v2 auto-triggers via VAD | Different trigger mechanisms |

**Note on turn-end detection**: With NullAudioOutputTransport and real-time pacing, `BotStoppedSpeakingFrame` becomes the authoritative turn-end signal (same as in the realtime pipeline). AUDIO END_TURN tells us audio *generation* is complete, but `BotStoppedSpeakingFrame` tells us audio *playout* is complete. Since we care about when the bot finishes "speaking" (from a timing perspective), `BotStoppedSpeakingFrame` is the signal we need. The AUDIO END_TURN signal is not particularly useful for turn timing.

---

## Part 3: Proposed Improvements

### 3.1 Add NullAudioOutputTransport

**Location in pipeline:**
```
paced_input → context_aggregator.user() → llm →
ToolCallRecorder → turn_detector → context_aggregator.assistant() →
output_transport → audio_buffer
```

**Benefits:**
- Generates `BotStoppedSpeakingFrame` based on actual audio output timing
- Paces audio "playback" to real-time
- Enables audio recording with accurate timing

**Implementation notes:**
- Nova Sonic outputs `TTSAudioRawFrame` (subclass of `OutputAudioRawFrame`)
- `NullAudioOutputTransport.write_audio_frame()` handles `OutputAudioRawFrame`, so TTSAudioRawFrame works without modification
- **Output sample rate: 24kHz** (confirmed from pipecat AWSNovaSonicLLMService defaults)
- Input sample rate: 16kHz (for user audio)

**CRITICAL: BOT_VAD_STOP_SECS must be set to 2.0s**

This threshold determines how long the audio queue must be **empty** before triggering `BotStoppedSpeakingFrame`. With NullAudioOutputTransport's real-time pacing:

1. Audio frames queue up and are consumed at real-time pace
2. Even if the LLM pauses during generation, the audio content plays continuously (no gaps)
3. `BotStoppedSpeakingFrame` fires only when the queue has been empty for 2s

The 2s threshold (vs default 0.35s) is needed because:
- We want high confidence the response is truly finished
- Brief delays in frame delivery shouldn't trigger premature turn advancement
- Provides margin for network jitter or processing delays

```python
# MUST set before creating NullAudioOutputTransport
import pipecat.transports.base_output as base_output_module
base_output_module.BOT_VAD_STOP_SECS = 2.0
logger.info("Set BOT_VAD_STOP_SECS to 2.0s for reliable end-of-response detection")
```

### 3.2 Add AudioBufferProcessor for Recording

**Purpose:** Record stereo audio (user=left, bot=right) to `conversation.wav`

**Implementation:**
- Create AudioBufferProcessor with sample_rate and num_channels=2
- Register `on_track_audio_data` handler
- Save to `recorder.run_dir / "conversation.wav"`

### 3.3 Simplify Turn Detection with TurnGate Pattern

**Current:** Complex timeout-based detection with many edge cases

**Proposed:** Use the same `TurnGate` pattern as the realtime pipeline, with `BotStoppedSpeakingFrame` as the primary turn-end signal:

```python
class TurnGate(FrameProcessor):
    """Gates turn advancement until bot finishes speaking.

    BotStoppedSpeakingFrame (from NullAudioOutputTransport) is the
    authoritative signal that all audio has been output.
    """
```

**Logic:**
1. Accumulate SPECULATIVE text as it arrives (via TTSTextFrame)
2. On `BotStoppedSpeakingFrame` → trigger turn end with accumulated text
3. Keep response timeout as a safety fallback

**Why BotStoppedSpeakingFrame is sufficient:**
- With real-time pacing, it fires ~2s after the last audio frame
- SPECULATIVE text arrives with audio, so transcript is already complete
- No need to wait for AUDIO END_TURN or FINAL text
- Same reliable mechanism used in the realtime pipeline

**Benefits:**
- Simpler logic (~50 lines vs ~350)
- Aligns with realtime pipeline pattern
- Turn advancement synchronized with audio completion
- No complex timeout chains or multiple signal coordination

### 3.4 Pipeline Structure After Changes

```python
pipeline = Pipeline([
    self.paced_input,
    self.context_aggregator.user(),
    self.llm,
    ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),  # Same as realtime
    self.turn_gate,               # Accumulates text, waits for BotStoppedSpeakingFrame
    self.context_aggregator.assistant(),
    self.output_transport,        # Paces audio, generates BotStoppedSpeakingFrame
    self.audio_buffer,            # Records audio
])
```

**Key insight:** The structure mirrors the realtime pipeline:
- Both use `ToolCallRecorder` for evaluation (records function calls for scoring)
- Both use `TurnGate` + `NullAudioOutputTransport` for turn detection
- Both use `AudioBufferProcessor` for recording

**Nova Sonic-specific handling in TurnGate:**
- Accumulates SPECULATIVE text from `TTSTextFrame` (arrives with audio)
- Does NOT wait for `NovaSonicTextTurnEndFrame` (end of generation is not useful for timing)
- Triggers turn end on `BotStoppedSpeakingFrame` with accumulated text

Note: Unlike the realtime pipeline which uses a separate `TTSStoppedAssistantTranscriptProcessor` for transcript capture, Nova Sonic's TurnGate handles both text accumulation and turn gating. This is simpler because SPECULATIVE text arrives via `TTSTextFrame` which can be processed directly.

---

## Part 4: Frame Flow Analysis

### 4.1 Current Nova Sonic Frame Flow

```
User Audio Input (16kHz):
  InputAudioRawFrame → LLM → (audio sent to server)

Nova Sonic Response:
  LLM emits:
    - LLMFullResponseStartFrame (response starting)
    - TTSTextFrame (SPECULATIVE text, with audio)
    - TTSAudioRawFrame (audio chunks, 24kHz)
    - NovaSonicTextTurnEndFrame (AUDIO END_TURN)
    - NovaSonicCompletionEndFrame (completionEnd)
    - TTSStoppedFrame (audio output complete)
    - LLMFullResponseEndFrame (response ending)
```

### 4.2 Proposed Frame Flow

```
User Audio Input (16kHz):
  PacedInputTransport → InputAudioRawFrame → LLM

Nova Sonic Response:
  LLM emits TTSTextFrame (SPECULATIVE) + TTSAudioRawFrame →
  TranscriptAccumulator (accumulates text) →
  TurnGate (passes frames through, waiting for BotStoppedSpeakingFrame) →
  NullAudioOutputTransport (paces audio at real-time, generates BotStoppedSpeakingFrame after 2s of empty queue) →
  AudioBufferProcessor (records audio)

Turn End:
  After all audio "played" + 2s silence:
  NullAudioOutputTransport emits BotStoppedSpeakingFrame (flows upstream) →
  TurnGate receives it → triggers turn end callback with accumulated text
```

**Note:** `NovaSonicTextTurnEndFrame` and other Nova Sonic-specific signals are no longer needed for turn detection. The `BotStoppedSpeakingFrame` mechanism handles turn timing uniformly.

### 4.3 Key Timing Considerations

| Event | Current | Proposed |
|-------|---------|----------|
| User audio complete | ~3-5s per turn | Same |
| Bot audio generation | Faster than real-time | Same |
| Bot audio "playback" | Not tracked | Paced to real-time via NullAudioOutputTransport |
| Turn end trigger | On NovaSonicTextTurnEndFrame + 1s delay | On BotStoppedSpeakingFrame (2s after last audio frame) |
| Transcript availability | Accumulated during response | Same (SPECULATIVE text arrives with audio) |

---

## Part 5: Testing Strategy

### 5.1 Quick Functional Tests (2-3 Turns)

```bash
# Run a short test
uv run multi-turn-eval run aiwf_medium_context \
    --model amazon.nova-2-sonic-v1:0 \
    --pipeline nova-sonic \
    --only-turns 0,1,2 \
    --verbose
```

**Success criteria:**
- All 3 turns complete without errors
- Transcript captured for each turn
- No premature turn advancement

### 5.2 Log Analysis for Frame Timing

Examine `run.log` for:

**User audio timing:**
```
grep "SENDING REAL AUDIO" run.log
grep "FINISHED SENDING AUDIO" run.log
```

**Bot audio timing (after adding NullAudioOutputTransport):**
```
grep "\[NullAudioOutput\]" run.log
```

**Turn transitions:**
```
grep "TurnGate" run.log
grep "BotStoppedSpeakingFrame" run.log
```

**Expected pattern per turn:**
1. `SENDING REAL AUDIO` - user audio starts
2. `FINISHED SENDING AUDIO` - user audio complete
3. `[NullAudioOutput] First audio frame` - bot starts speaking
4. `[NullAudioOutput] Frame N: ...` - periodic progress
5. `BotStoppedSpeakingFrame` - bot finished speaking
6. `[TurnGate] Triggering turn end` - turn advances

### 5.3 Audio Level Analysis on conversation.wav

**Goal:** Verify turn timing in the recording matches log timing.

**Analysis script outline:**

```python
#!/usr/bin/env python3
"""Analyze conversation.wav for turn segmentation and timing verification."""

import wave
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def load_stereo_wav(path):
    """Load stereo WAV file, return (user_audio, bot_audio, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        assert wf.getnchannels() == 2, "Expected stereo"
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        return audio[:, 0], audio[:, 1], sample_rate

def find_speech_segments(audio, sample_rate, threshold_db=-40, min_duration_ms=100):
    """Find speech segments using energy-based detection."""
    # Convert to energy (dB)
    frame_size = int(sample_rate * 0.025)  # 25ms frames
    hop_size = int(sample_rate * 0.010)    # 10ms hop

    segments = []
    # ... energy-based detection logic ...
    return segments

def parse_log_timing(log_path):
    """Parse run.log for SENDING/FINISHED audio events."""
    events = []
    with open(log_path) as f:
        for line in f:
            if "SENDING REAL AUDIO" in line or "FINISHED SENDING AUDIO" in line:
                # Parse timestamp and event
                # ...
                pass
    return events

def compare_timing(audio_segments, log_events, tolerance_ms=500):
    """Compare audio segments with log events."""
    mismatches = []
    for i, (audio_seg, log_event) in enumerate(zip(audio_segments, log_events)):
        delta = abs(audio_seg['start'] - log_event['time'])
        if delta > tolerance_ms:
            mismatches.append({
                'turn': i,
                'audio_start': audio_seg['start'],
                'log_time': log_event['time'],
                'delta_ms': delta
            })
    return mismatches

if __name__ == "__main__":
    import sys
    run_dir = Path(sys.argv[1])

    wav_path = run_dir / "conversation.wav"
    log_path = run_dir / "run.log"

    user_audio, bot_audio, sr = load_stereo_wav(wav_path)

    print(f"Audio duration: {len(user_audio) / sr:.2f}s")
    print(f"Sample rate: {sr}Hz")

    user_segments = find_speech_segments(user_audio, sr)
    bot_segments = find_speech_segments(bot_audio, sr)

    print(f"\nUser speech segments: {len(user_segments)}")
    for i, seg in enumerate(user_segments):
        print(f"  {i}: {seg['start_ms']:.0f}ms - {seg['end_ms']:.0f}ms")

    print(f"\nBot speech segments: {len(bot_segments)}")
    for i, seg in enumerate(bot_segments):
        print(f"  {i}: {seg['start_ms']:.0f}ms - {seg['end_ms']:.0f}ms")

    # Compare with log timing
    log_events = parse_log_timing(log_path)
    mismatches = compare_timing(user_segments, log_events)

    if mismatches:
        print("\nTiming mismatches detected:")
        for m in mismatches:
            print(f"  Turn {m['turn']}: audio={m['audio_start']:.0f}ms, log={m['log_time']:.0f}ms, delta={m['delta_ms']:.0f}ms")
    else:
        print("\nAll segments align within tolerance!")
```

### 5.4 Automated Test Script

```bash
#!/bin/bash
# test-nova-sonic-improvements.sh

set -e

# Run short test
echo "=== Running 3-turn Nova Sonic test ==="
uv run multi-turn-eval run aiwf_medium_context \
    --model amazon.nova-2-sonic-v1:0 \
    --pipeline nova-sonic \
    --only-turns 0,1,2 \
    --verbose

# Find the run directory
RUN_DIR=$(ls -td runs/aiwf_medium_context/*nova* | head -1)
echo "Run directory: $RUN_DIR"

# Check for conversation.wav
if [ -f "$RUN_DIR/conversation.wav" ]; then
    echo "=== conversation.wav found ==="
    file "$RUN_DIR/conversation.wav"
else
    echo "ERROR: conversation.wav not found"
    exit 1
fi

# Check log for expected frame patterns
echo ""
echo "=== Checking frame timing in logs ==="

echo "User audio events:"
grep -c "SENDING REAL AUDIO" "$RUN_DIR/run.log" || true

echo "Bot audio events:"
grep -c "\[NullAudioOutput\]" "$RUN_DIR/run.log" || true

echo "BotStoppedSpeakingFrame events:"
grep -c "BotStoppedSpeakingFrame" "$RUN_DIR/run.log" || true

echo "Turn gate events:"
grep -c "TurnGate" "$RUN_DIR/run.log" || true

# Run audio analysis
echo ""
echo "=== Audio analysis ==="
python scripts/analyze_conversation_wav.py "$RUN_DIR"

echo ""
echo "=== Test complete ==="
```

---

## Part 6: Implementation Checklist

### Phase 1: Add Audio Output Transport

- [ ] Import NullAudioOutputTransport in nova_sonic.py
- [ ] **CRITICAL: Set BOT_VAD_STOP_SECS = 2.0 BEFORE creating transport** (Nova Sonic has 2+ second pauses between audio segments)
- [ ] Create NullAudioOutputTransport instance with correct sample rate
- [ ] Add to pipeline after context_aggregator.assistant()
- [ ] Test: verify BotStoppedSpeakingFrame appears in logs (should fire ~2s after last audio frame)

### Phase 2: Add Audio Recording

- [ ] Import AudioBufferProcessor
- [ ] Create AudioBufferProcessor instance
- [ ] Register on_track_audio_data handler
- [ ] Call start_recording() in _queue_first_turn()
- [ ] Add to pipeline after NullAudioOutputTransport
- [ ] Test: verify conversation.wav is created

### Phase 3: Simplify Turn Detection

- [ ] Create NovaSonicTurnGate (based on TurnGate pattern)
- [ ] Accumulate SPECULATIVE text from TTSTextFrame (in process_frame)
- [ ] On BotStoppedSpeakingFrame → trigger turn end with accumulated text
- [ ] Keep response timeout as fallback (for error cases)
- [ ] Remove/replace NovaSonicTurnEndDetector with simpler TurnGate
- [ ] Test: verify turns advance correctly

Note: NovaSonicTextTurnEndFrame is NOT needed for turn detection. It signals end of generation, not end of playout. The BotStoppedSpeakingFrame from NullAudioOutputTransport is the authoritative signal.

### Phase 4: Testing

- [ ] Run 3-turn quick test
- [ ] Verify all turns complete
- [ ] Analyze run.log for frame timing
- [ ] Create analyze_conversation_wav.py script
- [ ] Verify audio timing matches log timing
- [ ] Run full 30-turn test
- [ ] Compare pass rate with previous implementation

---

## Part 7: Risk Assessment

### Low Risk
- Adding NullAudioOutputTransport (additive change)
- Adding AudioBufferProcessor (additive change)

### Medium Risk
- Simplifying turn detection (behavioral change)
- May need to handle Nova Sonic's multi-segment responses
- Need to ensure SPECULATIVE text is captured before END_TURN signal

### Mitigation
- Keep existing NovaSonicTurnEndDetector as fallback
- Use feature flag to switch between old/new detection
- Extensive logging during transition
- Side-by-side comparison of old vs new results

---

## Appendix A: Comparison of Frame Types

| Frame Type | Realtime Pipeline | Nova Sonic | Notes |
|------------|-------------------|------------|-------|
| InputAudioRawFrame | User audio input | User audio input | Same |
| OutputAudioRawFrame | Bot audio output | | Realtime models use this |
| TTSAudioRawFrame | | Bot audio output | Nova Sonic uses this |
| TTSTextFrame | | SPECULATIVE text | With audio |
| BotStoppedSpeakingFrame | From NullAudioOutputTransport | Not generated (yet) | Key addition |
| NovaSonicTextTurnEndFrame | | AUDIO END_TURN | Nova Sonic specific |

---

## Appendix B: Audio Sample Rate Matrix

| Component | Realtime (OpenAI) | Realtime (Gemini) | Nova Sonic |
|-----------|-------------------|-------------------|------------|
| User audio input | 24kHz | 24kHz | 16kHz |
| Bot audio output | 24kHz | 24kHz | 24kHz |
| Recording | 24kHz stereo | 24kHz stereo | 24kHz stereo |

**Note:** For Nova Sonic recording, user audio (16kHz) should be upsampled to 24kHz to match bot audio for stereo recording.

---

## Appendix C: Reconnection Handling

Nova Sonic has 8-minute connection timeout requiring special handling. The NullAudioOutputTransport must reset its timing state on reconnection:

```python
def reset_for_reconnection(self):
    """Reset timing state after reconnection."""
    self._next_send_time = 0.0
    self._total_audio_duration = 0.0
    self._total_sleep_time = 0.0
    self._frame_count = 0
```

This should be called from the `on_reconnected` callback.
