# Audio Alignment Analysis Guide

This document describes how to verify that user and bot audio are correctly aligned in benchmark recordings.

## Overview

The benchmark pipeline records a stereo WAV file (`conversation.wav`) with:
- **Left channel**: User audio (from pre-recorded turn files)
- **Right channel**: Bot audio (from the speech-to-speech model)

Both channels should be aligned to wall-clock time, meaning audio appears at the correct position relative to when it was sent/received.

## Quick Verification

Run the Silero VAD analysis script on a benchmark run:

```bash
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<run_dir>/ -v
```

This uses Silero VAD (neural network) for accurate speech boundary detection. Check that:
- User/bot segment counts match expected turn count
- V2V latencies are reasonable (typically 1000-1500ms for non-tool-call turns)

**Note**: There's also `analyze_conversation_wav.py` which uses energy-based VAD (RMS threshold). This is less accurate and tends to over-segment audio. Prefer `analyze_ttfb_silero.py` for TTFB analysis.

**Model names**: Use `gpt-realtime` (not `gpt-4o-realtime-preview`) for OpenAI Realtime benchmarks. See README.md for model/service mappings.

## Detailed Analysis

### 1. Check Recording Duration

The WAV duration should match the benchmark wall-clock time:

```bash
python3 -c "
import wave
with wave.open('runs/aiwf_medium_context/<run_dir>/conversation.wav', 'rb') as wf:
    print(f'Duration: {wf.getnframes() / wf.getframerate():.3f}s')
"
```

Compare to the elapsed time in the log (last timestamp minus first timestamp).

### 2. Extract Key Timestamps from Log

```bash
grep -E "(Recording baseline|SENDING REAL AUDIO|Bot audio initialized|Bot: Inserted.*silence)" \
    runs/aiwf_medium_context/<run_dir>/run.log | head -10
```

Key events:
- **Recording baseline reset**: T=0 for the WAV file
- **First SENDING REAL AUDIO**: When user audio transmission started
- **Bot audio initialized**: When first bot audio frame arrived
- **Bot: Inserted Xms silence**: Silence filled before first bot audio

### 3. Verify Sample Tracking

Check that audio samples track wall-clock time throughout the recording:

```bash
grep "Bot frame\|User frame" runs/aiwf_medium_context/<run_dir>/run.log
```

Output shows:
```
Bot frame 500: elapsed=137.33s, actual_samples=3296950, expected=3296038, diff=+38ms
```

- **diff** should be within +/- 50ms
- Positive diff means samples are slightly ahead (acceptable)
- Large negative diff indicates samples falling behind (problem)

### 4. Verify Turn 1 Timing

Calculate expected vs actual positions for the first turn:

```python
# From log:
recording_baseline = 0  # T=0
first_user_audio = 100  # ms (after pre-roll, from "SENDING REAL AUDIO" timestamp)
first_bot_audio = 5753  # ms (from "Bot audio initialized" timestamp)

# From VAD analysis of WAV:
user_speech_start = 610   # ms (includes silence in audio file)
bot_speech_start = 6410   # ms (includes OpenAI silent padding)

# Verify bot timing:
# Bot audio should start at position = first_bot_audio (silence was inserted)
# Bot speech = first_bot_audio + OpenAI_silent_padding
openai_padding = bot_speech_start - first_bot_audio  # ~650ms is normal
```

### 5. Verify V2V Matches Expected

For turn 1:
```
V2V_gap = bot_speech_start - user_speech_end
Service_TTFB = first_bot_audio - user_speech_end
Expected_V2V = Service_TTFB + OpenAI_silent_padding

# These should match:
assert abs(V2V_gap - Expected_V2V) < 50  # Within 50ms
```

## Expected Behavior

### Pre-roll Silence (User Audio)
- `pre_roll_ms=100` sends 100ms of silence before first user audio
- This allows the pipeline to stabilize before real audio

### OpenAI Silent Padding (Bot Audio)
- OpenAI Realtime sends 400-700ms of silent audio frames before actual speech
- This is model behavior, not a recording issue
- Service TTFB measures time to first audio BYTE (including silence)
- WAV V2V measures time to first audible SPEECH

### Silence Insertion
- `NullAudioOutputTransport` inserts silence to fill gaps > 10ms
- This ensures continuous wall-clock aligned recording
- Check log for "Inserted Xms silence" messages

## Troubleshooting

### Large User Audio Offset (>100ms)
1. Check frame traversal time for first frames
2. Verify `on_pipeline_started` event handler is triggering
3. Ensure recording baseline is set AFTER pipeline is ready

### Large Bot Audio Offset (>100ms)
1. Check sample tracking logs for drift
2. Verify silence insertion is happening correctly
3. Check for clock source mismatches (time.time vs time.monotonic)

### Cumulative Drift
1. Compare sample tracking at start vs end of recording
2. Check for sample rate mismatches between components
3. Verify resampling is working correctly

## Key Files

### Pipeline Components
- `src/multi_turn_eval/transports/null_audio_output.py`: Silence insertion and sample tracking
- `src/multi_turn_eval/transports/paced_input.py`: User audio transmission timing
- `src/multi_turn_eval/pipelines/realtime.py`: Recording baseline initialization

### Analysis Scripts
- `scripts/analyze_ttfb_silero.py`: **Primary analysis tool** - Uses Silero VAD (ONNX) for accurate speech boundary detection and V2V latency measurement
- `scripts/analyze_conversation_wav.py`: Secondary tool - Uses energy-based VAD (RMS threshold), useful for log-to-audio timestamp correlation but less accurate for segment detection

---

## In-Pipeline Speech Onset Detection

The pipeline includes RMS-based speech onset detection in `NullAudioOutputTransport` to measure silent padding in real-time.

### How It Works

When bot audio arrives, the pipeline:
1. Logs when first bot audio byte arrives (silence insertion log)
2. Checks each frame's RMS level against `BOT_SPEECH_RMS_THRESHOLD_DB` (-30dB)
3. Logs when RMS exceeds threshold as "Bot speech onset"

### Log Output

```
[NullAudioOutput] Bot: Inserted 5992ms silence ...
[NullAudioOutput] Bot speech onset: T+6040ms (sample_pos=6040ms, silent_padding=48ms, rms=-24.5dB)
```

- `T+Xms`: Wall-clock elapsed time since recording started
- `sample_pos`: Actual sample position in WAV file (should match T+ within ~15ms)
- `silent_padding`: Time from first bot audio byte to speech onset
- `rms`: RMS level that triggered detection

### Extracting Speech Onset Data

```bash
# Get bot speech onset events
grep "Bot speech onset" runs/aiwf_medium_context/<run_dir>/run.log

# Get bot audio arrival times (silence insertion)
grep "Bot: Inserted" runs/aiwf_medium_context/<run_dir>/run.log
```

### RMS Threshold Selection

The threshold affects what counts as "speech onset":

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| -40dB | Triggers on low-level pre-speech audio | Measures "time to any audio" |
| -30dB | Triggers closer to actual speech | Matches Silero VAD within ~50-100ms |
| -25dB | Only triggers on clear speech | May miss quiet speech starts |

**Recommendation**: Use -30dB (current default) for speech onset detection.

### Comparing RMS to Silero VAD

To validate RMS-based detection against Silero VAD:

```bash
# 1. Get RMS-based speech onset from logs
grep "Bot speech onset" runs/aiwf_medium_context/<run_dir>/run.log

# 2. Run Silero VAD analysis
uv run python scripts/analyze_ttfb_silero.py runs/aiwf_medium_context/<run_dir>/ -v

# 3. Compare per-turn: RMS sample_pos vs Silero bot segment start
```

Expected: RMS triggers 50-100ms before Silero with -30dB threshold.

### Detailed Silero VAD Analysis

To get exact Silero segment timestamps:

```python
import numpy as np
import wave
import torch
from scipy import signal

# Load audio
with wave.open('runs/aiwf_medium_context/<run_dir>/conversation.wav', 'rb') as wf:
    sr = wf.getframerate()
    n_frames = wf.getnframes()
    audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).reshape(-1, 2)

# Get bot channel (right)
bot_audio = audio[:, 1].astype(np.float32) / 32768.0

# Resample to 16kHz for Silero
target_sr = 16000
bot_16k = signal.resample(bot_audio, int(len(bot_audio) * target_sr / sr))

# Load Silero VAD
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=True)
(get_speech_timestamps, _, _, _, _) = utils

# Run VAD
segments = get_speech_timestamps(
    torch.from_numpy(bot_16k.astype(np.float32)),
    model,
    sampling_rate=target_sr,
    min_silence_duration_ms=2000,
    speech_pad_ms=0
)

for i, seg in enumerate(segments):
    start_ms = seg['start'] / target_sr * 1000
    end_ms = seg['end'] / target_sr * 1000
    print(f'Turn {i}: {start_ms:.0f}ms - {end_ms:.0f}ms')
```

---

## Silent Padding Analysis

OpenAI Realtime sends silent/low-level audio before actual speech begins. This "silent padding" varies by turn.

### Measuring Silent Padding

**Method 1: From pipeline logs (RMS-based)**
```bash
grep "Bot speech onset" run.log | grep "silent_padding"
# Output: silent_padding=48ms, silent_padding=31ms, ...
```

**Method 2: From WAV analysis (Silero-based)**
```
Silent padding = Silero bot speech start - Bot audio arrival time
```

### Typical Values

| Turn Type | Silent Padding (Silero) |
|-----------|------------------------|
| First response | 50-150ms |
| Subsequent turns | 400-700ms |

The difference is because Turn 0 starts immediately after connection, while later turns involve processing conversation history.

### Audio Flow Through Pipeline

Understanding the audio flow helps debug timing issues:

```
OpenAI sends OutputAudioRawFrame
    ↓
NullAudioOutputTransport.process_frame()
    ↓
_emit_bot_with_silence_fill()
    ├─→ RMS check on ORIGINAL frame
    └─→ super().process_frame()
            ↓
        BaseOutputTransport._handle_frame()
            ↓
        MediaSender.handle_audio_frame()
            ├─→ Resample to output sample rate
            ├─→ Buffer in _audio_buffer
            └─→ Chunk when buffer >= chunk_size → _audio_queue
                    ↓
                _audio_task_handler() pulls from queue
                    ├─→ write_audio_frame() (timing simulation)
                    └─→ push_frame() → sample counting
                            ↓
                        AudioBufferProcessor → WAV file
```

Key insight: RMS is checked on the original frame, sample counting happens after MediaSender chunking. These should match within ~15ms.

---

## Running Benchmarks

### Command Format

```bash
uv run multi-turn-eval run <benchmark> --model <model> --service <service> --pipeline <pipeline> [--only-turns X,Y,Z]
```

### Examples

```bash
# Full 30-turn benchmark
uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime --pipeline realtime

# Quick 3-turn test
uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime --pipeline realtime --only-turns 0,1,2
```

### Important: Never Use `tee` for Streaming Logs

When running benchmarks, **do not** pipe through `tee`:

```bash
# BAD - fills up context with streaming logs
uv run multi-turn-eval run ... 2>&1 | tee run.log

# GOOD - redirect to file, check periodically
uv run multi-turn-eval run ... > run.log 2>&1 &
ps aux | grep multi-turn-eval  # check if running
tail -20 run.log               # check progress
```

### Post-Run Analysis Checklist

1. **Check completion**: `tail -30 run.log`
2. **Get speech onset data**: `grep "Bot speech onset" run.log`
3. **Get silence insertion data**: `grep "Bot: Inserted" run.log`
4. **Run Silero analysis**: `uv run python scripts/analyze_ttfb_silero.py runs/.../ -v`
5. **Compare RMS vs Silero per turn**
