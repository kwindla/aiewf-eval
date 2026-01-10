# Plan: Verify OpenAI Silent Audio Padding

## Goal
~~Confirm that the ~500ms difference between service TTFB and WAV V2V is due to OpenAI sending silent audio frames before actual speech.~~

**Updated**: Investigate the ~500ms difference between Server TTFB and WAV V2V. Analysis revealed the difference comes from two sources, not just silent padding.

---

## Per-Turn Metrics Reference

These are the key timing metrics we track for each turn in a benchmark run.

### Primary Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| **Server TTFB** | `transcript.jsonl` → `ttfb_ms` | OpenAI's reported time from speech_stopped to first audio delta |
| **Pipeline TTFB** | Pipeline logs (derived) | User speech end → first bot audio byte arrival |
| **WAV V2V** | Silero VAD on `conversation.wav` | User speech end → bot speech start (audible speech) |
| **Silent Padding (RMS)** | Pipeline logs | First bot audio byte → RMS threshold crossing (-30dB) |
| **Silent Padding (VAD)** | WAV analysis | Bot audio tag position → Silero VAD speech detection |

### Raw Timestamps (from logs and WAV analysis)

| Timestamp | Source | Description |
|-----------|--------|-------------|
| User speech end | Silero VAD on WAV | When user stopped speaking (ms in WAV) |
| Bot tag position | Pipeline logs (`sample_pos`) | When first bot audio byte arrived (after silence insertion) |
| Bot RMS onset | Pipeline logs (`sample_pos`) | When RMS exceeded -30dB threshold |
| Bot speech start | Silero VAD on WAV | When Silero detected bot speech |

### Metric Relationships

```
WAV V2V = Pipeline TTFB + Silent Padding (Silero)
Pipeline TTFB ≈ Server TTFB + VAD_turn_detection_delay + network_latency
Silent Padding (RMS) ≈ Silent Padding (Silero) - 100ms  (RMS triggers earlier)
```

### Analysis Scripts

| Script | Metrics Provided |
|--------|------------------|
| `scripts/analyze_turn_metrics.py` | **All metrics** - Server TTFB, Pipeline TTFB, WAV V2V, Silent Padding (RMS & Silero), alignment check |
| `scripts/analyze_ttfb_silero.py` | WAV V2V only (Silero-based) |
| `scripts/detect_audio_tags.py` | Tag positions for alignment verification |

---

### `analyze_turn_metrics.py` - Comprehensive Per-Turn Analysis

**Usage:**
```bash
# Full analysis with per-turn breakdown
uv run python scripts/analyze_turn_metrics.py runs/<run_id> -v

# JSON output for programmatic use
uv run python scripts/analyze_turn_metrics.py runs/<run_id> --json
```

**Data Sources:**
1. `transcript.jsonl` → Server TTFB (`ttfb_ms`), tool call flags
2. `run.log` → Bot tag positions, RMS onset positions, silent padding
3. `conversation.wav` → Silero VAD segments, FFT-based tag detection

**Alignment Sanity Check:**
- Compares log tag positions (`sample_pos`) to WAV-detected tag positions
- Uses proximity matching (within 100ms) to handle false positives in WAV detection
- Reports alignment statistics and flags any tags outside ±20ms tolerance
- Identifies unmatched WAV tags (likely false positives from speech harmonics)

**Segment Matching:**
- Bot segments matched by finding Silero segment closest to bot tag position
- User segments matched by finding segment that ends just before bot speech starts
- This temporal matching handles cases where Silero merges adjacent user segments

**Output Columns:**

| Column | Header | Description | Calculation |
|--------|--------|-------------|-------------|
| Turn | `Turn` | Turn index (0-based) | Sequential from bot tags in log |
| Timestamp | `Timestamp` | WAV file position when user started speaking | `user_start_ms` from Silero VAD (displayed as seconds) |
| Server TTFB | `Srvr TTFB` | OpenAI's reported time-to-first-byte | From `transcript.jsonl` → `ttfb_ms` |
| Pipeline TTFB | `Pipe TTFB` | Time from user speech end to first bot audio byte | `bot_tag_log_ms - user_end_ms` (Silero) |
| WAV V2V | `WAV V2V` | Voice-to-voice latency (audible speech boundaries) | `bot_silero_start_ms - user_end_ms` |
| Pad RMS | `Pad RMS` | Silent padding measured by RMS threshold (-30dB) | `bot_rms_onset_ms - bot_tag_log_ms` (from logs) |
| Pad VAD | `Pad VAD` | Silent padding measured by Silero VAD | `bot_silero_start_ms - bot_tag_wav_ms` |
| Align | `Align` | Tag alignment between log and WAV (sanity check) | `bot_tag_log_ms - bot_tag_wav_ms` (should be 12-16ms) |
| Notes | (suffix) | `[T]` if turn has tool calls | From `transcript.jsonl` → `tool_calls` |

**Known Limitations:**
- Silero VAD may merge user segments with < 2s silence between them, causing segment count mismatches
- Turns with very short user speech (< 900ms) may show N/A for Silero-based metrics
- Some turns may show N/A for Pipeline TTFB/WAV V2V if the user segment is not detected
- False positives in WAV tag detection (typically low RMS, filtered by proximity matching)
- Turn 0 often shows Server TTFB = 0ms (OpenAI measurement artifact)

**Example Output (29-turn run):**
```
Turn |  Timestamp | Srvr TTFB | Pipe TTFB |   WAV V2V |  Pad RMS |  Pad VAD |  Align
-----+------------+-----------+-----------+-----------+----------+----------+-------
   0 |      1.0s |       0ms |    1117ms |    1632ms |    160ms |    530ms |   15ms
   1 |     33.9s |     886ms |    1479ms |    2080ms |     80ms |    617ms |   16ms
   2 |     71.0s |     439ms |     683ms |     704ms |     80ms |     37ms |   16ms
  ...
  11 |    327.1s |    1057ms |    2491ms |    2688ms |     80ms |    212ms |   15ms [T]
  ...
  28 |    699.9s |    1069ms |    1466ms |    1472ms |     40ms |     22ms |   16ms

Summary Statistics:
  Server TTFB:          median=671ms, range=0-1491ms
  Pipeline TTFB:        median=1059ms, range=642-2491ms
  WAV V2V:              median=1344ms, range=672-2688ms
  Silent Pad (RMS):     median=80ms, range=40-200ms
  Silent Pad (VAD):     median=188ms, range=22-629ms
```

`[T]` indicates turns with tool calls (typically higher latency).

---

## Three Metrics to Compare

### 1. Server TTFB
**Source**: `ttfb_ms` from transcript.jsonl (measured by pipecat client-side)

```
Server TTFB = time(receive response.audio.delta) - time(receive input_audio_buffer.speech_stopped)
```
Measured when OpenAI's WebSocket events arrive. Since both events travel the same network path, latency mostly cancels out, so this ≈ OpenAI's server-side processing time.

### 2. Pipeline TTFB
**Source**: Calculated from our logs

```
Actual user speech end = VADUserStoppedSpeakingFrame timestamp - VAD stop_secs
Pipeline TTFB = First bot audio frame arrival - Actual user speech end
```
We control `stop_secs` and know exactly when user speech ended.

### 3. WAV V2V
**Source**: Silero VAD analysis of conversation.wav

```
WAV V2V = Silero bot speech start - Silero user speech end
```
Measures audible speech boundaries, includes silent audio padding.

### Expected Relationships (Original Hypothesis)
- `Pipeline TTFB` ≈ `Server TTFB` (if OpenAI's VAD agrees with ours)
- `WAV V2V - Pipeline TTFB` ≈ OpenAI's silent audio padding (~500ms)

### Actual Relationships (Analysis Results)
From analysis of 30-turn benchmark run:

| Metric | Median | Range |
|--------|--------|-------|
| Server TTFB | 732ms | 26ms - 2364ms |
| Pipeline TTFB | 1159ms | 720ms - 2874ms |
| WAV V2V | 1312ms | 920ms - 3000ms |
| Silent Padding (direct) | 190ms | 8ms - 958ms |

**Key Finding**: The ~500ms difference between Server TTFB and WAV V2V comes from TWO sources:

1. **Server vs Pipeline TTFB gap (~427ms median)**:
   - OpenAI's VAD turn detection delay (~350ms) - time for server-side VAD to confirm end-of-speech
   - Network latency (~77ms) - round-trip for events vs our local measurement
   - This is NOT VAD disagreement; Server TTFB starts when OpenAI's `speech_stopped` event arrives

2. **Silent audio padding (~190ms median)**:
   - Time from first bot audio frame to actual speech
   - Highly variable (8ms to 958ms)
   - Verified: correlation r=0.981 between direct measurement and (WAV V2V - Pipeline TTFB)

## Problem with Simple Comparison (Resolved)
Comparing transcript `ttfb_ms` to Silero VAD V2V shows high variability (274ms to 1968ms). This was confounded by:
- Some turns have suspiciously low service TTFB (26-31ms) - likely OpenAI measurement artifacts
- Turn alignment between data sources may be off
- Tool call turns add extra latency

**Resolution**: Using Pipeline TTFB (calculated from our logs) instead of Server TTFB reveals that the gap is consistent (~153ms median for silent padding) with r=0.981 correlation.

## Approach: Compare All Three Metrics

Calculate Pipeline TTFB from logs, then compare all three metrics per turn.

### Implementation

Create a new analysis script `scripts/analyze_bot_silence.py` that:

1. **Parse run.log** to extract timing events:
   - **VADUserStoppedSpeakingFrame** timestamps (when VAD fires)
   - **VAD stop_secs** configuration (e.g., `BOT_VAD_STOP_SECS=2.0`)
   - **Bot audio arrival** times (from "Bot: Inserted Xms silence" or "Bot audio initialized")
   - Calculate: `Actual user speech end = VADUserStoppedSpeaking - stop_secs`
   - Calculate: `Pipeline TTFB = Bot audio arrival - Actual user speech end`

2. **Load transcript.jsonl** for Server TTFB (`ttfb_ms` per turn)

3. **Run Silero VAD** on conversation.wav to get WAV V2V per turn

4. **For each turn**, also measure silent padding directly:
   - **RMS method**: From bot audio arrival position, find where energy exceeds -30dB
   - **Silero method**: Use bot segment start from Silero analysis
   - Calculate: `Silent padding = Speech onset - Bot audio arrival`

5. **Output per-turn analysis**:
   ```
   Turn  Server TTFB  Pipeline TTFB  WAV V2V  Silent Padding (RMS)  Silent Padding (Silero)
   0     0ms          720ms          1664ms   597ms                 657ms
   1     376ms        520ms          992ms    480ms                 520ms
   ...
   ```

6. **Statistical summary**:
   - Mean/median for each metric
   - `WAV V2V - Pipeline TTFB` = silent audio padding (~153ms median, NOT 500ms)
   - `Pipeline TTFB - Server TTFB` = VAD turn detection + network latency (~427ms median)
   - Direct silent padding measurements for validation

### Key Implementation Details

```python
# 1. Parse VAD stop_secs configuration
stop_secs_pattern = r"BOT_VAD_STOP_SECS.*?([0-9.]+)"
# Example: "[AudioRecording] Set BOT_VAD_STOP_SECS to 2.0s"

# 2. Parse VADUserStoppedSpeakingFrame timestamps
vad_stopped_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*VADUserStoppedSpeakingFrame"
# Each match = when VAD fired for user speech end

# 3. Parse bot audio arrival times
bot_arrival_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Bot: Inserted (\d+)ms silence"
# OR for first turn: "Bot audio initialized"

# 4. Calculate Pipeline TTFB per turn
actual_user_speech_end = vad_stopped_timestamp - stop_secs
pipeline_ttfb = bot_arrival_timestamp - actual_user_speech_end

# 5. Get Silero boundaries by calling analyze_ttfb_silero internally
# This gives us bot_segments with start_ms for each response

# 6. Calculate speech onset using RMS threshold
def find_speech_onset(audio, sample_rate, start_sample, threshold_db=-30):
    """Find first sample where energy exceeds threshold."""
    window_samples = int(sample_rate * 0.05)  # 50ms windows
    for i in range(start_sample, len(audio) - window_samples, window_samples // 2):
        window = audio[i:i + window_samples]
        rms_db = 20 * np.log10(np.sqrt(np.mean(window**2)) / 32768 + 1e-10)
        if rms_db > threshold_db:
            return i
    return None
```

### Matching Turns

The script needs to match events across sources by turn index:
1. **VADUserStoppedSpeakingFrame[n]** → User speech end for turn n
2. **Bot silence insertion[n]** → Bot audio arrival for turn n
3. **Silero bot segment[n]** → Bot speech onset for turn n
4. **transcript.jsonl entry[n]** → Server TTFB for turn n

Events should be in chronological order, so turn matching is sequential.

### Analysis Results

**Hypothesis was partially correct**: Silent audio padding exists but is only ~190ms median, not ~500ms.

The larger ~500ms gap between Server TTFB and WAV V2V is explained by:
```
WAV V2V = Server TTFB + VAD_turn_detection + network_latency + silent_padding
        ≈ Server TTFB + 350ms + 77ms + 190ms
        ≈ Server TTFB + 617ms
```

**Per-turn data** showed:
- `WAV V2V - Pipeline TTFB` ≈ 153ms median (= silent padding)
- `Pipeline TTFB - Server TTFB` ≈ 427ms median (= VAD turn detection + network)
- Direct silent padding measurements corroborate the calculated values

### Files to Create/Modify

- **Create**: `scripts/analyze_bot_silence.py` - New analysis script comparing all three metrics
- **Update**: `docs/audio-alignment-analysis.md` - Add section on TTFB metrics and silent padding

### Verification (Completed)

Analysis was performed manually on `runs/aiwf_medium_context/20260109T093010_gpt-realtime_25ac82ec/`:

1. ✅ Extracted all three metrics per turn
2. ✅ Found `WAV V2V - Pipeline TTFB` ≈ 153ms median (silent padding, not 500ms)
3. ✅ Direct silent padding measurements (190ms median) corroborated the difference
4. ✅ Server vs Pipeline TTFB gap (~427ms) explained by VAD turn detection + network latency

### Next Steps

#### 1. Add RMS-based speech onset detection to NullAudioOutputTransport

Add a simple RMS check to detect when the first non-silent bot audio frame arrives from the service. This provides a pipeline-based measurement of silent padding.

**File**: `src/multi_turn_eval/transports/null_audio_output.py`

**Implementation**:
```python
# Constants
BOT_SPEECH_RMS_THRESHOLD_DB = -30  # dB threshold for speech detection (tuned from -40)

# New instance variables in __init__:
self._bot_first_speech_logged: bool = False  # Reset per turn
self._bot_audio_start_time: float = 0.0      # When first bot audio byte arrived

# In _emit_bot_with_silence_fill(), after first frame initialization:
if self._bot_sample_rate == 0:
    ...
    self._bot_audio_start_time = time.time()  # Record when bot audio started
    self._bot_first_speech_logged = False     # Reset for new turn

# After processing each frame, check for first non-silent frame:
if not self._bot_first_speech_logged:
    rms_db = self._calculate_rms_db(frame.audio)
    if rms_db > BOT_SPEECH_RMS_THRESHOLD_DB:
        speech_onset_time = time.time()
        silent_padding_ms = (speech_onset_time - self._bot_audio_start_time) * 1000
        elapsed_ms = (speech_onset_time - self._recording_start_time) * 1000
        logger.info(
            f"[NullAudioOutput] Bot speech onset: T+{elapsed_ms:.0f}ms "
            f"(silent_padding={silent_padding_ms:.0f}ms, rms={rms_db:.1f}dB)"
        )
        self._bot_first_speech_logged = True

# Helper method:
def _calculate_rms_db(self, audio_bytes: bytes) -> float:
    """Calculate RMS level in dB for audio frame."""
    import numpy as np
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio) == 0:
        return -100.0
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms < 1:
        return -100.0
    return 20 * np.log10(rms / 32768)
```

**Reset logic**: Reset `_bot_first_speech_logged` and `_bot_audio_start_time` when a new bot audio stream starts (detected by gap > some threshold, or when `_bot_sample_rate` is reset).

**Expected log output**:
```
[NullAudioOutput] Bot audio initialized: sample_rate=24000, ...
[NullAudioOutput] Bot: Inserted 5753ms silence ...
[NullAudioOutput] Bot speech onset: T+6410ms (silent_padding=657ms, rms=-28.3dB)
```

This gives us three timestamps per turn:
1. **Bot audio initialized** - when first audio byte arrived (after silence insertion)
2. **Bot speech onset** - when RMS exceeded threshold (actual speech)
3. **Silero VAD** - from WAV analysis (for cross-validation)

#### 2. Optional: Create analysis script

- Create `scripts/analyze_bot_silence.py` to automate comparison of all metrics
- Update `docs/audio-alignment-analysis.md` with TTFB metrics explanation

---

## Investigation: RMS Discrepancy (In Progress)

### Problem Discovered

When comparing in-pipeline RMS detection to WAV file analysis, we found a major discrepancy:

| Metric | Turn 0 | Turn 1 | Turn 2 |
|--------|--------|--------|--------|
| Pipeline TTFB + RMS silent padding | 1185ms | 843ms | 697ms |
| WAV V2V (Silero) | 1216ms | 1312ms | 672ms |
| **Difference** | 31ms ✓ | **469ms ✗** | 25ms ✓ |

For Turn 1, the pipeline detected speech at T+43968ms with RMS=-16.5dB, but WAV analysis shows only ~-70dB at that position. Silero VAD detects speech at 44480ms (user ear confirmed ~44450ms).

### Root Cause: Confirmed via Source Code Analysis

After reading MediaSender (`pipecat/src/pipecat/transports/base_output.py`) and AudioBufferProcessor (`pipecat/src/pipecat/processors/audio/audio_buffer_processor.py`), the root cause is confirmed:

**Complete Audio Flow:**

```
OpenAI sends OutputAudioRawFrame
    ↓
NullAudioOutputTransport.process_frame()
    ↓
_emit_bot_with_silence_fill()
    ├─→ RMS check on ORIGINAL frame (timing point A)
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
                    ├─→ write_audio_frame() (our timing simulation)
                    └─→ push_frame() → our override counts samples (timing point B)
                            ↓
                        AudioBufferProcessor
                            ├─→ Resample again to recording rate
                            └─→ Append to _bot_audio_buffer → WAV file
```

**The Problem:**
- **RMS check** happens at timing point A, on the ORIGINAL frame from OpenAI
- **Sample counting** (`_bot_actual_samples`) happens at timing point B, AFTER MediaSender's buffering/chunking
- MediaSender buffers audio until it accumulates `_audio_chunk_size` bytes before pushing downstream
- This means `sample_pos` in our logs reflects what's been pushed so far, which may lag behind the current frame being checked

**MediaSender Chunking Logic** (lines 531-540 of base_output.py):
```python
self._audio_buffer.extend(resampled)
while len(self._audio_buffer) >= self._audio_chunk_size:
    chunk = cls(bytes(self._audio_buffer[: self._audio_chunk_size]), ...)
    await self._audio_queue.put(chunk)
    self._audio_buffer = self._audio_buffer[self._audio_chunk_size:]
```

This explains why `sample_pos` can be behind `T+elapsed_ms` - MediaSender hasn't pushed the current frame yet, it's still buffering

### Fix Applied

Updated logging to show **actual sample position** in addition to wall-clock time:

```python
# Now logs both:
sample_position_ms = (self._bot_actual_samples / self._recording_sample_rate) * 1000
logger.info(
    f"[NullAudioOutput] Bot speech onset: T+{elapsed_ms:.0f}ms "
    f"(sample_pos={sample_position_ms:.0f}ms, silent_padding={silent_padding_ms:.0f}ms, rms={rms_db:.1f}dB)"
)
```

This will tell us the exact WAV position where RMS exceeded threshold, allowing direct comparison with WAV analysis.

### Re-run Analysis Results (2026-01-09)

After adding `sample_pos` logging, ran 3-turn benchmark (`20260109T125021_gpt-realtime_39c2c061`):

**Bot audio arrival (from silence insertion logs):**
- Turn 0: 5990ms
- Turn 1: 41570ms
- Turn 2: 73790ms

**RMS triggered at -40dB (sample_pos):**
- Turn 0: 5992ms → +2ms after first audio
- Turn 1: 41727ms → +157ms after first audio
- Turn 2: 73912ms → +122ms after first audio

**Silero detected speech:**
- Turn 0: 6048ms → +58ms after first audio
- Turn 1: 42176ms → +606ms after first audio
- Turn 2: 74432ms → +642ms after first audio

| Turn | First Audio | RMS +Xms | Silero +Xms | RMS Padding | Actual Padding |
|------|-------------|----------|-------------|-------------|----------------|
| 0 | 5990ms | +2ms | +58ms | 2ms | 58ms |
| 1 | 41570ms | +157ms | +606ms | 157ms | 606ms |
| 2 | 73790ms | +122ms | +642ms | 122ms | 642ms |

**Key Finding:**
- `sample_pos` and `T+elapsed_ms` match within 15ms - MediaSender buffering doesn't cause significant lag
- The -40dB RMS threshold triggers 450-500ms before Silero detects actual speech
- OpenAI sends ~600ms of silent/low-level audio before actual speech (turns 1-2)
- Our RMS check catches audio ~120-160ms after it starts, but this is still ~450-500ms before speech
- The -40dB threshold is too sensitive for measuring "time to actual speech"

**Conclusion:**
To measure time-to-speech (not time-to-audio-stream), we need either:
1. Higher RMS threshold (e.g., -30dB or -25dB)
2. Use Silero VAD in pipeline (more complex, adds latency)
3. Accept that RMS measures "time to non-silent audio" not "time to speech"

### Next Steps

1. ✅ Update logging to show sample position
2. ✅ Read MediaSender and AudioBufferProcessor source to understand audio flow
3. ✅ Re-run benchmark and compare sample_pos to WAV analysis
4. ✅ Raise RMS threshold to -30dB and re-test
5. ✅ Run full 30-turn benchmark with -30dB threshold

---

## RMS Threshold Tuning (2026-01-09)

### Changed Threshold from -40dB to -30dB

Updated `BOT_SPEECH_RMS_THRESHOLD_DB` in `null_audio_output.py` from -40dB to -30dB.

### 3-Turn Test Results with -30dB

Run: `20260109T133250_gpt-realtime_2f15f7e9`

| Turn | First Audio | RMS (-30dB) | Silero | RMS→Silero diff |
|------|-------------|-------------|--------|-----------------|
| 0 | 7092ms | +40ms | +140ms | 100ms |
| 1 | 50150ms | +38ms | +90ms | 52ms |
| 2 | 78350ms | +45ms | +146ms | 101ms |

**Comparison to -40dB threshold:**

| Turn | -40dB diff | -30dB diff |
|------|------------|------------|
| 0 | 56ms | 100ms |
| 1 | 449ms | 52ms |
| 2 | 520ms | 101ms |

The -30dB threshold is **much closer to Silero** - only 50-100ms difference vs 450-520ms with -40dB.

---

## Full 30-Turn Benchmark Results (2026-01-09)

### Run Details

- **Run ID**: `20260109T134501_gpt-realtime_2da208db`
- **Duration**: 747.8s (12.5 minutes)
- **Turns**: 30 (29 matched by Silero - turns 16/17 merged)
- **RMS Threshold**: -30dB

### All Metrics Per Turn

```
Turn | Srvr TTFB | Pipe TTFB | WAV V2V | Silent Pad | User End  | Bot Arrival | RMS Onset | Silero Bot
-----|-----------|-----------|---------|------------|-----------|-------------|-----------|----------
   0 |       0ms |    1289ms |  1696ms |      105ms |    4768ms |      6057ms |    6162ms |     6464ms
   1 |     482ms |     756ms |  1216ms |      163ms |   41280ms |     42036ms |   42199ms |    42496ms
   2 |     459ms |     714ms |   960ms |       73ms |   69248ms |     69962ms |   70035ms |    70208ms
   3 |     554ms |     740ms |   768ms |        0ms |   92608ms |     93348ms |   93348ms |    93376ms
   4 |     443ms |     648ms |   992ms |       78ms |  119296ms |    119944ms |  120022ms |   120288ms
   5 |      26ms |     732ms |   896ms |       37ms |  160672ms |    161404ms |  161441ms |   161568ms
   6 |     506ms |     778ms |   864ms |       33ms |  179520ms |    180298ms |  180331ms |   180384ms
   7 |     940ms |    1257ms |  1888ms |      174ms |  201216ms |    202473ms |  202647ms |   203104ms
   8 |     595ms |     813ms |   928ms |       36ms |  237824ms |    238637ms |  238673ms |   238752ms
   9 |      17ms |     791ms |   800ms |        0ms |  278432ms |    279223ms |  279223ms |   279232ms
  10 |     691ms |    1035ms |  1184ms |       48ms |  304864ms |    305899ms |  305947ms |   306048ms
  11 |     654ms |    1662ms |  1760ms |       41ms |  320672ms |    322334ms |  322375ms |   322432ms
  12 |     605ms |     823ms |   928ms |       37ms |  341440ms |    342263ms |  342300ms |   342368ms
  13 |     758ms |    1022ms |  1184ms |       29ms |  357024ms |    358046ms |  358075ms |   358208ms
  14 |     779ms |     987ms |  1024ms |        0ms |  391136ms |    392123ms |  392123ms |   392160ms
  15 |     804ms |    1127ms |  1152ms |        0ms |  408800ms |    409927ms |  409927ms |   409952ms
  16 |     830ms |    1009ms |  1024ms |        0ms |  422720ms |    423729ms |  423729ms |   423744ms
  17 |     687ms |    1618ms |  1024ms |        0ms |  422720ms |    424338ms |  424338ms |   423744ms
  18 |     678ms |     939ms |  1120ms |       47ms |  440416ms |    441355ms |  441402ms |   441536ms
  19 |     696ms |     946ms |  1248ms |       83ms |  457408ms |    458354ms |  458437ms |   458656ms
  20 |     656ms |     923ms |  1280ms |       75ms |  480576ms |    481499ms |  481574ms |   481856ms
  21 |     790ms |     939ms |  1216ms |       85ms |  512672ms |    513611ms |  513696ms |   513888ms
  22 |     660ms |    1039ms |  1280ms |       61ms |  544352ms |    545391ms |  545452ms |   545632ms
  23 |    1207ms |     877ms |  1120ms |       78ms |  574624ms |    575501ms |  575579ms |   575744ms
  24 |     725ms |    1461ms |  1920ms |       77ms |  594048ms |    595509ms |  595586ms |   595968ms
  25 |     696ms |     963ms |  1536ms |      156ms |  632064ms |    633027ms |  633183ms |   633600ms
  26 |     663ms |     973ms |  1024ms |        0ms |  653920ms |    654893ms |  654893ms |   654944ms
  27 |     744ms |     950ms |  1056ms |       36ms |  670816ms |    671766ms |  671802ms |   671872ms
  28 |     661ms |    1049ms |  1088ms |        0ms |  692256ms |    693305ms |  693305ms |   693344ms
  29 |       0ms |     927ms |   960ms |        0ms |  725952ms |    726879ms |  726879ms |   726912ms
```

### Column Definitions

| Column | Description |
|--------|-------------|
| **Srvr TTFB** | From transcript.jsonl - OpenAI's measurement (speech_stopped → first audio delta) |
| **Pipe TTFB** | User speech end (Silero) → First bot audio byte arrival |
| **WAV V2V** | User speech end (Silero) → Bot speech start (Silero) - true voice-to-voice |
| **Silent Pad** | First bot audio byte → RMS -30dB threshold crossing |
| **User End** | Silero VAD detected user speech end (ms in WAV) |
| **Bot Arrival** | When first bot audio byte arrived (RMS Onset - Silent Pad) |
| **RMS Onset** | When RMS exceeded -30dB threshold (sample_pos from logs) |
| **Silero Bot** | When Silero VAD detected bot speech start (ms in WAV) |

### Summary Statistics

| Metric | Median | Mean | Range |
|--------|--------|------|-------|
| Server TTFB | 670ms | 643ms | 17-1207ms |
| Pipeline TTFB | 948ms | 993ms | 648-1662ms |
| WAV V2V | 1104ms | 1171ms | 768-1920ms |
| Silent Padding (RMS -30dB) | 39ms | 52ms | 0-174ms |

### Metric Relationships

```
WAV V2V - Pipeline TTFB = 156ms median (≈ Silero silent padding)
Pipeline TTFB - Server TTFB = 290ms median (VAD delay + network)
```

### RMS vs Silero Alignment (30 turns)

| Metric | Value |
|--------|-------|
| Mean diff (Silero - RMS) | 151ms |
| Median diff | 127ms |
| Min/Max | 9ms / 457ms |
| Within 100ms | 13/29 (45%) |
| Within 200ms | 21/29 (72%) |
| Within 500ms | 29/29 (100%) |

The -30dB RMS threshold triggers **127ms before Silero VAD on average** (median), which is a reasonable approximation for in-pipeline speech onset detection.

### Silent Padding Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0ms (instant) | 9 | 30% |
| 1-50ms | 9 | 30% |
| 51-100ms | 8 | 27% |
| >100ms | 4 | 13% |

**Key insight**: 30% of turns have instant speech (no silent padding), while 70% have some padding (median 39ms with -30dB threshold).

---

## Operational Notes

### NEVER use `tee` when running benchmarks

When starting benchmark runs, **do not** pipe output through `tee`:

```bash
# BAD - fills up context with streaming logs
uv run python -m multi_turn_eval ... 2>&1 | tee run.log

# GOOD - run in background, check periodically
uv run python -m multi_turn_eval ... > run.log 2>&1 &
# Then check: tail -20 run.log
# Or: ps aux | grep multi_turn_eval
```

The log files are very verbose and will consume all available context if streamed. Instead:
1. Run the benchmark in background or separate terminal
2. Periodically check if the process is still running
3. Read the log file after completion

### Model name mapping

Use `gpt-realtime` (not `gpt-4o-realtime-preview`) for OpenAI Realtime benchmarks. See README.md for model/service mappings.

---

## How to Replicate This Testing

### Step 1: Run a Benchmark

```bash
# Quick 3-turn test
uv run multi-turn-eval run aiwf_medium_context \
  --model gpt-realtime \
  --service openai-realtime \
  --pipeline realtime \
  --only-turns 0,1,2 \
  > /tmp/benchmark.log 2>&1 &

# Full 30-turn benchmark
uv run multi-turn-eval run aiwf_medium_context \
  --model gpt-realtime \
  --service openai-realtime \
  --pipeline realtime \
  > /tmp/benchmark.log 2>&1 &

# Check progress
ps aux | grep multi-turn-eval
tail -20 /tmp/benchmark.log
grep "Recorded turn" /tmp/benchmark.log | tail -5
```

### Step 2: Find the Run Directory

```bash
# From end of log
tail -5 /tmp/benchmark.log
# Look for: Transcript: runs/aiwf_medium_context/<run_id>/transcript.jsonl

# Set variable for convenience
RUN_DIR="runs/aiwf_medium_context/<run_id>"
```

### Step 3: Extract RMS Speech Onset Data

```bash
# Get bot speech onset events (RMS-based)
grep "Bot speech onset" /tmp/benchmark.log

# Output format:
# [NullAudioOutput] Bot speech onset: T+6162ms (sample_pos=6162ms, silent_padding=105ms, rms=-29.5dB)
```

### Step 4: Extract Bot Audio Arrival Times

```bash
# Get silence insertion events (marks when first bot audio byte arrived)
grep "Bot: Inserted" /tmp/benchmark.log

# Output format:
# [NullAudioOutput] Bot: Inserted 6057ms silence (145368 samples): elapsed=6.06s, ...
```

### Step 5: Extract Server TTFB

```bash
# From transcript.jsonl
uv run python -c "
import json
with open('$RUN_DIR/transcript.jsonl') as f:
    for i, line in enumerate(f):
        if line.strip():
            t = json.loads(line)
            print(f'Turn {i}: {t.get(\"ttfb_ms\", 0)}ms')
"
```

### Step 6: Run Silero VAD Analysis

```python
# Save as analyze_silero.py or run inline
import numpy as np
import wave
import torch
from scipy import signal

RUN_DIR = "runs/aiwf_medium_context/<run_id>"

# Load audio
with wave.open(f'{RUN_DIR}/conversation.wav', 'rb') as wf:
    sr = wf.getframerate()
    n_frames = wf.getnframes()
    audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).reshape(-1, 2)

print(f'WAV: {n_frames / sr:.1f}s at {sr}Hz')

# Get channels (left=user, right=bot)
user_audio = audio[:, 0].astype(np.float32) / 32768.0
bot_audio = audio[:, 1].astype(np.float32) / 32768.0

# Resample to 16kHz for Silero
target_sr = 16000
user_16k = signal.resample(user_audio, int(len(user_audio) * target_sr / sr))
bot_16k = signal.resample(bot_audio, int(len(bot_audio) * target_sr / sr))

# Load Silero VAD
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=True)
(get_speech_timestamps, _, _, _, _) = utils

# Run VAD on both channels
user_segments = get_speech_timestamps(
    torch.from_numpy(user_16k.astype(np.float32)),
    model, sampling_rate=target_sr,
    min_silence_duration_ms=2000, speech_pad_ms=0
)
bot_segments = get_speech_timestamps(
    torch.from_numpy(bot_16k.astype(np.float32)),
    model, sampling_rate=target_sr,
    min_silence_duration_ms=2000, speech_pad_ms=0
)

print(f'\nUser segments: {len(user_segments)}')
for i, seg in enumerate(user_segments):
    start_ms = seg['start'] / target_sr * 1000
    end_ms = seg['end'] / target_sr * 1000
    print(f'  {i}: {start_ms:.0f}ms - {end_ms:.0f}ms')

print(f'\nBot segments: {len(bot_segments)}')
for i, seg in enumerate(bot_segments):
    start_ms = seg['start'] / target_sr * 1000
    end_ms = seg['end'] / target_sr * 1000
    print(f'  {i}: {start_ms:.0f}ms - {end_ms:.0f}ms')
```

### Step 7: Calculate All Metrics Per Turn

```python
# Combine all data sources into per-turn metrics

import json
import statistics

# 1. Load Server TTFB from transcript
with open(f'{RUN_DIR}/transcript.jsonl') as f:
    transcript = [json.loads(line) for line in f if line.strip()]
server_ttfb = [t.get('ttfb_ms', 0) or 0 for t in transcript]

# 2. RMS sample_pos values (from grep "Bot speech onset" output)
# Extract manually or parse from log
rms_sample_pos = [6162, 42199, ...]  # Fill from logs

# 3. Silent padding values (from grep "Bot speech onset" output)
silent_padding = [105, 163, ...]  # Fill from logs

# 4. Bot audio arrival = RMS onset - Silent padding
bot_audio_arrival = [rms - pad for rms, pad in zip(rms_sample_pos, silent_padding)]

# 5. Silero VAD results (from Step 6)
user_ends = [4768, 41280, ...]  # User segment end times
bot_starts = [6464, 42496, ...]  # Bot segment start times

# 6. Calculate derived metrics
for i in range(len(rms_sample_pos)):
    user_end = user_ends[i] if i < len(user_ends) else None
    silero_bot = bot_starts[i] if i < len(bot_starts) else None

    # Pipeline TTFB = Bot audio arrival - User speech end
    pipe_ttfb = bot_audio_arrival[i] - user_end if user_end else None

    # WAV V2V = Silero bot start - Silero user end
    wav_v2v = silero_bot - user_end if silero_bot and user_end else None

    print(f'Turn {i}: Server={server_ttfb[i]}ms, Pipe={pipe_ttfb}ms, V2V={wav_v2v}ms, Pad={silent_padding[i]}ms')
```

### Step 8: Verify Metric Relationships

Expected relationships to check:
```
WAV V2V ≈ Pipeline TTFB + Silero silent padding
Pipeline TTFB ≈ Server TTFB + 290ms (VAD delay + network)
RMS onset ≈ Silero bot start - 127ms (with -30dB threshold)
```

### Key Files Modified for This Testing

| File | Change |
|------|--------|
| `src/multi_turn_eval/transports/null_audio_output.py` | Added RMS-based speech onset detection with -30dB threshold |
| `docs/audio-alignment-analysis.md` | Added documentation on metrics and analysis procedures |

### Current RMS Threshold Setting

```python
# In src/multi_turn_eval/transports/null_audio_output.py
BOT_SPEECH_RMS_THRESHOLD_DB = -30.0  # Changed from -40.0
```

### Log Output Format Reference

```
# Bot audio arrival (silence insertion)
[NullAudioOutput] Bot: Inserted 6057ms silence (145368 samples): elapsed=6.06s, expected=145368, actual_before=0

# Bot speech onset (RMS detection)
[NullAudioOutput] Bot speech onset: T+6162ms (sample_pos=6162ms, silent_padding=105ms, rms=-29.5dB)

# Recording summary
[NullAudioOutput] Bot recording summary: actual_samples=17735979 (739.0s), silence_inserted=5036139 (209.8s), silence_frames=30
```

---

## Audio Tags for Turn Alignment Verification (2026-01-09)

### Purpose

Add short audio "tags" (2kHz sine bursts) mixed into the first frame of each turn to:
1. **Verify alignment** - Confirm silence insertion places audio at correct WAV positions
2. **Measure silent padding** - Tag marks first audio byte; gap to Silero speech = silent padding
3. **Visual debugging** - Tags visible as waveform spikes in audio editors
4. **Automated detection** - FFT-based detection in analysis scripts

### Implementation

**File**: `src/multi_turn_eval/transports/null_audio_output.py`

**Constants added**:
```python
AUDIO_TAG_FREQUENCY_HZ = 2000  # 2kHz sine wave - above typical speech fundamentals
AUDIO_TAG_DURATION_MS = 15    # 15ms duration
AUDIO_TAG_AMPLITUDE_DB = -12  # -12dB relative to full scale
```

**Methods added**:
- `_generate_audio_tag(sample_rate)` - Creates 2kHz sine burst with fade-in/out
- `_mix_tag_into_frame(frame_audio, tag)` - Mixes tag into audio without clipping

**Injection points**:
- `_emit_bot_with_silence_fill()` - Tags first frame after silence insertion (bot turn start)
- `_emit_user_with_silence_fill()` - Tags first frame after silence insertion (user turn start)

**Log output**:
```
[NullAudioOutput] Bot turn tag: sample_pos=6113ms, freq=2000Hz, duration=15ms
[NullAudioOutput] User turn tag: sample_pos=5261ms, freq=2000Hz, duration=15ms
```

### Detection Script

**File**: `scripts/detect_audio_tags.py`

Uses FFT to find 2kHz energy bursts in WAV files:
```bash
uv run python scripts/detect_audio_tags.py runs/<run_id>/conversation.wav --threshold 20
```

**Note**: Use threshold ≥20 to reduce false positives from speech harmonics around 2kHz.

### Key Finding: Tags Are NOT Detected as Speech by Silero

The 15ms tag duration is below Silero's `min_speech_duration_ms=900`, so tags are filtered out.
This allows us to measure the gap between tag position and Silero speech detection.

---

## Test Run with Audio Tags (2026-01-09)

### Run Details

- **Run ID**: `20260109T160948_gpt-realtime_8cfb7911`
- **Turns**: 3
- **Tags**: Injected on both user and bot tracks

### Comprehensive Metrics Comparison

```
Turn   Pipeline TTFB   WAV V2V    Diff       Server TTFB   Bot Tag     Silent Pad
----   -------------   -------    ----       -----------   -------     ----------
  0         1433ms      1632ms    +199ms           0ms      6113ms        287ms
  1         1094ms      1504ms    +410ms         724ms     45155ms        573ms
  2         1028ms      1280ms    +252ms         633ms     71448ms        424ms
```

### Metric Sources

| Metric | Source | Calculation |
|--------|--------|-------------|
| **Pipeline TTFB** | Log timestamps | `BotStartedSpeakingFrame - UserStoppedSpeaking(actual_end)` |
| **WAV V2V** | Silero on WAV | `Silero(bot_start) - Silero(user_end)` |
| **Server TTFB** | transcript.jsonl | OpenAI's reported `ttfb_ms` |
| **Bot Tag** | Pipeline logs | `sample_pos` when tag was mixed |
| **Silent Pad** | Tag + Silero | `Silero(bot_start) - Bot_Tag_Position` |

### Detailed Timestamps

```
Turn   User End (VAD)   User End (Silero)   Bot Tag    Bot Frame    Bot Speech (Silero)
----   --------------   -----------------   -------    ---------    -------------------
  0          4722ms            4768ms         6113ms      6155ms            6400ms
  1         44102ms           44224ms        45155ms     45196ms           45728ms
  2         70462ms           70560ms        71448ms     71490ms           71872ms
```

### Component Breakdown

The difference between Pipeline TTFB and WAV V2V is explained by two factors:

```
Turn   User Offset    Bot Silent Pad   Net Diff    Observed Diff
       (Sil-VAD)      (Sil-Tag)        (Bot-User)  (V2V-PTTFB)
----   -----------    --------------   --------    -------------
  0        +46ms          +287ms         +241ms        +199ms
  1       +122ms          +573ms         +451ms        +410ms
  2        +98ms          +424ms         +326ms        +252ms
```

**Note**: The ~40ms discrepancy between calculated and observed is due to the difference
between `BotStartedSpeakingFrame` timestamp and the actual tag position (~42ms).

### Silent Padding Verification (Sanity Check)

Tags allow direct measurement of silent padding in the WAV file:

```
Turn   Tag Position   Silero Bot Start   Silent Padding
----   ------------   ----------------   --------------
  0        6113ms           6400ms            287ms
  1       45155ms          45728ms            573ms
  2       71448ms          71872ms            424ms
```

This confirms OpenAI sends 287-573ms of silent audio before actual speech begins.

---

## Known Issues

### BUG: User Speech End Detection Discrepancy (46-122ms)

**Problem**: Silero VAD (on WAV) detects user speech ending 46-122ms LATER than pipeline VAD:

```
Turn 0: Pipeline=4722ms, Silero=4768ms, diff=+46ms
Turn 1: Pipeline=44102ms, Silero=44224ms, diff=+122ms
Turn 2: Pipeline=70462ms, Silero=70560ms, diff=+98ms
```

**Expected**: Both should use the same Silero model (ONNX) and produce results within ~30ms.

**Possible causes**:
1. Different VAD parameters between pipeline and analysis script
2. Different audio being analyzed (real-time stream vs recorded WAV)
3. Incorrect offset compensation in one of the calculations
4. Resampling differences (pipeline may use different sample rates)

**Pipeline VAD config** (from logs):
```
start_secs=0.2, stop_secs=0.8, confidence=0.7
```

**Analysis script Silero config**:
```python
min_silence_duration_ms=2000, min_speech_duration_ms=900, threshold=0.7, speech_pad_ms=0
```

**Note**: Pipeline logs show "actual end" which already subtracts the 800ms `stop_secs` offset.
The discrepancy should be investigated to ensure consistent VAD behavior.

---

## Next Tasks

### 1. Verify User/Bot Track Alignment in WAV File ✅ COMPLETED

**Goal**: Confirm that both tracks in `conversation.wav` are perfectly time-aligned.

**Method**:
1. Extract tag positions from pipeline logs (`sample_pos` values)
2. Detect tags in the WAV file using `scripts/detect_audio_tags.py`
3. Compare log positions to WAV detection positions
4. Correlation should be within the 20ms detection window resolution

**Expected result**: Tag positions from logs should match WAV detection within ~15-20ms.

**If mismatch found**: Indicates a bug in silence insertion or sample counting.

#### Verification Results (2026-01-09)

**Run**: `20260109T160948_gpt-realtime_8cfb7911` (3 turns with audio tags)

**From Pipeline Logs (`sample_pos`):**
```
User Turn 0: 5261ms
User Turn 1: 7894ms
User Turn 2: 66133ms
Bot Turn 0: 6113ms
Bot Turn 1: 45155ms
Bot Turn 2: 71448ms
```

**From WAV Detection (`scripts/detect_audio_tags.py --threshold 20`):**
```
User tags: 5245ms, 7880ms, 66120ms
Bot tags: 6100ms, 45140ms, 71435ms
```

**Alignment Comparison:**

| Channel | Turn | Log Position | WAV Detection | Difference |
|---------|------|--------------|---------------|------------|
| User | 0 | 5261ms | 5245ms | +16ms |
| User | 1 | 7894ms | 7880ms | +14ms |
| User | 2 | 66133ms | 66120ms | +13ms |
| Bot | 0 | 6113ms | 6100ms | +13ms |
| Bot | 1 | 45155ms | 45140ms | +15ms |
| Bot | 2 | 71448ms | 71435ms | +13ms |

**Result**: ✅ All differences are 13-16ms, well within the expected ≤15-20ms tolerance.

**Conclusion**: User and bot tracks in `conversation.wav` are properly time-aligned. The silence insertion and sample counting logic is working correctly.

#### Script Fix: False Positive Filtering

Initial detection found spurious tags at 16790-16905ms on the bot channel. Investigation showed:
- Genuine tags: RMS = -20 to -22 dB
- False positives: RMS = -57 to -58 dB

Added `--min-level` parameter (default -40dB) to `scripts/detect_audio_tags.py` to filter out low-level noise that may have 2kHz harmonics. This eliminated the false positives while preserving all genuine tag detections.

### 2. Verify Bot Audio Leading Silence Measurements ✅ COMPLETED

**Goal**: Confirm that two independent calculations of silent padding agree.

**Method 1 - Pipeline logs**:
```
Silent padding = Bot speech onset (RMS sample_pos) - Bot tag (sample_pos)
```
From logs: `sample_pos` in "Bot speech onset" and "Bot turn tag" entries.

**Method 2 - WAV file analysis**:
```
Silent padding = Silero(bot_start) - Tag_position
```
Tag marks first audio byte; Silero detects actual speech.

**Expected result**: Both methods should agree within ~50-100ms (accounting for RMS vs Silero
detection differences at -30dB vs neural network).

#### Bug Fix: Tag Window Skip

The original RMS detection showed 0ms silent padding because the -12dB audio tag exceeded the -30dB
RMS threshold, triggering immediately. Fixed in `null_audio_output.py` by:

1. Added `_bot_skip_rms_samples` counter
2. Set to tag duration (15ms worth of samples) when tag is injected
3. Skip RMS check while counter > 0, decrement as frames are processed

This allows RMS to measure the actual silent audio after the tag.

#### Verification Results (2026-01-09)

**Run**: `20260109T165417_gpt-realtime_96d2e213` (3 turns with tag window skip fix)

**Pipeline-based (RMS -30dB threshold):**

| Turn | Tag sample_pos | RMS sample_pos | RMS Silent Pad |
|------|----------------|----------------|----------------|
| 0 | 5609ms | 5689ms | 80ms |
| 1 | 48907ms | 48987ms | 80ms |
| 2 | 77557ms | 77637ms | 80ms |

**WAV-based (Silero VAD):**

| Turn | Tag (WAV) | Silero bot_start | Silero Silent Pad |
|------|-----------|------------------|-------------------|
| 0 | 5595ms | 5792ms | 197ms |
| 1 | 48895ms | 48960ms | 65ms |
| 2 | 77545ms | 77600ms | 55ms |

**Comparison:**

| Turn | RMS (pipeline) | Silero (WAV) | Difference |
|------|----------------|--------------|------------|
| 0 | 80ms | 197ms | 117ms (Silero later) |
| 1 | 80ms | 65ms | 15ms (RMS later) |
| 2 | 80ms | 55ms | 25ms (RMS later) |

**Results**:
- Turns 1-2: ✅ Excellent agreement (within 25ms)
- Turn 0: ⚠️ 117ms difference - slightly outside expected ~100ms tolerance

**Conclusion**: The two methods produce comparable results. The RMS threshold (-30dB) and Silero
neural network have different detection characteristics:
- RMS triggers on any audio above -30dB (catches quiet preamble sounds)
- Silero triggers on actual speech patterns (may miss quiet initial phonemes or catch later)

The ~15-117ms difference is acceptable for silent padding measurement. For precise measurements,
use the WAV-based Silero method as the reference.

#### Full 30-Turn Verification (2026-01-09)

**Run**: `20260109T190150_gpt-realtime_d4d4626a` (727s, 29 bot turns matched)

**Task 1: Track Alignment**

| Track | Tags | Min Diff | Max Diff | Mean Diff |
|-------|------|----------|----------|-----------|
| Bot | 29 | 12ms | 16ms | 13.6ms |
| User | 16 | 6ms | 16ms | 13.1ms |

✅ All 45 tags aligned within 16ms (well under 20ms tolerance).

Note: 1 false positive detected at 263470ms with rms=-38.4dB (just above -40dB threshold).

**Task 2: Silent Padding Comparison**

| Metric | RMS (-30dB) | Silero VAD |
|--------|-------------|------------|
| Median | 80ms | 169ms |
| Mean | 90ms | 240ms |
| Range | 40-200ms | 28-687ms |

**Difference Distribution:**
- Within 50ms: 6/29 (21%)
- Within 100ms: 14/29 (48%)
- Within 150ms: 17/29 (59%)

**Analysis**: RMS triggers 106ms earlier than Silero on median. This is expected because:
- RMS (-30dB) detects any audio above threshold, including quiet preamble sounds
- Silero detects actual speech patterns, triggering later on turns with quiet intros
- High Silero values (300-687ms) indicate OpenAI sends significant non-speech audio before speaking

**Conclusion**: Both methods work as designed. Use RMS for "time to audio stream start" and
Silero for "time to actual speech". The larger variability in Silero (28-687ms) reflects
natural variation in how OpenAI's TTS model starts responses.

### 3. Investigate User Speech End Detection Discrepancy

**Goal**: Reduce the 46-122ms discrepancy between pipeline VAD and Silero to ≤30ms.

**Investigation steps**:
1. Compare exact Silero parameters used in pipeline vs analysis script
2. Check if audio resampling differs between paths
3. Verify `stop_secs` offset is being applied correctly
4. Consider if real-time vs batch processing affects detection boundaries
5. Test with identical parameters on both paths

**Acceptance criteria**: User end time difference should be ≤30ms across all turns.
