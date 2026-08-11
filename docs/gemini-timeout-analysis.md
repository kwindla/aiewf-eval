# Gemini Live Timeout Analysis

Investigating why some runs have 0 timeouts, some have 1, and a few have 3.

## Root Cause: Variable Response Times

The 10-minute session limit is **fixed** and always triggers at exactly 10 minutes from connection. What varies is **Gemini's response time**, which differs significantly between runs.

### Response Time vs Timeout Count

| Run ID | Mean Latency/Turn | Total Latency | Timeouts | Time to First Event |
|--------|-------------------|---------------|----------|---------------------|
| b61c8b50 | 19.5s | 565s (9.4 min) | 0 | 574s → **completed** |
| 8c1822a8 | 23.3s | 652s (10.9 min) | 1 | 600s → timeout |
| a5cb91b4 | 25.7s | 720s (12.0 min) | 1 | 600s → timeout |
| 7c32cd99 | 33.1s | 893s (14.9 min) | 1 | 600s → timeout |
| 589beb33 | 31.8s | 892s (14.9 min) | 1 | 600s → timeout |
| bd81f362 | 32.9s | 889s (14.8 min) | 2 | 600s → timeout |
| 56cef1c1 | 32.5s | 910s (15.2 min) | 0* | *policy errors instead |
| 7df4fd84 | 49.6s | 1339s (22.3 min) | 3 | 600s → timeout |
| 58b6e52f | 53.3s | 1438s (24.0 min) | 1** | 600s → timeout |
| 756c3027 | 53.9s | 1402s (23.4 min) | 3 | 600s → timeout |

**Key insight**: Only b61c8b50 completed before the 10-minute mark (574 seconds). Every other run hit at least one timeout.

### Response Time Variability

Mean latency per turn ranges from **19.5s to 53.9s** - a 2.8x difference between fastest and slowest runs. This variability is entirely due to Gemini's response generation speed, which varies significantly:

- **Fast runs** (~20s/turn): Complete 30 turns in ~10 min, avoiding timeouts
- **Medium runs** (~30s/turn): Complete in ~15 min, hitting 1 timeout
- **Slow runs** (~50s/turn): Complete in ~25 min, hitting 2-3 timeouts

### Why Does Response Time Vary?

Possible factors (not verified):
1. **Server load** - Gemini may be slower during peak times
2. **Response complexity** - Some turns may generate longer responses
3. **Network latency variation** - Connection quality affects streaming
4. **Model behavior randomness** - Different generation paths

## Failure Modes Discovered

### 1. Session Timeout (10-minute limit)

Gemini Live has a hard 10-minute session limit. When reached:
- Connection closes with error code 1011
- pipecat automatically reconnects
- Current turn is retried

### 2. Empty Response (control tokens only)

Discovered in run 756c3027, turn 11. The model responded with TTS frames but only control tokens (`<ctrl46><ctrl46><ctrl46><ctrl46>`) - no actual audio.

**Symptoms:**
- TTSStartedFrame received
- TTSTextFrame with only control tokens
- TTSStoppedFrame with completion_tokens=0
- No TTSAudioRawFrame generated
- BotStoppedSpeakingFrame never fires (no audio to play)
- Conversation stalls for 6+ minutes until session timeout

**Timeline from run 756c3027:**
```
18:57:25.132 - UserStoppedSpeakingFrame sent
18:57:25.952 - TTSStartedFrame received
18:57:25.953 - TTSTextFrame: <ctrl46><ctrl46><ctrl46><ctrl46>
18:57:28.165 - TTSStoppedFrame, completion_tokens=0
... (5.5 minutes of silence) ...
19:03:09.196 - Connection error (1011) - 10-minute timeout
```

### 3. No Response (model never responds)

Discovered in run 16c34b4b, turn 15. The model never responded at all after user audio.

**Symptoms:**
- UserStoppedSpeakingFrame sent
- No TTSStartedFrame ever received
- Complete silence until session timeout

**Timeline from run 16c34b4b:**
```
21:47:51.912 - [USER_AUDIO_QUEUED] turn=15
21:47:54.968 - [VAD] UserStoppedSpeaking
21:47:55.327 - Audio finished sending
... (5 minutes of complete silence - no TTS response) ...
21:52:56.193 - Connection error (1011) - 10-minute timeout
```

## Implemented Fixes

### Empty Response Detection (immediate)

Added to `TurnGate` class in `realtime.py`:
- Track TTS state (TTSStartedFrame/TTSStoppedFrame)
- On TTSStoppedFrame, check if bot never started speaking
- If transcript contains only control tokens → trigger retry immediately

**Log markers:**
- `[EMPTY_RESPONSE] No bot audio generated, transcript='<ctrl46>...'` (from TurnGate)
- `[EMPTY_RESPONSE] turn=X retry_count=Y` (from RealtimePipeline)

### No Response Detection (15s timeout)

Added to `TurnGate` class in `realtime.py`:
- Start 15-second timer on VADUserStoppedSpeakingFrame
- If no TTSStartedFrame arrives within 15 seconds → trigger retry

**Log markers:**
- `[NO_RESPONSE] No TTS response after 15.0s` (from TurnGate)
- `[NO_RESPONSE] turn=X retry_count=Y` (from RealtimePipeline)

### Reconnection Detection (mid-turn only)

Session timeouts that force reconnection mid-turn are flagged as turn-taking failures.
Turn 0 reconnections are excluded (expected initial connection setup).

**Log markers:**
- `Gemini reconnected: scheduling turn X retry` (from RealtimePipeline)

### Unified Retry Logic

Both empty response and no response use the same retry mechanism:
1. Clear TurnGate state
2. Wait 2 seconds
3. Re-queue the same audio file
4. Increment retry count (max 3 retries)

### V2V Metrics for Retried Turns

For turns with retries, V2V is calculated from the **first** user audio end, not the retry:

1. Log `[USER_AUDIO_QUEUED] turn=X predicted_end=Y duration=Z` when audio queued
2. Parse recording baseline monotonic time from log
3. Convert `predicted_end` to WAV milliseconds
4. Calculate: `wav_v2v_ms = bot_silero_start_ms - first_user_end_wav_ms`

This captures the **total user-perceived latency** including failed attempts.

### Judge Integration

Updated `turn_taking.py` to detect these as turn-taking failures:
- `empty_response (N retries)` - model returned control tokens only
- `no_response (N retries)` - model never responded

## Timeout Mechanics

### Connection Timeline

```
0:00  - Connected to Gemini service
0:00  - 10-minute timer starts
...
10:00 - Session timeout (1011 error)
10:00 - Automatic reconnection
10:00 - New 10-minute timer starts
...
20:00 - Session timeout (1011 error)
...
```

### TTFB Impact on Metrics

When a timeout occurs during a turn, that turn's TTFB includes the reconnection time:

| Normal TTFB | ~1200-1400ms |
| Reconnection TTFB | 76,000 - 529,000ms (1.3 - 8.8 min) |

The reconnection time varies based on when in the turn the timeout occurred.

## Per-Run Details

### b61c8b50 (0 timeouts - the only clean run)
- Connected: 18:53:19
- Turn 29 recorded: 19:02:53
- Duration: **574 seconds (9.6 min)**
- Finished 26 seconds before timeout would have hit
- Mean latency: 19.5s/turn (fastest)

### 756c3027 (3 timeouts)
- Connected: 18:53:09
- Timeout 1: 19:03:09 (during turn 11)
- Timeout 2: 19:13:09 (during turn 15)
- Timeout 3: 19:23:08 (during turn 29)
- Mean latency: 53.9s/turn (slowest)

### 56cef1c1 (0 session timeouts, 10 policy errors)
- Hit policy violation errors instead of session timeouts
- Rapid reconnection failures every ~12 seconds
- Eventually recovered and completed all 30 turns

## Files Modified

| File | Changes |
|------|---------|
| `src/multi_turn_eval/pipelines/realtime.py` | TurnGate: TTS state tracking, empty/no response detection, timeouts. RealtimePipeline: retry logic, first user end tracking. |
| `scripts/analyze_turn_metrics.py` | Parse retry events, first user end times, recording baseline. Calculate adjusted V2V for retried turns. |
| `src/multi_turn_eval/judging/turn_taking.py` | Detect empty_response and no_response as turn failures. |

## Conclusion

The timeout behavior is **consistent** - all runs get exactly 10 minutes before timeout. The **inconsistency** is in Gemini's response times, which vary 2-3x between runs.

Only 1 in 10 runs completed fast enough to avoid any timeouts. The rest hit 1-3 timeouts depending on how slow Gemini's responses were.

With the implemented fixes:
- Empty responses are detected immediately when TTSStoppedFrame arrives (not 6+ minutes)
- No responses are detected within 15 seconds (not 5+ minutes)
- Mid-turn reconnections are flagged as turn-taking failures (turn 0 excluded)
- Retries happen automatically (up to 3 times)
- V2V metrics correctly reflect total user-perceived latency
- Judge flags these as turn-taking failures
