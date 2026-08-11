# Gemini Live Missing `turn_complete` Analysis

Date: 2026-03-26

This note analyzes a Gemini Live failure mode where a turn finished speaking locally but never emitted `TTSStoppedFrame` or `LLMFullResponseEndFrame`, causing the benchmark turn to hang until a later reconnect.

## Summary

The clearest failing example is turn 14 in:

- `runs/aiwf_medium_context/20260326_180617_gemini-3.1-flash-live-preview_high_x10_aborted_pre_botstop_flush_fix/20260326T192234_gemini-3.1-flash-live-preview_f6f35380`

What happened:

1. Gemini issued `submit_dietary_request` and the tool result was returned.
2. Gemini then streamed spoken audio/text for the follow-up answer.
3. Local output finished draining audio and emitted `BotStoppedSpeakingFrame`.
4. Pipecat never emitted `TTSStoppedFrame` or `LLMFullResponseEndFrame`.
5. The pipeline stalled until the session later reconnected and retried the turn.

The 10-minute reconnect was not the root cause. It was the recovery path after the turn had already become stuck.

## Log Evidence

### Bad Turn

Relevant lines from:

- `runs/aiwf_medium_context/20260326_180617_gemini-3.1-flash-live-preview_high_x10_aborted_pre_botstop_flush_fix/20260326T192234_gemini-3.1-flash-live-preview_f6f35380/run.log`

Key events:

- Tool call starts: `run.log:17377`
- Tool result recorded: `run.log:17384`
- Gemini continues streaming `TTSAudioRawFrame`, `LLMTextFrame`, and `TTSTextFrame` chunks after the tool result
- Local output transport logs `Bot stopped speaking`: `run.log:17853`
- Turn gate receives `BotStoppedSpeakingFrame`: `run.log:17854`

What is missing in that slice:

- No `TTSStoppedFrame`
- No `LLMFullResponseEndFrame`
- No transcript flush driven by those end-of-response frames
- No turn advancement

So the audio finished locally, but the LLM service never signaled logical response completion.

### Healthy Retry of the Same Step

Later in the same run, the retried version of that step completed normally:

- Tool call starts: `run.log:34829`
- Tool result recorded: `run.log:34836`
- `MetricsFrame`: `run.log:35380`
- `TTSStoppedFrame`: `run.log:35381`
- `LLMFullResponseEndFrame`: `run.log:35382`
- Transcript flush and turn end follow immediately after: `run.log:35386` through `run.log:35425`

That healthy retry shows the normal shape Pipecat expects from Gemini Live.

## Pipecat Core Control Flow

Relevant file:

- `pipecat/src/pipecat/services/google/gemini_live/llm.py`

### Receive Loop

The Gemini Live receive loop processes incoming server messages here:

- `llm.py:1356-1405`

Important behavior:

- `server_content.model_turn` is handled by `_handle_msg_model_turn()`
- `server_content.output_transcription` is handled by `_handle_msg_output_transcription()`
- `server_content.turn_complete` is handled by `_handle_msg_turn_complete()`
- `message.tool_call` is handled separately by `_handle_msg_tool_call()`

The receive loop assumes content-bearing fields and `turn_complete` may appear in different combinations on the same server message, but the close of the response still depends on `turn_complete`.

### Response Start Paths

`_handle_msg_model_turn()`:

- starts audio-mode response state
- emits `TTSStartedFrame`
- emits `LLMFullResponseStartFrame`
- pushes `TTSAudioRawFrame`

Reference:

- `llm.py:1644-1720`

`_handle_msg_output_transcription()`:

- handles output transcript text for audio mode
- can also start response state if transcription arrives before audio
- emits `TTSStartedFrame`
- emits `LLMFullResponseStartFrame`

Reference:

- `llm.py:1882-1915`

### Response End Path

`_handle_msg_turn_complete()` is the only normal close path:

- clears buffers
- ends `_bot_is_responding`
- emits `TTSStoppedFrame` in audio mode
- emits `LLMFullResponseEndFrame`

Reference:

- `llm.py:1747-1775`

There is no alternate non-interruption path that emits those frames.

### `BotStoppedSpeakingFrame` Is Explicitly Ignored

The Gemini Live service ignores downstream `BotStoppedSpeakingFrame`:

- `llm.py:1061-1063`

That means local output can know speech has ended, but the Gemini Live service will not use that signal to close the logical response.

## Output Transport Behavior

Relevant file:

- `pipecat/src/pipecat/transports/base_output.py`

The output transport emits `BotStoppedSpeakingFrame` when audio playback drains:

- local bot-stop event and frame creation: `base_output.py:652-673`

For TTS audio specifically:

- `TTSStoppedFrame` also triggers bot-stop handling if TTS audio was received: `base_output.py:721-726`

In the failing turn, we saw the plain local `Bot stopped speaking` path fire, which means the audio queue drained. The missing piece was the Gemini Live service's logical end-of-response signal.

## What We Can and Cannot Prove From the Logs

What we can say with high confidence:

- Gemini produced tool-call output and spoken response content.
- Local output finished speaking.
- Pipecat never emitted the end-of-response frames.
- In core, those frames only come from `_handle_msg_turn_complete()`.

What we cannot prove from current logging:

- The raw incoming Gemini websocket payload is not logged in the run output, so we cannot directly show a server message that lacked `server_content.turn_complete`.

So the conclusion is inferential, but the inference is strong:

- either Gemini Live sometimes omits `server_content.turn_complete`
- or Pipecat sometimes fails to receive/process that message

Either way, Pipecat core currently has no robust fallback for this response shape.

## Conclusion

This does not look like the previously documented 10-minute Gemini session timeout behavior.

The failure happens earlier:

- the response audio completes
- `BotStoppedSpeakingFrame` fires locally
- but no logical response-end signal is emitted

The most likely root cause is inconsistent Gemini Live response shape for this turn, specifically missing or unobserved `server_content.turn_complete`.

Pipecat core is also too brittle because it assumes `turn_complete` will always arrive to close the response.

## Recommended Pipecat Core Fix

The right long-term fix should be in Pipecat core, not only in the eval harness.

Suggested approach:

1. Keep `server_content.turn_complete` as the primary authoritative end-of-response signal.
2. Add a guarded fallback for audio mode when:
   - `_bot_is_responding` is still `True`
   - local downstream `BotStoppedSpeakingFrame` occurs
   - no new Gemini content arrives within a short grace timeout
   - no real `turn_complete` arrives during that window
3. On that timeout, synthesize:
   - `TTSStoppedFrame`
   - `LLMFullResponseEndFrame`
4. Clear response state and log that the fallback path was used.
5. Ensure interruptions still use their current behavior and do not double-emit end frames.

This would make Gemini Live handling robust to incomplete or inconsistent response shapes while preserving `turn_complete` as the preferred signal.

## Local Harness Workaround

As a benchmark-side mitigation, `src/multi_turn_eval/processors/tts_transcript.py` was updated to also flush buffered assistant text on `BotStoppedSpeakingFrame`.

That workaround prevents transcript loss in this failure mode, but it does not fix the underlying Gemini Live service assumption in Pipecat core.
