# Gemini Extended Thinking: harness findings

## Existing support was already important and correct

The existing Gemini service defers terminal pipeline frames when a provider message
has `turn_complete=true` and `interaction_status=IN_PROGRESS`. It accumulates spoken
transcript fragments until the terminal interaction. The smoke test exercised real
intermediate phases; a GPT-Live-style external completion classifier is unnecessary.

## Additions in this experiment

- Preserve setup, raw provider content/transcription, phase boundaries and commits.
- Declare functions explicitly NON_BLOCKING, matching Google's extended-thinking
  contract. The previous Pipecat Gemini-3 path omitted that declaration; successful
  old calls do not prove that the implicit default was blocking.
- Handle IDLE even when it arrives without turn_complete.
- Reconstruct whole spoken exchanges and assert equivalence before applying the
  unchanged Opus 5 task-quality rubric. Preserve all original data.
- Use actual speech windows and independently verify input waveform alignment.
  Keep first speech, final speech completion and missing/overlapping speech distinct.

## Closing bug demonstrated by the first smoke

Source: `BasePipeline._function_catchall`, end_session branch. After returning tool
success, the shared handler waits 50ms, writes an empty assistant transcript, and
cancels the pipeline. That bypasses provider IDLE and the normal audio-drain gate.

The first smoke's final exchange had the raw spoken fragment “It was a pleasure
helping you navigate the fair.” but its committed transcript was empty. No final
IDLE or normal exchange_commit was captured. The integrity audit rejected it, and
no 25-run campaign was launched from that smoke.

The isolated adapter now returns the same tool result and waits for IDLE and drained
playback. Normal spoken closure is recorded through the regular turn gate; a truly
silent tool-only closure has an IDLE-and-drain fallback. Early end_session still
ends as an incomplete model outcome, without advancing to another user question.
The first capture remains preserved; a second full smoke tests the correction.

This also changes how to interpret earlier “silent end_session” measurements: those
were silent/empty in the saved artifacts, not evidence that the model chose silence.
The shared cancellation policy could truncate closing speech. No old recordings or
scores have been repaired, replaced, or retroactively relabeled as model failures.

## Measurement limits

An IN_PROGRESS phase is not necessarily just filler; it may contain useful partial
content. Conversely, IDLE time is not the time a substantive word was heard. We do
not label either automatically as substantive-answer latency. First-speech V2V still
includes acknowledgments. Transcript receipts and provider state are not substitutes
for audible word alignment.
