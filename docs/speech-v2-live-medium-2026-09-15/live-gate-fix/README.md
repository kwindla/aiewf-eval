# GPT-Live input-end gate defect: isolated fix ready for review

The current adapter can commit a turn immediately when the input WAV ends if the
model produced an early acknowledgment and has been quiet for three seconds **during**
the user's remaining speech. That leaves no response opportunity after input end.

Concrete evidence: extension attempt 7, turns 8 and 9, committed within a millisecond
of `input_end`, with only “Mm-hmm.” and “Okay.” recorded. The next user input starts
about 20 ms later. The merged user speech and missing detected bot segments in audio
analysis exposed the issue. See [event-gap audit](audit.json) and the preserved run
`runs/aiwf_medium_context_speech_live/20260915T213511_gpt-live-1_dff26048`.

This is a harness timing defect, not a model/API failure or evidence that the frontend
prompt should change. It can affect subsequent conversation state and semantic scores.

## Concrete proposed fix

[Patch](proposed.patch): add `finish_input(now)` to update both `input_finished` and
`last_activity`, and call it when the input stream finishes. The same three-second
quiet interval now starts no earlier than the end of user input. Backend activity,
backend tool continuations, and playback still hold the turn open as before.

The [candidate module](openai_live_candidate.py) is isolated in this directory.
The active pipeline file has not changed. Ten [tests](tests.txt) passed, including
reproduction of the original defect, the repaired post-input interval, continued
backend gating, and the existing prompt-isolation/adapter tests.

This remains a quiet-period heuristic; GPT-Live has no authoritative audio-completion
event. The fix addresses the concrete premature-advance path, not every possible
ambiguity between a short answer and a backchannel.

## Cohort treatment

Finish and preserve the already-started captures under their original hashes. Do not
mix the candidate adapter into this cohort or selectively replace only failed-looking
runs. Keep these 25 GPT-Live conversations as diagnostic evidence. A fresh, uniformly
configured 25-attempt standard-prompt cohort is recommended before publishing a
canonical GPT-Live row. That would add 25 new captures beyond the 20 additions the
user requested; the concrete patch and tests are ready for that scope decision.

The separate Gemini medium campaign is unaffected and continues. The optimized-prompt
design is also independent; any future standard/optimized comparison should use the
same corrected playback gate, with the common prompt preserved for the standard arm.

## Application status

The user approved the fix and 25 additional fresh captures. The tested patch was
applied on September 15, 2026; nine adapter/prompt-isolation tests passed. The
[fresh standard cohort](../../gpt-live-fixed-gate-n25-2026-09-15/PROTOCOL.md)
is running at concurrency five. This earlier cohort remains diagnostic.
