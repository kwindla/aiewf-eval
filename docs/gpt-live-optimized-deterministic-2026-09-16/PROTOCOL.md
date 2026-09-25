# GPT-Live prompt-optimized: event and playback gate

Method: **live-playback-exchanges-v1**. Twenty-five fresh attempts at concurrency
five, 750 planned turns, no replacement of failures. The full smoke is excluded.

## Correction

Remove the semantic completion classifier entirely. Across the twelve original
attempts, it made 229 decisions: 221 approvals and eight incorrect rejections.
There was no observed correct rejection of waiting-only speech after the existing
backend/playback gate. The v2 classifier revision remains diagnostic, with its
cohort launcher stopped. Original recordings and results are preserved.

The optimized frontend and backend prompts remain unchanged. The standard Live
gate already waits for user input to finish, all observed delegated backend work
and tool continuations to finish, and local playback to stop. A three-second quiet
interval restarts on input completion, backend events, speech, and playback.
No LLM evaluates answer content before advancing the conversation.

This gate is deterministic code with a timing heuristic. It is **not** proof of
semantic delivery. OpenAI documents no output-audio-done event; backend completion
and local playback drain do not guarantee the frontend has nothing further to say.
Streaming speech may begin before backend completion. Do not require every answer
to start after backend completion, or require backend work for social turns and
clarification questions. A long delayed continuation can still exceed the quiet
window; inspect actual recordings and raw events rather than claim certainty.

Sources:
- [WebSocket output and playback](https://developers.openai.com/api/docs/guides/voice-websockets#send-and-receive-events)
- [Backend delegation lifecycle](https://developers.openai.com/api/docs/guides/live-delegation#handle-responses-delegation)

## Validation

Use a full 30-turn smoke, lossless spoken-text reconstruction, shared Opus 5 judge,
waveform V2V measurements, input alignment, prompt echoes, and equality of canonical
and incremental recordings. Replay logged backend and playback events to audit
every turn boundary. Block launch on measurement failures, not low model accuracy.
Freeze source hashes before launching 25 attempts. Pause new launches on a
completion timeout and allow already active captures to finish.

Preserve raw events, stereo recordings, separate incremental tracks, exact prompt
hashes, source snapshots, and backend diagnostic messages. Backend answers never
replace spoken answers for scoring. Record the shared judge prompt hash. Missing
planned turns count as failures; unjudged recorded exchanges prevent score
publication. First-speech V2V includes interim speech. Substantive-answer onset is
not yet a validated metric, and no human listening audit is claimed.

Standard prompts, the failed cohorts, and other model captures remain untouched.
