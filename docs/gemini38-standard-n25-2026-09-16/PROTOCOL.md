# Regular Gemini 3.8 Live: 25 completed conversations

Authorized September 16, 2026. Model `gemini-3.8-live`, with thinking configuration omitted: Google's regular model does not accept `low` or any thinking level.

- Unchanged shared speech-v2 prompt and shared Opus 5 judge.
- First 25 complete, harness-valid conversations, 30 turns each; no quality-based selection.
- One full conversation validated before increasing capture concurrency to five.
- Record stereo audio and raw server events; run quantitative waveform V2V and input alignment audits alongside the judge.
- Synchronous turnComplete completion, explicit user activity, startup guard, deferred end_session until playback completes. No interaction-IDLE requirement, replay, or recovery.
- Preserve every attempt. Report incomplete model/service conversations separately with one-based failed turn. Stop for review on harness/validation issues; at most 35 attempts without review.
- Judge and latency processing are concurrent. A completed conversation with a measurement issue is held for review, not replaced.

Google configuration reference: https://ai.google.dev/gemini-api/docs/live-api/thinking

The earlier extended-thinking medium harness row is removed from the current README; historical results and captures remain archived.

## Duplicate-result delivery review

The initial 35 attempts produced 21 validated full conversations, 13 harness-invalid timeouts after a duplicate tool result was filtered before provider delivery, and one turn-1 no-response timeout. All 21 complete conversations had zero filtered duplicate results and remain eligible. The capture-local correction sends the existing duplicate_skipped response directly to the provider for that call ID, without executing the action twice; prompts and judging are unchanged. First corrected conversation validates this path, then remaining captures resume. Review cap increased to 50 total captures; original attempt records remain immutable. See harness-review.json for evidence.
