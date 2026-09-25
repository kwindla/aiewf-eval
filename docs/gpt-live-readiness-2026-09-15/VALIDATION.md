**Superseded first-pass judgments:** see the [revised shared-judge validation](rejudged/VALIDATION.md). Original figures below are retained.

# GPT-Live standard-prompt validation

**Accuracy scores are first-pass Opus 5 results pending a shared-judge consistency review of the MCP counting question across all models. No model prompt or judge has been changed for that review. Runtime, API reliability, and audio measurements below are complete.**

**GPT-Live-1 + GPT-5.6-Terra, medium reasoning; Pipecat 1.10.0 / OpenAI SDK 3.14.0.**

Frontend and backend both received the full, byte-identical shared speech-v2 prompt and knowledge base. All six runs used the unchanged Opus 5 rubric, original audio, and original tool schemas. No recovery nudges, prompt adaptations, or replacement attempts.

| Run | Turns | Strict accuracy | Mean V2V | Median V2V | p90 V2V |
|---|---:|---:|---:|---:|---:|
| Single validation | 30/30 | 28/30 (93.3%) | 1.83 s | 1.71 s | 2.46 s |
| Concurrent 1 | 30/30 | 28/30 (93.3%) | 1.71 s | 1.62 s | 2.25 s |
| Concurrent 2 | 30/30 | 29/30 (96.7%) | 1.94 s | 1.90 s | 2.53 s |
| Concurrent 3 | 30/30 | 28/30 (93.3%) | 1.82 s | 1.87 s | 2.25 s |
| Concurrent 4 | 30/30 | 29/30 (96.7%) | 1.92 s | 1.86 s | 2.40 s |
| Concurrent 5 | 30/30 | 29/30 (96.7%) | 1.88 s | 1.76 s | 2.40 s |

## Five-conversation concurrency pilot

Completed 5/5 attempts; 143/150 strict passes (95.3%). All planned turns are in the denominator. API/pipeline errors: 0; confirmed closed sessions: 5/5.

Pooled V2V: mean 1.85 s, median 1.86 s, p90 2.37 s. Speech-overlap turns: 0; turn-boundary issues: 0; waveform alignment issues: 0.

V2V is measured from actual user-speech end to first actual assistant speech in the stereo WAV, including acknowledgements. The detector/thresholds are shared with the canonical benchmark; Live turn association uses recorded input/commit windows because the API emits continuous silence audio. These figures do not measure time to a substantive answer or tool completion.

The five Live sessions overlapped the tail of the existing OpenAI Realtime campaign; see `concurrency-pilot/background-load.json`. This is evidence for N=5 under the tested load, not a guarantee about sustained quotas or a canonical 25-attempt score.

## Errors and provenance

- Single validation, turn 23: No call was expected here; the assistant called vote_for_session early. The arguments (session_id 936902, Jennifer Smith) exactly match the expected call and the session details it quoted are KB-accurate, so the tool use itself is correct. However, 'the talk was by Harald Kirschner' is genuinely ambiguous across three KB sessions (June 3 workshop, June 4 MCP talk, June 5 Vibe Coding talk), and the policy requires a brief clarifying question rather than acting on a guess — the assistant committed an irreversible vote on an unverified assumption, so instruction_following fails.
- Single validation, turn 29: The user clearly signals they are finished, so end_session was expected, but it was never called at this or any earlier turn — a genuine missing tool call. The spoken closing itself is an appropriate, friendly sign-off with no factual errors, so instruction_following and kb_grounding pass.
- Concurrent 1, turn 13: Answers the food question and no tool call was expected, but it asserts that all June 5 meals are in Salons 9-15 Expo Hall; the KB places June 5 continental breakfast in Grand Assembly. That is an explicit factual error.
- Concurrent 1, turn 29: The user clearly signals the conversation is finished ('I don't have anything else'), so end_session should have been called. The assistant instead asked for redundant confirmation about wrapping up, so the expected call was never made and the turn did not advance the closing.
- Concurrent 2, turn 29: The user clearly signals they are finished ('I don't have anything else'), which is unambiguous intent to close, but end_session was never called in this or any other turn, so the expected tool use is genuinely missing. The spoken farewell itself is an appropriate, factually clean closing response, so instruction_following and kb_grounding stand.
- Concurrent 3, turn 4: The governing policy states: 'Do not calculate or infer total numbers of sessions, speakers, or topics. Describe the available kinds of sessions and their days instead.' The assistant instead produced an explicit count ('three sessions listed for the MCP track on June 4th'), which also misses Harald Kirschner's June 4, 3:00 PM MCP-track talk. This is the specific missing behavior: a descriptive overview of session kinds rather than a total. The individual session titles, times, and speakers it cited are all accurate, so grounding is credited.
- Concurrent 3, turn 29: The user clearly signals the end ('I don't have anything else'), which is the condition for end_session, but no end_session call was made at this or any prior turn, so the expected action is simply missing — tool_use_correct is false. The spoken closing itself is warm and appropriate and contradicts nothing it did, so instruction following and grounding stand.
- Concurrent 4, turn 29: The user clearly signaled the conversation was finished ('I don't have anything else'), which is the condition for calling end_session, but no end_session call was made and it was not made earlier. The verbal sign-off does not complete the required closing action, so both the tool use and the task completion fail here. No factual error was stated.
- Concurrent 5, turn 29: The user clearly signals the end of the conversation ('I don't have anything else'), which is an unambiguous intent to finish, but end_session was never called here or anywhere earlier, so there is no realignment credit. The reply also merely offers further help instead of closing out, failing to act on the user's clear conclusion.

All earlier custom-frontend probes are diagnostic only and excluded in full, including the silent name-turn failure and the interrupted unauthorized prompt revision. The user required the same frontend prompt for every model; all results in this report comply. The other five models retained their frozen prompt/rubric throughout.

See `standard-validation-results.json` for per-turn audio metrics, complete judge summaries, error evidence, and every run path. Prompt improvements belong in model-specific README notes and were not applied to these benchmark captures.
