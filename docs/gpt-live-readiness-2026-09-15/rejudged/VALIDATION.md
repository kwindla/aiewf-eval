# GPT-Live standard-prompt validation

Judge revision: `speech-v2-policy-compliance-1`. Original judgments remain in the capture directories; revised judgments and the full before/after audit are in [the shared revision directory](../../speech-v2-canonical-2026-09-15/rejudge-policy-compliance-1/AUDIT.md).
**GPT-Live-1 + GPT-5.6-Terra, medium reasoning; Pipecat 1.10.0 / OpenAI SDK 3.14.0.**

Frontend and backend both received the full, byte-identical shared speech-v2 prompt and knowledge base. All six saved conversations were rejudged with the same clarified Opus 5 speech rubric used for the other 125 attempts; original audio and tool schemas are unchanged. No recovery nudges, prompt adaptations, or replacement attempts.

| Run | Turns | Strict accuracy | Mean V2V | Median V2V | p90 V2V |
|---|---:|---:|---:|---:|---:|
| Single validation | 30/30 | 28/30 (93.3%) | 1.83 s | 1.71 s | 2.46 s |
| Concurrent 1 | 30/30 | 28/30 (93.3%) | 1.71 s | 1.62 s | 2.25 s |
| Concurrent 2 | 30/30 | 29/30 (96.7%) | 1.94 s | 1.90 s | 2.53 s |
| Concurrent 3 | 30/30 | 28/30 (93.3%) | 1.82 s | 1.87 s | 2.25 s |
| Concurrent 4 | 30/30 | 28/30 (93.3%) | 1.92 s | 1.86 s | 2.40 s |
| Concurrent 5 | 30/30 | 28/30 (93.3%) | 1.88 s | 1.76 s | 2.40 s |

## Five-conversation concurrency pilot

Completed 5/5 attempts; 141/150 strict passes (94.0%). All planned turns are in the denominator. API/pipeline errors: 0; confirmed closed sessions: 5/5.

Pooled V2V: mean 1.85 s, median 1.86 s, p90 2.37 s. Speech-overlap turns: 0; turn-boundary issues: 0; waveform alignment issues: 0.

V2V is measured from actual user-speech end to first actual assistant speech in the stereo WAV, including acknowledgements. The detector/thresholds are shared with the canonical benchmark; Live turn association uses recorded input/commit windows because the API emits continuous silence audio. These figures do not measure time to a substantive answer or tool completion.

The five Live sessions overlapped the tail of the existing OpenAI Realtime campaign; see `concurrency-pilot/background-load.json`. This is evidence for N=5 under the tested load, not a guarantee about sustained quotas or a canonical 25-attempt score.

## Errors and provenance

- Single validation, turn 23: Harald Kirschner has three listed sessions (June 3 workshop, June 4 MCP talk, June 5 Vibe Coding talk), so the target was genuinely ambiguous and policy required a brief clarification; instead the assistant guessed and announced a submitted vote. That is a concrete policy conflict, so instruction_following fails. The call itself used the correct session ID 936902 for the session the user confirms next turn, so it is credited as an early-but-correct tool call, and the session title/date it cited are factually accurate.
- Single validation, turn 29: The user clearly signals they are finished ('I don't have anything else'), so end_session({}) was required; it was never called at this or any earlier turn, so the expected tool use is missing. The assistant verbally signs off ('Take care') while leaving the session open, so the required closing action was not completed. No factual errors.
- Concurrent 1, turn 13: No tool was expected and none was made, but the answer states that on June 5 the meals are all in Salons 9-15: Expo Hall; the KB lists June 5 Continental Breakfast in Grand Assembly. That incorrect location is a factual contradiction and it misinforms the user on the very question asked.
- Concurrent 1, turn 29: The user clearly signaled the end of the conversation ('I don't have anything else'), so end_session should have been called. Instead the assistant asked whether to wrap up — a redundant confirmation when the user's intent was unambiguous — and the tool was never called anywhere in the conversation.
- Concurrent 2, turn 29: The user clearly signals completion ('I don't have anything else'), which the policy says should trigger end_session, but no end_session call was made at this or any prior turn, so the required terminating action is simply missing. The spoken sign-off is friendly and factually fine, hence kb_grounding stands, but failing to invoke the available end_session action contradicts the governing policy on ending the session.
- Concurrent 3, turn 4: The governing policy explicitly says not to calculate or infer total numbers of sessions and to describe kinds of sessions and days instead; the assistant instead gave a concrete count ('three sessions listed for the MCP track on June 4th'), which is the specific behavior the policy forbids. The individual sessions named are accurate to the KB, so there is no factual contradiction.
- Concurrent 3, turn 29: The user clearly signaled they were finished and the assistant said goodbye, but end_session was never called and was not called earlier, so the expected action is simply missing. The spoken farewell also contradicts the fact that the session was left open.
- Concurrent 4, turn 23: No tool call was expected and none was made, so tool use is fine. However, the KB lists three Harald Kirschner sessions (including a second talk on June 5), and the policy requires including all relevant entries when identifying a speaker's sessions. The assistant surfaced only the June 4 talk and proposed voting for it, which risks an incorrect vote; that is a concrete policy conflict. The details it did state were correctly qualified by date, so there is no factual contradiction.
- Concurrent 4, turn 29: The user clearly signalled they were finished ('I don't have anything else'), so end_session was required and was never called at this or any earlier turn. The reply also implies the conversation is closing ('That should do it from my side') while taking no closing action, a words/actions mismatch. No factual claims were made, so KB grounding stands.
- Concurrent 5, turn 4: The listed sessions and speakers are accurate, but the assistant gave explicit totals ('Three on June 4th', 'two MCP workshops'), directly contradicting the policy instruction not to calculate or infer total numbers of sessions. Instruction following fails on that specific policy conflict; the facts themselves are correct so grounding holds.
- Concurrent 5, turn 29: The user clearly signals completion ('I don't have anything else'), so end_session should have been called; it was not called here or at any earlier turn, so no realignment credit applies. The response also keeps the conversation open instead of closing it as the user intended. No factual errors.

All earlier custom-frontend probes are diagnostic only and excluded in full, including the silent name-turn failure and the interrupted unauthorized prompt revision. The user required the same frontend prompt for every model; all results in this report comply. The other five models retained their frozen model prompts throughout and received the identical judge revision.

See `standard-validation-results.json` for per-turn audio metrics, complete judge summaries, error evidence, and every run path. Prompt improvements belong in model-specific README notes and were not applied to these benchmark captures.
