# GPT-6 Luna completion investigation — 2026-10-18

The Luna results reveal a real Responses integration problem. The adapter discards provider-native assistant history, including `phase` values and output-item boundaries. In a bounded replay of two archived failure contexts, the current flattened history produced **6/6 inappropriate tool-only responses**; restoring the original assistant output items produced **6/6 relevant text answers and no tool calls**. This strongly implicates history reconstruction. It does not yet establish that a repaired full campaign will complete all 30 turns.

No benchmark runtime, prompt, configuration, source manifest, canonical conversation, or published grade was changed in this investigation. The paid replay is an excluded diagnostic, not replacement benchmark data.

## What actually ended the runs

The [existing completion audit (local archive)](../README.md) contains all 60 canonical ending cases and transcript hashes. All 60 end with an explicit `end_session({})` on a non-final scripted user turn. These are not timeouts, token-limit failures, missing provider responses, or completion classification mistakes. The harness deliberately cancels the conversation when that tool runs ([base.py:1404](../../../src/multi_turn_eval/pipelines/base.py#L1404)).

Turn numbers below are one-based:

| Effort | Ended on user turn | Number of runs | User input |
|---|---:|---:|---|
| none | 2 | 1 | Are there any workshops about Gemini? |
| none | 6 | 6 | It seems like there are so many great sessions. Can you suggest some sessions for me? |
| none | 7 | 2 | I'm particularly interested in Voice AI and in techniques for building reliable agents. |
| none | 8 | 3 | I'll be at the conference on June 4th all day. |
| none | 9 | 11 | Actually, I made a mistake. I will only be at the conference on June 5th. |
| none | 14 | 7 | Thanks for submitting both session suggestions. Is there food at the conference? |
| low | 26 | 7 | Yes — what time is lunch on June 5th? |
| low | 27 | 1 | Could you also confirm if there's an afterparty on June 4th? |
| low | 28 | 13 | Are there community meetups on the evening of June 5th? |
| low | 29 | 9 | One last detail: when is continental breakfast on June 4th? |

The low setting was effective: the canonical low runs report **18,635 reasoning tokens**, versus zero for none. The median observed scripted length is 28 for low and 9 for none. Low had 43 additional recovery responses across its 30 conversations; none had no recovery responses. Every final premature end call was on a scripted user turn, not on a synthetic recovery message. All premature end-call responses themselves report zero reasoning tokens, even in low mode; that fact does not show the setting was ignored on earlier turns.

The semantic distinction matters. Turn 14 begins with thanks, and the prompt explicitly lists “thank you” as a conversation-ending example ([system.py:30](../../../benchmarks/aiwf_medium_context/prompts/system.py#L30)). That wording can plausibly over-trigger despite the following food question. Turn 29 says “One last detail,” but still asks an unanswered question. All other endings are direct in-scope requests or preference updates with no plausible farewell. Prompt ambiguity can explain a subset of susceptibility; it does not explain the full pattern.

The underlying `end_session` tool description is only “End the current session.” ([tools.py:6](../../../benchmarks/_shared/tools.py#L6)). Its first position among the tools could make it an attractive erroneous fallback, but no ordering ablation was run. The prompt's sample name Sam also leaks into an erroneous pair of votes in the excluded stream diagnostic; this is contextual confusion, not a missing Jennifer field at that early stage.

## Concrete history loss

The existing [excluded full-stream capture (local raw runs; unarchived)](../README.md) establishes that the provider itself emitted the tool calls. Streamed and completed names match. However, that observation does not validate the history sent *into* the provider.

That capture contains nine requests, indexed 0–8. Request 5 responds to the session-recommendation question with two output messages:

1. A nonempty assistant message with `phase: "commentary"`, asking interests and availability.
2. An empty assistant message with `phase: "final_answer"`.

Request 6 sends these back as one plain assistant text message without a phase. Earlier final-answer phases are also omitted. Request 7 then calls `vote_for_session` twice, even though the user only says they will attend June 4. It invents the name Sam from the prompt examples. Request 8 responds to the June 5 correction with `end_session`.

The loss occurs in two places:

- The stream handler forwards text deltas and executes function calls, but does not retain complete output message/reasoning items for future context ([openai_responses.py:345](../../../src/multi_turn_eval/services/openai_responses.py#L345), [openai_responses.py:416](../../../src/multi_turn_eval/services/openai_responses.py#L416)).
- `_message_to_responses_items` reconstructs messages using only role and text. It drops empty messages and does not preserve `phase`, original item identities, or reasoning items ([openai_responses.py:111](../../../src/multi_turn_eval/services/openai_responses.py#L111)). A generic universal context aggregator supplies this flattened history ([text.py:326](../../../src/multi_turn_eval/pipelines/text.py#L326)).

The full system prompt, all expected user inputs, and tool definitions are present in the saved requests. None of these requests forces a specific tool or supplies a token cap. OpenAI uses the stateless reconstruction path; only the separate xAI Responses path enables the adapter's existing stored-response mode. The archived model ID is `gpt-6-luna` throughout.

Official OpenAI documentation recommends preserving original assistant phases and explains that losing them can cause early stopping. It also recommends retaining all response output items, including reasoning items when reasoning is used. See [assistant phase](https://developers.openai.com/api/docs/guides/reasoning#phase-parameter), [reasoning continuity](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls), and [GPT-6 migration guidance](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart). Those requirements support the integration diagnosis; the replay below provides direct local behavioral evidence.

## Bounded replay results

The parent agent authorized twelve paid calls. [replay_diagnostic.py](../../../ops/luna-completion-investigation-20261018/replay_diagnostic.py) replays requests 7 and 8 from the excluded stream capture, three times per context under each of two conditions:

- **Flattened:** the exact archived request.
- **Original items:** the same request settings, prompt, tools and current user input, rebuilding prior assistant output from each archived completed response.

The reconstructed condition preserves all original assistant phases, output message boundaries, empty final-answer messages, and API output fields. It removes only SDK parsing conveniences that are not API input fields. For the two parallel function calls before request 8, it restores the original contiguous provider call order followed by the original tool-result order. Assertions verify identical system/user sequences, identical tool-result sequences, and semantically identical function calls. Raw requests, responses and stream events are in [excluded-replay (local archive)](../README.md).

This is a combined history-preservation test, **not a phase-only ablation**. There are no reasoning items in the archived none-mode output, and every replay reports zero reasoning tokens. Restoring hidden reasoning therefore cannot explain this particular result.

| Archived context | History | Results across 3 repetitions |
|---|---|---|
| Request 7: June 4 availability | Flattened | Two unsolicited session-suggestion calls; one `end_session`; no text answers |
| Request 7: June 4 availability | Original items | Three text answers recommending June 4 sessions; zero tool calls |
| Request 8: June 5 correction | Flattened | One unsolicited session-suggestion call; one `end_session`; one unsolicited vote; no text answers |
| Request 8: June 5 correction | Original items | Three text answers updating recommendations to June 5; zero tool calls |

All twelve API responses completed successfully. There were no API validity errors. A diagnostic logging assumption about `_request_id` failed *after* the first seven responses were saved; those responses were recovered offline without another request. The remaining five calls completed after correcting the logger. Each raw record retains its response ID; stream response objects did not expose the HTTP request ID. The twelve-call limit was maintained.

Usage: **149,580 input tokens + 668 output tokens = 150,248 total tokens**. All outputs are retained, including any remaining factual errors; these diagnostic answers were not passed through the benchmark judge.

See [all replay summaries (local archive)](../README.md) and [completion marker (local archive)](../README.md).

## Why low lasts longer, and what remains unknown

The data show a clear reasoning-dependent difference in susceptibility: low gets much further under the same history reconstruction. The combination of omitted phases, flattened output messages, and omitted reasoning provides a credible mechanism for degradation across calls. The replay demonstrates that repairing provider-native history can remove the observed inappropriate tool selection at two none-mode failure contexts.

It does **not** establish which lost field is necessary or sufficient; why low fails specifically in turns 26–29; whether every one of the 60 canonical failures shares one cause; or the completion rate after a full repaired run. There is no saved full low-mode provider event trace comparable to the none diagnostic, so its phases cannot be reconstructed retrospectively. Greater inherent low-mode coherence and better resistance to malformed history are consistent with the observations but remain inferences.

Recovery is another confound in low mode. The harness inserts one “Please go ahead.” message when a required tool is missing or has different arguments ([base.py:1568](../../../src/multi_turn_eval/pipelines/base.py#L1568)). Some low conversations submitted vegan requests ahead of the scripted “Yes,” then answered that the request was already submitted and received a recovery nudge. This changes history and adds turns. It does not directly trigger any recorded premature end, and the none-mode replay demonstrates the history problem without recovery.

## Recommended next action

Repair Responses continuity before increasing Luna reasoning or changing the benchmark prompt. Preserve complete output items and phases at the provider boundary, keep opaque reasoning items in model history, and retain every tool result. Judging should still receive only visible assistant content and actual tool calls.

A provider-native history ledger with a reliable per-response input cursor is safer than reconstructing assistant state from generic text. A correctly implemented stored-response path is another option. **Do not simply switch on the existing stateful flag**: its “slice after the last assistant message” code ([openai_responses.py:233](../../../src/multi_turn_eval/services/openai_responses.py#L233)) can lose the first result when parallel calls appear in universal context as assistant-call/result/assistant-call/result. The saved Luna request 8 has exactly this shape. Delta tracking must preserve both results and the next user input.

Meaningful tests should cover commentary + empty final answer round-trips; consecutive assistant items; reasoning output continuity; multiple calls/results in one response; and new-user/recovery messages appended exactly once. Then run an excluded full 30-turn smoke in each Luna mode, recording all request and completed output items. Only after that should a fresh, separately registered canonical cohort replace the current README characterization.

The existing 0/30 measurements remain valid records of the tested integration, but they should not be interpreted as clean evidence that correctly integrated GPT-6 Luna cannot complete this conversation. Grok Voice uses a separate realtime adapter; this finding does not require stopping its current high-mode campaign.

[Evidence storage and local audit instructions](../README.md).
