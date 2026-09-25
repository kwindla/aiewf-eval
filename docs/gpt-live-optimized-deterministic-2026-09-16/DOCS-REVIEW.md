# GPT-Live completion signals: documentation review

Reviewed September 16, 2026 after the user asked whether backend completion can
be traced to pending speech finishing. No new captures launched during this review.
The replacement cohort had already stopped on its completion-timeout safeguard.

## Conclusion

The public API supports a timeline of backend progress, context injection, spoken
transcript intervals, and observed playback. It does not document a per-result
frontend generation-completed event or an authoritative complete-turn event.
This is narrower than saying there is no way to trace backend results to speech.

The missing proof is that no additional speech associated with the backend result
will be generated later. Playback drain proves that received/queued samples played;
it does not prove that the frontend has no more output to generate.

## Evidence

1. [Delegation lifecycle](https://developers.openai.com/api/docs/guides/live-delegation#handle-responses-delegation):
   nested backend events carry a delegation ID. Collect function calls from
   output_item.done and finish their result/continuation cycle. Backend completion
   does not establish spoken delivery.
2. [Context injection](https://developers.openai.com/api/docs/guides/live-conversations#understand-when-context-reaches-the-model):
   commentary/thinking/instruction append acknowledgments carry estimated context
   injection intervals, not speech completion. They do not prove full consumption.
   Switching to client delegation does not create a delivery acknowledgment.
3. [Transcript deltas](https://developers.openai.com/api/docs/guides/live-conversations#transcript-deltas):
   retain session-relative intervals and original fragments. There is no item ID or
   authoritative turn-completed event. Late fragments can require revised grouping.
4. [Primary WebSocket output](https://developers.openai.com/api/docs/guides/voice-websockets#send-and-receive-events):
   audio chunks have no timing fields or audio-done event. Track playback locally.
5. [Reflected sideband audio](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live#receive-reflected-audio):
   unlike primary WebSocket output, reflected audio has start/end session timestamps.
   These improve alignment but do not establish caller playback or result completion.
6. [Migration comparison](https://developers.openai.com/api/docs/guides/live-migration#adapt-the-connection-and-audio-lifecycle):
   explicitly distinguishes Realtime response.done/output_audio.done from Live,
   which has no corresponding per-spoken-response end event.
7. [Evaluation cookbook](https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation):
   CRAWL infers completion from ended caller input, completed delegated work,
   settled speech-backed transcript groups and a quiet tail. It also requires a
   new spoken turn after returned content becomes available; the guide acknowledges
   that a valid answer sharing an earlier caption group can therefore time out.
   RUN uses a continuous simulated caller rather than an external turn scheduler.

Installed OpenAI SDK 3.14.0 event definitions agree with these distinctions.
Output transcript events have an optional client_event_id for associated commands;
this is not a documented guarantee of a backend-result-to-complete-spoken-answer
mapping. Output audio events have no delegation/response identifier. The server
union has no frontend output-done event.

## Implications for this benchmark

The existing no-classifier gate waits for backend inactivity, stopped playback and
three seconds of quiet. That is a timing heuristic, not proof of delivered results.
A stronger audit should correlate returned backend content with subsequent spoken
fragments and played samples. Do not require a new caption group to start after
backend completion: streaming speech can start before final backend completion,
and the official cookbook warns about the caption-group assumption.

Content correctness belongs to offline judging. No runtime answer-quality model
is necessary to trace the event/audio sequence. Any advancement rule still needs
an explicit tested quiet-tail policy; neither these docs nor the SDK establishes
an end-to-end authoritative completion barrier. Keep the stopped cohort preserved
and investigate its timeout before resuming captures.
