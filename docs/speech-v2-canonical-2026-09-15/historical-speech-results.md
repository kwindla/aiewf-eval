# Historical speech results before speech-v2

Archived before the 2026-09-15 canonical refresh. These used the original shared prompt and older judge configuration; they are not directly comparable to speech-v2. Recording links are relative to the repository root.



| Model             | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|-------------------|-----------|----------|-------------|-----------|---------|------------------|------------------|---------------|------------------|
| gpt-realtime-2.1 (low) | 97.2% | 876/900 | 876/900 | 900/900 | 899/900 | 1504ms | 4288ms | 1537ms | 75ms |
| grok-voice-think-fast-1.0 | 95.3% | 288/300 | 289/300 | 299/300 | 296/300 | 2336ms | 4800ms | 2753ms | 239ms |
| gpt-realtime-1.5 | 93.3% | 282/300 | 280/300 | 300/300 | 299/300 | 1152ms | 2304ms | 2251ms | 96ms |
| gemini-3.1-flash-live (minimal) | 91.7% | 276/300 | 277/300 | 300/300 | 300/300 | 1632ms | 5664ms | 3172ms | 100ms |
| gpt-realtime | 86.7% | 271/300 | 260/300 | 300/300 | 296/300 | 1536ms | 4672ms | 2199ms | 341ms |
| gemini-live | 86.0% | 258/300 | 261/300 | 293/300 | 278/300 | 2624ms | 30000ms | 4082ms | 90ms |
| nova-2-sonic | * | 278/300 | 265/300 | 296/300 | * | 1280ms | 3232ms | 1689ms | 79ms |

For speech-to-speech models, we measure voice-to-voice latency by analyzing the conversation recording. We measure the overall time from the end of the user's speech to the beginning of the model's speech response.

For voice agent use cases, voice-to-voice latency needs to be under 1,500ms.

The voice-to-voice latency measured here is different from the TTFB reported by the Pipecat service for these models, because all of these models were tested in server-side VAD mode (the server-side turn delay is opaque to the Pipecat pipeline), and all of the models send initial silence bytes before actual speech audio. (Text-to-speech models do this, too. The initial silence segments are typically between 150ms and 250ms for standalone TTS models.)

We also include a "Turn Ok" column for these models, which counts how often the model does not respond at all when we expect it or responds with control characters, API refusals, or generic errors.

The Nova 2 Sonic model performs very well on instruction following and tool calling but has a high rate of safety refusals for normal content. It also has a connection limit of 8 minutes. Fixes for both of these issues are in flight from AWS.

gemini-3.1-flash-live-preview is a reasoning model. We show the results with minimal thinking enabled, above. Here are the results all thinking levels. Even minimal is too slow for voice agent use cases, today. With additional reasoning, the model's benchmark results improve, but latency increases. Note the tool use and instruction following regression from medium to high. This is something we see regularly in multi-turn testing of thinking models. The model is not always "smartest" at the highest thinking setting.

| Thinking | Pass Rate | Tool Use | Instruction | KB Ground | Turn Ok | Non-Tool V2V Med | Non-Tool V2V Max | Tool V2V Mean | Silence Pad Mean |
|----------|-----------|----------|-------------|-----------|---------|------------------|------------------|---------------|------------------|
| high | 91.0% | 281/300 | 273/300 | 300/300 | 297/300 | 2368ms | 15872ms | 3665ms | 119ms |
| medium | 96.0% | 291/300 | 288/300 | 300/300 | 296/300 | 2400ms | 10208ms | 3903ms | 113ms |
| low | 90.3% | 282/300 | 271/300 | 299/300 | 297/300 | 2176ms | 4672ms | 3602ms | 97ms |
| minimal | 91.7% | 276/300 | 277/300 | 300/300 | 300/300 | 1632ms | 5664ms | 3172ms | 100ms |

### Sample Recordings

Listen to full 30-turn benchmark conversations from each speech-to-speech model:

| Model         | Recording                                      |
|---------------|------------------------------------------------|
| Ultravox v0.7 | [ultravox-v0.7.mp3](../../samples/ultravox-v0.7.mp3) |
| GPT Realtime 1.5 | [gpt-realtime.mp3](../../samples/gpt-realtime.mp3)   |
| Grok Voice (think-fast 1.0) | [grok-realtime.mp3](../../samples/grok-realtime.mp3) |
| Gemini 3.1 Flash Live (minimal thinking) | [gemini-live.mp3](../../samples/gemini-live.mp3)     |
| Nova 2 Sonic  | [nova-2-sonic.mp3](../../samples/nova-2-sonic.mp3)   |

In the recordings, the left channel is the user's speech and the right channel is the model's speech. You'll hear a small audio tag (a short beep) at the start of each model speech segment. The benchmarking code adds this audio tag to make it easier to double-check track alignment and to provide a visual indicator of segment start for manually inspecting the recordings.

![Visually inspecting silence padding by looking at the audio waveform](../../samples/silence-padding-measurement.png)

*Visually inspecting silence padding by looking at the audio waveform. Here we're using the [Descript](https://descript.com/) editor.*

