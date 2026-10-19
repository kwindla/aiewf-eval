# Nova 2.5 Sonic: standard benchmark blocked at preflight

The standard speech-to-speech benchmark cannot currently run: AWS rejects the
unchanged speech-v2 system prompt and knowledge base with a content-filter error
before the first scripted question. This occurred both in the full harness and
in a separate native Bedrock diagnostic with the identical full prompt and tools.
The service does not identify the triggering passage; its precise cause is unknown.

| Check | Result |
|---|---|
| Account authentication and model lookup | Passed; `amazon.nova-2-5-sonic`, ACTIVE, us-east-1 |
| Native protocol, tool call, tool result, spoken continuation | Passed; `probe(value=323)`, success result, spoken “OK” |
| Standard full-task harness smoke | Provider content-filter rejection; zero scripted turns |
| Native unchanged full-prompt confirmation | Same provider content-filter rejection |
| Canonical conversations | 0 collected; target remains 25 complete conversations of 30 turns |
| Quality, waveform latency, turn-taking | Unavailable; no benchmark responses to score |
| Paid judge requests | 0 |

Both full-prompt requests returned:

> This request has been blocked by our content filters.

The full prompt is 41,161 characters, SHA-256
`0ff3ba3519554c8743095841a93af92c25d76ec2243d7abb4d1451578a701f9f`.
The harness rejection has request ID `ee95eaf4-0012-4634-a22f-ab6cdba1357f`;
the native confirmation has request ID `aca9599b-8d18-43d5-a3aa-deffa956a993`.
AWS's [Nova 2.5 Sonic service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-2-5-sonic/overview.html)
states that its safety filters cannot be configured or disabled.

The first excluded direct probe initially idled without streaming silence and
hit AWS's input-gap timeout. Adding continuous silent audio completed the native
protocol/tool probe. Both attempts are retained. This probe issue is separate
from the repeatable rejection of the full benchmark prompt.

Before testing, the adapter required compatibility repairs: remove a silent
fallback to Sonic v1 for model IDs without a colon; recognize versioned Nova IDs
in CLI routing; send endpointing through Pipecat's supported settings; inherit
Pipecat's stream-bound receive loop and single spoken-transcript emission; retain
the parent's completion bookkeeping; and honor disabled benchmark recovery so
an error cannot replace the prompt and discard conversation history. The final
adapter uses the supported settings object, preserving the same tested model,
voice, system prompt, sampling defaults, and endpointing wire values.

The seven focused routing/protocol tests pass. Full-suite validation is recorded
in [VERIFICATION.json (local archive)](../README.md). These checks establish routing and wire
compatibility; the blocked full-task smoke cannot establish a valid 30-turn
capture or its waveform alignment.

Collection is held at preflight. The benchmark prompt, knowledge base, tasks,
input audio, existing leaderboard, previous study artifacts, and all previous
grades remain unchanged. No pass rate or failure-rate row is published from
these excluded probes. Resumption requires AWS accepting the unchanged standard
prompt; a different prompt would require a separately described experiment.

[Protocol](PROTOCOL.md) · [Configuration](../../../ops/nova-2-5-sonic-20261019/configuration.json) ·
[Native probe (local archive)](../README.md) · [Harness record (local archive)](../README.md) ·
[Native full-prompt evidence (local archive)](../README.md) ·
[Evidence hashes (local archive)](../README.md)

[Evidence storage and local audit instructions](../README.md).
