# Nova 2.5 Sonic: full-reference USER-context benchmark

**86.0% strict pass**, 25 complete 30-turn conversations
(750 turns), Opus 5.5 Medium. The 95% conversation-bootstrap interval is
82.9–88.8% (20,000 draws; seed 20261018).

| Measure | Result |
|---|---:|
| Strict quality | 645/750 |
| Tool use | 676/750 |
| Instruction following | 658/750 |
| KB grounding | 750/750 |
| Audio turn-taking | 750/750 |
| Unsuccessful canonical attempts | 0/25 (0.0%) |
| Earlier adapter-defect captures, excluded from canonical denominator | 3 |
| First-speech positive P50 / P95 | 1152ms / 1618ms |
| Non-tool median / maximum | 1120ms / 1888ms |
| Tool mean | 1485ms |
| Silence pad mean | 181ms |
| Positive / nonpositive / missing timing samples | 750 / 0 / 0 |

The exact speech-v2 policy is sent as SYSTEM, followed by the complete unchanged
knowledge base as a noninteractive USER history message. Their concatenation is
the original 41,161-character prompt. No facts, tools, scripted turns or input
audio were removed. The [filter investigation](../filter-diagnostics/REPORT.md)
explains why placing the whole reference in SYSTEM was blocked. This message
placement is an explicit provider configuration difference from standard rows.

For a single reproduction with AWS credentials configured:

```bash
MTE_NOVA_KB_CONTEXT_ROLE=user MTE_ENABLE_RECOVERY=0 uv run multi-turn-eval run aiwf_medium_context_speech_v2 --model amazon.nova-2-5-sonic --pipeline nova-sonic
```

The corrected excluded full-task smoke passed before generation was frozen. The
first 30-turn capture lacked local VAD timing markers; a second excluded capture
was stopped after confirming a startup clock offset. Those captures, their source
versions, and failed audits remain preserved. The final smoke validates the shared
VAD/logger wiring and synchronized startup. Selection uses
the first 25 complete conversations in attempt order; every unsuccessful attempt
is retained. There is no quality-based replacement, generation recovery, prompt
truncation, or grading retry based on score. Accuracy is conditional on completion.
The original blocked smoke and diagnostic probes are excluded from canonical
attempt counts and remain available in the parent study.

8 of the 25 complete conversations made no tool calls, although all 25
requests declared the identical five tools. These conversations remain included;
completion is separate from correct tool use and instruction following. The
[serving-tool audit (local archive)](../../README.md) records every case.

The first frozen adapter profile exposed deferred native tool-result delivery in
three captures. Those are retained as harness exclusions in the
[exclusion ledger (local archive)](../../README.md), with raw artifact hashes and
their original source freeze. The repaired profile sends each native tool result
immediately and marks its ID completed, preventing a later context push from
resending it. A new excluded full-task smoke passed before this profile's freeze.
Only the repaired profile supplies the 25 scored conversations.

Latency and turn-taking use the actual stereo recordings and unchanged analyzer.
Nova explicitly arms one recording marker per sequential input WAV, so an early
response cannot lose its marker while waiting for local VAD-stop. The earlier
28-marker smoke and its failed audit are retained; this annotation repair was
validated by the final full-task smoke before the canonical source freeze.
All 30 user input clips in each complete capture pass the existing three-landmark
waveform correlation and alignment checks. Signed and positive latency statistics
are retained in the aggregate. Comparison with other rows is descriptive across
independent cohorts; the confidence interval describes these sampled conversations.

Successful Opus judge requests cost $4.611891. Judging credit cooldown events:
0, each waiting at least 90 seconds. Private
reasoning is not used as grading evidence.

[Protocol](PROTOCOL.md) · [Configuration](../../../../ops/nova-2-5-sonic-20261019/user-context-v2/configuration.json) ·
[Attempts (local archive)](../../README.md) · [Frozen inputs (local archive)](../../README.md) ·
[Aggregate](summary.json) · [Capture hashes (local archive)](../../README.md)

[Evidence storage and local audit instructions](../../README.md).
