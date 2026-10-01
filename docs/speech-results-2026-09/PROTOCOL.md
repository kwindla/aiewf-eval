# September speech protocol

## Cohorts and sampling

Each published configuration scores the first 25 complete, harness-valid conversations,
30 turns each. Selection never depends on judge quality. Accuracy excludes the entire
incomplete conversation, including its observed prefix. Failed conversations remain in
the attempt ledger; model/service failure rate is failed / harness-valid attempts.
Confirmed harness defects are excluded from that denominator and disclosed separately.
Failed turn numbers are one-based in reports, with source indices preserved in evidence.

The nine published configurations contain 225 scored conversations, six model/service
failures, and 14 harness-excluded attempts. The retired earlier-harness Gemini medium
cohort is not one of those nine. The [manifest](manifest.json) records all 245 current
cohort attempts and the broader diagnostic run inventory.

Capture concurrency was capped at five sessions per provider. Recovery nudges and turn
replay were disabled. Top-ups are new logged conversations; a complete capture with an
unresolved validation issue is held for review, not replaced for a better score. Run
limits triggered review during capture; they were not performance-based selection.
The regular Gemini duplicate-result defect and the earlier startup defect are
explicitly identified in the ledger.

## Prompts and judge

The three benchmark packages share the original tasks, knowledge base, scripted audio,
and tools. `aiwf_medium_context_speech_v2` separates speech behavior instructions and
its voice rubric from the text benchmark. The standard Live package inherits that
same full frontend prompt and gives the same prompt to its Terra medium backend.
The text benchmark and README text-model scores are unchanged.

All final quality scores use `claude-opus-5` and the shared revised voice rubric:

- Standard speech prompt SHA-256: `0ff3ba3519554c8743095841a93af92c25d76ec2243d7abb4d1451578a701f9f`.
- Final judge prompt SHA-256: `c5a8a3b3c3240172cc392529a1784b59d2913a32e2ec2ba843a456415b75ef27`.

Strict turn success requires tool use, instruction following, and knowledge-base
grounding all to pass. Turn-taking is separately reported. The earlier initial judge
revision in historical capture protocols was superseded by the shared rejudge; its
hash must not be mistaken for the final scoring rubric.

For optimized Live and interaction-aware Gemini medium, the judge receives a lossless
spoken-transcript reconstruction. Interim speech remains present; backend text is
never substituted for spoken answers. Ambiguous ordering requires review. Those
reconstruction judges skip turn-taking: it is **unscored/null**, not 750 passes.
Separate waveform checks assess recorded speech boundaries and timing.

### Optimized prompt structure

`aiwf_medium_context_speech_live_optimized` preserves the task but uses a separate
frontend/backend structure. The frontend contains voice behavior and delegation
instructions; the backend contains the full knowledge base and answering instructions.
The configuration is explicitly labeled in the main table and uses the same quality
rubric. It also uses a different completion gate, so its comparison with standard Live
is not a prompt-only controlled experiment.

- Frontend SHA-256: `332a125c8fbc1a98be1302cf0dc10de22fb24004d56105d75990a9b96ed74ce0`.
- Backend SHA-256: `26a768b2c8faf9e4a27ade00b4f111a4a4e9c5cbdc27bd5a4a9992b0ba712c71`.

The final prompt text lives in the benchmark package. The original design and superseded
classifier experiments remain in the archive; no runtime classifier is used by the
published optimized cohort. This final summary replaces a second design document.

## Completion gates

- Standard GPT-Live: backend lifecycle and drained playback, with a fresh three-second
  quiet interval after user input. Backend: `gpt-5.6-terra`, medium reasoning; voice Marin.
- Optimized GPT-Live: five-second quiet interval plus evidence of frontend speech after
  backend content becomes available. This is an opportunity-to-speak heuristic, not proof
  that all backend facts were communicated. Both Live profiles use a 90-second response
  timeout. API backend completion does not mean speech playback has finished.
- Gemini 3.8 extended-thinking medium: explicit user activity, NON_BLOCKING tools,
  interaction-status IDLE, and closing playback. `turnComplete` alone does not establish
  the end of an asynchronous interaction.
- Gemini 3.8 regular: thinking configuration omitted, synchronous completion and drained
  playback, startup guard, deferred end_session, and duplicate-call acknowledgment without
  re-executing the action. It uses the existing 15-second no-audio timeout.
- Earlier Gemini low/minimal cohorts retain their capture configurations. Replacement
  captures add only the documented startup guard. They are not pooled with the newer
  interaction-aware medium configuration.

See [live speech support](../live-speech-support.md) for the exact capture profiles and
offline entry points. Use those profile-specific scripts to reproduce the published
harness behavior. The generic realtime command is not interchangeable with every profile.

## Audio metrics and validation

Voice-to-voice latency is actual user speech ending to the model's first recorded speech,
using the existing Silero detector and parameters. It includes interim acknowledgments,
not just substantive answers. Waveform correlation validates logged input windows against
scripted input recordings. Audio analysis is offline and is not paced in real time.

Published latency statistics use measurable positive observations. Signed statistics,
nonpositive counts, and missing measurements remain in [aggregate.json](aggregate.json).
Unavailable silence-padding values remain null. No missing metric is replaced with zero.
Tool/non-tool grouping follows recorded tool calls. The resolved alignment and brief-answer
VAD holds are described in [REPORT.md](REPORT.md#measurement-and-judging).

This consolidation does not rerun captures, judging, or latency analysis. It retains the
source results, display rounding, and failure accounting. README table values are checked
against the compact aggregate; ledger counts and prompt hashes are tested independently.

## Provenance and storage

The publication source is commit `3fed83e41c09c91c28f2751a82f1011542f11063`, plus
preserved on-disk campaign evidence. This is not a claim that every historical capture
ran at that commit: individual frozen protocols and run metadata record the actual code,
prompts, and dependency versions. The newer captures used Pipecat 1.10.0, google-genai
2.23.0, and OpenAI 3.14.0 where applicable. Capture protocol references remain in each
cohort's configuration and in the archive manifest.

`manifest.json` identifies the verified local archive and its checksum, compact attempt
ledger, artifact paths, and raw run paths. Paths under `git/` or `disk/` are relative to the
archive root; `runs/` paths are relative to the source repository and remain **unarchived**.
Their archival and the off-host destination are pending user decisions. No recordings or
per-conversation judge dumps are embedded in this PR.

The old one-shot campaign publisher is retired. This package contains no replacement
that rewrites scores automatically; result publication requires consistency checks.
