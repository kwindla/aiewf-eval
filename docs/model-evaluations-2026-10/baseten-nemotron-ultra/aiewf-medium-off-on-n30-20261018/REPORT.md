# Fresh Nemotron Ultra on Baseten: thinking off and on

Collected 30 newly generated conversations per mode using Baseten's released
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`, with no historical transcripts reused.
Completed analysis: 2026-10-18T16:31:38.043439+00:00.

| Mode | API effort | Complete | Observed scripted turns | Strict, fixed 900 turns | Strict, complete conversations only | TTFAT P50 | TTFAT P95 | TTFAT max |
|---|---|---|---|---|---|---|---|---|
| off | none | 19/30 | 647/900 | 544/900 (60.4%) | 86.0% | 465ms | 771ms | 1533ms |
| on | high | 30/30 | 900/900 | 827/900 (91.9%) | 91.9% | 926ms | 2116ms | 8679ms |

The fixed denominator retains all 30 assigned conversations per mode. A complete
conversation has all 30 scripted turns with assistant text or an actual tool call
at each turn. Missing
future turns after early termination count as failures. Completion-conditional
quality uses only complete conversations and should not be read as reliability.
Provider-rejected judge inputs remain explicitly unscored, if any.

Error rates among observed scripted turns (missing future turns are additional
failures in the primary score above):

| Mode | Tool-use errors | Instruction errors | KB grounding errors |
|---|---|---|---|
| off | 3.1% | 13.3% | 6.0% |
| on | 0.1% | 4.1% | 4.9% |

Most frequent failed thinking-on turns (one-based user turn numbers; per-case
grades below contain the specific reasons):

| User turn | Question | Failed conversations |
|---|---|---|
| 5 | How many sessions are in that track? | 22/30 |
| 21 | Is there a talk by Charles Frye? | 16/30 |
| 20 | Well, since I can't access the mobile maps, can you tell me how to get to Salon 2? | 13/30 |
| 22 | Where is the second one? | 12/30 |
| 9 | Actually, I made a mistake. I will only be at the conference on June 5th. | 3/30 |

## Observed tool behavior

The seventh scripted user turn (index 6) says: "I'm particularly interested in
Voice AI and in techniques for building reliable agents." It asks for no action.
The following tool calls were recorded on that turn, independently of judge scores:

| Mode | Recorded tools on the interests turn |
|---|---|
| off | end_session: 11, request_tech_support: 3, submit_session_suggestion: 1, vote_for_session: 2 |
| on | none |

Premature end_session counts by one-based user turn were
**off: user turn 7: 11**; **on: none**. Raw transcripts remain
unchanged. These observations identify API-visible behavior; they do not isolate
the cause within model weights, chat templates, or provider tool parsing.

## Design and serving checks

The [frozen protocol](PROTOCOL.md), [configuration](../../../../ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/configuration.json), and
[schedule](../../../../ops/baseten-nemotron-ultra/aiewf-medium-off-on-n30-20261018/schedule.tsv) specify 30 randomized temporal pairs, one off and one on
conversation per pair. Collection was strictly sequential at the shared Baseten
endpoint, with temperature 1.0, top-p 0.95, max_tokens 16384, no filler, the same
30 scripted turns, and the same benchmark instructions/knowledge base/tools.
No numeric reasoning budget was requested. The API model catalog reports FP4
weights and a deployed context limit of 202800 tokens; exact provider engine and
weight revisions are not exposed by this catalog. Its snapshot is in
[smoke/model-catalog.json (local archive)](../../README.md).

Direct streaming verified zero reasoning tokens for off and separate reasoning
for on. Full-harness diagnostics are preserved in smoke/harness-off.json and
smoke/harness-on.json and excluded from the registered cohorts. Early termination
was not used to tune or rerun smokes.
There were 60 registered collection attempts.
All attempts, including exclusions, are recorded in [artifacts/attempts.tsv (local archive)](../../README.md).

The entire initial unpaced [operational pilot (local archive)](../../README.md)
was stopped after an explicit provider HTTP 429, before any judging. All its
cases were excluded together. Effective limits were verified at 120 RPM and
500,000 combined TPM. The registered restart uses the same generation settings
with three-second request spacing before latency measurement and up to three
additional HTTP 429 retries before streaming (10/20/40 seconds). Retry waits
remain in measured latency. Registered off/on explicit retry counts were
0/0;
terminal rate-limit affected conversation counts were
0/0.

## Judging and uncertainty

Every observed scripted turn was graded with Opus 5.5, medium effort, adaptive
thinking, no tools/project settings, and the frozen README text rubric.
Generating-model identity, mode, paths, and prior scores were withheld. Explicit
reasoning sections are excluded by the standard judge formatter; raw captures
remain intact. Recovery records are not additional scored turns and were omitted
from judge input, matching the existing formatter. First valid grades are kept;
credit failures wait at least 90 seconds before retrying.

Whole-conversation bootstrap 95% intervals (20,000 resamples):

- Off, fixed denominator: 48.1–72.3%.
- On, fixed denominator: 90.4–93.2%.
- On minus off: 31.4 percentage points;
  paired-conversation interval 19.2–44.1 percentage points.

Intervals account for conversation-level clustering, not judge uncertainty or
future provider changes. Per-conversation grades and source hashes are linked
below and frozen in [judging/inputs.json (local archive)](../../README.md).

## Latency and usage

TTFAT is recorded time to first answer text or tool-call token; raw TTFT is the
first stream chunk. First scripted responses are excluded; recorded recoveries
are included. Positive finite values supply P50, linear P95, and maximum. No
missing timing is replaced with zero. Early-ending runs cover fewer turns, so
latency alone must not be interpreted as a quality improvement. Counts and
complete-only latency/quality are in [aggregate.json](summary.json).

SDK-reported successful judge cost: **$12.13**. Registered-conversation
generation cost estimates, using the frozen API catalog's per-token rates and
recorded cache/usage, are $1.89 off and
$3.06 on. These are estimates, not billing
receipts, and exclude diagnostic smokes and any unrecorded failed-request usage.

## Per-conversation results

| Slot | Mode | Observed turns | Missing future turns | Strict passes / fixed 30 | TTFAT P50 | Evidence |
|---|---|---|---|---|---|---|
| 1 | on | 30 | 0 | 28/30 | 958ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 2 | off | 30 | 0 | 25/30 | 484ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 3 | off | 7 | 23 | 5/30 | 445ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 4 | on | 30 | 0 | 28/30 | 920ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 5 | on | 30 | 0 | 27/30 | 950ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 6 | off | 30 | 0 | 26/30 | 453ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 7 | off | 7 | 23 | 5/30 | 457ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 8 | on | 30 | 0 | 30/30 | 900ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 9 | on | 30 | 0 | 26/30 | 919ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 10 | off | 30 | 0 | 27/30 | 448ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 11 | on | 30 | 0 | 28/30 | 890ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 12 | off | 7 | 23 | 5/30 | 444ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 13 | on | 30 | 0 | 26/30 | 816ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 14 | off | 30 | 0 | 24/30 | 454ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 15 | off | 30 | 0 | 25/30 | 458ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 16 | on | 30 | 0 | 27/30 | 972ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 17 | off | 7 | 23 | 5/30 | 434ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 18 | on | 30 | 0 | 27/30 | 834ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 19 | on | 30 | 0 | 29/30 | 787ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 20 | off | 30 | 0 | 27/30 | 450ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 21 | on | 30 | 0 | 29/30 | 804ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 22 | off | 7 | 23 | 5/30 | 408ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 23 | on | 30 | 0 | 29/30 | 740ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 24 | off | 30 | 0 | 26/30 | 464ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 25 | on | 30 | 0 | 29/30 | 972ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 26 | off | 30 | 0 | 26/30 | 462ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 27 | on | 30 | 0 | 25/30 | 768ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 28 | off | 30 | 0 | 28/30 | 482ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 29 | on | 30 | 0 | 28/30 | 883ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 30 | off | 7 | 23 | 5/30 | 460ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 31 | on | 30 | 0 | 27/30 | 813ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 32 | off | 30 | 0 | 20/30 | 456ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 33 | on | 30 | 0 | 28/30 | 966ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 34 | off | 30 | 0 | 24/30 | 483ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 35 | on | 30 | 0 | 26/30 | 1301ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 36 | off | 7 | 23 | 5/30 | 474ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 37 | on | 30 | 0 | 29/30 | 1067ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 38 | off | 7 | 23 | 5/30 | 528ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 39 | off | 30 | 0 | 27/30 | 470ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 40 | on | 30 | 0 | 28/30 | 979ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 41 | on | 30 | 0 | 27/30 | 844ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 42 | off | 30 | 0 | 24/30 | 472ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 43 | on | 30 | 0 | 29/30 | 883ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 44 | off | 30 | 0 | 25/30 | 482ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 45 | on | 30 | 0 | 28/30 | 848ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 46 | off | 7 | 23 | 4/30 | 452ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 47 | on | 30 | 0 | 27/30 | 1064ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 48 | off | 30 | 0 | 27/30 | 465ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 49 | on | 30 | 0 | 26/30 | 1050ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 50 | off | 30 | 0 | 23/30 | 490ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 51 | on | 30 | 0 | 28/30 | 973ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 52 | off | 30 | 0 | 29/30 | 468ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 53 | on | 30 | 0 | 26/30 | 982ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 54 | off | 7 | 23 | 5/30 | 435ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 55 | off | 30 | 0 | 28/30 | 463ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 56 | on | 30 | 0 | 28/30 | 880ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 57 | on | 30 | 0 | 27/30 | 1155ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 58 | off | 7 | 23 | 5/30 | 438ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 59 | off | 30 | 0 | 29/30 | 476ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |
| 60 | on | 30 | 0 | 27/30 | 960ms | [grade (local archive)](../../README.md) · [failures (local raw runs; unarchived)](../../README.md) |

The existing README Ultra rows are unchanged pending review of this experiment.

[Evidence storage and local audit instructions](../../README.md).
