# September 2026 speech results

These are the published September cohorts, consolidated without recalculating scores. Each row contains 25 completed conversations / 750 turns. Quality is conditional on completion; incomplete prefixes are excluded and reliability is reported separately. Machine-readable values: [aggregate.json](aggregate.json). Attempt ledger and archive locations: [manifest.json](manifest.json).

| Configuration | Strict passes | Accuracy | Failed / eligible | Harness excluded | First-speech mean / median | Positive coverage |
|---|---:|---:|---:|---:|---:|---:|
| GPT-Live prompt-optimized + Terra medium* | 739/750 | 98.5% | 2/27 | 0 | 2686 / 2112 ms | 748/750 |
| Grok Voice Think Fast | 733/750 | 97.7% | 0/25 | 0 | 2655 / 2528 ms | 730/750 |
| GPT-Realtime-2.1 low | 708/750 | 94.4% | 0/25 | 0 | 1615 / 1536 ms | 750/750 |
| GPT-Live standard + Terra medium | 707/750 | 94.3% | 0/25 | 0 | 1870 / 1728 ms | 748/750 |
| GPT-Realtime-1.5 | 705/750 | 94.0% | 0/25 | 0 | 1279 / 1152 ms | 725/750 |
| Gemini 3.8 Live (regular; default) | 697/750 | 92.9% | 1/26 | 13 | 1616 / 1536 ms | 749/750 |
| Gemini 3.8 Extended Thinking low | 697/750 | 92.9% | 0/25 | 1 | 1492 / 1440 ms | 743/750 |
| Gemini 3.1 minimal | 656/750 | 87.5% | 1/26 | 0 | 1447 / 1344 ms | 721/750 |
| Gemini 3.8 Extended Thinking medium (interaction-aware) | 698/750 | 93.1% | 2/27 | 0 | 1752 / 1632 ms | 750/750 |

## Failures

Turn numbers are one-based. No-response timeouts are not classified as explicit API errors.
- GPT-Live prompt-optimized + Terra medium*: turn 11, after 10 completed turns; `TimeoutError`; `20260916T123039_gpt-live-1_5dc5eeb2`.
- GPT-Live prompt-optimized + Terra medium*: turn 11, after 10 completed turns; `TimeoutError`; `20260916T140645_gpt-live-1_fd2c66d8`.
- Gemini 3.8 Live (regular; default): turn 1, after 0 completed turns; `no_audio_timeout`; `20260916T151738_gemini-3.8-live_927feec6`.
- Gemini 3.1 minimal: turn 25, after 24 completed turns; `no_audio_timeout`; `20260915T144826_gemini-3.1-flash-live-preview_8a11ff4e`.
- Gemini 3.8 Extended Thinking medium (interaction-aware): turn 1, after 0 completed turns; `no_audio_timeout`; `20260916T074653_gemini-3.8-live-extended-thinking_c2d7a24b`.
- Gemini 3.8 Extended Thinking medium (interaction-aware): turn 1, after 0 completed turns; `no_audio_timeout`; `20260916T080944_gemini-3.8-live-extended-thinking_bd55a0b2`.

## Regular Gemini

Thirteen captures are excluded for suppressed duplicate-tool acknowledgments. The 21 completed conversations before the fix had no suppressed duplicate results; four complete validated conversations were then added. One separate turn-1 timeout remains in the failure denominator. Thinking configuration is omitted for this model. See the per-attempt exclusions and original harness review reference in the manifest.

## Measurement and judging

The optimized GPT-Live and interaction-aware Gemini transcript judges do not score turn-taking. Their `scores.turn_taking` values are null and `turn_taking_scored` is false; the historical default true values are not measurements. Their quality rubric is shared, and quantitative audio validation is separate.

First speech may be an acknowledgment. Summaries retain measurable positive latency only; signed statistics and missing/nonpositive counts remain in aggregate.json. Silence-padding values are unavailable for some cohorts and are null, never zero-filled. Earlier Gemini captures retain their documented silent-closing limitations.

Gemini extended-thinking low has one excluded startup-invalid capture. The standard Live input-window hold was resolved through waveform alignment without altering latencies. One optimized Live answer was audible but missed by VAD, so the conversation remains scored and its latency missing.

## Historical diagnostics

Earlier-harness Gemini medium, broken-gate Live captures, smoke tests, interim classifier experiments, old attempt-based aggregates, and prior speech tables remain historical evidence and are not pooled into current rows. The README’s single-validation and August Gemini reliability observations are recorded separately under `readme_numeric_notes` in aggregate.json.

The earlier fixed-gate aggregate retains a stale inherited eligibility flag; the final completed-conversation aggregate and documented alignment review supersede that intermediate flag. This consolidation uses the final aggregate for its published scores.

Historical REPORTs, protocols, frozen dependency environments, source snapshots, and per-conversation docs outputs remain in the local archive identified by manifest.json. The original optimized-prompt design is referenced there; the final design is summarized in [PROTOCOL.md](PROTOCOL.md#optimized-prompt-structure).

## Artifact availability

The maintainer-held local archive has ID `aiewf-eval-archive/2026-09-speech` and `MANIFEST.json` SHA-256 `cdb45ebc09b2671a6659ec1471d3e3ac83192498fcd5e99d90942bb7ba7efa60`. Its Git and on-disk snapshots are distinct. Off-host storage has not been selected.

The manifest lists 398 raw run directories referenced by committed campaign files (reported size 26.7 GB), plus 57 additional existing run directories referenced by archived on-disk diagnostics. Raw runs are NOT archived: their run IDs and repository-relative `runs/` paths are inventory references, pending a user decision. Some individual linked recordings/transcripts were materialized in the docs archive; that does not archive the full run directories.

The stale one-shot `publish.py` is retired and not shipped. Tests compare every current README speech-table cell to aggregate.json, validate ledger denominators, prompt hashes, and local links. No judge was rerun and no result number was changed.
