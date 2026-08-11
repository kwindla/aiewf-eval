# Muse Glimmer 30B AIEWF medium-context N=30

Date: 2026-08-10

## Result

All 30 conversations completed all 30 scripted turns. Claude Opus 4.5 judged
764/900 turns as strict passes (84.9%). This is effectively unchanged from the
earlier capped, sampler-mismatched N=25 result of 636/750 (84.8%), while the new
run uses Meta's recommended sampler, native high reasoning strength, and no
request-level output cap.

| Metric | Result |
|---|---:|
| Complete conversations | 30/30 |
| Scripted turns judged | 900/900 |
| Strict turn pass | 764/900 (84.9%) |
| Tool use | 774/900 (86.0%) |
| Instruction following | 774/900 (86.0%) |
| KB grounding | 897/900 (99.7%) |
| Per-run strict score | median 26/30, range 23-28 |
| Recovery responses | 126 |

The recurring misses were scripted tool turns 24 (30/30 runs), 12 (27/30), 15
(26/30), and 17 (26/30). Turn 11 failed in 9 runs, `end_session` on turn 29
failed in 8, turn 21 failed in 6, and turns 19 and 25 each failed twice.

Recovery responses are benchmark-injected follow-ups after a scripted turn
does not complete its expected tool action. They remain in the raw transcript
for diagnosis but are excluded from the fixed 900-turn judge and timing cohort.

## Pipecat timing

TTFT is Pipecat `raw_ttfb_ms`, the first streamed reasoning token. TTFAT is
Pipecat `ttfb_ms`, the first answer-text or tool-call event. The table uses only
the 900 scripted turns and excludes recovery responses.

| Metric | P50 | P95 | Max |
|---|---:|---:|---:|
| TTFT | 178 ms | 291 ms | 4,741 ms |
| TTFAT | 232 ms | 6,488 ms | 11,586 ms |
| Full response latency | 832 ms | 7,576 ms | 13,019 ms |

On turn 0 of each conversation, TTFT was 598 ms P50 / 625 ms P95 and TTFAT was
4,533 ms P50 / 8,652 ms P95. The first cold run accounts for the 4,741 ms
maximum turn-0 TTFT.

Compared with the earlier capped N=25 run, TTFT is essentially identical
(178/292 ms P50/P95 previously). TTFAT P50 is also unchanged (231 ms
previously), while TTFAT P95 increased from 5,791 ms to 6,488 ms, exposing more
of the native high-strength reasoning tail.

## Serving

- RTX 5090, 32,768 context, one slot, Q8_0 K/V cache, all layers on GPU.
- DFlash `--spec-draft-n-max 15`, with 22,044 MiB VRAM used at readiness.
- Native reasoning enabled with `reasoning_strength=high` and
  `--reasoning-budget -1`.
- Temperature 1.0, top-p 0.95, top-k 64, min-p 0.0.
- No request-level `max_tokens` field.
- 1,026 total model responses: 900 scripted turns plus 126 recovery responses.
- 204,517 decoded tokens at 142.39 weighted tokens/s.
- Weighted DFlash acceptance: 18.89% (151,030 / 799,500 proposals).
- Collection completed in about 29 minutes with no retries.

## Artifacts

- `included-runs.txt`: exact 30-run manifest.
- `attempts.tsv`: collection completeness and timing.
- `judging.tsv`: per-run strict score and final judge status.
- `aggregate.txt` and `aggregate.json`: standard benchmark aggregation.
- `server.log`: llama.cpp timings and DFlash acceptance.
- `logs/`: per-run collection and judge logs.
- `worker.sh`: exact sequential serving and collection commands.
- `judge.sh`: sequential judging and aggregation commands.

The companion port-to-port report is at
`/home/khkramer/src/gb-benchmarks/port-to-port/runs/muse-glimmer-30b-natural-high-card-nomax-dflash15-n25-20260810T213830Z/REPORT.md`.
