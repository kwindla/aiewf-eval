# Qwen3.6-27B BaseTen benchmark handoff

## Outcome

The cached APC+MTP Qwen3.6-27B deployment was tested on both benchmarks and
then scaled to zero.

The benchmarks favor different thinking settings:

| Benchmark | Thinking on / `high` | Thinking off / `none` | Practical conclusion |
|---|---:|---:|---|
| Port-to-port, N=25 per arm | 25/25 task complete; 24/25 strict success | 14/25 task complete and strict success | Native thinking is required for reliable long-horizon tool control |
| AIEWF medium, N=30 per arm | 86.0% strict turn pass; 23/30 complete | 97.3% strict turn pass; 30/30 complete | Thinking off is better for this scripted conversational benchmark |

This is not a contradiction. Port-to-port requires a long sequence of
stateful tool decisions. With thinking disabled, Qwen often narrated an
intended action instead of calling its tool: 429 of 1,191 turns (36.0%).
Native thinking reduced that to 5 of 847 turns (0.6%).

AIEWF asks mostly short factual and transactional questions in a fixed
30-turn dialogue. Thinking off handled all 30 conversations end to end.
Thinking on was slightly more accurate on turns where it actually responded
(98.5% strict pass over 786 observed turns versus 97.3% over 900), but it
failed conversation control:

- five thinking-on conversations called `end_session` after scheduled turn 15;
- two thinking-on conversations hit the 45-second idle timeout, after 2 and 14
  valid responses;
- those seven failures left 114 scheduled future turns missing.

Under the predeclared fixed-denominator policy, missing future turns fail.
That changes the thinking-on AIEWF result from strong observed-turn accuracy
to 86.0% strict pass over the full 900-turn denominator.

## AIEWF final result

Each setting contains 30 canonical conversations and 900 scheduled turns.
The collection order was frozen as 30 temporal pairs, each containing one
`high` and one `none` conversation in randomized order. Conversations were
strictly sequential; the two settings never overlapped on the H100.

| Setting | Strict pass | Cluster-bootstrap 95% CI | Complete | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|
| Native thinking on | 86.0% | 76.4–94.1% | 23/30 (76.7%) | 1,924 ms | 9,469 ms |
| Thinking off | 97.3% | 96.6–98.1% | 30/30 (100.0%) | 668 ms | 822 ms |

Thinking-on minus thinking-off strict pass was **-11.3 points**, with a paired
whole-conversation bootstrap 95% interval of **-20.8 to -3.1 points**.
Thinking-on minus thinking-off completion was **-23.3 points**, with a paired
95% interval of **-40.0 to -10.0 points**.

TTFAT is content-aware and conditional on an observed assistant response.
Missing turns remain accuracy failures and are not assigned invented latency.

## Port-to-port final result

Each setting contains 25 canonical conversations: three pilots plus 22 valid
top-up runs.

| Setting | Task complete | Strict success | Median primary score | Turn P50 | Turn P90 | Total-time P50 |
|---|---:|---:|---:|---:|---:|---:|
| Native thinking on | 25/25 (100%) | 24/25 (96%) | 90 | 1.61 s | 7.70 s | 144.8 s |
| Thinking off | 14/25 (56%) | 14/25 (56%) | 72 | 1.00 s | 2.52 s | 215.3 s |

The completion difference was +44 points for thinking on (Newcombe 95%
interval +22.1 to +62.9 points; two-sided Fisher exact p=0.000239). All 11
incomplete thinking-off runs exhausted the 50-turn cap. The evidence points
to model control under `thinking=none`, not BaseTen, vLLM, or tool-schema
breakage.

## Configuration and resource status

- Model: `Qwen/Qwen3.6-27B`
- Provider: BaseTen
- Accelerator: one H100, BF16
- Runtime: vLLM 0.26
- Prefix caching: enabled with aligned Mamba cache mode
- MTP speculative decoding: two tokens
- AIEWF sampling: temperature 0.6, top-p 0.95, max tokens 8,192
- AIEWF filler: none
- Deployment `wxpnlg5`: confirmed `SCALED_TO_ZERO`, zero replicas

No OpenAI pro model was tested or used as a judge. The AIEWF judge was
`claude-opus-4-5`; the port-to-port judge was `claude-sonnet-4-6`.

## Artifacts

### AIEWF

- `canonical.tsv`: exact 60-run manifest
- `judging/COMPLETE.json`: completed judging marker
- `analysis/REPORT.md`: concise fixed-denominator result
- `analysis/aggregates.json`: machine-readable protocol, results, effects, and
  per-run audit records
- `analysis/aggregates.tsv`: arm-level summary
- `analysis/effects.tsv`: paired high-minus-none effects
- `analysis/included-runs.tsv`: exact run membership and artifact hashes

### Port-to-port

The canonical port-to-port campaign is in
`../gb-benchmarks/port-to-port/runs/qwen36-baseten-vllm026-apc-mtp-n30-20260728T083618Z/`.

- `canonical-n25-per-setting.paths`: exact 50-run manifest
- `eval-canonical-n25-per-setting/table.md`: published score table
- `eval-canonical-n25-per-setting/aggregate.json`: machine-readable aggregate
- `eval-canonical-n25-per-setting/enriched_runs.jsonl`: judged per-run records
- `analysis-canonical-n25-per-setting.md`: full failure-pattern analysis
