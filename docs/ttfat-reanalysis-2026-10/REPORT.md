# Historical TTFAT reanalysis — October 2026

This reanalysis excludes each conversation’s first scripted response. Accuracy, cohort membership, recovery eligibility, and percentile formulas retain each campaign’s definitions. Baseline: `ea9ed5a`; reproduction target: `9fcb3a9`. No captures or judging were rerun.

**Coverage limitation:** all 47 README text rows (141 latency cells) were audited. Thirty rows have traceable v2 values; four retain the existing documented Nemotron exceptions; thirteen older rows lack an established exact published cohort in this audit and are unchanged. These thirteen are not certified as v2. See the row ledger below.

## Reproduction

The 14 campaign Python changes in the source comprise 12 analysis files and two campaign test files. The capture-capable bakeoff module is imported only for its offline summarizer. All twelve analysis paths were exercised, including dependent comparison/report generation.

Numeric JSON aggregates reproduce the source. Non-byte-identical JSON differences are timestamps, executing-source and derived artifact hashes, and equivalent paired-Gemma method metadata. Source Kimi COMPLETE.json report hashes were stale; regeneration repairs them. Text differences are rendering whitespace and explicit v2 scope wording; exact per-file diffs are in [manifest.json](manifest.json).

The paired-Gemma direct CLI refuses the changed source hash. Its offline reanalysis path verifies every original source SHA against historical Git blobs, retains cohort/configuration/transcript/judgment checks, and records the current analyzer separately in [paired-gemma-provenance.json](paired-gemma-provenance.json). Frozen source manifests are unchanged.

The shared helper `multi_turn_eval.latency_reporting.include_response_latency` implements only turn-zero selection. Scripted-only campaigns keep recovery excluded; Muse and the generic summaries retain recovery. The Qwen/Kimi/pooled-Gemma validation and percentile methods remain explicit.

| Output | Matches 9fcb3a9 bytes | Analyzer |
|---|---|---|
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | yes | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/REPORT.md` | yes | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/aggregates.json` | no; see manifest diff | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/aggregates.tsv` | yes | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | yes | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | no; see manifest diff | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-31b-sglang/README.md` | no; see manifest diff | `scripts/render_ttfat_reanalysis_prose.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/README.md` | no; see manifest diff | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | yes | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | yes | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | yes | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/aggregates.json` | yes | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | no; see manifest diff | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/COMPLETE.json` | no; see manifest diff | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/REPORT.md` | yes | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.json` | no; see manifest diff | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.tsv` | yes | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPARISON.md` | yes | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPLETE.json` | no; see manifest diff | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | yes | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | no; see manifest diff | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.tsv` | yes | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | no; see manifest diff | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | yes | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | yes | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.tsv` | yes | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/effects.tsv` | yes | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/included-runs.tsv` | yes | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | no; see manifest diff | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | yes | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | yes | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | no; see manifest diff | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/aggregate.json` | yes | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/aggregate.txt` | no; see manifest diff | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | yes | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-high.json` | yes | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-high.txt` | no; see manifest diff | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-low.json` | yes | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-low.txt` | no; see manifest diff | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-medium.json` | yes | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-medium.txt` | no; see manifest diff | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-xhigh.json` | yes | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-xhigh.txt` | no; see manifest diff | `scripts/benchmark_summary.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | yes | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |

## README cell audit

Each triplet is P50 / P95 / max in milliseconds. Run IDs and source paths are recorded in [readme-audit.json](readme-audit.json). Existing v2 outputs retain their campaign estimator; raw cohort audits use the frozen scripted-only linear-P95 convention.

| Model | Before | After | Status | Source |
|---|---|---|---|---|
| nemotron-3-ultra (128) | 541 / 712 / 1302 | 541 / 712 / 1302 | legacy exception | exact cohort not established / historical exception |
| claude-sonnet-4-6 | 850 / 4126 / 9396 | 850 / 4126 / 9396 | unresolved cohort provenance | exact cohort not established / historical exception |
| claude-fable-5 (low) | 3535 / 5148 / 8815 | 3582 / 5148 / 8815 | updated | `docs/ten-run-allowlists/claude-fable-5-low-2026-06-09.txt` |
| claude-fable-5 (default) | 3956 / 6496 / 13602 | 3966 / 6680 / 13602 | updated | `docs/ten-run-allowlists/claude-fable-5-default-2026-06-09.txt` |
| glm-5.2 (none) | 936 / 2140 / 7567 | 924 / 2091 / 7567 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| glm-5.3 (low reasoning) | 2376 / 4761 / 8007 | 2376 / 4761 / 8007 | verified | `ops/zai-glm-5.3/aiewf-medium-low-n30-20260819/run-dirs.txt` |
| nemotron-3-ultra (96) | 529 / 655 / 1259 | 529 / 655 / 1259 | legacy exception | exact cohort not established / historical exception |
| kimi-k2.6 (thinking on) | 1560 / 5404 / 13622 | 1560 / 5404 / 13622 | verified | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` |
| qwen3.8-27b (thinking off, FP8) | 649 / 801 / 2161 | 649 / 801 / 2161 | verified | `ops/baseten-qwen38-27b-sglang/aiewf-medium-none-n30-20260816/analysis/aggregates.json` |
| claude-haiku-4-5 | 637 / 1615 / 3152 | 637 / 1615 / 3152 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-5.1 | 739 / 1492 / 4244 | 739 / 1492 / 4244 | unresolved cohort provenance | exact cohort not established / historical exception |
| qwen3.8-27b (thinking off, NVFP4) | 101 / 318 / 592 | 101 / 318 / 592 | verified | `ops/local-qwen38-27b-nvfp4-sglang/aiewf-medium-none-n30-20260816/analysis/aggregates.json` |
| gpt-5.6-terra (medium) | 927 / 2149 / 4167 | 927 / 2149 / 4167 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-5.5 (none) | 875 / 2177 / 5623 | 868 / 2197 / 5623 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| qwen3.6-27b (thinking off) | 667 / 769 / 1920 | 667 / 769 / 1920 | verified | `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` |
| deepseek-v4-pro-0813 (low) | 752 / 1477 / 3545 | 752 / 1477 / 3545 | verified | `ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json` |
| gemini-3.6-flash (minimal) | 798 / 984 / 1472 | 799 / 985 / 1472 | updated | `docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |
| nemotron-3-super-120b (512) | 687 / 1210 / 2254 | 687 / 1210 / 2254 | legacy exception | exact cohort not established / historical exception |
| gpt-5.4 (low) | 782 / 1706 / 2698 | 782 / 1706 / 2698 | unresolved cohort provenance | exact cohort not established / historical exception |
| deepseek-v4-flash-0731 (low) | 677 / 1452 / 4687 | 677 / 1444 / 4687 | updated | `ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json` |
| gemma-4-31b-it (thinking off) | 489 / 609 / 38250 | 489 / 609 / 38250 | verified | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` |
| gpt-5.6-sol (none) | 1098 / 2625 / 6344 | 1090 / 2619 / 6344 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| gpt-4.1 | 536 / 1771 / 5056 | 536 / 1771 / 5056 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-5.4 (none, +96 dots) | 694 / 2273 / 17264 | 688 / 2210 / 17264 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| inkling (none) | 447 / 727 / 1813 | 445 / 652 / 1158 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| gpt-4o | 546 / 1369 / 4897 | 546 / 1369 / 4897 | unresolved cohort provenance | exact cohort not established / historical exception |
| kimi-k2.6 (thinking off) | 475 / 842 / 4458 | 475 / 842 / 4458 | verified | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.json` |
| deepseek-v4-flash-0731 (high) | 763 / 1871 / 8702 | 762 / 1871 / 8702 | updated | `ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json` |
| nemotron-3.5-lightning (thinking on, NVFP4) | 1464 / 5787 / 29869 | 1464 / 5786 / 29869 | updated | `ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811/artifacts/analysis/aggregates.json` |
| gemini-3.5-flash (minimal) | 892 / 1183 / 1721 | 892 / 1183 / 1721 | verified | `docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |
| claude-sonnet-5 | 1204 / 2465 / 6955 | 1200 / 2524 / 6955 | updated | `docs/ten-run-allowlists/claude-sonnet-5-disabled-2026-07-01.txt` |
| qwen3.6-35b-a3b (thinking off, FP8) | 764 / 1233 / 35664 | 764 / 1233 / 35664 | verified | `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` |
| gpt-5.6-terra (none) | 621 / 1870 / 5665 | 614 / 1855 / 5665 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| nemotron-3-nano-30b (512) | 940 / 1912 / 2821 | 940 / 1912 / 2821 | legacy exception | exact cohort not established / historical exception |
| gpt-5.4 (none) | 689 / 1723 / 6571 | 682 / 1680 / 6571 | updated | `docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| gemini-2.5-flash (thinking off) | 550 / 850 / 2352 | 548 / 848 / 2352 | updated | `docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/aggregates.json` |
| gpt-5.2 | 624 / 1171 / 2509 | 624 / 1171 / 2509 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-5.6-luna (none) | 671 / 2304 / 12017 | 671 / 2304 / 12017 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-oss-120b (groq) | 98 / 217 / 2117 | 98 / 217 / 2117 | unresolved cohort provenance | exact cohort not established / historical exception |
| muse-glimmer-30b (thinking low, GGUF) | 231 / 1752 / 5474 | 231 / 1752 / 5474 | verified | `runs/muse-glimmer-reasoning-strength-n30-20260811/aggregate-low.json` |
| poolside/laguna-s-2.1 (thinking off) | 295 / 620 / 21032 | 293 / 580 / 3140 | updated | `docs/filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json` |
| gpt-4.1-mini | 851 / 2135 / 5945 | 851 / 2135 / 5945 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-5-mini | 682 / 1132 / 1904 | 682 / 1132 / 1904 | unresolved cohort provenance | exact cohort not established / historical exception |
| gpt-4o-mini | 553 / 1947 / 6497 | 553 / 1947 / 6497 | unresolved cohort provenance | exact cohort not established / historical exception |
| gemma-4-26b-a4b-it (thinking off) | 578 / 634 / 31574 | 578 / 634 / 31574 | verified | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` |
| gemini-3.5-flash-lite (minimal) | 591 / 679 / 928 | 591 / 677 / 928 | updated | `docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |
| nemotron-3.5-lightning (thinking off, NVFP4) | 62 / 70 / 80 | 62 / 70 / 80 | verified | `ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811/artifacts/analysis/aggregates.json` |

## Changed published numbers

One row per changed JSON scalar or rendered table number; repeated publications are listed separately. The artifact and metric path identify the campaign and cell. The artifact’s run-path list in manifest.json, or its adjacent canonical manifest, identifies the input cohort. Raw sample arrays are evidence, not table statistics, and are not expanded into one row per sample.

| Campaign artifact | Metric | Before (v1) | After (v2) | Analyzer |
|---|---|---:|---:|---|
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/dots96/ttfat_observations` | 180 | 174 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/dots96/ttfat_p50_ms` | 682.5 | 680.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/dots96/ttfat_p95_ms` | 846.3499999999996 | 808.4499999999999 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/nofiller/ttfat_observations` | 900 | 870 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/nofiller/ttfat_p50_ms` | 668.0 | 667.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/arms/nofiller/ttfat_p95_ms` | 822.0 | 768.55 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/decision_control/ttfat_observations` | 300 | 290 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/decision_control/ttfat_p50_ms` | 667.0 | 664.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_27b/decision_control/ttfat_p95_ms` | 828.45 | 774.6500000000001 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/dots96/ttfat_observations` | 862 | 832 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/dots96/ttfat_p50_ms` | 667.5 | 666.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/dots96/ttfat_p95_ms` | 852.8499999999998 | 812.4499999999999 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/nofiller/ttfat_observations` | 900 | 870 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/nofiller/ttfat_p50_ms` | 765.0 | 764.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/arms/nofiller/ttfat_p95_ms` | 1254.6999999999994 | 1232.6999999999985 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/decision_control/ttfat_observations` | 300 | 290 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/decision_control/ttfat_p50_ms` | 775.0 | 773.0 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `docs/filler-study-data/qwen36-dots-2026-07-28/aggregates.json` | `/models/qwen36_35b/decision_control/ttfat_p95_ms` | 1659.300000000001 | 1702.300000000001 | `docs/filler-study-data/qwen36-dots-2026-07-28/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/aggregates.json` | `/aggregate/ttfat_ms_observed_responses_only/count` | 900 | 870 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/aggregates.json` | `/aggregate/ttfat_ms_observed_responses_only/p50` | 597.0 | 596.0 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/aggregates.json` | `/aggregate/ttfat_ms_observed_responses_only/p95` | 670.0 | 635.0 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/dots96/ttfat_ms_observed_responses_only/count` | 882 | 852 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/dots96/ttfat_ms_observed_responses_only/p50` | 582.0 | 581.0 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/dots96/ttfat_ms_observed_responses_only/p95` | 812.0 | 638.8999999999999 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/nofiller/ttfat_ms_observed_responses_only/count` | 894 | 864 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/nofiller/ttfat_ms_observed_responses_only/p50` | 579.5 | 578.0 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/aggregates-full.json` | `/arms/nofiller/ttfat_ms_observed_responses_only/p95` | 802.6999999999998 | 633.5499999999997 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/extension_n120/count` | 3600 | 3480 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/extension_n120/p50_ms` | 495.0 | 494.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/extension_n120/p95_ms` | 778.0 | 615.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/original_n30/count` | 900 | 870 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/original_n30/max_ms` | 3718.0 | 1283.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/original_n30/p50_ms` | 430.0 | 429.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/cohort_latency/original_n30/p95_ms` | 564.1499999999999 | 539.55 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/latency/count` | 4500 | 4350 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/latency/p50_ms` | 490.0 | 489.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json` | `/latency/p95_ms` | 718.1500000000005 | 608.5500000000002 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/aggregates.json` | `/latency/count` | 900 | 870 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/aggregates.json` | `/latency/max_ms` | 3718.0 | 1283.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/aggregates.json` | `/latency/p50_ms` | 430.0 | 429.0 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/aggregates.json` | `/latency/p95_ms` | 564.1499999999999 | 539.55 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/p50` | 479.5 | 475.0 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/p95` | 854.0 | 842.0 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/raw_ttfb_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/raw_ttfb_ms/p50` | 666.0 | 657.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/raw_ttfb_ms/p95` | 1146.0 | 1132.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/reasoning_delay_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/reasoning_delay_ms/p50` | 854.0 | 825.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/reasoning_delay_ms/p95` | 4586.0 | 4726.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/p50` | 1595.5 | 1560.5 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/aggregates.json` | `/ttfat_ms/p95` | 5252.0 | 5404.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_off/ttfat_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_off/ttfat_ms/p50` | 479.5 | 475.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_off/ttfat_ms/p95` | 854.0 | 842.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/raw_ttfb_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/raw_ttfb_ms/p50` | 666.0 | 657.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/raw_ttfb_ms/p95` | 1146.0 | 1132.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/reasoning_delay_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/reasoning_delay_ms/p50` | 854.0 | 825.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/reasoning_delay_ms/p95` | 4586.0 | 4726.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/ttfat_ms/count` | 900 | 870 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/ttfat_ms/p50` | 1595.5 | 1560.5 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/thinking_on/ttfat_ms/p95` | 5252.0 | 5404.0 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/comparison.json` | `/ttfat_ratio_on_over_off` | 3.327424400417101 | 3.285263157894737 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/high/ttfat/coverage_of_fixed_turn_denominator_pct` | 87.33333333333333 | 84.0 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/high/ttfat/observations` | 786 | 756 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/high/ttfat/p50_ms` | 1924.0 | 1841.5 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/high/ttfat/p95_ms` | 9468.5 | 9481.25 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/none/ttfat/coverage_of_fixed_turn_denominator_pct` | 100.0 | 96.66666666666667 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/none/ttfat/max_ms` | 5429.0 | 1920.0 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/none/ttfat/observations` | 900 | 870 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/none/ttfat/p50_ms` | 668.0 | 667.0 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/arms/none/ttfat/p95_ms` | 822.0 | 768.55 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/high_p50_ms` | 1924.0 | 1841.5 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/high_p95_ms` | 9468.5 | 9481.25 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/none_p50_ms` | 668.0 | 667.0 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/none_p95_ms` | 822.0 | 768.55 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/p50_high_minus_none_ms` | 1256.0 | 1174.5 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/effects_high_minus_none/ttfat/p95_high_minus_none_ms` | 8646.5 | 8712.7 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/0/ttfat_observations` | 16 | 15 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/1/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/2/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/3/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/4/ttfat_observations` | 2 | 1 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/5/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/6/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/7/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/8/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/9/ttfat_observations` | 14 | 13 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/10/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/11/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/12/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/13/ttfat_observations` | 16 | 15 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/14/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/15/ttfat_observations` | 16 | 15 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/16/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/17/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/18/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/19/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/20/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/21/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/22/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/23/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/24/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/25/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/26/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/27/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/28/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/29/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/30/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/31/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/32/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/33/ttfat_observations` | 16 | 15 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/34/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/35/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/36/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/37/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/38/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/39/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/40/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/41/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/42/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/43/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/44/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/45/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/46/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/47/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/48/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/49/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/50/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/51/ttfat_observations` | 16 | 15 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/52/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/53/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/54/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/55/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/56/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/57/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/58/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/aggregates.json` | `/included_runs/59/ttfat_observations` | 30 | 29 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/extension/summary/latency/count` | 3600 | 3480 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/extension/summary/latency/max_ms` | 3893.0 | 415.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/extension/summary/latency/p50_ms` | 129.0 | 128.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/extension/summary/latency/p95_ms` | 337.0 | 319.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/frozen/summary/latency/count` | 900 | 870 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/frozen/summary/latency/max_ms` | 3757.0 | 409.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/campaigns/frozen/summary/latency/p95_ms` | 326.04999999999995 | 308.54999999999995 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/pooled/latency/count` | 4500 | 4350 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/pooled/latency/max_ms` | 3893.0 | 415.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/pooled/latency/p50_ms` | 128.0 | 127.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/bf16_kv/pooled/latency/p95_ms` | 336.0 | 318.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/extension/summary/latency/count` | 3600 | 3480 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/extension/summary/latency/max_ms` | 3280.0 | 413.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/extension/summary/latency/p95_ms` | 310.0 | 291.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/frozen/summary/latency/count` | 900 | 870 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/frozen/summary/latency/max_ms` | 3136.0 | 390.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/campaigns/frozen/summary/latency/p95_ms` | 295.0 | 281.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/pooled/latency/count` | 4500 | 4350 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/pooled/latency/max_ms` | 3280.0 | 413.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/arms/fp8_kv/pooled/latency/p95_ms` | 309.0 | 290.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/baseten_reference/latency/count` | 4500 | 4350 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/baseten_reference/latency/p50_ms` | 490.0 | 489.0 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/aggregates.json` | `/baseten_reference/latency/p95_ms` | 718.1500000000005 | 608.5500000000002 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/raw_ttft_ms/max` | 4711.0 | 338.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/raw_ttft_ms/median` | 181.0 | 179.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/raw_ttft_ms/p95` | 286.0 | 268.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/reasoning_delay_ms/p95` | 6481.0 | 6037.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/response_latency_ms/max` | 18093.0 | 16971.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/response_latency_ms/median` | 861.5 | 811.5 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/response_latency_ms/p95` | 8666.0 | 7354.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/ttfat_ms/max` | 16626.0 | 15594.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/ttfat_ms/median` | 236.0 | 234.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/high/ttfat_ms/p95` | 7719.0 | 6176.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/raw_ttft_ms/max` | 4713.0 | 354.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/raw_ttft_ms/median` | 177.0 | 176.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/raw_ttft_ms/p95` | 275.0 | 252.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/reasoning_delay_ms/p95` | 1906.0 | 1732.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/response_latency_ms/max` | 9017.0 | 6597.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/response_latency_ms/median` | 804.0 | 778.5 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/response_latency_ms/p95` | 3976.0 | 2727.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/ttfat_ms/max` | 7713.0 | 5474.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/ttfat_ms/median` | 231.0 | 228.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/low/ttfat_ms/p95` | 3105.0 | 1938.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/raw_ttft_ms/max` | 4734.0 | 336.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/raw_ttft_ms/median` | 181.0 | 179.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/raw_ttft_ms/p95` | 290.0 | 274.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/response_latency_ms/max` | 12059.0 | 8967.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/response_latency_ms/median` | 854.0 | 833.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/response_latency_ms/p95` | 6910.0 | 5786.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/ttfat_ms/max` | 9432.0 | 7510.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/ttfat_ms/median` | 236.0 | 234.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/medium/ttfat_ms/p95` | 5958.0 | 4632.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/raw_ttft_ms/max` | 4710.0 | 344.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/raw_ttft_ms/median` | 179.0 | 178.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/raw_ttft_ms/p95` | 285.0 | 268.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/reasoning_delay_ms/p95` | 6147.0 | 6138.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/response_latency_ms/max` | 14635.0 | 14518.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/response_latency_ms/median` | 837.5 | 814.5 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/response_latency_ms/p95` | 8781.0 | 7046.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/ttfat_ms/median` | 234.0 | 232.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/analysis.json` | `/arms/xhigh/ttfat_ms/p95` | 7767.0 | 6358.0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 1` | 597 | 596 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 2` | 670 | 635 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/REPORT.md` | `gemma-4-26b-a4b-it (thinking off) / cell 7, number 1` | 597 | 596 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/analysis/REPORT.md` | `gemma-4-26b-a4b-it (thinking off) / cell 8, number 1` | 670 | 635 | `ops/baseten-gemma4-26b-a4b-vllm/analysis/analyze.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 12 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 30 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 100.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 21 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 30 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 100.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 24 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 30 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 100.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 15 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 29 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 96.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 17 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 28 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 93.3 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 19 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 18 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 60.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 11 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 3 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 10.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 25 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 3, number 1` | 2 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 4, number 1` | 6.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `no filler / cell 2, number 1` | 26 | 27 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 21 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 30 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 100.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 24 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 30 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 100.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 12 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 29 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 96.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 15 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 27 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 90.0 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 17 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 25 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 83.3 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 19 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 20 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 66.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 11 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 8 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 26.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 13 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 3, number 1` | 2 | 1 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 4, number 1` | 6.7 | 3.3 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/REPORT-full.md` | `+96 dots / cell 2, number 1` | 14 | 16 | `ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analyze_stage.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/README.md` | `4,346/4,500 (96.58%) / cell 3, number 1` | 490 | 489 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/README.md` | `4,346/4,500 (96.58%) / cell 4, number 1` | 718 | 609 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 1` | 490 | 489 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 2` | 718 | 609 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Original / cell 2, number 1` | 900 | 870 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Original / cell 3, number 1` | 430 | 429 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Original / cell 4, number 1` | 564 | 540 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Original / cell 5, number 1` | 3718 | 1283 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Extension / cell 2, number 1` | 3600 | 3480 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Extension / cell 3, number 1` | 495 | 494 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Extension / cell 4, number 1` | 778 | 615 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Pooled / cell 2, number 1` | 4500 | 4350 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Pooled / cell 3, number 1` | 490 | 489 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/REPORT.md` | `Pooled / cell 4, number 1` | 718 | 609 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 1` | 430 | 429 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 2` | 564 | 540 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `TTFAT P50 / P95 / max / cell 2, number 3` | 3718 | 1283 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `gemma-4-31b-it (thinking off) / cell 7, number 1` | 430 | 429 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `gemma-4-31b-it (thinking off) / cell 8, number 1` | 564 | 540 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/REPORT.md` | `gemma-4-31b-it (thinking off) / cell 9, number 1` | 3718 | 1283 | `ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang NEXTN/MTP / cell 4, number 1` | 422 | 420 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang NEXTN/MTP / cell 5, number 1` | 537 | 521 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang NEXTN/MTP / cell 6, number 1` | 3181 | 567 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `vLLM APC, no MTP / cell 4, number 1` | 434 | 429 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `vLLM APC, no MTP / cell 5, number 1` | 648 | 550 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `vLLM APC, no MTP / cell 6, number 1` | 2374 | 968 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang APC, no MTP / cell 4, number 1` | 496 | 495 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang APC, no MTP / cell 5, number 1` | 728 | 647 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-gemma4-31b-sglang/bakeoff-20260806-normalized/REPORT.md` | `SGLang APC, no MTP / cell 6, number 1` | 4559 | 2731 | `ops/baseten-gemma4-31b-sglang/run_bakeoff.py:summarize_run` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/REPORT.md` | `Scripted-turn TTFAT P50 / P95 / max / cell 2, number 1` | 480 | 475 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/REPORT.md` | `Scripted-turn TTFAT P50 / P95 / max / cell 2, number 2` | 854 | 842 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/REPORT.md` | `kimi-k2.6 (thinking off) / cell 7, number 1` | 480 | 475 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/REPORT.md` | `kimi-k2.6 (thinking off) / cell 8, number 1` | 854 | 842 | `ops/baseten-kimi-k2.6/aiewf-medium-none-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPARISON.md` | `TTFAT P50 / cell 2, number 1` | 1596 | 1560 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPARISON.md` | `TTFAT P50 / cell 3, number 1` | 480 | 475 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPARISON.md` | `TTFAT P95 / cell 2, number 1` | 5252 | 5404 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/COMPARISON.md` | `TTFAT P95 / cell 3, number 1` | 854 | 842 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/compare_arms.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Scripted-turn TTFAT P50 / P95 / max / cell 2, number 1` | 1596 | 1560 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Scripted-turn TTFAT P50 / P95 / max / cell 2, number 2` | 5252 | 5404 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Raw first-chunk TTFB P50 / P95 / max / cell 2, number 1` | 666 | 657 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Raw first-chunk TTFB P50 / P95 / max / cell 2, number 2` | 1146 | 1132 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Reasoning delay P50 / P95 / max / cell 2, number 1` | 854 | 825 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `Reasoning delay P50 / P95 / max / cell 2, number 2` | 4586 | 4726 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `kimi-k2.6 (thinking on) / cell 7, number 1` | 1596 | 1560 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/REPORT.md` | `kimi-k2.6 (thinking on) / cell 8, number 1` | 5252 | 5404 | `ops/baseten-kimi-k2.6/aiewf-medium-thinking-n30-20260806/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `Arm / cell 4, number 1` | 29 | 50 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 2, number 1` | 76.7 | 756 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 4, number 1` | 76.7 | 1842 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 6, number 1` | 114 | 18023 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 2, number 1` | 100.0 | 870 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 4, number 1` | 100.0 | 667 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 6, number 1` | 0 | 1920 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 2, number 1` | 786 | 756 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 3, number 1` | 87.3 | 84.0 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 4, number 1` | 1924 | 1842 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `High / native thinking-on / cell 5, number 1` | 9468 | 9481 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 2, number 1` | 900 | 870 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 3, number 1` | 100.0 | 96.7 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 4, number 1` | 668 | 667 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 5, number 1` | 822 | 769 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/REPORT.md` | `None / thinking-off / cell 6, number 1` | 5429 | 1920 | `ops/baseten-qwen36-27b-vllm/aiewf-medium-qwen36-baseten-vllm026-apc-mtp-n30-20260728T110824Z/analysis/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | `BaseTen BF16 weights/KV + MTP / cell 4, number 1` | 490 | 489 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | `BaseTen BF16 weights/KV + MTP / cell 4, number 2` | 718 | 609 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | `Local NVFP4 + FP8 KV / cell 4, number 2` | 309 | 290 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | `Local NVFP4 + BF16 KV / cell 4, number 1` | 128 | 127 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/README.md` | `Local NVFP4 + BF16 KV / cell 4, number 2` | 336 | 318 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | `BaseTen BF16 weights/KV + MTP / cell 7, number 1` | 490 | 489 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | `BaseTen BF16 weights/KV + MTP / cell 8, number 1` | 718 | 609 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | `Local FP8 KV / cell 8, number 1` | 309 | 290 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | `Local BF16 KV / cell 7, number 1` | 128 | 127 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/REPORT.md` | `Local BF16 KV / cell 8, number 1` | 336 | 318 | `ops/local-gemma4-31b-nvfp4-sglang/pooled-n150-analysis-20260807/analyze.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `TTFT / cell 2, number 1` | 178 | 176 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `TTFT / cell 3, number 1` | 291 | 271 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `TTFT / cell 4, number 1` | 4,741 | 333 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `TTFAT / cell 2, number 1` | 232 | 229 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `TTFAT / cell 3, number 1` | 6,488 | 5,957 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `Full response latency / cell 2, number 1` | 832 | 806 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-card-high-nomax-dflash15-32k-n30-20260810T214000Z/REPORT.md` | `Full response latency / cell 3, number 1` | 7,576 | 6,914 | `scripts/reanalyze_historical_ttfat.py muse / scripts/render_ttfat_reanalysis_prose.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 5, number 1` | 231 | 4 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 5, number 2` | 3105 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 6, number 1` | 30 | 0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 6, number 2` | 1906 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 5, number 1` | 236 | 8 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 5, number 2` | 5958 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 6, number 1` | 30 | 0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 6, number 2` | 4456 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 5, number 1` | 236 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 5, number 2` | 7719 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 6, number 1` | 30 | 0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 6, number 2` | 6481 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 5, number 1` | 234 | 11 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 5, number 2` | 7767 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 6, number 1` | 30 | 0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 6, number 2` | 6147 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 3, number 1` | 24 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 4, number 1` | 7 | 2 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 4, number 2` | 24 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 5, number 1` | 17 | 4 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 5, number 2` | 24 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `low / cell 6, number 2` | 24 | 6 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 3, number 1` | 21 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 4, number 1` | 5 | 1 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 4, number 2` | 21 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 5, number 1` | 16 | 8 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 5, number 2` | 21 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `medium / cell 6, number 2` | 21 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 3, number 1` | 20 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 4, number 1` | 5 | 1 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 4, number 2` | 20 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 5, number 1` | 15 | 9 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 5, number 2` | 20 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `high / cell 6, number 2` | 20 | 10 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 3, number 1` | 17 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 4, number 1` | 4 | 2 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 4, number 2` | 17 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 5, number 1` | 12 | 11 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 5, number 2` | 17 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 6, number 1` | 1 | 0 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `runs/muse-glimmer-reasoning-strength-n30-20260811/REPORT.md` | `xhigh / cell 6, number 2` | 17 | 13 | `ops/local-muse-glimmer-reasoning-sweep-20260811/analyze.py` |
| `README.md` | `claude-fable-5 (low) / TTFAT P50` | 3535 | 3582 | `scripts/audit_readme_ttfat.py; source: docs/ten-run-allowlists/claude-fable-5-low-2026-06-09.txt` |
| `README.md` | `claude-fable-5 (default) / TTFAT P50` | 3956 | 3966 | `scripts/audit_readme_ttfat.py; source: docs/ten-run-allowlists/claude-fable-5-default-2026-06-09.txt` |
| `README.md` | `claude-fable-5 (default) / TTFAT P95` | 6496 | 6680 | `scripts/audit_readme_ttfat.py; source: docs/ten-run-allowlists/claude-fable-5-default-2026-06-09.txt` |
| `README.md` | `glm-5.2 (none) / TTFAT P50` | 936 | 924 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `glm-5.2 (none) / TTFAT P95` | 2140 | 2091 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.5 (none) / TTFAT P50` | 875 | 868 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.5 (none) / TTFAT P95` | 2177 | 2197 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gemini-3.6-flash (minimal) / TTFAT P50` | 798 | 799 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |
| `README.md` | `gemini-3.6-flash (minimal) / TTFAT P95` | 984 | 985 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |
| `README.md` | `deepseek-v4-flash-0731 (low) / TTFAT P95` | 1452 | 1444 | `scripts/audit_readme_ttfat.py; source: ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json` |
| `README.md` | `gpt-5.6-sol (none) / TTFAT P50` | 1098 | 1090 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.6-sol (none) / TTFAT P95` | 2625 | 2619 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.4 (none, +96 dots) / TTFAT P50` | 694 | 688 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.4 (none, +96 dots) / TTFAT P95` | 2273 | 2210 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `inkling (none) / TTFAT P50` | 447 | 445 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `inkling (none) / TTFAT P95` | 727 | 652 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `inkling (none) / TTFAT max` | 1813 | 1158 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `deepseek-v4-flash-0731 (high) / TTFAT P50` | 763 | 762 | `scripts/audit_readme_ttfat.py; source: ops/baseten-deepseek-v4/aiewf-medium-3arm-n30-20260817/analysis/aggregates.json` |
| `README.md` | `nemotron-3.5-lightning (thinking on, NVFP4) / TTFAT P95` | 5787 | 5786 | `scripts/audit_readme_ttfat.py; source: ops/local-nemotron35-lightning-sglang/aiewf-medium-binary-n30-20260811/artifacts/analysis/aggregates.json` |
| `README.md` | `claude-sonnet-5 / TTFAT P50` | 1204 | 1200 | `scripts/audit_readme_ttfat.py; source: docs/ten-run-allowlists/claude-sonnet-5-disabled-2026-07-01.txt` |
| `README.md` | `claude-sonnet-5 / TTFAT P95` | 2465 | 2524 | `scripts/audit_readme_ttfat.py; source: docs/ten-run-allowlists/claude-sonnet-5-disabled-2026-07-01.txt` |
| `README.md` | `gpt-5.6-terra (none) / TTFAT P50` | 621 | 614 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.6-terra (none) / TTFAT P95` | 1870 | 1855 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.4 (none) / TTFAT P50` | 689 | 682 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gpt-5.4 (none) / TTFAT P95` | 1723 | 1680 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/dot-stability-n30-2026-07-20/analyze.py` |
| `README.md` | `gemini-2.5-flash (thinking off) / TTFAT P50` | 550 | 548 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/aggregates.json` |
| `README.md` | `gemini-2.5-flash (thinking off) / TTFAT P95` | 850 | 848 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/aggregates.json` |
| `README.md` | `poolside/laguna-s-2.1 (thinking off) / TTFAT P50` | 295 | 293 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json` |
| `README.md` | `poolside/laguna-s-2.1 (thinking off) / TTFAT P95` | 620 | 580 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json` |
| `README.md` | `poolside/laguna-s-2.1 (thinking off) / TTFAT max` | 21032 | 3140 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/laguna-s21-openrouter-2026-07-22/aggregates.json` |
| `README.md` | `gemini-3.5-flash-lite (minimal) / TTFAT P95` | 679 | 677 | `scripts/audit_readme_ttfat.py; source: docs/filler-study-data/gemini-minimal-dots-2026-07-21/aggregates.json` |

## Prose and filler study

All generated prose edits are recorded as diffs in [prose-changes.json](prose-changes.json). The main README prose changes Flash high P50 763 → 762 ms and Nemotron thinking-on P95 5787 → 5786 ms to match their existing v2 aggregates. Muse’s 231/1752 and 236/5850 ms quotations already match v2 and are unchanged. Other model-specific historical observations retain their stated cohorts.

The Qwen filler-study rows now regenerate as 667 and 764 ms, matching the README. The historical aggregate and Markdown/HTML generator are rerun; no numeric cell is hand-edited. Other filler-study latency figures retain their original analysis scope; the main README’s newly corrected values are separately audited here.

## Reproduce

Use an isolated worktree based on ttfat-first-response, its project environment, and the raw runs plus campaign state manifests. Input directories must be real worktree directories with read-only file links/copies if an analyzer enforces repository containment; do not symlink the entire runs directory. Inputs required beyond tracked files include the Qwen-dots and focused-dot `state/` trees and the paired-Gemma campaign.log. The original raw runs remain unchanged.

```bash
uv run python -m scripts.reanalyze_historical_ttfat campaigns
uv run python -m scripts.reanalyze_historical_ttfat paired-gemma
uv run python -m scripts.reanalyze_historical_ttfat muse
uv run python -m scripts.reanalyze_historical_ttfat bakeoff
uv run python -m scripts.audit_readme_ttfat --write
uv run python -m scripts.render_ttfat_reanalysis_prose
uv run python scripts/build_filler_report.py
uv run python -m scripts.build_ttfat_reanalysis_report
```

Ordinary campaign subprocesses run through offline_analysis.py, which disallows network/subprocess calls and writes outside the worktree. Supplementary commands invoke only analysis functions and local Git reads. Timestamped metadata may change on replay; all result numbers must remain stable.

## Remaining provenance gap

The thirteen unresolved README rows are explicitly listed above. Retaining their published values does not certify first-response exclusion. Recover the exact original inclusion manifests before changing those rows; do not substitute a later filler-study control, an arbitrary model glob, or a cohort chosen to reproduce a score. The four Nemotron legacy notes remain unchanged.
