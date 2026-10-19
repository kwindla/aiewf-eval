# Recovery context and judge migration

Status: validation passed; canonical rejudging and reconciliation complete.

The judge previously omitted harness recovery exchanges. The corrected format retains their public answers, actual tool calls and recorded outcomes as unscored chronological evidence. It preserves the original scripted targets and underlying text/voice policy and knowledge-base rubrics. Recovery success does not retroactively excuse an unnecessary original deferral. An earlier valid recovery can establish state or fulfill a later expected action.

The active scope is 1,686 assignments across 62 configurations: 1,619 complete and 67 incomplete. The leaderboard still has 50 ranked text rows and 11 voice rows; Luna none remains unranked. All 698 recovery-bearing assignments were independently rejudged. The other 988 requests reuse their original grades after complete historical system/user request reconstruction and byte-identity checks. Their old metadata is unchanged.

All generation cohorts, raw transcripts, original snapshots and grades, completion classifications, recorded latency and audio annotations remain unchanged. No new conversations were generated. Original studies remain historical; this directory owns the corrected aggregate and publication entry point.

## Independent diagnostic validation

The plan was reviewed independently by Astra and Fable; actual Fable reviewer was `claude-fable-5-1`. Fresh blind reference sessions used explicit `gpt-6-astra` and actual `claude-fable-5-1`, no tools and independently rendered anonymized raw-record packets. The Codex CLI records the requested Astra model and token usage, but does not expose a server-returned model identifier or USD billing; that provenance limit is retained in reference metadata.

Selection was frozen before candidate calls: 24 complete conversations, 48 target turns, with 12 text (8 recovery, 4 clean) and 12 voice conversations. A disjoint 24-conversation reserve remains untouched. Known calibration, inspected Luna/Ultra and development examples were excluded conservatively. Deterministic old-label failure/pass stratification deliberately overrepresents errors and is not a random accuracy sample.

The fixed consensus contains 123/144 target dimensions: 64 text, 59 voice, 43 recovery, and 18 negatives. The 21 predeclared uncertain/disputed dimensions remain recorded, excluded before any candidate grade. These are independent model references, not human ground truth. Model-family overlap and consensus selection limit accuracy claims.

| Diagnostic pass | Overall agreement | Recovery agreement | Clean text agreement | Voice agreement |
|---|---:|---:|---:|---:|
| validation-A | 121/123 (98.4%) | 42/43 (97.7%) | 21/21 (100.0%) | 58/59 (98.3%) |
| validation-B | 122/123 (99.2%) | 43/43 (100.0%) | 20/21 (95.2%) | 59/59 (100.0%) |

Both passes individually cleared the preregistered 95% agreement gate. Full-conversation repeat discordance was 14/2160 dimensions (0.65%), below the 2% gate. Both development repetitions passed all clear support-ticket and delayed dietary/vote expectations. All diagnostic grades remain separate; each changed case received its own first-valid canonical request.

Every discrepancy and explanation is retained in [VALIDATION.json (local archive)](../README.md), with reference disagreements in [references/FROZEN.json (local archive)](../README.md). Passing these gates supports adoption with documented interpretation uncertainty; it does not establish perfect correctness on every label.

## Reconciled scores

Across all active assignments, 816 semantic dimension labels changed: 571 false-to-true and 245 true-to-false. These changes combine corrected context with ordinary judge variability; they are not a pure causal estimate. All 275 voice assignments have no recovery and retain their original canonical grades; voice was independently checked in diagnostic validation.

Strict and component scores, complete-only and fixed-assignment denominators, and 20,000-draw conversation-bootstrap intervals are recomputed from the reconciled per-turn grades. Missing future turns remain failures only in fixed-assignment summaries; incomplete conversations remain excluded from the ranked conditional table.

| Configuration | Prior completed-only pass | Corrected pass | Change (points) | Complete / selected or assigned | Rejudged |
|---|---:|---:|---:|---:|---:|
| claude-haiku-5-5 (thinking off, low) | 91.1% | 94.7% | +3.56 | 29/30 | 23 |
| gpt-5-mini | 78.5% | 80.9% | +2.42 | 11/11 | 11 |
| nemotron-3-nano-30b (512, NVFP4) | 69.5% | 71.9% | +2.38 | 7/7 | 7 |
| gpt-5.6-terra (none) | 90.8% | 92.7% | +1.92 | 26/30 | 30 |
| gpt-4o-mini | 81.2% | 83.0% | +1.82 | 11/11 | 11 |
| muse-glimmer-30b (thinking low, GGUF) | 83.3% | 85.1% | +1.78 | 30/30 | 30 |
| muse-spark-open (low, NVFP4) | 95.1% | 96.7% | +1.56 | 30/30 | 30 |
| gpt-4o | 86.7% | 88.2% | +1.52 | 11/11 | 7 |
| gpt-5.4 (none) | 91.7% | 93.0% | +1.33 | 30/30 | 30 |
| gpt-4.1 | 94.6% | 95.9% | +1.28 | 26/30 | 28 |
| kimi-k2.6 (thinking off) | 87.4% | 88.7% | +1.22 | 30/30 | 27 |
| qwen3.8-27b (thinking off, NVFP4) | 93.9% | 95.1% | +1.22 | 30/30 | 12 |
| gemma-4-26b-a4b-it (thinking off) | 82.2% | 83.4% | +1.22 | 30/30 | 30 |
| gpt-5.5 (none) | 97.2% | 98.2% | +0.92 | 29/30 | 15 |
| deepseek-v4-flash-0731 (low) | 96.4% | 97.3% | +0.89 | 30/30 | 14 |
| qwen3.6-35b-a3b (thinking off, FP8) | 81.7% | 82.6% | +0.89 | 30/30 | 24 |
| nemotron-3.5-lightning (thinking off, NVFP4) | 70.4% | 69.6% | -0.83 | 16/16 | 15 |
| gpt-5.6-sol (none) | 96.1% | 96.9% | +0.83 | 28/30 | 27 |
| gpt-5.4 (low) | 96.9% | 97.7% | +0.78 | 30/30 | 14 |
| gemini-3.5-flash-lite (minimal) | 93.6% | 94.3% | +0.78 | 30/30 | 14 |
| gpt-5.1 | 92.8% | 93.5% | +0.67 | 25/25 | 17 |
| inkling (none) | 93.8% | 94.4% | +0.67 | 30/30 | 20 |
| gpt-5.2 | 87.9% | 87.3% | -0.61 | 11/11 | 10 |
| gemma-4-31b-it (thinking off) | 95.9% | 96.4% | +0.56 | 30/30 | 24 |
| nemotron-3.5-lightning (thinking on, NVFP4) | 83.3% | 83.9% | +0.56 | 30/30 | 6 |
| gpt-4.1-mini | 82.3% | 81.8% | -0.51 | 13/30 | 30 |
| gpt-6-luna (low) | 91.4% | 91.8% | +0.46 | 29/30 | 20 |
| poolside/laguna-s-2.1 (thinking off) | 79.0% | 79.3% | +0.34 | 29/29 | 29 |
| claude-haiku-5-5 (low) | 96.9% | 97.2% | +0.33 | 30/30 | 8 |
| deepseek-v4-flash-0731 (high) | 98.2% | 98.6% | +0.33 | 30/30 | 7 |
| gpt-5.6-luna (none) | 84.7% | 84.4% | -0.25 | 27/30 | 29 |
| deepseek-v4-pro-0813 (low) | 98.7% | 98.4% | -0.22 | 30/30 | 5 |
| gemini-3.6-flash (minimal) | 95.3% | 95.1% | -0.22 | 30/30 | 7 |
| nemotron-3-super-120b (512, NVFP4) | 90.7% | 90.9% | +0.22 | 30/30 | 6 |
| gpt-5.4 (none, +96 dots) | 96.2% | 96.4% | +0.22 | 30/30 | 12 |
| gemini-2.5-flash (thinking off) | 81.6% | 81.4% | -0.11 | 30/30 | 30 |
| gemini-3.5-flash (minimal) | 94.6% | 94.7% | +0.11 | 30/30 | 3 |
| qwen3.6-27b (thinking off) | 92.0% | 92.1% | +0.11 | 30/30 | 4 |
| qwen3.8-27b (thinking off, FP8) | 96.1% | 96.0% | -0.11 | 30/30 | 11 |
| nemotron-3-ultra (thinking on) | 91.9% | 91.8% | -0.11 | 30/30 | 2 |
| muse-spark-1.3 (minimal) | 99.4% | 99.6% | +0.11 | 30/30 | 12 |
| glm-5.2 (none) | 95.7% | 95.8% | +0.11 | 30/30 | 3 |
| glm-5.3 (low reasoning) | 97.8% | 97.7% | -0.11 | 30/30 | 2 |
| gemini-3.7-flash (low) | 99.9% | 99.9% | +0.00 | 30/30 | 0 |
| gemini-3.8-flash (low) | 99.9% | 99.9% | +0.00 | 30/30 | 0 |
| kimi-k2.6 (thinking on) | 98.8% | 98.8% | +0.00 | 30/30 | 0 |
| claude-sonnet-5-5 (no up-front thinking, low) | 99.9% | 99.9% | +0.00 | 30/30 | 0 |
| claude-sonnet-5-5 (low) | 99.9% | 99.9% | +0.00 | 30/30 | 0 |
| muse-spark-1.3 (low) | 100.0% | 100.0% | +0.00 | 30/30 | 1 |
| gpt-5.6-terra (medium) | 99.9% | 99.9% | +0.00 | 26/30 | 0 |
| gpt-6-luna (none) | — | — | — | 0/30 | 1 |

Full per-turn deltas are retained in [changed-labels.json (local archive)](../README.md). The confirmed development example now recognizes the recorded successful support call; it no longer claims no ticket existed. The separate delayed-action example still fails the original unnecessary dietary/vote deferrals.

## Provenance and operations

The inventory verifies all source/snapshot/grade hashes, actual prior judge model and effort, frozen rubric hashes, audio annotations, scripted IDs and request hashes. Recovery ancestry is checked against source order; 108 missing explicit parents are recorded as unambiguous positional inferences. Zero active holds remain. Public reasoning normalization covers recovery and scripted text; actual calls/results remain intact.

The original 34-entry hash-verified public-ending restoration manifest is immutable and applied before formatting with per-source application-count checks. The separate recovery-ending audit found no positive native-wire evidence of omitted public output. Sixty-four older recovery-ending captures have no native Responses audit; this is a documented evidence limit, not proof of complete native output. Empty tool-only responses are not automatically omissions or failures. Unavailable encrypted reasoning is neither exposed nor claimed verified.

The locked driver uses eight concurrent requests, a shared persisted 90-second cooldown for credit failures, operational retries only, retained attempts and first-valid atomic result ownership. The canonical wrapper additionally captures SDK message IDs and can recover a validated pending result after a crash. It does not change candidate prompts or acceptance rules. [canonical-registration.json (local archive)](../README.md) freezes this audit-only entry point.

Known Opus judging cost for this migration: **$143.2719**, including diagnostics and all cost-bearing retained attempts. Fable plan/reference review cost: **$10.2351**. Astra CLI token usage is retained, but USD billing is unavailable. Opus attempts without a reported cost: 0.

| Opus phase | Reported cost |
|---|---:|
| canonical | $134.4198 |
| development-A | $0.6019 |
| development-B | $0.2553 |
| validation-A | $4.9278 |
| validation-B | $3.0671 |

## Reproduce and verify

Run from the worktree with `PYTHONPATH="$PWD/src:$PWD"` using its `.venv/bin/python`. `preflight.py` reconstructs historical requests; `references.py`, `run_validation.py` and `validate.py` preserve separate reference/diagnostic ownership. Resume canonical judging with `canonical.py`. Reconcile with `analyze.py`, generate this report with `report.py`, then publish with `publish.py`. Never rerun old publishers to update the corrected regime.

Machine-readable coverage and publication verification are in [VERIFICATION.json (local archive)](../README.md). Prior archived bytes and 2,988 historical evidence files must retain their original hashes. Tests, chart consistency, whitespace and idempotent publication are required before the final completion marker.

[Evidence storage and local audit instructions](../README.md).
