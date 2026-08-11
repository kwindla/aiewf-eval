# Batch Workflow Plan

## Review Context

The review findings that triggered this plan were:

- [P2] Flag incomplete runs instead of scoring them as `1/1` in `scripts/group_report.py`.
  When a benchmark run exits early but still produces `claude_judged.jsonl`, the script counts only the observed rows and then uses that count as the denominator for every score. An aborted 1-turn run can therefore render as `1/1` and `100.0%` instead of being marked incomplete. Because `run.sh` generates `REPORT.md` from this script, a failed sweep can look like a perfect run and skew the batch aggregate.

- [P3] Include silence-pad samples even when V2V is missing in `scripts/group_report.py`.
  If a turn has `silent_pad_silero_ms` but no `wav_v2v_ms`, the early `continue` skips the silence value too, so `SilPad Mean` is biased low or becomes `N/A` for runs with alignment gaps or greeting-only turns. Existing `turn_metrics.json` files already contain turns with that shape, so the report disagrees with the existing summary logic for those runs.

## In-Place Repair Plan Considered

Before we decided to remove the new batch wrapper/report path from this branch, the in-place repair plan was:

1. Persist the run's intended scripted-turn scope at start time from `src/multi_turn_eval/cli.py`. The CLI already knows the benchmark and parsed `--only-turns`, so it can hand that to `src/multi_turn_eval/recording/transcript_recorder.py` to write new `runtime.json` fields such as `benchmark_name`, `selected_turn_indices`, and `expected_scripted_turn_count`.

2. Keep `runtime.json.turns` backward-compatible, but stop treating it as completeness metadata. In `src/multi_turn_eval/recording/transcript_recorder.py`, that counter includes recovery transcript rows, so it is not a safe denominator for grouped reports.

3. Refactor `scripts/group_report.py` to load `claude_summary.json` plus the new runtime metadata first. Use:

   - observed judged turns = `claude_summary.turns_scored`
   - expected turns = `runtime.expected_scripted_turn_count`

4. Mark incomplete runs explicitly in the rendered report from `scripts/group_report.py` and exclude them from `AGG` by default. Recommended behavior:

   - show per-run counts against expected turns, for example `1/3`
   - render `Pass Rate` as `INCOMPLETE`
   - add a short note that incomplete runs were excluded from aggregate totals

5. Fix the silence-pad bug in `scripts/group_report.py` by collecting `silent_pad_silero_ms` independently of `wav_v2v_ms`, matching the existing behavior in `scripts/benchmark_summary.py`.

6. Add focused `pytest` coverage for:

   - full batch run
   - `--only-turns` batch with correct denominator
   - aborted partial run
   - turn with silence pad but missing V2V
   - mixed batch where incomplete runs are listed but excluded from aggregate

Recommendation from that approach:
Render incomplete rows as visible but non-aggregated, rather than treating missing turns as failures. That is the least misleading behavior for both full sweeps and intentional `--only-turns` slices.

## Chosen Implementation Plan

1. Remove `run.sh` from the branch so we do not ship a Gemini-specific batch wrapper as part of the main workflow.

2. Remove `scripts/group_report.py` from the branch, since it exists only to support `run.sh` batch reporting and is the code that triggered the review findings.

3. Update `README.md` to document the recommended batching pattern using explicit shell loops around:

   - `uv run multi-turn-eval run`
   - `uv run python scripts/analyze_turn_metrics.py` for speech runs when needed
   - `uv run multi-turn-eval judge`
   - existing aggregation utilities such as `scripts/aggregate_existing_runs.py` and `scripts/benchmark_summary.py`

4. Add both full-run and `--only-turns` batch examples to `README.md`, so partial sweeps remain supported in a transparent way without inventing a new report format.

5. Audit the branch for any references to `run.sh` or `group_report.py` and remove or update them so there are no stale pointers.

6. Validate the documented commands and confirm the branch no longer introduces the misleading batch-report path that review flagged.

## Testing

### Scope

- Default benchmark for this sweep: `aiwf_medium_context`
- All target configurations in this sweep should use explicit `--pipeline` values rather than relying on pipeline auto-detection.
- All speech-to-speech runs in this sweep should run `scripts/analyze_turn_metrics.py` before judging so the latency artifacts are available per run.
- There is no separate "aggregate judge" command. The aggregate is computed from individually judged runs plus per-run metrics.
- Any fixes during this testing phase must happen in benchmark code only. Do not modify Pipecat core.
- The plan doc is the authoritative workflow document. Raw execution logs, exact commands, run directories, and per-run notes should go in `docs/batch-sweep-results-2026-03-27.md`.

### Configuration Matrix

| README Label | CLI Model | Service | Pipeline | Benchmark | Extra Args | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ultravox-v0.7` | `ultravox-v0.7` | `ultravox-realtime` | `realtime` | `aiwf_medium_context` | none | Direct README mapping |
| `gpt-realtime-1.5` | `gpt-realtime-1.5` | `openai-realtime` | `realtime` | `aiwf_medium_context` | none | Use the direct model tag |
| `gpt-realtime` | `gpt-realtime` | `openai-realtime` | `realtime` | `aiwf_medium_context` | none | Direct README mapping |
| `gemini-live` | `gemini-2.5-flash-native-audio-preview-12-2025` | `gemini-live` | `realtime` | `aiwf_medium_context` | none | Direct README mapping |
| `nova-2-sonic` | `amazon.nova-2-sonic-v1:0` | none | `nova-sonic` | `aiwf_medium_context` | none | Built-in pipeline, no `--service` |
| `grok-realtime` | `grok-realtime` | none | `grok-realtime` | `aiwf_medium_context` | none | Dedicated Grok pipeline, no `--service` |
| `gemini-3.1-live (minimal)` | `gemini-3.1-flash-live-preview` | `gemini-live` | `realtime` | `aiwf_medium_context` | `--thinking minimal` | Separate README-style row |
| `gemini-3.1-live (low)` | `gemini-3.1-flash-live-preview` | `gemini-live` | `realtime` | `aiwf_medium_context` | `--thinking low` | Separate README-style row |
| `gemini-3.1-live (medium)` | `gemini-3.1-flash-live-preview` | `gemini-live` | `realtime` | `aiwf_medium_context` | `--thinking medium` | Separate README-style row |
| `gemini-3.1-live (high)` | `gemini-3.1-flash-live-preview` | `gemini-live` | `realtime` | `aiwf_medium_context` | `--thinking high` | Separate README-style row |

### Sanity-Check Protocol

Default sanity-check turns: `--only-turns 0,1`

For each target configuration, one at a time:

1. Run a 2-turn sanity check with the exact configuration from the matrix above and `--only-turns 0,1`.

2. Capture the exact `run_dir` emitted by `multi-turn-eval run`. Do not rely on heuristic directory lookup if the CLI already printed the path.

3. Run both per-run metrics commands on that exact `run_dir`:

   - `uv run python scripts/analyze_turn_metrics.py "$run_dir" -v 2>&1 | tee "$run_dir/metrics.txt"`
   - `uv run python scripts/analyze_turn_metrics.py "$run_dir" --json > "$run_dir/turn_metrics.json"`

4. Judge the run with:

   - `uv run multi-turn-eval judge "$run_dir"`

5. If any issues appear, fix them in benchmark code only, keep notes on the fix, and rerun the same sanity check for the same configuration before moving on.

6. If a configuration appears too difficult to fix in a reasonable amount of time, note the blocker clearly and move on to the next configuration.

### Single Full-Run Protocol

For each target configuration that passes the sanity-check gate:

1. Run one full benchmark with the exact configuration from the matrix above and no `--only-turns`.

2. Capture the exact `run_dir` emitted by the CLI.

3. Run per-run metrics on that exact `run_dir`:

   - `uv run python scripts/analyze_turn_metrics.py "$run_dir" -v 2>&1 | tee "$run_dir/metrics.txt"`
   - `uv run python scripts/analyze_turn_metrics.py "$run_dir" --json > "$run_dir/turn_metrics.json"`

4. Judge the run with:

   - `uv run multi-turn-eval judge "$run_dir"`

5. If the full run reveals issues, make benchmark-code fixes, keep notes, rerun the sanity check for that configuration, then rerun the full benchmark.

### Ten-Run Protocol

For each configuration that passes the single full-run gate:

1. Within a single configuration, run 10 full runs sequentially.

2. At this stage only, it is acceptable to run different configurations in parallel because different providers hit different endpoints. Target parallelism: 2 to 3 configurations at a time.

3. After each of the 10 runs:

   - capture the exact `run_dir`
   - run `scripts/analyze_turn_metrics.py` in both text and JSON modes
   - run `multi-turn-eval judge` on that run

4. Aggregate the 10-run set using existing aggregation tooling. There is no separate aggregate judge step; the aggregate is computed from individually judged runs.

5. Prefer parallel groups that spread load across providers rather than hitting the same provider repeatedly at once.

6. Record the exact shell commands used for all 10 runs, all 10 judging steps, and the aggregation step.

7. If a configuration regresses during the 10-run sweep and becomes too difficult to stabilize, note the blocker, stop the sweep for that configuration, and move on.

### Results Reporting

- The final README-style speech-to-speech table should include only configurations that completed 10 judged full runs successfully.
- For 10-run reporting, aggregate only an explicit allowlist of the 10 full run directories for that configuration. Do not rely on a broad model glob by default.
- Do not include `--only-turns` runs in README aggregates.
- If pattern-based aggregation is used for convenience, constrain it by exact timestamp window or maintain an explicit exclude list so sanity runs and partial runs are not pulled into the final results.
- Each Gemini 3.1 thinking level should appear as its own row:

  - `gemini-3.1-live (minimal)`
  - `gemini-3.1-live (low)`
  - `gemini-3.1-live (medium)`
  - `gemini-3.1-live (high)`

- Blocked or incomplete configurations should stay out of the main README table and should instead be recorded in this plan doc's progress section with blocker details.
- Keep the README table format aligned with the existing speech-to-speech section in `README.md`.

### Test Recordkeeping

- Keep a running note of benchmark-code fixes made per configuration in `docs/batch-sweep-results-2026-03-27.md`.
- Keep the exact shell commands used for sanity checks, full runs, judging, and 10-run aggregation in `docs/batch-sweep-results-2026-03-27.md`.
- Keep explicit notes for blocked or skipped configurations and why they were not completed in `docs/batch-sweep-results-2026-03-27.md`.
- Keep exact run directories for any run used in the final README table in `docs/batch-sweep-results-2026-03-27.md`.

## Progress

### Fixes

- [x] Remove `run.sh`
- [x] Remove `scripts/group_report.py`
- [x] Update `README.md` batching guidance
- [x] Audit stale references
- [x] Validate documented commands

### Results Log

- Raw execution details live in `docs/batch-sweep-results-2026-03-27.md`.

### Per-Config Status

#### `ultravox-v0.7`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T212559_ultravox-v0.7_3308712e`
  - runtime warnings observed; tracked in results log

#### `gpt-realtime-1.5`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T215701_gpt-realtime-1.5_24ce011d`
  - prior mapped fallback `gpt-realtime-2026-01-12` returned `model_not_found`, but the direct tag works and should be the canonical CLI value

#### `gpt-realtime`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T212938_gpt-realtime_bd6599d8`
  - validated benchmark-side OpenAI constructor fix

#### `gemini-live`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T213342_gemini-2.5-flash-native-audio-preview-12-2025_fe608778`

#### `nova-2-sonic`

- Status: sanity check passed after benchmark-side fixes
- Notes:
  - `runs/aiwf_medium_context/20260327T214410_amazon.nova-2-sonic-v1_0_644fb3a5`
  - required CLI compatibility and Nova session-start compatibility fixes in `src/multi_turn_eval/pipelines/nova_sonic.py`

#### `grok-realtime`

- Status: sanity check passed after benchmark-side fixes
- Notes:
  - `runs/aiwf_medium_context/20260327T214031_grok-realtime_d04d3b93`
  - required xAI URL and session-update compatibility fixes in `src/multi_turn_eval/pipelines/grok_realtime.py`

#### `gemini-3.1-live (minimal)`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T214724_gemini-3.1-flash-live-preview_9d3a9e74`

#### `gemini-3.1-live (low)`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T214918_gemini-3.1-flash-live-preview_a024112c`

#### `gemini-3.1-live (medium)`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T215130_gemini-3.1-flash-live-preview_2af8954b`

#### `gemini-3.1-live (high)`

- Status: sanity check passed
- Notes:
  - `runs/aiwf_medium_context/20260327T215332_gemini-3.1-flash-live-preview_23b315b3`

### Blockers

- None currently.
