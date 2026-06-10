# Capture playbook — data collection for report/analysis development

Recipes for collecting benchmark data whose *purpose is improving the
report/analysis tooling itself* (benchmark_summary.py, analyze_turn_metrics.py,
future report generators), not adding README rows. Each capture below
deliberately exercises a different slice of the transcript schema so report
code can be developed against real, varied data.

## What a run captures

Every run directory (`runs/aiwf_medium_context/<ts>_<model>_<hash>/`) contains:

- `transcript.jsonl` — per-turn: `ttfb_ms` (first *visible* token),
  `raw_ttfb_ms` (first stream chunk; see `multi_turn_eval/metrics.py` for
  per-service trigger nuance), `latency_ms`, `tokens` (incl. cache
  read/creation), `tool_calls`/`tool_results`, `reconnection_count`.
- `run.log` — config lines ("Configured <model> with ..."), retry events,
  cache token logs, exact request payloads when `MTE_LOG_ANTHROPIC_PAYLOADS=1`.
- After judging: `claude_summary.json`, `claude_judged.jsonl`, `claude_analysis.md`.

Analysis entry points:

```bash
uv run multi-turn-eval judge runs/aiwf_medium_context/<run_dir>
uv run python scripts/analyze_turn_metrics.py <run_dir> -v        # text runs: transcript-only mode (TTFB stats incl. P95)
uv run python scripts/analyze_turn_metrics.py <run_dir> --json    # machine-readable, per-turn + summary
uv run python scripts/benchmark_summary.py <run_dir> [<run_dir>...]  # pass-rate aggregation
```

## Capture set A — TTFT decomposition (raw vs visible vs thinking)

Goal: data for raw/visible TTFT columns and thinking-delay distributions in
reports. Three fable configs that put the thinking delay at ~1.5s, ~14ms, and
in between; sonnet as the no-thinking control.

```bash
# 1. Thinking-heavy: visible-raw gap is the thinking delay
MTE_ANTHROPIC_EFFORT=high uv run multi-turn-eval run aiwf_medium_context \
  --model claude-fable-5 --service anthropic

# 2. Voice-optimized: thinking suppressed, gap ~0
MTE_ANTHROPIC_EFFORT=low MTE_ANTHROPIC_THINKING_DISPLAY=omitted \
  MTE_ANTHROPIC_VOICE_STEERING=1 uv run multi-turn-eval run aiwf_medium_context \
  --model claude-fable-5 --service anthropic

# 3. No-thinking control: raw ≈ visible by construction
uv run multi-turn-eval run aiwf_medium_context --model claude-sonnet-4-6 --service anthropic
```

Analysis development target: `analyze_turn_metrics.py --json` now emits both
`server_ttfb_ms` and `raw_ttfb_ms` per turn — a report can derive
`thinking_delay = server - raw` and plot its distribution per config.

## Capture set B — cross-provider raw-TTFB comparability

Goal: data to validate (or correct) cross-provider TTFT comparisons. The raw
trigger differs per service (Anthropic: first stream event; Cerebras: first
chunk-with-choices; Google: first chunk-with-candidates; vLLM: no raw value —
a known gap worth fixing if reports need it).

```bash
MTE_CEREBRAS_REASONING_EFFORT=none uv run multi-turn-eval run aiwf_medium_context \
  --model moonshotai-kimi-k2.6 --service cerebras       # instant mode
uv run multi-turn-eval run aiwf_medium_context \
  --model moonshotai-kimi-k2.6 --service cerebras       # thinking mode (default)
uv run multi-turn-eval run aiwf_medium_context \
  --model gemini-2.5-flash --service google
```

## Capture set C — cache-warmup curves

Goal: per-turn cache token data for a cache-effectiveness section in reports.
Prompt caching is on by default for Anthropic; contrast with it off:

```bash
uv run multi-turn-eval run aiwf_medium_context --model claude-sonnet-4-6 --service anthropic
MTE_ANTHROPIC_PROMPT_CACHING=0 uv run multi-turn-eval run aiwf_medium_context \
  --model claude-sonnet-4-6 --service anthropic
```

The transcript `tokens` field carries `cache_read_input_tokens` /
`cache_creation_input_tokens` per turn; TTFT-vs-turn-index against cache reads
shows the warmup curve.

## Capture set D — failure-mode corpus (cheap, partial runs)

Goal: transcripts containing retries, short runs, and missing-field turns, so
report code handles imperfect data gracefully. Use `--only-turns` for cheap
partial captures and keep any runs that hit errors — don't delete them; they're
exactly the fixtures report code needs.

```bash
uv run multi-turn-eval run aiwf_medium_context --model gpt-4.1-mini --service openai
# (historically ends early via premature end_session — produces <30-turn transcripts)
uv run multi-turn-eval run aiwf_medium_context --model claude-fable-5 \
  --service anthropic --only-turns 0,1,2
```

## Conventions

- 10-20 min per full 30-turn run; run in background, never block-wait
  (see CLAUDE.md).
- Track multi-run sets as allowlist files in `docs/ten-run-allowlists/`
  (one run dir per line) and snapshot aggregates into `docs/ten-run-aggregates/`.
- Before any multi-run capture, pin/clear every `MTE_*` knob you're not
  sweeping (leftover exports silently change the config; the harness logs a
  warning for non-fable + `MTE_ANTHROPIC_EFFORT`, and `scripts/run_fable5_sweep.sh`
  shows the `env -u` pinning pattern).
- Sanity-check any new config with `--only-turns 0,1,2` before a full run;
  first-turn TTFT is cold and unrepresentative.

## Known analysis gaps these captures will expose (candidate roadmap)

1. `benchmark_summary.py` has no TTFT columns for text runs (V2V columns are
   speech-only) — README TTFT for text rows is currently computed ad hoc.
2. vLLM service gates `ttfb_ms` but emits no `raw_ttfb_ms`.
3. No per-turn thinking-delay or cache-warmup visualization yet.
4. Tool-call turns vs non-tool turns aren't split in text-mode latency stats
   (the audio path has the [T] marker; transcript-only mode now carries
   `has_tool_call` in its JSON).
