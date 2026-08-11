# Claude Agent SDK LLM-as-a-Judge: How It Works

This document describes the Claude Agent SDK-based judge used to score multi-turn transcripts. The implementation lives in `src/multi_turn_eval/judging/claude_judge.py` and is invoked by the `multi-turn-eval judge` CLI.

## High-level goal

The judge scores each assistant turn on four dimensions:

- `turn_taking` (audio timing correctness; precomputed)
- `tool_use_correct` (expected function call correctness)
- `instruction_following` (did the assistant advance the task)
- `kb_grounding` (factual correctness)

It uses Claude (via the Agent SDK) to reason about golden expectations and function call alignment, then writes structured outputs for later analysis.

## Key inputs

1. **Run directory** (argument to `judge`):
   - `transcript.jsonl` (required)
   - `conversation.wav` (optional; enables turn-taking analysis)
2. **Expected turns**: imported from `turns.py` unless explicitly passed in.
3. **Environment**: `ANTHROPIC_API_KEY` must be set for the Claude SDK.

### transcript.jsonl format (produced by the recorder)
Each line is a single turn, recorded by `TranscriptRecorder` (`src/multi_turn_eval/recording/transcript_recorder.py`). Relevant fields:

- `turn`: zero-based index
- `model_name`
- `user_text`
- `assistant_text`
- `tool_calls`: list of `{"name": ..., "args": ...}` (duplicates preserved)
- `tool_results`: list of tool results (not used by the judge)
- `ttfb_ms`, `latency_ms`, `tokens` (not used by the judge)

Tool calls are recorded by `ToolCallRecorder` (`src/multi_turn_eval/processors/tool_call_recorder.py`) to preserve duplicates for evaluation while preventing them from polluting the live context.

### Expected turn format
The `turns.py` module provides a list of dicts with:

- `golden_text`: the reference response
- `required_function_call`: `{name, args}` or `None`

## Execution flow

### 1) Load data
`judge_with_claude` loads `transcript.jsonl` and the expected turns. If `--only-turns` is used, it filters both the transcript and prompt formatting to those turn indices.

### 2) Optional turn-taking analysis
If `conversation.wav` exists, `turn_taking.analyze_turn_taking` runs `scripts/analyze_turn_metrics.py` and detects per-turn audio timing anomalies (overlaps, negative TTFB, missing audio, reconnections, etc.).

- Failed turns are surfaced to the judge prompt as `turn_taking: false`.
- The final scoring always treats this precomputed value as the source of truth (it overrides Claude's output).

### 3) Build the judge prompt
`format_turns_for_claude` constructs a structured text prompt that includes:

1. A summary of turn-taking failures (if any)
2. A list of expected function calls by turn
3. Each conversation turn with:
   - turn-taking status
   - user text and assistant text
   - golden response (if present)
   - expected function call
   - actual function calls

This formatted prompt is appended with additional instructions to ensure the output includes **exactly one judgment per turn**.

### 4) Call Claude via Agent SDK
The judge sends the prompt using the Claude Agent SDK:

- `system_prompt`: `JUDGE_SYSTEM_PROMPT` (includes the two-phase realignment instructions)
- `model`: `JUDGE_MODEL` (`claude-opus-4-5` by default)
- `permission_mode`: `bypassPermissions`

The SDK streams responses; the code concatenates all text blocks into a single string.

### 5) Parse JSON and normalize scores
The judge extracts the JSON object from Claude's response by searching for the first `{` and last `}`. It then:

- Reads `final_judgments`, `realignment_notes`, and `function_call_tracking`.
- Builds a `judgments` dict keyed by turn index with:
  - `scores.turn_taking` (overridden by audio analysis if available)
  - `scores.tool_use_correct`
  - `scores.instruction_following`
  - `scores.kb_grounding`
  - `reasoning`
- Attaches `turn_taking_issues` when provided by the audio analysis.
- Validates that **all expected turns were judged** (hard error if any are missing).

### 6) Write outputs
`write_outputs` produces three files in the run directory:

1. `claude_judged.jsonl`
   - Original transcript line + `scores` + `claude_reasoning`
2. `claude_summary.json`
   - Aggregate pass counts, judge metadata, realignment flags, turn-taking failures
3. `claude_analysis.md`
   - Human-readable summary, turn-taking issues, realignment notes, per-turn failure details

## Realignment logic (LLM-driven)
The judge prompt instructs Claude to run a two-phase evaluation:

1. **Initial pass**: compare each turn to the golden expectation.
2. **Realignment pass**: detect early/late tool calls and avoid double-penalizing later turns.

The LLM outputs a `function_call_tracking` map and `realignment_notes`. The Python code itself does not re-score turns; it trusts Claude’s final judgments (except for `turn_taking`, which is precomputed).

## CLI entry point
The command below invokes the judge through `src/multi_turn_eval/cli.py`:

```
uv run multi-turn-eval judge runs/<benchmark>/<timestamp>_<model>
```

Optional flags:

- `--only-turns 0,1,2` (subset of turns)
- `--debug` (verbose diagnostics)

## Relevant files

- `src/multi_turn_eval/judging/claude_judge.py` (main judge implementation)
- `src/multi_turn_eval/judging/turn_taking.py` (audio timing analysis)
- `src/multi_turn_eval/recording/transcript_recorder.py` (transcript format)
- `src/multi_turn_eval/processors/tool_call_recorder.py` (tool call logging)
- `src/multi_turn_eval/cli.py` (CLI entry point)
- `turns.py` (golden expectations + required function calls)
