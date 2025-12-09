#!/usr/bin/env python3
"""
Claude Agent SDK-based transcript judge (simplified version).

This script evaluates conversation transcripts using Claude Sonnet 4.5 via the
Claude Agent SDK. Uses natural language output with JSON parsing instead of
custom MCP tools for better reliability.

Usage:
    uv run judge_transcript_claude_v2.py runs/20251119T051205
    uv run judge_transcript_claude_v2.py runs/20251119T051205 --only-turns 0,1,2
    uv run judge_transcript_claude_v2.py runs/20251119T051205 --debug
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except ImportError:
    print("ERROR: claude-agent-sdk not installed.", file=sys.stderr)
    print("Install with: uv add claude-agent-sdk", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

JUDGE_VERSION = "claude-agent-sdk-v2-simple"
JUDGE_MODEL = "claude-opus-4-5"

# System prompt for Claude as judge
JUDGE_SYSTEM_PROMPT = """# Role
You are an expert evaluator for conversational AI systems. You will judge a multi-turn conversation between a user and an AI assistant for the AI Engineer World's Fair 2025.

# Evaluation Dimensions

For each turn, evaluate three dimensions:

1. **tool_use_correct** (bool):
   - TRUE if the assistant correctly called the expected function with semantically equivalent arguments
   - TRUE if no function call was expected and none was made
   - FALSE otherwise
   - For argument matching, use semantic equivalence (not verbatim):
     - "can't access location maps" ≈ "cannot access the location maps"
     - "OpenTelemetry tracing" ≈ "session about open telemetry tracing"
   - Session IDs must match exactly

2. **instruction_following** (bool):
   - TRUE if assistant directly answers the question OR advances the task by gathering required info
   - TRUE if assistant properly deflects out-of-scope questions with a polite redirect
   - FALSE if assistant neither answers nor advances the workflow
   - Examples of TRUE:
     - Recommending 2 sessions when asked for recommendations
     - Asking for missing parameters before calling a function
     - Listing speaker's sessions when asked about that speaker
     - Deflecting "What day is it?" with "I can only help with the conference"

3. **kb_grounding** (bool):
   - TRUE unless assistant states an explicit factual error relative to GOLDEN_TEXT
   - TRUE if assistant provides additional correct information not in golden
   - TRUE if assistant provides partial information without contradictions
   - FALSE only for clear factual contradictions (wrong dates, times, speakers, etc.)
   - Omissions are acceptable; extra details are acceptable

# Important Guidelines

- **Prefer semantic equivalence** over exact string matching
- **Be generous with kb_grounding** - penalize only clear contradictions
- For recommendations, accept valid alternatives even if different from golden
- Consider the full conversation context when evaluating
- Empty or very short responses usually indicate instruction_following failure
- If assistant refuses/deflects and states no factual claims, kb_grounding should be TRUE

# Output Format

Output ONLY JSON objects, one per line, nothing else - no explanations, no markdown.

For turn 0, output:
{"turn": 0, "tool_use_correct": true, "instruction_following": true, "kb_grounding": true, "reasoning": "explanation"}

For turn 1, output:
{"turn": 1, "tool_use_correct": false, "instruction_following": true, "kb_grounding": true, "reasoning": "explanation"}

And so on for ALL turns. Start immediately with the JSON for turn 0.
"""


# ============================================================================
# Data Loading
# ============================================================================

def load_transcript(run_dir: Path) -> List[Dict[str, Any]]:
    """Load transcript.jsonl from run directory."""
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No transcript.jsonl in {run_dir}")

    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ============================================================================
# Turn Formatting
# ============================================================================

def format_turns_for_claude(
    records: List[Dict[str, Any]],
    expected_turns: List[Dict[str, Any]],
    only_turns: Optional[set[int]] = None,
) -> str:
    """Format conversation turns as structured text for Claude to analyze."""
    lines = []

    for rec in records:
        turn_idx = rec["turn"]

        # Skip turns not in the filter set
        if only_turns is not None and turn_idx not in only_turns:
            continue

        if turn_idx >= len(expected_turns):
            continue

        expected = expected_turns[turn_idx]

        lines.append(f"## Turn {turn_idx}")
        lines.append(f"User: {rec['user_text']}")
        lines.append(f"Assistant: {rec['assistant_text']}")

        golden = expected.get('golden_text', '')
        if golden:
            lines.append(f"Golden: {golden}")

        # Expected function call
        expected_fc = expected.get('required_function_call')
        if expected_fc:
            fc_str = json.dumps(expected_fc)
            lines.append(f"Expected function: {fc_str}")
        else:
            lines.append("Expected function: none")

        # Actual function calls
        actual_calls = rec.get('tool_calls', [])
        if actual_calls:
            calls_str = json.dumps(actual_calls)
            lines.append(f"Actual functions: {calls_str}")
        else:
            lines.append("Actual functions: none")

        lines.append("")  # Blank line between turns

    return "\n".join(lines)


# ============================================================================
# Claude Judge
# ============================================================================

async def judge_with_claude(
    run_dir: Path,
    only_turns: Optional[set[int]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Main judging function using Claude Agent SDK with simple query()."""

    # Load data
    records = load_transcript(run_dir)
    from turns import turns as expected_turns

    # Filter records if only_turns specified
    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    if not records:
        raise ValueError("No turns to judge")

    model_name = records[0].get("model_name", "unknown")

    if debug:
        print(f"Judging {len(records)} turns...", file=sys.stderr)

    # Format turns
    formatted_turns = format_turns_for_claude(records, expected_turns, only_turns)

    # Create prompt
    prompt = f"{formatted_turns}\n\nPlease evaluate each turn and output one JSON object per line."

    # Configure options
    options = ClaudeAgentOptions(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=JUDGE_MODEL,
        permission_mode="bypassPermissions",
    )

    # Query Claude
    all_text = []
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            # Extract text from message
            if isinstance(message.content, str):
                all_text.append(message.content)
            elif isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'text'):
                        all_text.append(block.text)

    response_text = "".join(all_text)

    if debug:
        print(f"Claude response:\n{response_text[:500]}...", file=sys.stderr)

    # Parse JSON lines
    judgments = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue

        # Remove markdown code blocks if present
        if line.startswith('```'):
            continue
        if line.endswith('```'):
            continue

        try:
            judgment = json.loads(line)
            turn_num = judgment.get('turn')
            if turn_num is not None:
                judgments[turn_num] = {
                    "scores": {
                        "tool_use_correct": judgment.get('tool_use_correct', False),
                        "instruction_following": judgment.get('instruction_following', False),
                        "kb_grounding": judgment.get('kb_grounding', False),
                    },
                    "reasoning": judgment.get('reasoning', ''),
                }
                if debug:
                    print(f"✓ Parsed judgment for turn {turn_num}", file=sys.stderr)
        except json.JSONDecodeError as e:
            if debug:
                print(f"Failed to parse line: {line[:100]}... Error: {e}", file=sys.stderr)
            continue

    # Validate all turns were judged
    expected_turn_numbers = {r["turn"] for r in records}
    judged_turn_numbers = set(judgments.keys())
    missing = expected_turn_numbers - judged_turn_numbers

    if missing:
        raise ValueError(
            f"Failed to get judgments for turns: {sorted(missing)}. "
            f"Expected {len(expected_turn_numbers)} judgments, got {len(judgments)}."
        )

    return {
        "judgments": judgments,
        "summary": f"Evaluated {len(judgments)} turns successfully.",
        "model_name": model_name,
    }


# ============================================================================
# Output Generation
# ============================================================================

def write_outputs(
    run_dir: Path,
    records: List[Dict[str, Any]],
    judgments: Dict[int, Dict[str, Any]],
    summary: str,
    model_name: str,
) -> None:
    """Write all output files."""

    # 1. claude_judged.jsonl
    with (run_dir / "claude_judged.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            turn = rec["turn"]
            judgment = judgments[turn]
            f.write(json.dumps({
                **rec,
                "scores": judgment["scores"],
                "claude_reasoning": judgment["reasoning"],
            }, ensure_ascii=False) + "\n")

    # 2. claude_summary.json
    passes = {
        "tool_use_correct": sum(
            1 for j in judgments.values() if j["scores"]["tool_use_correct"]
        ),
        "instruction_following": sum(
            1 for j in judgments.values() if j["scores"]["instruction_following"]
        ),
        "kb_grounding": sum(
            1 for j in judgments.values() if j["scores"]["kb_grounding"]
        ),
    }

    summary_data = {
        "model_name": model_name,
        "claude_passes": passes,
        "turns_scored": len(judgments),
        "judge_version": JUDGE_VERSION,
        "judge_model": JUDGE_MODEL,
        "judged_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    (run_dir / "claude_summary.json").write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # 3. claude_analysis.md
    total = len(judgments)
    lines = [
        f"# Claude Agent SDK Evaluation",
        f"",
        f"**Model**: {model_name}",
        f"**Turns**: {total}",
        f"**Judge**: {JUDGE_MODEL}",
        f"**Judge Version**: {JUDGE_VERSION}",
        f"**Judged**: {summary_data['judged_at']}",
        f"",
        f"## Summary Metrics",
        f"",
        f"- **Tool Use Correct**: {passes['tool_use_correct']}/{total} ({passes['tool_use_correct']/total*100:.1f}%)",
        f"- **Instruction Following**: {passes['instruction_following']}/{total} ({passes['instruction_following']/total*100:.1f}%)",
        f"- **KB Grounding**: {passes['kb_grounding']}/{total} ({passes['kb_grounding']/total*100:.1f}%)",
        f"",
        f"## Per-Turn Failures",
        f"",
    ]

    # Add failure details
    has_failures = False
    for rec in records:
        turn = rec["turn"]
        judgment = judgments[turn]
        scores = judgment["scores"]

        if not all(scores.values()):
            has_failures = True
            failed_dimensions = [k for k, v in scores.items() if not v]

            lines.append(f"### Turn {turn}")
            lines.append(f"")
            lines.append(f"**User**: {rec['user_text']}")
            lines.append(f"")
            lines.append(f"**Assistant**: {rec['assistant_text'][:300]}{'...' if len(rec['assistant_text']) > 300 else ''}")
            lines.append(f"")
            lines.append(f"**Failed Dimensions**: {', '.join(failed_dimensions)}")
            lines.append(f"")
            lines.append(f"**Claude's Reasoning**: {judgment['reasoning']}")
            lines.append(f"")

    if not has_failures:
        lines.append("*No failures - all turns passed all evaluation dimensions!*")

    (run_dir / "claude_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Judge conversation transcripts using Claude Agent SDK"
    )
    parser.add_argument(
        "run_dir",
        help="Path to runs/<timestamp> directory containing transcript.jsonl"
    )
    parser.add_argument(
        "--only-turns",
        default="",
        help="Comma-separated list of turn indices to judge (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate ANTHROPIC_API_KEY
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse only_turns filter
    only_turns: Optional[set[int]] = None
    if args.only_turns.strip():
        try:
            only_turns = {int(x.strip()) for x in args.only_turns.split(',') if x.strip()}
            if args.debug:
                print(f"Filtering to turns: {sorted(only_turns)}", file=sys.stderr)
        except ValueError as e:
            print(f"ERROR: Invalid --only-turns format: {e}", file=sys.stderr)
            sys.exit(1)

    # Load records (for output generation)
    records = load_transcript(run_dir)
    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    # Run judgment
    try:
        result = asyncio.run(judge_with_claude(run_dir, only_turns, args.debug))
    except Exception as e:
        print(f"ERROR: Judgment failed: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Write outputs
    write_outputs(
        run_dir,
        records,
        result["judgments"],
        result["summary"],
        result["model_name"],
    )

    # Print summary to stdout (matches current judge behavior)
    summary_path = run_dir / "claude_summary.json"
    print(summary_path.read_text())

    if args.debug:
        print(f"\n✓ Wrote outputs:", file=sys.stderr)
        print(f"  - {run_dir / 'claude_judged.jsonl'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_summary.json'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_analysis.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
