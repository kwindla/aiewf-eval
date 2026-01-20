#!/usr/bin/env python3
"""
Claude Agent SDK-based transcript judge (v3 with turn realignment).

This version adds intelligence to handle turn misalignment issues where:
- A function call happens earlier than expected (premature)
- Subsequent turns should not be penalized for the "missing" call

The judge uses a two-phase approach:
1. Initial pass: Compare each turn against golden expectations
2. Realignment pass: Detect early/late function calls and adjust scoring

Usage via CLI:
    uv run multi-turn-eval judge runs/aiwf_medium_context/20251215T202910_gemini-...
    uv run multi-turn-eval judge runs/... --only-turns 0,1,2
    uv run multi-turn-eval judge runs/... --debug
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

JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
JUDGE_MODEL = "claude-opus-4-5"

# System prompt for the two-phase judge
JUDGE_SYSTEM_PROMPT = """# Role
You are an expert evaluator for conversational AI systems. You will judge a multi-turn conversation between a user and an AI assistant for the AI Engineer World's Fair 2025.

# CRITICAL: Evaluate ALL Turns

**You MUST output a judgment for EVERY turn provided in the input.** Do not stop early or skip turns. Even if the conversation seems to have gone off-track, continue evaluating all remaining turns. The final_judgments array must contain exactly one entry for each turn in the input.

# Two-Phase Evaluation Process

You will evaluate in TWO phases:

## PHASE 1: Initial Turn-by-Turn Analysis
For each turn, evaluate against the golden expectation and note any discrepancies.

## PHASE 2: Realignment Analysis
After the initial pass, look for "turn misalignment" patterns:
- **Early function calls**: A function was called earlier than expected (e.g., at turn N instead of N+1)
- **Late function calls**: A function was called later than expected (e.g., at turn N+1 instead of N)
- **Cascading effects**: If a function was called early, subsequent turns expecting that call should NOT be penalized
- **Semantic equivalence**: Even if timing differs, did the conversation accomplish the same goals?

# Evaluation Dimensions

For each turn, evaluate FOUR dimensions:

1. **turn_taking** (bool):
   - This dimension is PRE-COMPUTED based on audio timing analysis
   - If marked as a turn-taking failure in the input, set to FALSE
   - If not marked, set to TRUE
   - Turn-taking failures indicate audio timing issues (interruptions, overlaps, missing audio)

2. **tool_use_correct** (bool):
   - TRUE if the assistant correctly called the expected function with semantically equivalent arguments
   - TRUE if no function call was expected and none was made
   - TRUE if a function call was expected but was already made in an earlier turn (realignment case)
   - TRUE if a late function call is made at this turn (the call eventually happened, credit this turn)
   - FALSE if a function call was expected, not made, and NOT already made earlier
   - FALSE if the assistant's words imply waiting for confirmation but it acts without waiting
   - FALSE if the assistant asks for unnecessary confirmation instead of making the expected function call
   - For argument matching, use semantic equivalence (not verbatim)
   - Session IDs must match exactly

3. **instruction_following** (bool):
   - TRUE if assistant directly answers the question OR advances the task
   - TRUE if assistant properly deflects out-of-scope questions
   - TRUE if the turn is part of a realigned workflow that still accomplishes the goal
   - FALSE if assistant's words contradict its actions (says "Does that work?" but doesn't wait)
   - FALSE if assistant neither answers nor advances the workflow
   - FALSE if the assistant asks for unnecessary confirmation when it already has all needed information
   - **IMPORTANT**: If a turn has turn_taking=FALSE, be lenient on instruction_following since garbled audio may cause transcription issues

4. **kb_grounding** (bool):
   - TRUE unless assistant states an explicit factual error
   - TRUE if assistant provides additional correct information
   - FALSE only for clear factual contradictions (wrong dates, times, locations, speakers)

# Critical: Detecting Words-Actions Mismatch

A turn should FAIL instruction_following if the assistant's text implies one behavior but its actions show another:
- Says "I'll wait for confirmation" but calls the function immediately
- Says "Could you confirm?" but doesn't actually wait for the response
- Says "Does that work?" in the same turn where it confirms completion

# Critical: Handling Early Function Calls

When you detect an early function call:
1. Note which function was called and at which turn
2. In subsequent turns, if that same function was "expected", mark tool_use_correct as TRUE (already satisfied)
3. Add a note in reasoning explaining the realignment

# Critical: Handling Late Function Calls

When you detect a late function call (assistant asked for unnecessary confirmation instead of acting):
1. Penalize the turn where the function SHOULD have been called (tool_use_correct=FALSE, instruction_following=FALSE)
2. Credit the turn where the function was ACTUALLY called (tool_use_correct=TRUE)
3. Continue evaluating ALL subsequent turns normally
4. Add a note in function_call_tracking with status "late"

Example: If vote_for_session was expected at turn 24 but called at turn 25:
- Turn 24: tool_use_correct=FALSE (didn't call when it should have), instruction_following=FALSE (asked unnecessary confirmation)
- Turn 25: tool_use_correct=TRUE (function was called correctly)
- Turns 26-29: Evaluate normally, do NOT skip these turns

# Critical: Empty Assistant Text with Tool Calls

A turn with empty assistant_text but a valid tool call is still a valid turn. The assistant may have called the function without generating speech. Evaluate the tool call normally.

# Output Format

Output a JSON object with this structure:
```json
{
  "phase1_analysis": [
    {"turn": 0, "initial_tool_use": true, "initial_instruction": true, "initial_kb": true, "notes": "..."},
    ...
  ],
  "realignment_notes": "Description of any detected misalignments and how they were resolved",
  "function_call_tracking": {
    "submit_dietary_request": {"expected_turn": 15, "actual_turn": 14, "status": "early"},
    ...
  },
  "final_judgments": [
    {"turn": 0, "reasoning": "...", "turn_taking": true, "tool_use_correct": true, "instruction_following": true, "kb_grounding": true},
    ...
  ]
}
```

Note: The `turn_taking` field should match what was provided in the input (pre-computed from audio timing analysis).

Output ONLY this JSON object, no markdown code blocks, no explanations outside the JSON.
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
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# Turn Formatting
# ============================================================================

def format_turns_for_claude(
    records: List[Dict[str, Any]],
    expected_turns: List[Dict[str, Any]],
    only_turns: Optional[set[int]] = None,
    turn_taking_data: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Format conversation turns with full context for realignment analysis.

    Args:
        records: List of transcript records
        expected_turns: List of expected turn data
        only_turns: Optional set of turn indices to include
        turn_taking_data: Optional dict mapping turn index to turn-taking analysis
    """
    lines = []

    # First, provide turn-taking failure summary if any
    if turn_taking_data:
        failed_turns = [idx for idx, data in turn_taking_data.items() if not data.get("turn_taking", True)]
        if failed_turns:
            lines.append("# Turn-Taking Failures (Pre-computed from Audio Analysis)")
            lines.append("")
            lines.append("The following turns have audio timing issues that may affect transcription quality:")
            for idx in sorted(failed_turns):
                issues = turn_taking_data[idx].get("issues", [])
                lines.append(f"- Turn {idx}: {', '.join(issues) if issues else 'timing issue'}")
            lines.append("")
            lines.append("For these turns, set `turn_taking: false` in your output.")
            lines.append("Be lenient on `instruction_following` for turns with turn_taking failures.")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Provide a summary of all expected function calls
    lines.append("# Expected Function Calls Summary")
    lines.append("")
    for i, exp in enumerate(expected_turns):
        fc = exp.get('required_function_call')
        if fc:
            lines.append(f"- Turn {i}: {fc['name']}({json.dumps(fc['args'])})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Then provide each turn's details
    lines.append("# Conversation Turns")
    lines.append("")

    for rec in records:
        turn_idx = rec["turn"]

        # Skip turns not in the filter set
        if only_turns is not None and turn_idx not in only_turns:
            continue

        if turn_idx >= len(expected_turns):
            continue

        expected = expected_turns[turn_idx]

        lines.append(f"## Turn {turn_idx}")

        # Add turn-taking status if available
        if turn_taking_data and turn_idx in turn_taking_data:
            tt_data = turn_taking_data[turn_idx]
            tt_ok = tt_data.get("turn_taking", True)
            if not tt_ok:
                issues = tt_data.get("issues", [])
                lines.append(f"**Turn-Taking**: FAILURE ({', '.join(issues)})")
            else:
                lines.append("**Turn-Taking**: OK")
        else:
            lines.append("**Turn-Taking**: OK (no audio analysis)")

        lines.append(f"**User**: {rec['user_text']}")
        lines.append(f"**Assistant**: {rec['assistant_text']}")
        lines.append("")

        golden = expected.get('golden_text', '')
        if golden:
            lines.append(f"**Golden Response**: {golden}")
            lines.append("")

        # Expected function call
        expected_fc = expected.get('required_function_call')
        if expected_fc:
            fc_str = json.dumps(expected_fc)
            lines.append(f"**Expected Function**: {fc_str}")
        else:
            lines.append("**Expected Function**: none")

        # Actual function calls
        actual_calls = rec.get('tool_calls', [])
        if actual_calls:
            calls_str = json.dumps(actual_calls)
            lines.append(f"**Actual Functions**: {calls_str}")
        else:
            lines.append("**Actual Functions**: none")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Claude Judge
# ============================================================================

async def judge_with_claude(
    run_dir: Path,
    only_turns: Optional[set[int]] = None,
    debug: bool = False,
    expected_turns: Optional[List[Dict[str, Any]]] = None,
    skip_turn_taking: bool = False,
) -> Dict[str, Any]:
    """Main judging function using two-phase realignment approach.

    Args:
        run_dir: Path to the run directory containing transcript.jsonl
        only_turns: Optional set of turn indices to judge
        debug: Enable debug logging
        expected_turns: Optional list of expected turns. If not provided, imports from turns module.
        skip_turn_taking: If True, skip turn-taking analysis (for runs without WAV files)

    Returns:
        Dict with judgments, realignment_notes, function_tracking, turn_taking_analysis, summary, and model_name.
    """

    # Load data
    records = load_transcript(run_dir)

    # Get expected turns from parameter or import
    if expected_turns is None:
        from turns import turns as expected_turns

    # Filter records if only_turns specified
    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    if not records:
        raise ValueError("No turns to judge")

    model_name = records[0].get("model_name", "unknown")

    if debug:
        print(f"Judging {len(records)} turns with realignment analysis...", file=sys.stderr)

    # Run turn-taking analysis if WAV file exists
    turn_taking_data: Optional[Dict[int, Dict[str, Any]]] = None
    turn_taking_analysis = None
    if not skip_turn_taking:
        wav_path = run_dir / "conversation.wav"
        if wav_path.exists():
            if debug:
                print("Running turn-taking analysis...", file=sys.stderr)
            try:
                from .turn_taking import analyze_turn_taking
                turn_taking_analysis = analyze_turn_taking(run_dir)
                if turn_taking_analysis.error:
                    if debug:
                        print(f"Turn-taking analysis error: {turn_taking_analysis.error}", file=sys.stderr)
                else:
                    turn_taking_data = {
                        idx: result.to_dict()
                        for idx, result in turn_taking_analysis.per_turn.items()
                    }
                    if debug and turn_taking_analysis.failed_turns:
                        print(f"Turn-taking failures: {turn_taking_analysis.failed_turns}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"Turn-taking analysis failed: {e}", file=sys.stderr)

    # Format turns (with turn-taking data if available)
    formatted_turns = format_turns_for_claude(records, expected_turns, only_turns, turn_taking_data)

    # Create prompt
    prompt = f"""{formatted_turns}

Please perform your two-phase evaluation:
1. First, analyze each turn against its golden expectation
2. Then, identify any turn misalignments (early/late function calls)
3. Apply realignment adjustments to avoid double-penalizing
4. Output the final JSON with judgments for ALL {len(records)} turns

CRITICAL: Your final_judgments array MUST contain exactly {len(records)} entries (turns 0-{len(records)-1}).

Remember:
- If a function is called early (before expected turn), subsequent turns should not be penalized for the "missing" call
- If a function is called late (after expected turn), penalize the turn that should have called it, credit the turn that did call it, then continue evaluating all remaining turns
- If the assistant says "Does that work?" but doesn't wait for confirmation, that's an instruction_following failure
- If the assistant asks for unnecessary confirmation when it has all needed info, that's a tool_use_correct AND instruction_following failure
- Be generous with kb_grounding unless there's a clear factual error
- Empty assistant_text with a valid tool call is still a valid turn - evaluate the tool call
"""

    # Configure options - use extended thinking for complex reasoning
    options = ClaudeAgentOptions(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=JUDGE_MODEL,
        permission_mode="bypassPermissions",
    )

    # Query Claude
    all_text = []
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            if isinstance(message.content, str):
                all_text.append(message.content)
            elif isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'text'):
                        all_text.append(block.text)

    response_text = "".join(all_text)

    if debug:
        print(f"Claude response length: {len(response_text)} chars", file=sys.stderr)
        print(f"First 1000 chars:\n{response_text[:1000]}", file=sys.stderr)

    # Parse the JSON response
    # Try to find JSON object in the response
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start == -1 or json_end == 0:
        raise ValueError(f"No JSON found in response: {response_text[:500]}")

    json_str = response_text[json_start:json_end]

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parse error: {e}", file=sys.stderr)
            print(f"Attempted to parse: {json_str[:500]}...", file=sys.stderr)
        raise ValueError(f"Failed to parse JSON response: {e}")

    # Extract final judgments
    final_judgments = result.get('final_judgments', [])
    realignment_notes = result.get('realignment_notes', '')
    function_tracking = result.get('function_call_tracking', {})

    if debug:
        print(f"\nRealignment notes: {realignment_notes}", file=sys.stderr)
        print(f"Function tracking: {json.dumps(function_tracking, indent=2)}", file=sys.stderr)

    # Convert to our standard format
    judgments = {}
    for j in final_judgments:
        turn_num = j.get('turn')
        if turn_num is not None:
            # Get turn_taking from Claude's response, defaulting to True if not provided
            turn_taking = j.get('turn_taking', True)

            # If we have turn_taking_data, use that as the source of truth
            if turn_taking_data and turn_num in turn_taking_data:
                turn_taking = turn_taking_data[turn_num].get('turn_taking', True)

            judgments[turn_num] = {
                "scores": {
                    "turn_taking": turn_taking,
                    "tool_use_correct": j.get('tool_use_correct', False),
                    "instruction_following": j.get('instruction_following', False),
                    "kb_grounding": j.get('kb_grounding', False),
                },
                "reasoning": j.get('reasoning', ''),
            }

            # Add turn-taking issues if available
            if turn_taking_data and turn_num in turn_taking_data:
                issues = turn_taking_data[turn_num].get('issues', [])
                if issues:
                    judgments[turn_num]["turn_taking_issues"] = issues

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
        "realignment_notes": realignment_notes,
        "function_tracking": function_tracking,
        "turn_taking_analysis": turn_taking_analysis.to_dict() if turn_taking_analysis else None,
        "summary": f"Evaluated {len(judgments)} turns with realignment.",
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
    realignment_notes: str = "",
    function_tracking: Optional[Dict[str, Any]] = None,
    turn_taking_analysis: Optional[Dict[str, Any]] = None,
) -> None:
    """Write all output files.

    Args:
        run_dir: Path to the run directory
        records: List of transcript records
        judgments: Dict mapping turn number to judgment data
        summary: Summary string (for backward compat, not used in output)
        model_name: Name of the model being judged
        realignment_notes: Optional notes about turn realignment (v3 feature)
        function_tracking: Optional dict tracking function call timing (v3 feature)
        turn_taking_analysis: Optional turn-taking analysis result (v4 feature)
    """
    if function_tracking is None:
        function_tracking = {}

    # 1. claude_judged.jsonl
    with (run_dir / "claude_judged.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            turn = rec["turn"]
            judgment = judgments[turn]
            output_rec = {
                **rec,
                "scores": judgment["scores"],
                "claude_reasoning": judgment["reasoning"],
            }
            # Include turn-taking issues if present
            if "turn_taking_issues" in judgment:
                output_rec["turn_taking_issues"] = judgment["turn_taking_issues"]
            f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

    # 2. claude_summary.json
    passes = {
        "turn_taking": sum(
            1 for j in judgments.values() if j["scores"].get("turn_taking", True)
        ),
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

    # Count turns with turn-taking failures that also failed instruction_following
    # (these may be excusable)
    turn_taking_affected_instruction = sum(
        1 for j in judgments.values()
        if not j["scores"].get("turn_taking", True) and not j["scores"]["instruction_following"]
    )

    summary_data = {
        "model_name": model_name,
        "claude_passes": passes,
        "turns_scored": len(judgments),
        "judge_version": JUDGE_VERSION,
        "judge_model": JUDGE_MODEL,
        "judged_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "realignment_applied": bool(function_tracking),
        "function_tracking": function_tracking,
        "turn_taking_failures": turn_taking_analysis.get("failed_turns", []) if turn_taking_analysis else [],
        "turn_taking_affected_instruction": turn_taking_affected_instruction,
    }

    (run_dir / "claude_summary.json").write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # 3. claude_analysis.md
    total = len(judgments)
    lines = [
        f"# Claude Agent SDK Evaluation (v4 with Turn-Taking)",
        f"",
        f"**Model**: {model_name}",
        f"**Turns**: {total}",
        f"**Judge**: {JUDGE_MODEL}",
        f"**Judge Version**: {JUDGE_VERSION}",
        f"**Judged**: {summary_data['judged_at']}",
        f"",
        f"## Summary Metrics",
        f"",
        f"- **Turn-Taking**: {passes['turn_taking']}/{total} ({passes['turn_taking']/total*100:.1f}%)",
        f"- **Tool Use Correct**: {passes['tool_use_correct']}/{total} ({passes['tool_use_correct']/total*100:.1f}%)",
        f"- **Instruction Following**: {passes['instruction_following']}/{total} ({passes['instruction_following']/total*100:.1f}%)",
        f"- **KB Grounding**: {passes['kb_grounding']}/{total} ({passes['kb_grounding']/total*100:.1f}%)",
        f"",
    ]

    # Add turn-taking analysis summary
    if turn_taking_analysis and turn_taking_analysis.get("failed_turns"):
        failed_turns = turn_taking_analysis["failed_turns"]
        lines.extend([
            f"## Turn-Taking Analysis",
            f"",
            f"**{len(failed_turns)} turns** had audio timing issues:",
            f"",
        ])
        per_turn = turn_taking_analysis.get("per_turn", {})
        for turn_idx in failed_turns:
            turn_data = per_turn.get(str(turn_idx), per_turn.get(turn_idx, {}))
            issues = turn_data.get("issues", [])
            lines.append(f"- Turn {turn_idx}: {', '.join(issues) if issues else 'timing issue'}")
        lines.append("")
        if turn_taking_affected_instruction > 0:
            lines.append(f"*{turn_taking_affected_instruction} instruction_following failures may be caused by turn-taking issues.*")
            lines.append("")

    # Add realignment notes if any
    if realignment_notes:
        lines.extend([
            f"## Realignment Analysis",
            f"",
            realignment_notes,
            f"",
        ])

    if function_tracking:
        lines.extend([
            f"## Function Call Tracking",
            f"",
            "| Function | Expected Turn | Actual Turn | Status |",
            "|----------|---------------|-------------|--------|",
        ])
        for func_name, tracking in function_tracking.items():
            exp = tracking.get('expected_turn', '?')
            act = tracking.get('actual_turn', '?')
            status = tracking.get('status', '?')
            lines.append(f"| {func_name} | {exp} | {act} | {status} |")
        lines.append("")

    lines.extend([
        f"## Per-Turn Failures",
        f"",
    ])

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
            # Add turn-taking issues if relevant
            if "turn_taking" in failed_dimensions and "turn_taking_issues" in judgment:
                lines.append(f"**Turn-Taking Issues**: {', '.join(judgment['turn_taking_issues'])}")
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
# Main CLI (for standalone use)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Judge conversation transcripts using Claude Agent SDK (v4 with turn-taking)"
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
        result.get("realignment_notes", ""),
        result.get("function_tracking", {}),
        result.get("turn_taking_analysis"),
    )

    # Print summary
    total = len(result["judgments"])
    passes = {
        "turn_taking": sum(1 for j in result["judgments"].values() if j["scores"].get("turn_taking", True)),
        "tool_use": sum(1 for j in result["judgments"].values() if j["scores"]["tool_use_correct"]),
        "instruction": sum(1 for j in result["judgments"].values() if j["scores"]["instruction_following"]),
        "kb": sum(1 for j in result["judgments"].values() if j["scores"]["kb_grounding"]),
    }

    print(f"Judged {total} turns (with turn-taking analysis)")
    print(f"  Turn-taking: {passes['turn_taking']}/{total}")
    print(f"  Tool use: {passes['tool_use']}/{total}")
    print(f"  Instruction following: {passes['instruction']}/{total}")
    print(f"  KB grounding: {passes['kb']}/{total}")

    turn_taking_analysis = result.get("turn_taking_analysis")
    if turn_taking_analysis and turn_taking_analysis.get("failed_turns"):
        print(f"\nTurn-taking failures: {turn_taking_analysis['failed_turns']}")

    if result.get("realignment_notes"):
        print(f"\nRealignment applied: {result['realignment_notes'][:200]}...")

    if args.debug:
        print(f"\n✓ Wrote outputs:", file=sys.stderr)
        print(f"  - {run_dir / 'claude_judged.jsonl'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_summary.json'}", file=sys.stderr)
        print(f"  - {run_dir / 'claude_analysis.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
