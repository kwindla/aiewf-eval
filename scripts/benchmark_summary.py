#!/usr/bin/env python3
"""Generate benchmark summary table for multiple runs.

Usage:
    # Summarize specific runs by pattern
    uv run python scripts/benchmark_summary.py "runs/aiwf_medium_context/20260111T*ultravox*"

    # Multiple patterns for different models
    uv run python scripts/benchmark_summary.py \
        "runs/aiwf_medium_context/20260111T*ultravox*" \
        "runs/aiwf_medium_context/20260111T*gpt-realtime*" \
        "runs/aiwf_medium_context/20260111T*grok-realtime*" \
        "runs/aiwf_medium_context/20260110T22*gemini*" \
        "runs/aiwf_medium_context/20260110T23*gemini*"

    # Exclude specific runs
    uv run python scripts/benchmark_summary.py "runs/aiwf_medium_context/20260111T*grok*" \
        --exclude "runs/aiwf_medium_context/20260111T005820_grok-realtime_5fbc589b"
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark summary table")
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Glob patterns for run directories",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Run directories to exclude (can be specified multiple times)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of ASCII table",
    )
    return parser.parse_args()


MODEL_ALIASES = {
    "gemini-2.5-flash-native-audio-preview-12-2025": "gemini-live",
}


def get_model_name(run_dir: Path) -> str:
    """Extract model name from run directory name."""
    # Format: YYYYMMDDTHHmmss_model-name_hash
    parts = run_dir.name.split("_", 1)
    if len(parts) >= 2:
        # Remove trailing hash if present
        model_parts = parts[1].rsplit("_", 1)
        if len(model_parts) == 2 and len(model_parts[1]) == 8:
            name = model_parts[0]
        else:
            name = parts[1]
        # Apply alias if available
        return MODEL_ALIASES.get(name, name)
    return run_dir.name


def get_run_data(run_dir: Path) -> dict | None:
    """Extract scores and V2V data from run."""
    judged_file = run_dir / "claude_judged.jsonl"
    if not judged_file.exists():
        return None

    scores = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(judged_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                if "scores" not in data or not isinstance(data["scores"], dict):
                    continue
                for key, value in data["scores"].items():
                    if isinstance(value, bool):
                        scores[key]["total"] += 1
                        if value:
                            scores[key]["correct"] += 1
            except json.JSONDecodeError:
                continue

    # Get V2V metrics from analyze script
    non_tool_v2vs = []
    tool_v2vs = []
    silence_pads = []

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/analyze_turn_metrics.py",
                str(run_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            metrics = json.loads(result.stdout)
            for turn in metrics.get("turns", []):
                v2v = turn.get("wav_v2v_ms")
                silence = turn.get("silent_pad_silero_ms")
                has_tool = turn.get("has_tool_call", False)

                if v2v is not None and v2v > 0:
                    if has_tool:
                        tool_v2vs.append(v2v)
                    else:
                        non_tool_v2vs.append(v2v)

                if silence is not None:
                    silence_pads.append(silence)
    except Exception:
        pass

    return {
        "scores": dict(scores),
        "non_tool_v2vs": non_tool_v2vs,
        "tool_v2vs": tool_v2vs,
        "silence_pads": silence_pads,
    }


def aggregate_runs(runs: list[Path]) -> dict:
    """Aggregate data across runs."""
    all_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    all_non_tool_v2vs = []
    all_tool_v2vs = []
    all_silence_pads = []
    valid_runs = 0

    for run in runs:
        data = get_run_data(run)
        if not data:
            continue

        valid_runs += 1

        for key, val in data["scores"].items():
            all_scores[key]["correct"] += val["correct"]
            all_scores[key]["total"] += val["total"]

        all_non_tool_v2vs.extend(data["non_tool_v2vs"])
        all_tool_v2vs.extend(data["tool_v2vs"])
        all_silence_pads.extend(data["silence_pads"])

    return {
        "valid_runs": valid_runs,
        "total_runs": len(runs),
        "scores": dict(all_scores),
        "non_tool_v2vs": all_non_tool_v2vs,
        "tool_v2vs": all_tool_v2vs,
        "silence_pads": all_silence_pads,
    }


def format_score(scores: dict, key: str) -> str:
    """Format score as 'correct/total'."""
    s = scores.get(key, {})
    return f"{s.get('correct', 0)}/{s.get('total', 0)}"


def format_ms(values: list, stat: str = "median") -> str:
    """Format milliseconds statistic."""
    if not values:
        return "N/A"
    if stat == "median":
        return f"{int(statistics.median(values))}ms"
    elif stat == "max":
        return f"{int(max(values))}ms"
    elif stat == "mean":
        return f"{int(statistics.mean(values))}ms"
    return "N/A"


def calculate_pass_rate(scores: dict, total_turns: int) -> float:
    """Calculate pass rate as minimum of all categories."""
    if total_turns == 0:
        return 0.0
    tu = scores.get("tool_use_correct", {}).get("correct", 0)
    ifu = scores.get("instruction_following", {}).get("correct", 0)
    kb = scores.get("kb_grounding", {}).get("correct", 0)
    return min(tu, ifu, kb) / total_turns * 100


def print_ascii_table(model_results: dict[str, dict]):
    """Print results as ASCII table."""
    # Calculate column widths
    col_widths = {
        "model": 17,
        "tool": 7,
        "instr": 11,
        "kb": 8,
        "turn": 7,
        "pass": 8,
        "v2v_med": 13,
        "v2v_max": 13,
        "tool_v2v": 10,
        "silence": 12,
    }

    # Header
    sep = "-" * (sum(col_widths.values()) + 3 * len(col_widths) + 1)
    print(sep)
    print(
        f"| {'Model':<{col_widths['model']}} "
        f"| {'Tool':<{col_widths['tool']}} "
        f"| {'Instruction':<{col_widths['instr']}} "
        f"| {'KB':<{col_widths['kb']}} "
        f"| {'Turn':<{col_widths['turn']}} "
        f"| {'Pass':<{col_widths['pass']}} "
        f"| {'Non-Tool V2V':<{col_widths['v2v_med']}} "
        f"| {'Non-Tool V2V':<{col_widths['v2v_max']}} "
        f"| {'Tool V2V':<{col_widths['tool_v2v']}} "
        f"| {'Silence Pad':<{col_widths['silence']}} |"
    )
    print(
        f"| {'':<{col_widths['model']}} "
        f"| {'Use':<{col_widths['tool']}} "
        f"| {'':<{col_widths['instr']}} "
        f"| {'Ground':<{col_widths['kb']}} "
        f"| {'Ok':<{col_widths['turn']}} "
        f"| {'Rate':<{col_widths['pass']}} "
        f"| {'Med':<{col_widths['v2v_med']}} "
        f"| {'Max':<{col_widths['v2v_max']}} "
        f"| {'Mean':<{col_widths['tool_v2v']}} "
        f"| {'Mean':<{col_widths['silence']}} |"
    )
    print(sep)

    # Data rows
    for model_name, data in model_results.items():
        scores = data["scores"]
        total_turns = scores.get("tool_use_correct", {}).get("total", 0)
        pass_rate = calculate_pass_rate(scores, total_turns)

        print(
            f"| {model_name:<{col_widths['model']}} "
            f"| {format_score(scores, 'tool_use_correct'):<{col_widths['tool']}} "
            f"| {format_score(scores, 'instruction_following'):<{col_widths['instr']}} "
            f"| {format_score(scores, 'kb_grounding'):<{col_widths['kb']}} "
            f"| {format_score(scores, 'turn_taking'):<{col_widths['turn']}} "
            f"| {pass_rate:>5.1f}% "
            f"| {format_ms(data['non_tool_v2vs'], 'median'):<{col_widths['v2v_med']}} "
            f"| {format_ms(data['non_tool_v2vs'], 'max'):<{col_widths['v2v_max']}} "
            f"| {format_ms(data['tool_v2vs'], 'mean'):<{col_widths['tool_v2v']}} "
            f"| {format_ms(data['silence_pads'], 'mean'):<{col_widths['silence']}} |"
        )
        print(sep)


def main():
    args = parse_args()

    # Collect all run directories from patterns
    all_runs = []
    exclude_set = set(Path(p).resolve() for p in args.exclude)

    for pattern in args.patterns:
        if not pattern.strip():
            continue
        runs = list(Path().glob(pattern))
        for run in runs:
            if run.is_dir() and run.resolve() not in exclude_set:
                all_runs.append(run)

    if not all_runs:
        print("No run directories found matching the given patterns.", file=sys.stderr)
        sys.exit(1)

    # Group runs by model
    runs_by_model: dict[str, list[Path]] = defaultdict(list)
    for run in all_runs:
        model = get_model_name(run)
        runs_by_model[model].append(run)

    # Aggregate results for each model
    model_results = {}
    for model, runs in sorted(runs_by_model.items()):
        print(f"Processing {model} ({len(runs)} runs)...", file=sys.stderr)
        model_results[model] = aggregate_runs(sorted(runs))

    # Output
    if args.json:
        print(json.dumps(model_results, indent=2))
    else:
        print()
        print_ascii_table(model_results)
        print()

        # Print notes about incomplete runs
        for model, data in model_results.items():
            if data["valid_runs"] < data["total_runs"]:
                print(
                    f"Note: {model} has {data['valid_runs']}/{data['total_runs']} valid runs"
                )


if __name__ == "__main__":
    main()
