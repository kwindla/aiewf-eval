#!/usr/bin/env python3
"""Aggregate results from existing runs without re-running evaluations.

Usage:
    # Aggregate all ultravox runs
    uv run python scripts/aggregate_existing_runs.py --model ultravox-v0.7 --benchmark aiwf_medium_context

    # Aggregate specific runs by pattern
    uv run python scripts/aggregate_existing_runs.py --pattern "runs/aiwf_medium_context/*ultravox*"

    # Aggregate runs from a specific time range
    uv run python scripts/aggregate_existing_runs.py --model ultravox-v0.7 --after 20251216T160000
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate results from existing runs")
    parser.add_argument("--model", help="Model name to aggregate (e.g., ultravox-v0.7)")
    parser.add_argument(
        "--benchmark",
        default="aiwf_medium_context",
        help="Benchmark name (default: aiwf_medium_context)",
    )
    parser.add_argument(
        "--pattern",
        help="Glob pattern for run directories (e.g., 'runs/aiwf_medium_context/*ultravox*')",
    )
    parser.add_argument(
        "--after",
        help="Only include runs after this timestamp (format: YYYYMMDDTHHmmss)",
    )
    parser.add_argument(
        "--before",
        help="Only include runs before this timestamp (format: YYYYMMDDTHHmmss)",
    )
    parser.add_argument(
        "--output", help="Output file for aggregate results (default: auto-generated)"
    )
    return parser.parse_args()


def load_run_summary(run_dir: Path) -> dict:
    """Load summary from a judged run."""
    summary_path = run_dir / "claude_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def load_ttfb_from_transcript(run_dir: Path) -> list[float]:
    """Load TTFB values from transcript file."""
    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.exists():
        return []

    ttfb_values = []
    with transcript_path.open() as f:
        for line in f:
            record = json.loads(line)
            if not record.get("recovery_turn") and int(record.get("turn", -1)) == 0:
                continue
            ttfb_ms = record.get("ttfb_ms")
            if ttfb_ms is not None:
                ttfb_values.append(ttfb_ms)

    return ttfb_values


def load_turn_pass_from_judged(run_dir: Path) -> tuple[int, int]:
    """Count strict per-turn passes from claude_judged.jsonl."""
    judged_path = run_dir / "claude_judged.jsonl"
    if not judged_path.exists():
        return 0, 0

    passed = 0
    total = 0
    with judged_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores = rec.get("scores")
            if not isinstance(scores, dict):
                continue
            total += 1
            if (
                scores.get("tool_use_correct", False)
                and scores.get("instruction_following", False)
                and scores.get("kb_grounding", False)
            ):
                passed += 1

    return passed, total


def find_run_directories(args) -> list[Path]:
    """Find run directories based on search criteria."""
    if args.pattern:
        # Use explicit pattern
        run_dirs = list(Path().glob(args.pattern))
    elif args.model:
        # Search by model name
        runs_base = Path("runs") / args.benchmark
        if not runs_base.exists():
            logger.error(f"Runs directory not found: {runs_base}")
            return []

        # Find all directories containing the model name
        model_safe = args.model.replace("/", "_").replace(":", "_")
        run_dirs = [
            d for d in runs_base.iterdir() if d.is_dir() and model_safe in d.name
        ]
    else:
        logger.error("Either --model or --pattern must be specified")
        return []

    # Filter by time if specified
    if args.after:
        after_time = datetime.strptime(args.after, "%Y%m%dT%H%M%S")
        run_dirs = [
            d
            for d in run_dirs
            if datetime.fromtimestamp(d.stat().st_mtime) >= after_time
        ]

    if args.before:
        before_time = datetime.strptime(args.before, "%Y%m%dT%H%M%S")
        run_dirs = [
            d
            for d in run_dirs
            if datetime.fromtimestamp(d.stat().st_mtime) <= before_time
        ]

    # Sort by modification time
    run_dirs.sort(key=lambda d: d.stat().st_mtime)

    return run_dirs


def aggregate_results(run_dirs: list[Path], model: str) -> dict:
    """Aggregate results from multiple runs."""
    run_entries = []
    all_ttfb = []
    skipped = []

    for run_dir in run_dirs:
        # Skip runs that don't have a summary (failed or incomplete runs)
        summary = load_run_summary(run_dir)
        if summary is None:
            logger.warning(f"Skipping run without summary: {run_dir}")
            skipped.append(run_dir)
            continue

        turns_scored = summary.get("turns_scored", 0)
        passes = summary.get("claude_passes", {})

        # Prefer strict turn-pass from summary, then judged file, then fallback.
        turn_pass_count = 0
        turn_pass_total = 0
        turn_pass = summary.get("turn_pass")
        if isinstance(turn_pass, dict):
            turn_pass_count = int(turn_pass.get("count", 0) or 0)
            turn_pass_total = int(turn_pass.get("total", 0) or 0)

        if turn_pass_total <= 0:
            turn_pass_count, turn_pass_total = load_turn_pass_from_judged(run_dir)

        if turn_pass_total <= 0 and turns_scored > 0:
            turn_pass_total = turns_scored
            turn_pass_count = min(
                passes.get("tool_use_correct", 0),
                passes.get("instruction_following", 0),
                passes.get("kb_grounding", 0),
            )

        run_entries.append(
            {
                "summary": summary,
                "turns_scored": turns_scored,
                "passes": passes,
                "turn_pass_count": turn_pass_count,
                "turn_pass_total": turn_pass_total,
            }
        )

        # Extract TTFB from transcript
        ttfb_values = load_ttfb_from_transcript(run_dir)
        all_ttfb.extend(ttfb_values)

    # Check if we have any valid summaries
    if not run_entries:
        raise ValueError(
            "No valid run summaries found. All runs may have failed or not been judged yet."
        )

    # Calculate total turns across all runs
    total_turns = sum(e["turns_scored"] for e in run_entries)

    # Aggregate pass counts
    tool_use_total = sum(
        e["passes"].get("tool_use_correct", 0) for e in run_entries
    )
    instruction_total = sum(
        e["passes"].get("instruction_following", 0) for e in run_entries
    )
    kb_grounding_total = sum(
        e["passes"].get("kb_grounding", 0) for e in run_entries
    )
    turn_pass_total = sum(e["turn_pass_count"] for e in run_entries)
    turn_pass_denominator = sum(e["turn_pass_total"] for e in run_entries)

    # Calculate rates
    tool_use_rate = (tool_use_total / total_turns * 100) if total_turns > 0 else 0
    instruction_rate = (instruction_total / total_turns * 100) if total_turns > 0 else 0
    kb_grounding_rate = (
        (kb_grounding_total / total_turns * 100) if total_turns > 0 else 0
    )
    overall_pass_rate = (
        (turn_pass_total / turn_pass_denominator * 100)
        if turn_pass_denominator > 0
        else 0
    )

    # Median of per-run strict turn pass rates
    run_pass_rates = []
    for entry in run_entries:
        turns = entry["turn_pass_total"]
        if turns > 0:
            run_pass_rates.append(entry["turn_pass_count"] / turns * 100)

    median_rate = statistics.median(run_pass_rates) if run_pass_rates else 0

    # Calculate TTFB statistics
    ttfb_med = statistics.median(all_ttfb) if all_ttfb else None
    ttfb_p95 = (
        statistics.quantiles(all_ttfb, n=20)[18] if len(all_ttfb) >= 20 else None
    )  # 95th percentile
    ttfb_max = max(all_ttfb) if all_ttfb else None

    return {
        "model": model,
        "total_turns": total_turns,
        "num_runs": len(run_entries),
        "num_skipped": len(skipped),
        "skipped_runs": [str(d) for d in skipped],
        "turn_pass": {
            "count": turn_pass_total,
            "total": turn_pass_denominator,
            "rate": overall_pass_rate,
        },
        "tool_use": {
            "count": tool_use_total,
            "total": total_turns,
            "rate": tool_use_rate,
        },
        "instruction": {
            "count": instruction_total,
            "total": total_turns,
            "rate": instruction_rate,
        },
        "kb_grounding": {
            "count": kb_grounding_total,
            "total": total_turns,
            "rate": kb_grounding_rate,
        },
        "pass_rate": overall_pass_rate,
        "median_rate": median_rate,
        "ttfb_med": ttfb_med,
        "ttfb_p95": ttfb_p95,
        "ttfb_max": ttfb_max,
    }


def format_ms(value_ms: float | None) -> str:
    """Format milliseconds for display."""
    if value_ms is None:
        return "N/A"
    return f"{int(value_ms)}ms"


def print_results_table(results: dict):
    """Print results in a formatted table."""
    r = results

    print("\n" + "=" * 100)
    print(f"Aggregate Results: {r['model']}")
    if r["num_skipped"] > 0:
        print(f"Runs: {r['num_runs']} successful, {r['num_skipped']} skipped")
    else:
        print(f"Runs: {r['num_runs']}")
    print(
        f"Total turns: {r['total_turns']} ({r['num_runs']} × {r['total_turns'] // r['num_runs']})"
    )
    print("=" * 100)
    print()

    # Header
    print(
        "| Model                         | Turn Pass | Tool Use  | Instruction | KB Ground | Pass Rate | Median Rate | TTFB Med | TTFB P95 | TTFB Max |"
    )
    print(
        "|-------------------------------|-----------|-----------|-------------|-----------|-----------|-------------|----------|----------|----------|"
    )

    # Data row
    model_name = r["model"][:29].ljust(29)
    turn_pass = f"{r['turn_pass']['count']}/{r['turn_pass']['total']}".ljust(9)
    tool_use = f"{r['tool_use']['count']}/{r['total_turns']}".ljust(9)
    instruction = f"{r['instruction']['count']}/{r['total_turns']}".ljust(11)
    kb_grounding = f"{r['kb_grounding']['count']}/{r['total_turns']}".ljust(9)
    pass_rate = f"{r['pass_rate']:.1f}%".ljust(9)
    median_rate = f"{r['median_rate']:.1f}%".ljust(11)
    ttfb_med = format_ms(r["ttfb_med"]).ljust(8)
    ttfb_p95 = format_ms(r["ttfb_p95"]).ljust(8)
    ttfb_max = format_ms(r["ttfb_max"]).ljust(8)

    print(
        f"| {model_name} | {turn_pass} | {tool_use} | {instruction} | {kb_grounding} | {pass_rate} | {median_rate} | {ttfb_med} | {ttfb_p95} | {ttfb_max} |"
    )
    print()

    # Individual metrics
    print("Detailed Breakdown:")
    print(
        f"  Turn Pass (strict):    {r['turn_pass']['count']}/{r['turn_pass']['total']} ({r['turn_pass']['rate']:.1f}%)"
    )
    print(
        f"  Tool Use:              {r['tool_use']['count']}/{r['total_turns']} ({r['tool_use']['rate']:.1f}%)"
    )
    print(
        f"  Instruction Following: {r['instruction']['count']}/{r['total_turns']} ({r['instruction']['rate']:.1f}%)"
    )
    print(
        f"  KB Grounding:          {r['kb_grounding']['count']}/{r['total_turns']} ({r['kb_grounding']['rate']:.1f}%)"
    )
    print()

    if r["num_skipped"] > 0:
        print(f"Skipped runs ({r['num_skipped']}):")
        for run_dir in r["skipped_runs"]:
            print(f"  - {run_dir}")
        print()


def main():
    args = parse_args()

    print(f"\n{'=' * 100}")
    print("Aggregating Existing Runs")
    print(f"{'=' * 100}\n")

    # Find run directories
    run_dirs = find_run_directories(args)

    if not run_dirs:
        print("❌ No run directories found matching criteria")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    print()

    # Determine model name
    model = args.model
    if not model and run_dirs:
        # Try to infer from first run directory name
        first_dir = run_dirs[0].name
        # Format: YYYYMMDDTHHmmss_model-name
        parts = first_dir.split("_", 1)
        if len(parts) == 2:
            model = parts[1]

    if not model:
        model = "unknown"

    # Aggregate results
    try:
        results = aggregate_results(run_dirs, model)
        print_results_table(results)

        # Save aggregate results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                Path("runs")
                / args.benchmark
                / f"aggregate_{model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Aggregate results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        print(f"\n❌ Failed to aggregate results: {e}")
        sys.exit(1)

    print(f"\n{'=' * 100}")
    print("✅ Aggregation completed!")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
