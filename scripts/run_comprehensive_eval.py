#!/usr/bin/env python3
"""Run comprehensive evaluation with N runs, automatic judging, and aggregate statistics.

Usage:
    uv run python scripts/run_comprehensive_eval.py --model ultravox-v0.7 --service ultravox-realtime --runs 10
    uv run python scripts/run_comprehensive_eval.py --model gpt-realtime --service openai-realtime --runs 10 --benchmark aiwf_medium_context
"""

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation with multiple runs"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., ultravox-v0.7, gpt-realtime)"
    )
    parser.add_argument(
        "--service",
        required=True,
        help="Service name (e.g., ultravox-realtime, openai-realtime)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of evaluation runs (default: 10)"
    )
    parser.add_argument(
        "--benchmark",
        default="aiwf_medium_context",
        help="Benchmark to run (default: aiwf_medium_context)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-opus-4-5",
        help="Model for judging (default: claude-opus-4-5)",
    )
    parser.add_argument(
        "--pipeline", help="Pipeline type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run attempts one at a time until the requested number of complete judged runs is collected",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Maximum total attempts when using --sequential (default: 3x --runs)",
    )
    parser.add_argument(
        "--delay-between-attempts-seconds",
        type=float,
        default=15.0,
        help="Delay between sequential attempts in seconds (default: 15)",
    )
    return parser.parse_args()


async def run_single_eval(
    benchmark: str,
    model: str,
    service: str,
    pipeline: str = None,
    verbose: bool = False,
):
    """Run a single evaluation and return the run directory."""
    from multi_turn_eval.cli import _run

    logger.info(f"Starting evaluation run for {model}")

    try:
        run_dir = await _run(
            benchmark_name=benchmark,
            model=model,
            service=service,
            pipeline_type=pipeline,
            only_turns=None,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Evaluation run failed: {e}")
        raise
    logger.info(f"Evaluation completed: {run_dir}")
    return run_dir


async def judge_run(
    run_dir: Path, benchmark: str, judge_model: str = "claude-opus-4-5"
):
    """Judge a single run and return the summary."""
    from multi_turn_eval.cli import load_benchmark
    from multi_turn_eval.judging.claude_judge import (
        judge_with_claude,
        load_transcript,
        write_outputs,
    )

    logger.info(f"Judging run: {run_dir}")

    # Load benchmark for expected turns
    BenchmarkConfig = load_benchmark(benchmark)
    benchmark_obj = BenchmarkConfig()
    expected_turns = benchmark_obj.turns

    # Load transcript
    records = load_transcript(run_dir)

    # Run judge
    result = await judge_with_claude(
        run_dir,
        only_turns=None,
        debug=False,
        expected_turns=expected_turns,
        judge_model=judge_model,
    )

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
        result.get("judge_model", judge_model),
    )

    # Load and return summary
    summary_path = run_dir / "claude_summary.json"
    summary = json.loads(summary_path.read_text())
    logger.info(f"Judging completed: {run_dir}")
    return summary


def load_run_summary(run_dir: Path) -> dict:
    """Load summary from a judged run."""
    summary_path = run_dir / "claude_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
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


def aggregate_results(
    run_dirs: list[Path], model: str, expected_turns_per_run: int | None = None
) -> dict:
    """Aggregate results from multiple runs."""
    run_entries = []
    all_ttfb = []

    for run_dir in run_dirs:
        # Skip runs that don't have a summary (failed or incomplete runs)
        summary_path = run_dir / "claude_summary.json"
        if not summary_path.exists():
            logger.warning(f"Skipping run without summary: {run_dir}")
            continue

        summary = load_run_summary(run_dir)
        turns_scored = summary.get("turns_scored", 0)
        passes = summary.get("claude_passes", {})

        if expected_turns_per_run is not None and turns_scored != expected_turns_per_run:
            logger.warning(
                "Skipping incomplete run {} (turns_scored={} expected={})".format(
                    run_dir, turns_scored, expected_turns_per_run
                )
            )
            continue

        # Prefer turn-pass from summary, fall back to judged file, then min() approximation.
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
        raise ValueError("No valid run summaries found. All runs may have failed.")

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

    # Calculate pass rates (percentage of total turns)
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

    # Calculate median rate (median of individual run pass rates)
    run_pass_rates = []
    for entry in run_entries:
        turns = entry["turn_pass_total"]
        if turns > 0:
            run_rate = (entry["turn_pass_count"] / turns) * 100
            run_pass_rates.append(run_rate)

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


def print_results_table(results: dict, total_attempted: int = None):
    """Print results in a formatted table."""
    r = results

    print("\n" + "=" * 100)
    print(f"Comprehensive Evaluation Results: {r['model']}")
    if total_attempted and total_attempted != r["num_runs"]:
        print(
            f"Runs: {r['num_runs']}/{total_attempted} successful × {r['total_turns'] // r['num_runs']} turns = {r['total_turns']} total turns"
        )
    else:
        print(
            f"Runs: {r['num_runs']} × {r['total_turns'] // r['num_runs']} turns = {r['total_turns']} total turns"
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
        f"  Turn Pass (strict):   {r['turn_pass']['count']}/{r['turn_pass']['total']} ({r['turn_pass']['rate']:.1f}%)"
    )
    print(
        f"  Tool Use:             {r['tool_use']['count']}/{r['total_turns']} ({r['tool_use']['rate']:.1f}%)"
    )
    print(
        f"  Instruction Following: {r['instruction']['count']}/{r['total_turns']} ({r['instruction']['rate']:.1f}%)"
    )
    print(
        f"  KB Grounding:         {r['kb_grounding']['count']}/{r['total_turns']} ({r['kb_grounding']['rate']:.1f}%)"
    )
    print()


async def main():
    args = parse_args()
    from multi_turn_eval.cli import load_benchmark

    print(f"\n{'=' * 100}")
    print(f"Running Comprehensive Evaluation")
    print(f"{'=' * 100}")
    print(f"Model:       {args.model}")
    print(f"Service:     {args.service}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Runs:        {args.runs}")
    print(f"Judge Model: {args.judge_model}")
    if args.sequential:
        print("Mode:        sequential")
        print(
            "Max Attempts: {}".format(args.max_attempts or (args.runs * 3))
        )
        print(
            "Delay:       {}s".format(
                int(args.delay_between_attempts_seconds)
                if args.delay_between_attempts_seconds.is_integer()
                else args.delay_between_attempts_seconds
            )
        )
    else:
        print("Mode:        parallel")
    print(f"{'=' * 100}\n")

    BenchmarkConfig = load_benchmark(args.benchmark)
    benchmark_obj = BenchmarkConfig()
    expected_turns = len(benchmark_obj.turns)

    if args.sequential:
        max_attempts = args.max_attempts or (args.runs * 3)
        run_dirs = []
        attempts = 0

        print(f"\n{'=' * 100}")
        print(
            f"Collecting {args.runs} complete judged runs sequentially (target: {expected_turns} turns each)"
        )
        print(f"{'=' * 100}\n")

        while len(run_dirs) < args.runs and attempts < max_attempts:
            attempts += 1
            print(
                f"Attempt {attempts}/{max_attempts}: collected {len(run_dirs)}/{args.runs} complete runs"
            )

            run_dir = None
            usable = False

            try:
                run_dir = await run_single_eval(
                    benchmark=args.benchmark,
                    model=args.model,
                    service=args.service,
                    pipeline=args.pipeline,
                    verbose=args.verbose,
                )
                summary = await judge_run(run_dir, args.benchmark, args.judge_model)
                turns_scored = int(summary.get("turns_scored", 0) or 0)
                if turns_scored == expected_turns:
                    run_dirs.append(run_dir)
                    usable = True
                    print(
                        f"✅ Accepted {run_dir.name} ({len(run_dirs)}/{args.runs} complete runs)"
                    )
                else:
                    logger.warning(
                        "Skipping incomplete run {} (turns_scored={} expected={})".format(
                            run_dir, turns_scored, expected_turns
                        )
                    )
                    print(
                        f"⚠️  Skipping incomplete run {run_dir.name}: judged {turns_scored}/{expected_turns} turns"
                    )
            except Exception as e:
                logger.error(f"Attempt {attempts} failed: {e}")
                print(f"⚠️  Attempt {attempts} failed: {e}")

            if len(run_dirs) < args.runs and attempts < max_attempts:
                delay = args.delay_between_attempts_seconds
                if delay > 0:
                    if usable:
                        print(f"Cooling down for {delay:g}s before the next attempt...\n")
                    else:
                        print(f"Retrying after {delay:g}s...\n")
                    await asyncio.sleep(delay)

        if not run_dirs:
            print("\n❌ No complete judged runs were collected. Exiting.")
            sys.exit(1)

        print(f"\n{'=' * 100}")
        print(
            f"Collected {len(run_dirs)}/{args.runs} complete judged runs in {attempts} attempts"
        )
        print(f"{'=' * 100}\n")
    else:
        # Run all evaluations in parallel
        print(f"\n{'=' * 100}")
        print(f"Running {args.runs} evaluations in parallel...")
        print(f"{'=' * 100}\n")

        eval_tasks = [
            run_single_eval(
                benchmark=args.benchmark,
                model=args.model,
                service=args.service,
                pipeline=args.pipeline,
                verbose=args.verbose,
            )
            for _ in range(args.runs)
        ]

        # Gather results, allowing failures
        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        # Filter successful runs
        run_dirs = []
        for i, result in enumerate(eval_results):
            if isinstance(result, Exception):
                logger.error(f"Run {i + 1} failed: {result}")
                print(f"\n⚠️  Run {i + 1} failed, continuing with remaining runs...\n")
            else:
                run_dirs.append(result)

        if not run_dirs:
            print("\n❌ All runs failed. Exiting.")
            sys.exit(1)

        print(f"\n{'=' * 100}")
        print(f"Completed {len(run_dirs)}/{args.runs} evaluation runs")
        print(f"{'=' * 100}\n")

        # Judge all runs in parallel
        print(f"\n{'=' * 100}")
        print(f"Judging {len(run_dirs)} results in parallel...")
        print(f"{'=' * 100}\n")

        judge_tasks = [
            judge_run(run_dir, args.benchmark, args.judge_model) for run_dir in run_dirs
        ]

        # Gather judging results, allowing failures
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Log any judging failures
        for i, result in enumerate(judge_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to judge {run_dirs[i]}: {result}")
                print(f"\n⚠️  Judging failed for run {i + 1}, skipping...\n")

    # Aggregate and display results
    print(f"\n{'=' * 100}")
    print("Aggregating Results")
    print(f"{'=' * 100}\n")

    try:
        results = aggregate_results(
            run_dirs,
            args.model,
            expected_turns_per_run=expected_turns,
        )
        print_results_table(results, total_attempted=args.runs)

        # Save aggregate results
        output_path = (
            Path("runs")
            / args.benchmark
            / f"aggregate_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Aggregate results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        print("\n❌ Failed to aggregate results")
        sys.exit(1)

    print(f"\n{'=' * 100}")
    print("✅ Comprehensive evaluation completed!")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    asyncio.run(main())
