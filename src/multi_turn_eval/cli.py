"""Multi-turn LLM evaluation framework CLI.

Usage:
    uv run multi-turn-eval run aiwf_long_context --model claude-sonnet-4-5 --service anthropic
    uv run multi-turn-eval judge runs/aiwf_long_context/20251213T123456_claude-sonnet-4-5
    uv run multi-turn-eval list-benchmarks
"""

import asyncio
import importlib
import json
import logging
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Service Aliases
# ============================================================================

SERVICE_ALIASES = {
    "openai": "pipecat.services.openai.llm.OpenAILLMService",
    "openai-realtime": "pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService",
    "openrouter": "pipecat.services.openai.llm.OpenAILLMService",  # OpenRouter uses OpenAI-compatible API
    "anthropic": "pipecat.services.anthropic.llm.AnthropicLLMService",
    "google": "pipecat.services.google.llm.GoogleLLMService",
    "gemini-live": "multi_turn_eval.pipelines.realtime.GeminiLiveLLMServiceWithReconnection",
    "bedrock": "pipecat.services.aws.llm.AWSBedrockLLMService",
    "groq": "pipecat.services.groq.llm.GroqLLMService",
    "cerebras": "pipecat.services.cerebras.llm.CerebrasLLMService",
    "ultravox-realtime": "pipecat.services.ultravox.llm.UltravoxRealtimeLLMService",
}


# ============================================================================
# Pipeline Registry
# ============================================================================

PIPELINE_CLASSES = {
    "text": "multi_turn_eval.pipelines.text.TextPipeline",
    "realtime": "multi_turn_eval.pipelines.realtime.RealtimePipeline",
    "grok-realtime": "multi_turn_eval.pipelines.grok_realtime.GrokRealtimePipeline",
    "nova-sonic": "multi_turn_eval.pipelines.nova_sonic.NovaSonicPipeline",
}


# ============================================================================
# Utility Functions
# ============================================================================


def load_service_class(service: str) -> type:
    """Load service class from fully qualified name or alias."""
    class_name = SERVICE_ALIASES.get(service, service)
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def load_benchmark(name: str):
    """Load benchmark by name from benchmarks/ directory."""
    try:
        module = importlib.import_module(f"benchmarks.{name}.config")
        return module.BenchmarkConfig
    except ModuleNotFoundError as e:
        raise click.UsageError(f"Benchmark '{name}' not found: {e}")


def list_available_benchmarks() -> list[str]:
    """Discover available benchmarks by scanning benchmarks/ directory."""
    # Find the benchmarks directory relative to the package or current working directory
    cwd_benchmarks = Path.cwd() / "benchmarks"

    benchmarks = []
    if cwd_benchmarks.exists():
        for d in cwd_benchmarks.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "config.py").exists():
                benchmarks.append(d.name)

    return sorted(benchmarks)


def get_pipeline_class(pipeline_type: str) -> type:
    """Load pipeline class by type name."""
    class_name = PIPELINE_CLASSES.get(pipeline_type)
    if not class_name:
        raise click.UsageError(
            f"Unknown pipeline: {pipeline_type}. Available: {list(PIPELINE_CLASSES.keys())}"
        )
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def infer_pipeline(model: str) -> str:
    """Infer default pipeline from model name pattern."""
    m = model.lower()
    # Grok realtime uses dedicated pipeline for xAI-specific protocol handling
    if m.startswith("grok") and "realtime" in m:
        return "grok-realtime"
    if "realtime" in m:
        return "realtime"
    if "native-audio" in m or "live" in m:
        return "realtime"
    if "ultravox" in m:
        return "realtime"
    if "nova-sonic" in m or "nova_sonic" in m:
        return "nova-sonic"
    return "text"


def create_run_directory(benchmark_name: str, model: str) -> Path:
    """Create timestamped run directory."""
    import uuid

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    # Add a short unique suffix to prevent collisions in parallel runs
    unique_suffix = str(uuid.uuid4())[:8]
    # Sanitize model name for filesystem (replace / and :)
    safe_model = model.replace("/", "_").replace(":", "_")
    run_dir = (
        Path("runs") / benchmark_name / f"{timestamp}_{safe_model}_{unique_suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False):
    """Configure logging to both console and run directory."""
    level = logging.DEBUG if verbose else logging.INFO

    # Remove default loguru handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level="INFO" if not verbose else "DEBUG",
        format="<level>{message}</level>",
    )

    # File handler (always DEBUG for debugging failed runs)
    logger.add(
        run_dir / "run.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {name}: {message}",
    )


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if len(values) >= 20:
        return statistics.quantiles(values, n=20)[18]
    return max(values)


def _summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _p95(values),
        "min": min(values),
        "max": max(values),
    }


def write_native_summary(run_dir: Path, model: str) -> dict | None:
    """Write native summary from transcript.jsonl to native_summary.json."""
    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.exists():
        return None

    rows = []
    with transcript_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return None

    ttfb_values = []
    latency_values = []
    tool_call_turns = 0
    tool_calls_total = 0
    empty_assistant_turns = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for r in rows:
        ttfb = r.get("ttfb_ms")
        if isinstance(ttfb, (int, float)):
            ttfb_values.append(float(ttfb))

        latency = r.get("latency_ms")
        if isinstance(latency, (int, float)):
            latency_values.append(float(latency))

        tool_calls = r.get("tool_calls") or []
        if tool_calls:
            tool_call_turns += 1
            tool_calls_total += len(tool_calls)

        assistant_text = r.get("assistant_text")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            empty_assistant_turns += 1

        tokens = r.get("tokens") or {}
        prompt_tokens += int(tokens.get("prompt_tokens") or 0)
        completion_tokens += int(tokens.get("completion_tokens") or 0)
        total_tokens += int(tokens.get("total_tokens") or 0)

    turns = len(rows)
    summary = {
        "model_name": model,
        "run_dir": str(run_dir),
        "turns": turns,
        "native_performance": {
            "tool_call_turns": tool_call_turns,
            "tool_calls_total": tool_calls_total,
            "empty_assistant_turns": empty_assistant_turns,
            "assistant_response_rate": (
                (turns - empty_assistant_turns) / turns if turns else None
            ),
        },
        "ttfb_ms": _summarize_numeric(ttfb_values),
        "latency_ms": _summarize_numeric(latency_values),
        "tokens": {
            "prompt_total": prompt_tokens,
            "completion_total": completion_tokens,
            "total": total_tokens,
            "avg_total_per_turn": (total_tokens / turns) if turns else None,
        },
    }

    out_path = run_dir / "native_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_native_aggregate(
    output_dir: Path,
    benchmark_name: str,
    model: str,
    run_dirs: list[Path],
    summaries: list[dict],
) -> Path | None:
    """Write aggregate native summary for a multi-run invocation."""
    if not run_dirs or not summaries:
        return None

    ttfb_values = []
    latency_values = []
    total_turns = 0
    tool_call_turns = 0
    tool_calls_total = 0
    empty_assistant_turns = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for s in summaries:
        total_turns += int(s.get("turns", 0))
        native_perf = s.get("native_performance", {})
        tool_call_turns += int(native_perf.get("tool_call_turns", 0))
        tool_calls_total += int(native_perf.get("tool_calls_total", 0))
        empty_assistant_turns += int(native_perf.get("empty_assistant_turns", 0))

        tokens = s.get("tokens", {})
        prompt_tokens += int(tokens.get("prompt_total", 0))
        completion_tokens += int(tokens.get("completion_total", 0))
        total_tokens += int(tokens.get("total", 0))

        transcript_path = Path(s["run_dir"]) / "transcript.jsonl"
        if not transcript_path.exists():
            continue
        with transcript_path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ttfb = row.get("ttfb_ms")
                if isinstance(ttfb, (int, float)):
                    ttfb_values.append(float(ttfb))
                latency = row.get("latency_ms")
                if isinstance(latency, (int, float)):
                    latency_values.append(float(latency))

    aggregate = {
        "model_name": model,
        "benchmark": benchmark_name,
        "runs": len(run_dirs),
        "run_dirs": [str(d) for d in run_dirs],
        "turns": total_turns,
        "native_performance": {
            "tool_call_turns": tool_call_turns,
            "tool_calls_total": tool_calls_total,
            "empty_assistant_turns": empty_assistant_turns,
            "assistant_response_rate": (
                (total_turns - empty_assistant_turns) / total_turns
                if total_turns
                else None
            ),
        },
        "ttfb_ms": _summarize_numeric(ttfb_values),
        "latency_ms": _summarize_numeric(latency_values),
        "tokens": {
            "prompt_total": prompt_tokens,
            "completion_total": completion_tokens,
            "total": total_tokens,
            "avg_total_per_turn": (total_tokens / total_turns) if total_turns else None,
        },
    }

    out_path = output_dir / "native_aggregate.json"
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return out_path


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """Multi-turn LLM evaluation framework."""
    pass


@cli.command()
@click.argument("benchmark_name")
@click.option("--model", required=True, help="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
@click.option("--service", help="Service class name or alias (e.g., openai, anthropic)")
@click.option(
    "--pipeline",
    help="Pipeline type (text, realtime, nova-sonic). Auto-detected if not specified.",
)
@click.option("--only-turns", help="Comma-separated turn indices to run (e.g., 0,1,2)")
@click.option(
    "--runs",
    type=int,
    default=1,
    show_default=True,
    help="Number of full benchmark runs to execute sequentially",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline: Optional[str],
    only_turns: Optional[str],
    runs: int,
    verbose: bool,
):
    """Run a benchmark against an LLM.

    Examples:
        uv run multi-turn-eval run aiwf_long_context --model claude-sonnet-4-5 --service anthropic
        uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai
        uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime --pipeline realtime
    """
    asyncio.run(_run(benchmark_name, model, service, pipeline, only_turns, runs, verbose))


async def _run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    runs: int = 1,
    verbose: bool = False,
):
    """Async implementation of the run command."""
    if runs < 1:
        raise click.UsageError("--runs must be >= 1")

    # Load benchmark
    BenchmarkConfig = load_benchmark(benchmark_name)

    # Infer pipeline if not specified
    if not pipeline_type:
        pipeline_type = infer_pipeline(model)
        click.echo(f"Auto-detected pipeline: {pipeline_type}")

    pipeline_cls = get_pipeline_class(pipeline_type)

    # Check if pipeline requires a service
    requires_service = getattr(pipeline_cls, "requires_service", True)
    if requires_service and not service:
        raise click.UsageError(f"--service is required for {pipeline_type} pipeline")

    # Load service class if provided
    service_class = load_service_class(service) if service else None

    # Parse turn indices if provided
    turn_indices = None
    if only_turns:
        turn_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Running only turns: {turn_indices}")

    base_run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {base_run_dir}")
    if runs > 1:
        iterations_base = base_run_dir / "iterations"
        iterations_base.mkdir(parents=True, exist_ok=True)
    else:
        iterations_base = None

    completed_runs: list[Path] = []
    native_summaries: list[dict] = []
    for run_index in range(1, runs + 1):
        if runs > 1:
            click.echo(f"\n=== Run {run_index}/{runs} ===")

        if runs > 1:
            run_dir = iterations_base / f"run_{run_index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Iteration directory: {run_dir}")
        else:
            run_dir = base_run_dir

        # Setup logging
        setup_logging(run_dir, verbose)

        # Create recorder
        from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder

        recorder = TranscriptRecorder(run_dir, model)

        # Run the pipeline
        try:
            benchmark = BenchmarkConfig()
            pipeline_instance = pipeline_cls(benchmark)
            await pipeline_instance.run(
                recorder=recorder,
                model=model,
                service_class=service_class,
                service_name=service,
                turn_indices=turn_indices,
            )
            completed_runs.append(run_dir)
            click.echo("Completed benchmark run")
            click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")
            native_summary = write_native_summary(run_dir, model)
            if native_summary is not None:
                native_summaries.append(native_summary)
                click.echo(f"  Native summary: {run_dir / 'native_summary.json'}")
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise click.ClickException(str(e))
        finally:
            recorder.close()

    if runs > 1:
        aggregate_path = write_native_aggregate(
            output_dir=base_run_dir,
            benchmark_name=benchmark_name,
            model=model,
            run_dirs=completed_runs,
            summaries=native_summaries,
        )
        click.echo("\nCompleted iteration directories:")
        for run_dir in completed_runs:
            click.echo(f"  - {run_dir}")
        if aggregate_path is not None:
            click.echo(f"Native aggregate: {aggregate_path}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--only-turns", help="Comma-separated turn indices to judge (e.g., 0,1,2)")
@click.option(
    "--only-runs",
    help="For parent run dirs: comma-separated iteration selectors (e.g., 1,3,9 or run_01,run_03)",
)
@click.option("--judge-model", default="claude-opus-4-5", help="Model for judging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def judge(
    run_dir: str,
    only_turns: Optional[str],
    only_runs: Optional[str],
    judge_model: str,
    debug: bool,
):
    """Judge a completed benchmark run.

    Examples:
        uv run multi-turn-eval judge runs/aiwf_long_context/20251213T123456_claude-sonnet-4-5
        uv run multi-turn-eval judge runs/aiwf_long_context/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2
    """
    run_path = Path(run_dir)

    # Parse turn indices
    turn_indices_set: Optional[set[int]] = None
    if only_turns:
        turn_indices_set = {int(i.strip()) for i in only_turns.split(",")}

    # Infer benchmark from path: runs/{benchmark}/{timestamp}_{model}/
    benchmark_name = run_path.parent.name

    # Load benchmark for expected turns
    try:
        BenchmarkConfig = load_benchmark(benchmark_name)
        benchmark = BenchmarkConfig()
        expected_turns = benchmark.turns
    except Exception:
        # Fall back to legacy turns module
        click.echo(f"Could not load benchmark '{benchmark_name}', using legacy turns module")
        from turns import turns as expected_turns

    # Run judge
    from multi_turn_eval.judging.claude_judge import judge_with_claude, load_transcript, write_outputs

    def judge_one(run_path_single: Path) -> dict:
        transcript_path_single = run_path_single / "transcript.jsonl"
        if not transcript_path_single.exists():
            raise click.UsageError(f"No transcript found at {transcript_path_single}")

        records = load_transcript(run_path_single)
        if turn_indices_set is not None:
            records = [r for r in records if r["turn"] in turn_indices_set]

        try:
            result = asyncio.run(
                judge_with_claude(
                    run_path_single,
                    only_turns=turn_indices_set,
                    debug=debug,
                    expected_turns=expected_turns,
                )
            )
        except Exception as e:
            raise click.ClickException(f"Judgment failed for {run_path_single}: {e}")

        write_outputs(
            run_path_single,
            records,
            result["judgments"],
            result["summary"],
            result["model_name"],
            result.get("realignment_notes", ""),
            result.get("function_tracking", {}),
            result.get("turn_taking_analysis"),
        )

        summary_path_single = run_path_single / "claude_summary.json"
        return json.loads(summary_path_single.read_text())

    transcript_path = run_path / "transcript.jsonl"
    if transcript_path.exists():
        # Single-run directory (legacy and runs=1 behavior)
        summary = judge_one(run_path)
        passes = summary.get("claude_passes", {})
        total = summary.get("turns_scored", 0)

        click.echo(f"Judged {total} turns (with turn-taking analysis)")
        click.echo(f"  Turn-taking: {passes.get('turn_taking', total)}/{total}")
        click.echo(f"  Tool use: {passes.get('tool_use_correct', 0)}/{total}")
        click.echo(f"  Instruction following: {passes.get('instruction_following', 0)}/{total}")
        click.echo(f"  KB grounding: {passes.get('kb_grounding', 0)}/{total}")

        turn_taking_failures = summary.get("turn_taking_failures", [])
        if turn_taking_failures:
            click.echo(f"\nTurn-taking failures: {turn_taking_failures}")
        return

    # Parent directory with iterations/run_XX
    iterations_dir = run_path / "iterations"
    iteration_dirs = []
    if iterations_dir.exists():
        iteration_dirs = sorted(
            [
                d
                for d in iterations_dir.iterdir()
                if d.is_dir() and (d / "transcript.jsonl").exists()
            ]
        )

    if not iteration_dirs:
        raise click.UsageError(
            f"No transcript found at {transcript_path} and no iteration transcripts under {iterations_dir}"
        )

    all_iteration_dirs = list(iteration_dirs)

    if only_runs:
        selected: set[str] = set()
        for raw in only_runs.split(","):
            token = raw.strip()
            if not token:
                continue
            if token.isdigit():
                selected.add(f"run_{int(token):02d}")
            else:
                selected.add(token)
        filtered = [d for d in iteration_dirs if d.name in selected]
        if not filtered:
            raise click.UsageError(
                f"--only-runs matched no iteration dirs. Available: {', '.join(d.name for d in iteration_dirs)}"
            )
        iteration_dirs = filtered

    if only_runs:
        click.echo(
            f"Found {len(all_iteration_dirs)} iterations total. Judging {len(iteration_dirs)} selected runs..."
        )
    else:
        click.echo(f"Found {len(iteration_dirs)} iterations. Judging each run...")
    iteration_summaries: list[dict] = []
    successful_iteration_dirs: list[Path] = []
    failed_iterations: list[dict] = []
    for idx, iteration_dir in enumerate(iteration_dirs, start=1):
        click.echo(f"\n[{idx}/{len(iteration_dirs)}] {iteration_dir.name}")
        try:
            summary = judge_one(iteration_dir)
            iteration_summaries.append(summary)
            successful_iteration_dirs.append(iteration_dir)
        except Exception as e:
            failed_iterations.append({"run": iteration_dir.name, "error": str(e)})
            click.echo(f"  Failed: {e}")

    if not iteration_summaries and only_runs:
        click.echo("No selected iterations were successfully judged in this invocation.")

    # Aggregate judged output at parent run dir using all iteration summaries available.
    aggregate_iteration_dirs: list[Path] = []
    aggregate_iteration_summaries: list[dict] = []
    unreadable_existing_summaries: list[dict] = []
    for d in all_iteration_dirs:
        summary_path = d / "claude_summary.json"
        if not summary_path.exists():
            continue
        try:
            aggregate_iteration_summaries.append(json.loads(summary_path.read_text()))
            aggregate_iteration_dirs.append(d)
        except Exception as e:
            unreadable_existing_summaries.append({"run": d.name, "error": str(e)})

    if not aggregate_iteration_summaries:
        raise click.ClickException(
            "No iteration claude_summary.json files available to aggregate. "
            "Run judge on at least one iteration first."
        )

    pass_keys = ["turn_taking", "tool_use_correct", "instruction_following", "kb_grounding"]
    total_turns = sum(int(s.get("turns_scored", 0)) for s in aggregate_iteration_summaries)
    aggregate_passes = {
        key: sum(int(s.get("claude_passes", {}).get(key, 0)) for s in aggregate_iteration_summaries)
        for key in pass_keys
    }
    aggregate_failures = []
    turn_taking_affected_instruction = 0
    for s, d in zip(aggregate_iteration_summaries, aggregate_iteration_dirs):
        for t in s.get("turn_taking_failures", []):
            aggregate_failures.append({"run": d.name, "turn": t})
        turn_taking_affected_instruction += int(s.get("turn_taking_affected_instruction", 0))

    aggregate_summary = {
        "model_name": aggregate_iteration_summaries[0].get("model_name"),
        "claude_passes": aggregate_passes,
        "turns_scored": total_turns,
        "judge_version": "claude-agent-sdk-v4-turn-taking-aggregate",
        "judge_model": judge_model,
        "judged_at": datetime.utcnow().isoformat() + "Z",
        "is_aggregate": True,
        "iteration_count": len(aggregate_iteration_dirs),
        "iterations": [d.name for d in aggregate_iteration_dirs],
        "total_iterations_in_folder": len(all_iteration_dirs),
        "missing_summary_iterations": [
            d.name for d in all_iteration_dirs if not (d / "claude_summary.json").exists()
        ],
        "unreadable_summary_iterations": unreadable_existing_summaries,
        "failed_iteration_count": len(failed_iterations),
        "failed_iterations": failed_iterations,
        "turn_taking_failures": aggregate_failures,
        "turn_taking_affected_instruction": turn_taking_affected_instruction,
    }

    (run_path / "claude_summary.json").write_text(json.dumps(aggregate_summary, indent=2) + "\n")

    analysis_lines = [
        "# Claude Aggregate Evaluation",
        "",
        f"**Parent Run**: `{run_path}`",
        f"**Iterations In Folder**: {len(all_iteration_dirs)}",
        f"**Iterations Requested This Run**: {len(iteration_dirs)}",
        f"**Iterations Judged This Run**: {len(successful_iteration_dirs)}",
        f"**Iterations Failed**: {len(failed_iterations)}",
        f"**Iterations Included In Aggregate**: {len(aggregate_iteration_dirs)}",
        f"**Turns Scored**: {total_turns}",
        "",
        "## Aggregate Metrics",
        "",
        f"- **Turn-Taking**: {aggregate_passes['turn_taking']}/{total_turns}",
        f"- **Tool Use Correct**: {aggregate_passes['tool_use_correct']}/{total_turns}",
        f"- **Instruction Following**: {aggregate_passes['instruction_following']}/{total_turns}",
        f"- **KB Grounding**: {aggregate_passes['kb_grounding']}/{total_turns}",
        "",
        "## Failed Iterations",
        "",
    ]
    if failed_iterations:
        analysis_lines.extend([f"- `{f['run']}`: {f['error']}" for f in failed_iterations])
    else:
        analysis_lines.append("- None")
    analysis_lines.extend([
        "",
        "## Missing/Unreadable Existing Summaries",
        "",
    ])
    missing_summary_iterations = [
        d.name for d in all_iteration_dirs if not (d / "claude_summary.json").exists()
    ]
    if missing_summary_iterations:
        for name in missing_summary_iterations:
            analysis_lines.append(f"- Missing `claude_summary.json`: `{name}`")
    if unreadable_existing_summaries:
        for item in unreadable_existing_summaries:
            analysis_lines.append(f"- Unreadable summary `{item['run']}`: {item['error']}")
    if not missing_summary_iterations and not unreadable_existing_summaries:
        analysis_lines.append("- None")
    analysis_lines.extend([
        "",
        "Per-iteration outputs remain in `iterations/run_XX/`.",
    ])
    (run_path / "claude_analysis.md").write_text("\n".join(analysis_lines) + "\n")

    click.echo(f"\nJudged {len(successful_iteration_dirs)}/{len(iteration_dirs)} requested iterations at parent path")
    click.echo(f"  Summary: {run_path / 'claude_summary.json'}")
    click.echo(f"  Analysis: {run_path / 'claude_analysis.md'}")
    if failed_iterations:
        click.echo(f"  Failed iterations: {len(failed_iterations)}")
    click.echo(
        f"  Aggregate includes: {len(aggregate_iteration_dirs)}/{len(all_iteration_dirs)} iterations in folder"
    )
    click.echo(f"  Turn-taking: {aggregate_passes.get('turn_taking', total_turns)}/{total_turns}")
    click.echo(f"  Tool use: {aggregate_passes.get('tool_use_correct', 0)}/{total_turns}")
    click.echo(
        f"  Instruction following: {aggregate_passes.get('instruction_following', 0)}/{total_turns}"
    )
    click.echo(f"  KB grounding: {aggregate_passes.get('kb_grounding', 0)}/{total_turns}")


@cli.command("list-benchmarks")
def list_benchmarks():
    """List available benchmarks."""
    benchmarks = list_available_benchmarks()
    if not benchmarks:
        click.echo("No benchmarks found in benchmarks/ directory")
        return

    click.echo("Available benchmarks:")
    for name in benchmarks:
        try:
            BenchmarkConfig = load_benchmark(name)
            description = getattr(BenchmarkConfig, "description", "")
            click.echo(f"  {name}: {description}")
        except Exception:
            click.echo(f"  {name}")


@cli.command("list-pipelines")
def list_pipelines():
    """List available pipelines."""
    click.echo("Available pipelines:")
    for name, cls_path in PIPELINE_CLASSES.items():
        click.echo(f"  {name}: {cls_path}")


@cli.command("list-aliases")
def list_aliases():
    """List service aliases."""
    click.echo("Service aliases:")
    for alias, cls_path in SERVICE_ALIASES.items():
        click.echo(f"  {alias}: {cls_path}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
