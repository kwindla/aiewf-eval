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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
):
    """Run a benchmark against an LLM.

    Examples:
        uv run multi-turn-eval run aiwf_long_context --model claude-sonnet-4-5 --service anthropic
        uv run multi-turn-eval run aiwf_medium_context --model gpt-4o --service openai
        uv run multi-turn-eval run aiwf_medium_context --model gpt-realtime --service openai-realtime --pipeline realtime
    """
    asyncio.run(_run(benchmark_name, model, service, pipeline, only_turns, verbose))


async def _run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
):
    """Async implementation of the run command."""
    # Load benchmark
    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()

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

    # Create output directory
    run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {run_dir}")

    # Setup logging
    setup_logging(run_dir, verbose)

    # Create recorder
    from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder

    recorder = TranscriptRecorder(run_dir, model)

    # Parse turn indices if provided
    turn_indices = None
    if only_turns:
        turn_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Running only turns: {turn_indices}")

    # Run the pipeline
    try:
        pipeline_instance = pipeline_cls(benchmark)
        await pipeline_instance.run(
            recorder=recorder,
            model=model,
            service_class=service_class,
            service_name=service,
            turn_indices=turn_indices,
        )
        click.echo(f"Completed benchmark run")
        click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise click.ClickException(str(e))
    finally:
        recorder.close()


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--only-turns", help="Comma-separated turn indices to judge (e.g., 0,1,2)")
@click.option("--judge-model", default="claude-opus-4-5", help="Model for judging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def judge(
    run_dir: str,
    only_turns: Optional[str],
    judge_model: str,
    debug: bool,
):
    """Judge a completed benchmark run.

    Examples:
        uv run multi-turn-eval judge runs/aiwf_long_context/20251213T123456_claude-sonnet-4-5
        uv run multi-turn-eval judge runs/aiwf_long_context/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2
    """
    run_path = Path(run_dir)

    # Infer benchmark from path: runs/{benchmark}/{timestamp}_{model}/
    benchmark_name = run_path.parent.name

    # Load transcript
    transcript_path = run_path / "transcript.jsonl"
    if not transcript_path.exists():
        raise click.UsageError(f"No transcript found at {transcript_path}")

    # Parse turn indices
    turn_indices_set: Optional[set[int]] = None
    if only_turns:
        turn_indices_set = {int(i.strip()) for i in only_turns.split(",")}

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

    records = load_transcript(run_path)
    if turn_indices_set is not None:
        records = [r for r in records if r["turn"] in turn_indices_set]

    try:
        result = asyncio.run(
            judge_with_claude(
                run_path,
                only_turns=turn_indices_set,
                debug=debug,
                expected_turns=expected_turns,
            )
        )
    except Exception as e:
        raise click.ClickException(f"Judgment failed: {e}")

    # Write outputs
    write_outputs(
        run_path,
        records,
        result["judgments"],
        result["summary"],
        result["model_name"],
        result.get("realignment_notes", ""),
        result.get("function_tracking", {}),
        result.get("turn_taking_analysis"),
    )

    # Print summary
    summary_path = run_path / "claude_summary.json"
    summary = json.loads(summary_path.read_text())
    passes = summary.get("claude_passes", {})
    total = summary.get("turns_scored", 0)

    click.echo(f"Judged {total} turns (with turn-taking analysis)")
    click.echo(f"  Turn-taking: {passes.get('turn_taking', total)}/{total}")
    click.echo(f"  Tool use: {passes.get('tool_use_correct', 0)}/{total}")
    click.echo(f"  Instruction following: {passes.get('instruction_following', 0)}/{total}")
    click.echo(f"  KB grounding: {passes.get('kb_grounding', 0)}/{total}")

    # Report turn-taking failures if any
    turn_taking_failures = summary.get("turn_taking_failures", [])
    if turn_taking_failures:
        click.echo(f"\nTurn-taking failures: {turn_taking_failures}")


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
