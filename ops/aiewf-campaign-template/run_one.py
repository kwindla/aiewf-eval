#!/usr/bin/env python3
"""Run one AIEWF conversation into an explicitly supplied output directory."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


async def run_conversation(config: dict, run_dir: Path) -> None:
    from multi_turn_eval.cli import (
        get_pipeline_class,
        load_benchmark,
        load_service_class,
        reject_openai_pro_model,
        setup_logging,
    )
    from multi_turn_eval.recording.transcript_recorder import TranscriptRecorder

    model = config["model"]
    service = config["service"]
    pipeline_name = config["pipeline"]
    reject_openai_pro_model(model, service)
    benchmark = load_benchmark(config["benchmark"])()
    pipeline_class = get_pipeline_class(pipeline_name)
    if getattr(pipeline_class, "requires_service", True) and not service:
        raise ValueError(f"service is required for pipeline {pipeline_name!r}")
    service_class = load_service_class(service) if service else None

    run_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(run_dir, verbose=False)
    print(f"Output directory: {run_dir}", flush=True)
    recorder = TranscriptRecorder(run_dir, model)
    try:
        pipeline = pipeline_class(benchmark)
        await pipeline.run(
            recorder=recorder,
            model=model,
            service_class=service_class,
            service_name=service,
            turn_indices=None,
        )
        print("Completed benchmark run", flush=True)
        print(f"  Transcript: {run_dir / 'transcript.jsonl'}", flush=True)
    finally:
        recorder.close()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arms = config.get("arms") or {}
    if args.arm not in arms:
        raise ValueError(f"unknown configured arm: {args.arm!r}")
    output_root_value = (config.get("paths") or {}).get("run_output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("paths.run_output_root is missing")
    root = Path(__file__).resolve().parents[2]
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    run_dir = args.run_dir.expanduser().resolve()
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("run directory is outside configured run_output_root") from exc
    asyncio.run(run_conversation(config, run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
