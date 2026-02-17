#!/usr/bin/env python3
"""Aggregate native performance and latency stats from transcript.jsonl files.

This script reads run directories under runs/{benchmark}/... and computes:
- Native transcript metrics (turn count, tool call usage, token totals)
- Latency metrics from transcript fields (ttfb_ms, latency_ms)
- Optional judged metrics from claude_summary.json when available

Usage:
  uv run python scripts/aggregate_transcript_native.py --benchmark aiwf_medium_context
  uv run python scripts/aggregate_transcript_native.py --model moonshotai/kimi-k2-instruct-0905
  uv run python scripts/aggregate_transcript_native.py --pattern "runs/aiwf_medium_context/*kimi*" --jsonl-out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate native transcript stats across benchmark runs"
    )
    parser.add_argument(
        "--benchmark",
        default="aiwf_medium_context",
        help="Benchmark folder under runs/ (default: aiwf_medium_context)",
    )
    parser.add_argument(
        "--model",
        help="Model name filter (unsanitized, e.g. moonshotai/kimi-k2-instruct-0905)",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for run directories (can be used multiple times)",
    )
    parser.add_argument(
        "--output",
        default="runs/transcript_native_summary.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--jsonl-out",
        help="Optional output JSONL path with one line per run summary",
    )
    parser.add_argument(
        "--include-judge",
        action="store_true",
        help="Include claude_summary.json fields when present",
    )
    return parser.parse_args()


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    # Keep behavior aligned with existing eval scripts.
    if len(values) >= 20:
        return statistics.quantiles(values, n=20)[18]
    return max(values)


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
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
        "p95": p95(values),
        "min": min(values),
        "max": max(values),
    }


def infer_model_from_dir(run_dir: Path) -> str:
    # Format: YYYYMMDDTHHmmss_model_name_hash
    parts = run_dir.name.split("_", 1)
    if len(parts) < 2:
        return run_dir.name
    tail = parts[1]
    maybe_model, maybe_hash = tail.rsplit("_", 1)
    if len(maybe_hash) == 8:
        return maybe_model.replace("_", "/")
    return tail.replace("_", "/")


def load_transcript(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_run(run_dir: Path, include_judge: bool) -> dict[str, Any] | None:
    rows = load_transcript(run_dir)
    if not rows:
        return None

    model_name = rows[0].get("model_name") or infer_model_from_dir(run_dir)
    ttfb_values = [
        float(r["ttfb_ms"])
        for r in rows
        if r.get("ttfb_ms") is not None and isinstance(r.get("ttfb_ms"), (int, float))
    ]
    latency_values = [
        float(r["latency_ms"])
        for r in rows
        if r.get("latency_ms") is not None
        and isinstance(r.get("latency_ms"), (int, float))
    ]

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    tool_call_turns = 0
    tool_calls_total = 0
    empty_assistant_turns = 0

    for r in rows:
        tool_calls = r.get("tool_calls") or []
        if tool_calls:
            tool_call_turns += 1
            tool_calls_total += len(tool_calls)

        assistant_text = r.get("assistant_text")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            empty_assistant_turns += 1

        tok = r.get("tokens") or {}
        prompt_tokens += int(tok.get("prompt_tokens") or 0)
        completion_tokens += int(tok.get("completion_tokens") or 0)
        total_tokens += int(tok.get("total_tokens") or 0)

    run_summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "turns": len(rows),
        "native_performance": {
            "tool_call_turns": tool_call_turns,
            "tool_calls_total": tool_calls_total,
            "empty_assistant_turns": empty_assistant_turns,
            "assistant_response_rate": (
                (len(rows) - empty_assistant_turns) / len(rows) if rows else None
            ),
        },
        "latency_ms": summarize_numeric(latency_values),
        "ttfb_ms": summarize_numeric(ttfb_values),
        "tokens": {
            "prompt_total": prompt_tokens,
            "completion_total": completion_tokens,
            "total": total_tokens,
            "avg_total_per_turn": (total_tokens / len(rows)) if rows else None,
        },
    }

    if include_judge:
        summary_path = run_dir / "claude_summary.json"
        if summary_path.exists():
            try:
                run_summary["judge_summary"] = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                run_summary["judge_summary"] = None

    return run_summary


def find_runs(args: argparse.Namespace) -> list[Path]:
    run_dirs: list[Path] = []

    if args.pattern:
        for pattern in args.pattern:
            run_dirs.extend([p for p in Path().glob(pattern) if p.is_dir()])
    else:
        base = Path("runs") / args.benchmark
        if base.exists():
            run_dirs.extend([p for p in base.iterdir() if p.is_dir()])

    if args.model:
        safe = args.model.replace("/", "_").replace(":", "_")
        run_dirs = [d for d in run_dirs if safe in d.name]

    run_dirs = sorted(set(run_dirs), key=lambda d: d.stat().st_mtime)
    return run_dirs


def aggregate_by_model(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for r in run_summaries:
        model = r["model_name"]
        g = grouped.setdefault(
            model,
            {
                "model_name": model,
                "runs": 0,
                "turns": 0,
                "ttfb_values": [],
                "latency_values": [],
                "tool_call_turns": 0,
                "tool_calls_total": 0,
                "empty_assistant_turns": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        g["runs"] += 1
        g["turns"] += r["turns"]
        g["tool_call_turns"] += r["native_performance"]["tool_call_turns"]
        g["tool_calls_total"] += r["native_performance"]["tool_calls_total"]
        g["empty_assistant_turns"] += r["native_performance"]["empty_assistant_turns"]
        g["prompt_tokens"] += r["tokens"]["prompt_total"]
        g["completion_tokens"] += r["tokens"]["completion_total"]
        g["total_tokens"] += r["tokens"]["total"]

        # Rehydrate approximate distributions from run-level values only if present.
        # Prefer exact per-turn data from runs instead of medians.
        # We collect values by re-reading from already summarized fields is not possible,
        # so these are filled in a second pass below.

    # second pass to accumulate exact per-turn arrays
    for r in run_summaries:
        model = r["model_name"]
        run_dir = Path(r["run_dir"])
        rows = load_transcript(run_dir)
        grouped[model]["ttfb_values"].extend(
            [
                float(x.get("ttfb_ms"))
                for x in rows
                if isinstance(x.get("ttfb_ms"), (int, float))
            ]
        )
        grouped[model]["latency_values"].extend(
            [
                float(x.get("latency_ms"))
                for x in rows
                if isinstance(x.get("latency_ms"), (int, float))
            ]
        )

    out: list[dict[str, Any]] = []
    for model in sorted(grouped.keys()):
        g = grouped[model]
        turns = g["turns"]
        out.append(
            {
                "model_name": model,
                "runs": g["runs"],
                "turns": turns,
                "native_performance": {
                    "tool_call_turns": g["tool_call_turns"],
                    "tool_calls_total": g["tool_calls_total"],
                    "empty_assistant_turns": g["empty_assistant_turns"],
                    "assistant_response_rate": (
                        (turns - g["empty_assistant_turns"]) / turns if turns else None
                    ),
                },
                "ttfb_ms": summarize_numeric(g["ttfb_values"]),
                "latency_ms": summarize_numeric(g["latency_values"]),
                "tokens": {
                    "prompt_total": g["prompt_tokens"],
                    "completion_total": g["completion_tokens"],
                    "total": g["total_tokens"],
                    "avg_total_per_turn": (g["total_tokens"] / turns) if turns else None,
                },
            }
        )
    return out


def main() -> int:
    args = parse_args()
    run_dirs = find_runs(args)
    if not run_dirs:
        print("No matching run directories found.")
        return 1

    run_summaries: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        s = summarize_run(run_dir, include_judge=args.include_judge)
        if s is None:
            skipped.append(str(run_dir))
            continue
        run_summaries.append(s)

    if not run_summaries:
        print("No usable transcript.jsonl files found in matching runs.")
        return 1

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "benchmark": args.benchmark,
            "model": args.model,
            "patterns": args.pattern,
        },
        "totals": {
            "matching_run_dirs": len(run_dirs),
            "processed_runs": len(run_summaries),
            "skipped_runs": len(skipped),
        },
        "skipped_runs": skipped,
        "models": aggregate_by_model(run_summaries),
        "runs": run_summaries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    if args.jsonl_out:
        jsonl_path = Path(args.jsonl_out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for row in run_summaries:
                f.write(json.dumps(row) + "\n")

    print(f"Wrote JSON summary: {out_path}")
    if args.jsonl_out:
        print(f"Wrote run-level JSONL: {args.jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

