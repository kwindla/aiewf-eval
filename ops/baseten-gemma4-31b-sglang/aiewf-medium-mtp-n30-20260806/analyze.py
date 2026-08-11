#!/usr/bin/env python3
"""Generate fixed-900-turn aggregates for the frozen Gemma 4 31B cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
CANONICAL = HERE / "canonical.tsv"
JUDGE_COMPLETE = HERE / "judging/COMPLETE.json"
OUTPUT_JSON = HERE / "aggregates.json"
OUTPUT_REPORT = HERE / "REPORT.md"
MODEL = "google/gemma-4-31B-it"
TARGET = 30
N_TURNS = 30
DENOMINATOR = TARGET * N_TURNS
COMPONENTS = ("tool_use_correct", "instruction_following", "kb_grounding")
BOOTSTRAPS = 20_000
SEED = 20260806


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(per_conversation_correct: list[int], *, salt: int) -> list[float]:
    rng = random.Random(SEED + salt)
    estimates = []
    for _ in range(BOOTSTRAPS):
        correct = sum(
            per_conversation_correct[rng.randrange(TARGET)] for _ in range(TARGET)
        )
        estimates.append(correct / DENOMINATOR * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    return function.get("name") if isinstance(function, dict) else None


def load_conversation(manifest: dict[str, str]) -> dict[str, Any]:
    run_dir = (ROOT / manifest["run_dir"]).resolve()
    transcript_path = run_dir / "transcript.jsonl"
    judgment_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    transcript = read_jsonl(transcript_path)
    scheduled = [row for row in transcript if row.get("recovery_turn") is not True]
    recovery = [row for row in transcript if row.get("recovery_turn") is True]
    turns = [row.get("turn") for row in scheduled]
    if turns != list(range(len(turns))):
        raise RuntimeError(f"invalid scheduled-turn sequence: {run_dir}")
    if any(row.get("model_name") != MODEL for row in transcript):
        raise RuntimeError(f"model mismatch: {run_dir}")
    judged = read_jsonl(judgment_path)
    if [row.get("turn") for row in judged] != turns:
        raise RuntimeError(f"judgment coverage mismatch: {run_dir}")
    summary = json.loads(summary_path.read_text())
    if summary.get("model_name") != MODEL:
        raise RuntimeError(f"judge summary model mismatch: {run_dir}")
    scores = {name: [False] * N_TURNS for name in COMPONENTS}
    for row in judged:
        turn = int(row["turn"])
        for name in COMPONENTS:
            value = row["scores"].get(name)
            if not isinstance(value, bool):
                raise RuntimeError(f"invalid {name} score: {run_dir} turn {turn}")
            scores[name][turn] = value
    strict = [all(scores[name][turn] for name in COMPONENTS) for turn in range(N_TURNS)]
    latencies = [
        float(row["ttfb_ms"])
        for row in scheduled
        if isinstance(row.get("ttfb_ms"), (int, float))
    ]
    scheduled_end = [
        int(row["turn"])
        for row in scheduled
        if any(tool_name(call) == "end_session" for call in row.get("tool_calls") or [])
    ]
    recovery_end = [
        int(row["turn"])
        for row in recovery
        if isinstance(row.get("turn"), int)
        and any(tool_name(call) == "end_session" for call in row.get("tool_calls") or [])
    ]
    if scheduled_end:
        end_kind, end_turn = "scripted", min(scheduled_end)
    elif recovery_end:
        end_kind, end_turn = "recovery", min(recovery_end)
    else:
        end_kind, end_turn = "missing", -1
    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "thinking_tokens",
    )
    tokens = {
        name: sum(int((row.get("tokens") or {}).get(name) or 0) for row in transcript)
        for name in token_fields
    }
    return {
        "slot": int(manifest["slot"]),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "classification": manifest["classification"],
        "observed_turns": len(scheduled),
        "scores": scores,
        "strict": strict,
        "latencies": latencies,
        "end_kind": end_kind,
        "end_turn": end_turn,
        "recovery_rows": len(recovery),
        "tokens": tokens,
        "transcript_sha256": sha256(transcript_path),
        "judgment_sha256": sha256(judgment_path),
        "summary_sha256": sha256(summary_path),
    }


def metric(conversations: list[dict[str, Any]], name: str | None) -> dict[str, Any]:
    per_conversation = [
        sum(conversation["strict"] if name is None else conversation["scores"][name])
        for conversation in conversations
    ]
    correct = sum(per_conversation)
    return {
        "correct": correct,
        "total": DENOMINATOR,
        "rate_percent": correct / DENOMINATOR * 100,
        "error_percent": (DENOMINATOR - correct) / DENOMINATOR * 100,
        "conversation_cluster_bootstrap_95_percent": bootstrap_ci(
            per_conversation, salt=correct
        ),
    }


def main() -> int:
    config = json.loads(CONFIG.read_text())
    if config.get("model") != MODEL or config.get("target_eligible_runs") != TARGET:
        raise RuntimeError("frozen configuration mismatch")
    judge_complete = json.loads(JUDGE_COMPLETE.read_text())
    if judge_complete.get("canonical_runs") != TARGET:
        raise RuntimeError("judging completion marker mismatch")
    canonical = read_tsv(CANONICAL)
    if [int(row["slot"]) for row in canonical] != list(range(1, TARGET + 1)):
        raise RuntimeError("canonical cohort is not exactly slots 1..30")
    conversations = [load_conversation(row) for row in canonical]
    strict = metric(conversations, None)
    components = {name: metric(conversations, name) for name in COMPONENTS}
    latencies = [value for conversation in conversations for value in conversation["latencies"]]
    latency = {
        "scope": "observed_scripted_turns_only",
        "count": len(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
    }
    per_turn_errors = []
    for turn in range(N_TURNS):
        errors = sum(not conversation["strict"][turn] for conversation in conversations)
        per_turn_errors.append({"turn": turn, "errors": errors, "error_percent": errors / TARGET * 100})
    end_counts = Counter(conversation["end_kind"] for conversation in conversations)
    end_turns = Counter(
        str(conversation["end_turn"])
        for conversation in conversations
        if conversation["end_turn"] >= 0
    )
    token_fields = tuple(conversations[0]["tokens"])
    token_totals = {
        name: sum(conversation["tokens"][name] for conversation in conversations)
        for name in token_fields
    }
    payload = {
        "schema_version": 1,
        "model": MODEL,
        "provider": "BaseTen",
        "serving": "SGLang v0.5.16 NEXTN/MTP on 2xH100",
        "n_conversations": TARGET,
        "fixed_turn_denominator": DENOMINATOR,
        "full_30_turn_conversations": sum(
            conversation["observed_turns"] == N_TURNS for conversation in conversations
        ),
        "strict": strict,
        "components": components,
        "latency": latency,
        "end_session": {"kind_counts": dict(end_counts), "turn_counts": dict(end_turns)},
        "per_turn_errors": per_turn_errors,
        "token_totals": token_totals,
        "conversations": conversations,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    ci = strict["conversation_cluster_bootstrap_95_percent"]
    worst = sorted(per_turn_errors, key=lambda row: (-row["errors"], row["turn"]))[:8]
    worst_rows = "\n".join(
        f"| {row['turn']} | {row['errors']}/{TARGET} | {row['error_percent']:.1f}% |"
        for row in worst
        if row["errors"]
    ) or "| — | 0/30 | 0.0% |"
    report = f"""# Gemma 4 31B BaseTen SGLang campaign

The frozen no-filler, thinking-off campaign scored
{strict['correct']}/{DENOMINATOR} strict turns ({strict['rate_percent']:.1f}%,
conversation-cluster bootstrap 95% CI {ci[0]:.1f}–{ci[1]:.1f}%).

| Metric | Result |
|---|---:|
| Canonical conversations | {TARGET} |
| Full 30-turn conversations | {payload['full_30_turn_conversations']}/{TARGET} |
| Strict pass | {strict['correct']}/{DENOMINATOR} ({strict['rate_percent']:.1f}%) |
| Tool error | {components['tool_use_correct']['error_percent']:.1f}% |
| Instruction error | {components['instruction_following']['error_percent']:.1f}% |
| KB error | {components['kb_grounding']['error_percent']:.1f}% |
| TTFAT P50 / P95 / max | {latency['p50_ms']:.0f} / {latency['p95_ms']:.0f} / {latency['max_ms']:.0f} ms |
| Thinking tokens | {token_totals['thinking_tokens']} |

README row:

| Model | Pass Rate | Any Error | Tool Error | Instruction Error | KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma-4-31b-it (thinking off) | {strict['rate_percent']:.1f}% | {strict['error_percent']:.1f}% | {components['tool_use_correct']['error_percent']:.1f}% | {components['instruction_following']['error_percent']:.1f}% | {components['kb_grounding']['error_percent']:.1f}% | {latency['p50_ms']:.0f}ms | {latency['p95_ms']:.0f}ms | {latency['max_ms']:.0f}ms | BaseTen |

Most error-prone scripted turns:

| Turn | Conversations with any error | Error rate |
|---:|---:|---:|
{worst_rows}

Missing future turns after a model-caused early exit count as failures in all
four displayed accuracy measures. Latency is reported only where a scripted
turn produced a measured response. Whole conversations, not individual turns,
are the bootstrap resampling unit.
"""
    OUTPUT_REPORT.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
