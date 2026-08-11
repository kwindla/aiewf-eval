#!/usr/bin/env python3
"""Pool canonical BaseTen Gemma N=30 + extension N=120 and analyze N=150."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806"
EXTENSION = ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n120-extension-20260807"
OUTPUT_JSON = HERE / "aggregates.json"
OUTPUT_REPORT = HERE / "REPORT.md"
OUTPUT_CANONICAL = HERE / "canonical-pooled.tsv"
MODEL = "google/gemma-4-31B-it"
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
N_TURNS = 30
TARGET = 150
COMPONENTS = ("tool_use_correct", "instruction_following", "kb_grounding")
BOOTSTRAPS = 20_000
SEED = 20260807


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(per_conversation_correct: list[int], *, salt: int) -> list[float]:
    n = len(per_conversation_correct)
    denominator = n * N_TURNS
    rng = random.Random(SEED + salt)
    estimates = []
    for _ in range(BOOTSTRAPS):
        correct = sum(per_conversation_correct[rng.randrange(n)] for _ in range(n))
        estimates.append(correct / denominator * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    return function.get("name") if isinstance(function, dict) else None


def validate_source(directory: Path, target: int) -> list[dict[str, str]]:
    config = json.loads((directory / "configuration.json").read_text())
    if config.get("model") != MODEL or config.get("target_eligible_runs") != target:
        raise RuntimeError(f"frozen configuration mismatch: {directory}")
    complete = json.loads((directory / "judging/COMPLETE.json").read_text())
    if complete.get("canonical_runs") != target:
        raise RuntimeError(f"judge completion mismatch: {directory}")
    if complete.get("judge_model") != JUDGE_MODEL:
        raise RuntimeError(f"judge model mismatch: {directory}")
    if complete.get("judge_version") != JUDGE_VERSION:
        raise RuntimeError(f"judge version mismatch: {directory}")
    rows = read_tsv(directory / "canonical.tsv")
    if [int(row["slot"]) for row in rows] != list(range(1, target + 1)):
        raise RuntimeError(f"canonical slots mismatch: {directory}")
    return rows


def load_conversation(
    manifest: dict[str, str], *, cohort: str, cohort_slot: int, pooled_slot: int
) -> dict[str, Any]:
    run_dir = (ROOT / manifest["run_dir"]).resolve()
    run_dir.relative_to(ROOT.resolve())
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
    if summary.get("judge_model") != JUDGE_MODEL:
        raise RuntimeError(f"judge summary judge-model mismatch: {run_dir}")
    if summary.get("judge_version") != JUDGE_VERSION:
        raise RuntimeError(f"judge summary version mismatch: {run_dir}")
    if summary.get("turns_scored") != len(turns):
        raise RuntimeError(f"judge summary turn count mismatch: {run_dir}")
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
        "pooled_slot": pooled_slot,
        "cohort": cohort,
        "cohort_slot": cohort_slot,
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


def metric(
    conversations: list[dict[str, Any]], name: str | None, *, salt: int
) -> dict[str, Any]:
    denominator = len(conversations) * N_TURNS
    per_conversation = [
        sum(conversation["strict"] if name is None else conversation["scores"][name])
        for conversation in conversations
    ]
    correct = sum(per_conversation)
    return {
        "correct": correct,
        "total": denominator,
        "rate_percent": correct / denominator * 100,
        "error_percent": (denominator - correct) / denominator * 100,
        "conversation_cluster_bootstrap_95_percent": bootstrap_ci(
            per_conversation, salt=salt
        ),
    }


def all_metrics(conversations: list[dict[str, Any]], *, salt: int) -> dict[str, Any]:
    return {
        "strict": metric(conversations, None, salt=salt),
        "components": {
            name: metric(conversations, name, salt=salt + index + 1)
            for index, name in enumerate(COMPONENTS)
        },
    }


def latency_metrics(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        value for conversation in conversations for value in conversation["latencies"]
    ]
    return {
        "scope": "observed_scripted_turns_only",
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "over_5000_ms": sum(value > 5_000 for value in values),
        "over_10000_ms": sum(value > 10_000 for value in values),
        "over_20000_ms": sum(value > 20_000 for value in values),
    }


def main() -> int:
    original_rows = validate_source(ORIGINAL, 30)
    extension_rows = validate_source(EXTENSION, 120)
    conversations = []
    for cohort, rows in (("original_n30", original_rows), ("extension_n120", extension_rows)):
        for row in rows:
            conversations.append(
                load_conversation(
                    row,
                    cohort=cohort,
                    cohort_slot=int(row["slot"]),
                    pooled_slot=len(conversations) + 1,
                )
            )
    if len(conversations) != TARGET:
        raise RuntimeError(f"expected {TARGET} pooled conversations")
    if len({conversation["run_dir"] for conversation in conversations}) != TARGET:
        raise RuntimeError("duplicate run directory in pooled cohort")
    write_tsv(
        OUTPUT_CANONICAL,
        (
            "pooled_slot",
            "cohort",
            "cohort_slot",
            "run_dir",
            "classification",
            "observed_turns",
            "transcript_sha256",
            "judgment_sha256",
            "summary_sha256",
        ),
        [
            {key: conversation[key] for key in (
                "pooled_slot", "cohort", "cohort_slot", "run_dir", "classification",
                "observed_turns", "transcript_sha256", "judgment_sha256", "summary_sha256"
            )}
            for conversation in conversations
        ],
    )
    pooled_metrics = all_metrics(conversations, salt=100)
    cohort_metrics = {
        "original_n30": all_metrics(conversations[:30], salt=200),
        "extension_n120": all_metrics(conversations[30:], salt=300),
    }
    latency = latency_metrics(conversations)
    cohort_latency = {
        "original_n30": latency_metrics(conversations[:30]),
        "extension_n120": latency_metrics(conversations[30:]),
    }
    per_turn_errors = []
    for turn in range(N_TURNS):
        row: dict[str, Any] = {"turn": turn}
        strict_errors = sum(not conversation["strict"][turn] for conversation in conversations)
        row.update(
            {
                "strict_errors": strict_errors,
                "strict_error_percent": strict_errors / TARGET * 100,
            }
        )
        for component in COMPONENTS:
            errors = sum(
                not conversation["scores"][component][turn]
                for conversation in conversations
            )
            row[f"{component}_errors"] = errors
            row[f"{component}_error_percent"] = errors / TARGET * 100
        per_turn_errors.append(row)
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
    strict = pooled_metrics["strict"]
    components = pooled_metrics["components"]
    payload = {
        "schema_version": 1,
        "model": MODEL,
        "provider": "BaseTen",
        "serving": "SGLang v0.5.16 NEXTN/MTP on 2xH100",
        "n_conversations": TARGET,
        "fixed_turn_denominator": TARGET * N_TURNS,
        "full_30_turn_conversations": sum(
            conversation["observed_turns"] == N_TURNS for conversation in conversations
        ),
        "bootstrap": {
            "unit": "conversation",
            "replicates": BOOTSTRAPS,
            "seed": SEED,
            "interval": "percentile 95%",
        },
        "strict": strict,
        "components": components,
        "cohort_metrics": cohort_metrics,
        "latency": latency,
        "cohort_latency": cohort_latency,
        "end_session": {"kind_counts": dict(end_counts), "turn_counts": dict(end_turns)},
        "per_turn_errors": per_turn_errors,
        "token_totals": token_totals,
        "source_artifacts": {
            "original_n30": {
                "directory": str(ORIGINAL.relative_to(ROOT)),
                "configuration_sha256": sha256(ORIGINAL / "configuration.json"),
                "canonical_sha256": sha256(ORIGINAL / "canonical.tsv"),
                "judge_complete_sha256": sha256(ORIGINAL / "judging/COMPLETE.json"),
            },
            "extension_n120": {
                "directory": str(EXTENSION.relative_to(ROOT)),
                "configuration_sha256": sha256(EXTENSION / "configuration.json"),
                "canonical_sha256": sha256(EXTENSION / "canonical.tsv"),
                "judge_complete_sha256": sha256(EXTENSION / "judging/COMPLETE.json"),
            },
        },
        "pooled_canonical_sha256": sha256(OUTPUT_CANONICAL),
        "conversations": conversations,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    ci = strict["conversation_cluster_bootstrap_95_percent"]
    worst = sorted(per_turn_errors, key=lambda row: (-row["strict_errors"], row["turn"]))[:8]
    worst_rows = "\n".join(
        f"| {row['turn']} | {row['strict_errors']}/{TARGET} | "
        f"{row['strict_error_percent']:.1f}% | {row['tool_use_correct_errors']} | "
        f"{row['instruction_following_errors']} | {row['kb_grounding_errors']} |"
        for row in worst if row["strict_errors"]
    ) or f"| — | 0/{TARGET} | 0.0% | 0 | 0 | 0 |"
    cohort_rows = []
    for label, n in (("Original", 30), ("Extension", 120), ("Pooled", 150)):
        selected = pooled_metrics if label == "Pooled" else cohort_metrics[
            "original_n30" if label == "Original" else "extension_n120"
        ]
        value = selected["strict"]
        value_ci = value["conversation_cluster_bootstrap_95_percent"]
        cohort_rows.append(
            f"| {label} | {n} | {value['correct']}/{value['total']} | "
            f"{value['rate_percent']:.2f}% | {value_ci[0]:.2f}–{value_ci[1]:.2f}% |"
        )
    latency_rows = []
    for label, selected in (
        ("Original", cohort_latency["original_n30"]),
        ("Extension", cohort_latency["extension_n120"]),
        ("Pooled", latency),
    ):
        latency_rows.append(
            f"| {label} | {selected['count']} | {selected['p50_ms']:.0f} | "
            f"{selected['p95_ms']:.0f} | {selected['max_ms']:.0f} | "
            f"{selected['over_10000_ms']} |"
        )
    report = f"""# Gemma 4 31B BaseTen pooled N=150 campaign

The pooled canonical no-filler, thinking-off cohort scored
{strict['correct']}/{strict['total']} strict turns ({strict['rate_percent']:.2f}%,
conversation-cluster bootstrap 95% CI {ci[0]:.2f}–{ci[1]:.2f}%).

| Cohort | Conversations | Strict turns | Pass rate | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(cohort_rows)}

| Metric | Pooled result |
|---|---:|
| Full 30-turn conversations | {payload['full_30_turn_conversations']}/{TARGET} |
| Any error | {strict['error_percent']:.2f}% |
| Tool error | {components['tool_use_correct']['error_percent']:.2f}% |
| Instruction error | {components['instruction_following']['error_percent']:.2f}% |
| KB error | {components['kb_grounding']['error_percent']:.2f}% |
| TTFAT P50 / P95 / max | {latency['p50_ms']:.0f} / {latency['p95_ms']:.0f} / {latency['max_ms']:.0f} ms |
| Thinking tokens | {token_totals['thinking_tokens']} |

Latency by cohort (observed scripted turns):

| Cohort | Turns | P50 ms | P95 ms | Max ms | >10s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(latency_rows)}

Most error-prone scripted turns:

| Turn | Conversations with any error | Error rate | Tool | Instruction | KB |
|---:|---:|---:|---:|---:|---:|
{worst_rows}

Missing future turns after a model-caused early exit count as failures in all
accuracy measures. Latency is reported only where a scripted turn produced a
measured response. Whole conversations, not turns, are the bootstrap unit
({BOOTSTRAPS:,} deterministic resamples).
"""
    OUTPUT_REPORT.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
