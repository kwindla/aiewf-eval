#!/usr/bin/env python3
"""Auditable fixed-900-turn analysis for the Gemma 4 26B A4B campaign.

``preflight`` is read-only and accepts an incomplete contiguous manifest.
``final`` requires all 30 canonical runs and valid judge artifacts, then writes
the aggregate artifacts in this directory. Inclusion is defined exclusively by
``../canonical.tsv``; this script never discovers runs by globbing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[1]
CONFIG_PATH = CAMPAIGN / "configuration.json"
FROZEN_PATH = CAMPAIGN / "frozen-order.tsv"
CANONICAL_PATH = CAMPAIGN / "canonical.tsv"
COMPLETE_PATH = CAMPAIGN / "judging/COMPLETE.json"
JUDGE_INPUTS_PATH = CAMPAIGN / "judging/canonical-inputs.tsv"
JUDGE_SOURCE_PATH = CAMPAIGN / "judging/judge-source-sha256.txt"

MODEL = "google/gemma-4-26B-A4B-it"
TARGET = 30
N_TURNS = 30
DENOMINATOR = TARGET * N_TURNS
EXPECTED_JUDGE_MODEL = "claude-opus-4-5"
EXPECTED_JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
SCORE_COMPONENTS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
)
BOOTSTRAPS = 20_000
SEED = 20260730


@dataclass(frozen=True)
class Conversation:
    slot: int
    run_dir: Path
    classification: str
    observed_turns: tuple[int, ...]
    component_scores: dict[str, tuple[bool, ...]]
    strict_scores: tuple[bool, ...]
    ttfat_ms: tuple[float, ...]
    full_scheduled_coverage: bool
    strict_protocol_completion: bool
    transcript_sha256: str
    judgment_sha256: str
    summary_sha256: str


def fail(message: str) -> None:
    raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def scheduled_map(
    rows: Iterable[dict[str, Any]], *, path: Path
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"invalid scheduled turn {turn!r}: {path}")
        if turn in result:
            fail(f"duplicate scheduled turn {turn}: {path}")
        result[turn] = row
    if sorted(result) != list(range(len(result))):
        fail(f"scheduled turns are not a contiguous prefix: {path}")
    return result


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run path escapes repository: {value}")
    return resolved


def tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def end_session_turns(rows: dict[int, dict[str, Any]]) -> tuple[int, ...]:
    return tuple(
        turn
        for turn, row in sorted(rows.items())
        if any(
            tool_name(call) == "end_session"
            for call in (row.get("tool_calls") or [])
            if isinstance(call, dict)
        )
    )


def validate_config(*, require_serving_verified: bool = False) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "benchmark": "aiwf_medium_context",
        "model": MODEL,
        "provider": "BaseTen",
        "endpoint": (
            "https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1"
        ),
        "filler": None,
        "target_eligible_runs": TARGET,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(
                f"configuration mismatch for {key}: "
                f"expected {value!r}, found {config.get(key)!r}"
            )
    arm = config.get("arm") or {}
    if arm.get("name") != "none" or arm.get("enable_thinking") is not False:
        fail("analysis accepts only the frozen thinking-off arm")
    if require_serving_verified:
        serving = config.get("serving") or {}
        if serving.get("verified") is not True:
            fail("final analysis requires the recorded serving smoke gate")
        if str(serving.get("vllm_version", "")).startswith("PENDING"):
            fail("final analysis requires the exact live vLLM version")
        if str((serving.get("mtp") or {}).get("status", "")).startswith("PENDING"):
            fail("final analysis requires the recorded MTP disposition")
    return config


def load_manifest(*, require_complete: bool) -> list[dict[str, Any]]:
    validate_config(require_serving_verified=require_complete)
    frozen = read_tsv(FROZEN_PATH)
    if frozen != [
        {"slot": str(slot), "mode": "none"} for slot in range(1, TARGET + 1)
    ]:
        fail("frozen order is not exactly 30 sequential thinking-off slots")
    rows = read_tsv(CANONICAL_PATH)
    if [int(row["slot"]) for row in rows] != list(range(1, len(rows) + 1)):
        fail("canonical manifest must be a contiguous prefix")
    if require_complete and len(rows) != TARGET:
        fail(f"final analysis requires 30 canonical runs; found {len(rows)}")
    seen: set[Path] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        slot = int(row["slot"])
        if row["mode"] != "none":
            fail(f"unexpected arm at slot {slot}: {row['mode']}")
        run_dir = resolve_run_dir(row["run_dir"])
        if run_dir in seen:
            fail(f"duplicate canonical run directory: {run_dir}")
        seen.add(run_dir)
        result.append({**row, "slot": slot, "run_dir_path": run_dir})
    return result


def validate_run_log(run_dir: Path) -> None:
    path = run_dir / "run.log"
    if not path.is_file():
        fail(f"missing standard run.log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    required = (
        "base_url=https://model-qel1y223.api.baseten.co/deployment/qz4zpye/sync/v1",
        f"model={MODEL}",
        "thinking=False",
        "T=1.0",
        "top_p=0.95",
        "top_k=64",
        "max_tokens=8192",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"run provenance is missing {missing}: {path}")


def load_conversation(entry: dict[str, Any]) -> Conversation:
    run_dir = entry["run_dir_path"]
    transcript_path = run_dir / "transcript.jsonl"
    judged_path = run_dir / "claude_judged.jsonl"
    summary_path = run_dir / "claude_summary.json"
    validate_run_log(run_dir)

    transcript = scheduled_map(read_jsonl(transcript_path), path=transcript_path)
    if not transcript:
        fail(f"canonical transcript has no scheduled turns: {transcript_path}")
    for turn, row in transcript.items():
        if row.get("model_name") != MODEL:
            fail(f"model mismatch in {transcript_path} turn {turn}")
    if int(entry["turns"]) != len(transcript):
        fail(f"canonical turn count mismatch at slot {entry['slot']}")

    judged = scheduled_map(read_jsonl(judged_path), path=judged_path)
    if sorted(judged) != sorted(transcript):
        fail(f"judgment coverage mismatch at slot {entry['slot']}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("judge_model") != EXPECTED_JUDGE_MODEL:
        fail(f"judge model mismatch at slot {entry['slot']}")
    if summary.get("judge_version") != EXPECTED_JUDGE_VERSION:
        fail(f"judge version mismatch at slot {entry['slot']}")
    if summary.get("model_name") != MODEL:
        fail(f"judged model mismatch at slot {entry['slot']}")
    if summary.get("turns_scored") != len(transcript):
        fail(f"judge turn count mismatch at slot {entry['slot']}")

    components: dict[str, list[bool]] = {
        component: [False] * N_TURNS for component in SCORE_COMPONENTS
    }
    strict = [False] * N_TURNS
    latency: list[float] = []
    for turn in sorted(transcript):
        scores = judged[turn].get("scores")
        if not isinstance(scores, dict):
            fail(f"missing scores at slot {entry['slot']} turn {turn}")
        for component in SCORE_COMPONENTS:
            value = scores.get(component)
            if not isinstance(value, bool):
                fail(
                    f"non-boolean {component} at slot {entry['slot']} turn {turn}"
                )
            components[component][turn] = value
        strict[turn] = all(components[key][turn] for key in SCORE_COMPONENTS)
        ttfat = transcript[turn].get("ttfb_ms")
        if isinstance(ttfat, (int, float)) and not isinstance(ttfat, bool):
            if math.isfinite(float(ttfat)) and float(ttfat) >= 0:
                latency.append(float(ttfat))

    full_coverage = sorted(transcript) == list(range(N_TURNS))
    end_turns = end_session_turns(transcript)
    strict_completion = full_coverage and end_turns == (N_TURNS - 1,)
    expected_classification = (
        "complete_30" if full_coverage else "fixed_denominator_short"
    )
    if entry["classification"] != expected_classification:
        fail(
            f"classification mismatch at slot {entry['slot']}: "
            f"{entry['classification']!r} != {expected_classification!r}"
        )
    return Conversation(
        slot=entry["slot"],
        run_dir=run_dir,
        classification=entry["classification"],
        observed_turns=tuple(sorted(transcript)),
        component_scores={
            key: tuple(values) for key, values in components.items()
        },
        strict_scores=tuple(strict),
        ttfat_ms=tuple(latency),
        full_scheduled_coverage=full_coverage,
        strict_protocol_completion=strict_completion,
        transcript_sha256=sha256(transcript_path),
        judgment_sha256=sha256(judged_path),
        summary_sha256=sha256(summary_path),
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def bootstrap_ci(per_conversation_correct: list[int]) -> tuple[float, float]:
    if not per_conversation_correct:
        return (0.0, 0.0)
    rng = random.Random(SEED + sum(per_conversation_correct))
    n = len(per_conversation_correct)
    estimates = []
    for _ in range(BOOTSTRAPS):
        total = sum(per_conversation_correct[rng.randrange(n)] for _ in range(n))
        estimates.append(total / (n * N_TURNS) * 100.0)
    return (
        float(percentile(estimates, 0.025) or 0.0),
        float(percentile(estimates, 0.975) or 0.0),
    )


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return ((center - spread) * 100, (center + spread) * 100)


def score_summary(
    conversations: list[Conversation], key: str
) -> dict[str, Any]:
    if key == "strict_pass":
        per_conversation = [sum(conv.strict_scores) for conv in conversations]
    else:
        per_conversation = [
            sum(conv.component_scores[key]) for conv in conversations
        ]
    correct = sum(per_conversation)
    total = len(conversations) * N_TURNS
    low, high = bootstrap_ci(per_conversation)
    return {
        "correct": correct,
        "total": total,
        "rate_percent": correct / total * 100 if total else 0.0,
        "error_percent": (total - correct) / total * 100 if total else 0.0,
        "cluster_bootstrap_95_percent": [low, high],
    }


def aggregate(conversations: list[Conversation]) -> dict[str, Any]:
    strict = score_summary(conversations, "strict_pass")
    component_results = {
        component: score_summary(conversations, component)
        for component in SCORE_COMPONENTS
    }
    latencies = [
        latency
        for conversation in conversations
        for latency in conversation.ttfat_ms
    ]
    full = sum(conv.full_scheduled_coverage for conv in conversations)
    protocol = sum(conv.strict_protocol_completion for conv in conversations)
    return {
        "runs": len(conversations),
        "scheduled_turns": len(conversations) * N_TURNS,
        "observed_turns": sum(len(conv.observed_turns) for conv in conversations),
        "missing_turns_counted_as_failures": (
            len(conversations) * N_TURNS
            - sum(len(conv.observed_turns) for conv in conversations)
        ),
        "strict_pass": strict,
        "components": component_results,
        "completion": {
            "full_scheduled_coverage": {
                "count": full,
                "total": len(conversations),
                "rate_percent": full / len(conversations) * 100
                if conversations
                else 0.0,
                "wilson_95_percent": list(wilson(full, len(conversations))),
            },
            "strict_protocol": {
                "count": protocol,
                "total": len(conversations),
                "rate_percent": protocol / len(conversations) * 100
                if conversations
                else 0.0,
                "wilson_95_percent": list(wilson(protocol, len(conversations))),
            },
        },
        "ttfat_ms_observed_responses_only": {
            "count": len(latencies),
            "p50": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
    }


def rounded(value: float | None) -> str:
    if value is None:
        return "N/A"
    return str(int(round(value)))


def readme_row(result: dict[str, Any]) -> str:
    strict = result["strict_pass"]
    components = result["components"]
    latency = result["ttfat_ms_observed_responses_only"]
    return (
        "| gemma-4-26b-a4b-it (thinking off) "
        f"| {strict['rate_percent']:.1f}% "
        f"| {strict['error_percent']:.1f}% "
        f"| {components['tool_use_correct']['error_percent']:.1f}% "
        f"| {components['instruction_following']['error_percent']:.1f}% "
        f"| {components['kb_grounding']['error_percent']:.1f}% "
        f"| {rounded(latency['p50'])}ms "
        f"| {rounded(latency['p95'])}ms "
        f"| {rounded(latency['max'])}ms "
        "| BaseTen |"
    )


def write_outputs(
    config: dict[str, Any],
    conversations: list[Conversation],
    result: dict[str, Any],
) -> None:
    manifest_hashes = {
        "configuration.json": sha256(CONFIG_PATH),
        "frozen-order.tsv": sha256(FROZEN_PATH),
        "canonical.tsv": sha256(CANONICAL_PATH),
        "judging/COMPLETE.json": sha256(COMPLETE_PATH),
        "judging/canonical-inputs.tsv": sha256(JUDGE_INPUTS_PATH),
        "judging/judge-source-sha256.txt": sha256(JUDGE_SOURCE_PATH),
        "analysis/analyze.py": sha256(Path(__file__).resolve()),
    }
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "campaign_id": config["campaign_id"],
        "model": MODEL,
        "provider": "BaseTen",
        "benchmark": "aiwf_medium_context",
        "arm": "none",
        "thinking_enabled": False,
        "filler": None,
        "fixed_denominator": DENOMINATOR,
        "sampling": config["sampling"],
        "serving": config["serving"],
        "judge": {
            "model": EXPECTED_JUDGE_MODEL,
            "version": EXPECTED_JUDGE_VERSION,
        },
        "manifest_hashes": manifest_hashes,
        "aggregate": result,
        "readme_candidate_row": readme_row(result),
        "runs": [
            {
                "slot": conv.slot,
                "run_dir": str(conv.run_dir.relative_to(ROOT)),
                "classification": conv.classification,
                "observed_turns": len(conv.observed_turns),
                "strict_passes": sum(conv.strict_scores),
                "full_scheduled_coverage": conv.full_scheduled_coverage,
                "strict_protocol_completion": conv.strict_protocol_completion,
                "transcript_sha256": conv.transcript_sha256,
                "judgment_sha256": conv.judgment_sha256,
                "summary_sha256": conv.summary_sha256,
            }
            for conv in conversations
        ],
    }
    (HERE / "aggregates.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    latency = result["ttfat_ms_observed_responses_only"]
    completion = result["completion"]
    write_tsv(
        HERE / "aggregates.tsv",
        (
            "model",
            "arm",
            "runs",
            "scheduled_turns",
            "observed_turns",
            "strict_pass_percent",
            "full_coverage",
            "strict_protocol_completion",
            "ttfat_p50_ms",
            "ttfat_p95_ms",
            "ttfat_max_ms",
        ),
        [
            {
                "model": MODEL,
                "arm": "none",
                "runs": result["runs"],
                "scheduled_turns": result["scheduled_turns"],
                "observed_turns": result["observed_turns"],
                "strict_pass_percent": f"{result['strict_pass']['rate_percent']:.4f}",
                "full_coverage": (
                    f"{completion['full_scheduled_coverage']['count']}/"
                    f"{completion['full_scheduled_coverage']['total']}"
                ),
                "strict_protocol_completion": (
                    f"{completion['strict_protocol']['count']}/"
                    f"{completion['strict_protocol']['total']}"
                ),
                "ttfat_p50_ms": rounded(latency["p50"]),
                "ttfat_p95_ms": rounded(latency["p95"]),
                "ttfat_max_ms": rounded(latency["max"]),
            }
        ],
    )
    write_tsv(
        HERE / "included-runs.tsv",
        (
            "slot",
            "run_dir",
            "classification",
            "observed_turns",
            "strict_passes",
            "full_scheduled_coverage",
            "strict_protocol_completion",
            "transcript_sha256",
            "judgment_sha256",
            "summary_sha256",
        ),
        payload["runs"],
    )

    strict = result["strict_pass"]
    components = result["components"]
    ci = strict["cluster_bootstrap_95_percent"]
    report = f"""# Gemma 4 26B A4B — AIEWF medium-context result

This report covers 30 canonical, strictly sequential thinking-off conversations
on the dedicated BaseTen deployment. Every conversation contributes 30
scheduled turns. Missing future turns after an early exit are failures; latency
is summarized only for observed model responses.

| Measure | Result |
|---|---:|
| Strict turn pass | {strict['correct']}/{strict['total']} ({strict['rate_percent']:.1f}%) |
| Whole-conversation bootstrap 95% CI | {ci[0]:.1f}–{ci[1]:.1f}% |
| Tool error | {components['tool_use_correct']['error_percent']:.1f}% |
| Instruction error | {components['instruction_following']['error_percent']:.1f}% |
| KB error | {components['kb_grounding']['error_percent']:.1f}% |
| Full scheduled coverage | {completion['full_scheduled_coverage']['count']}/{completion['full_scheduled_coverage']['total']} |
| Strict protocol completion | {completion['strict_protocol']['count']}/{completion['strict_protocol']['total']} |
| TTFAT P50 / P95 / max | {rounded(latency['p50'])} / {rounded(latency['p95'])} / {rounded(latency['max'])} ms |

## Candidate README row

The analyzer does not edit `README.md`. After reviewing the result, insert this
row at the correct score-sorted position:

```text
{readme_row(result)}
```

## Provenance

- Model: `{MODEL}`
- Endpoint: `{config['endpoint']}`
- Sampling: temperature 1.0, top-p 0.95, top-k 64, max tokens 8,192
- Thinking: explicitly disabled
- Filler: none
- Judge: `{EXPECTED_JUDGE_MODEL}` / `{EXPECTED_JUDGE_VERSION}`
- Fixed denominator: 30 × 30 = 900 scheduled turns
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "final"))
    args = parser.parse_args()
    config = validate_config(require_serving_verified=args.mode == "final")
    entries = load_manifest(require_complete=args.mode == "final")
    print(f"Manifest preflight: canonical={len(entries)}/{TARGET}")
    if args.mode == "preflight":
        judged = 0
        for entry in entries:
            run_dir = entry["run_dir_path"]
            if (
                (run_dir / "claude_judged.jsonl").is_file()
                and (run_dir / "claude_summary.json").is_file()
            ):
                judged += 1
        print(
            f"Read-only preflight complete: judged_artifact_sets="
            f"{judged}/{len(entries)}"
        )
        return 0

    if not COMPLETE_PATH.is_file():
        fail("final analysis requires judging/COMPLETE.json")
    complete = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    if complete.get("canonical_runs") != TARGET:
        fail("judging completion marker does not cover 30 runs")
    if not JUDGE_INPUTS_PATH.is_file() or not JUDGE_SOURCE_PATH.is_file():
        fail("final analysis requires frozen judge inputs and source hashes")
    if complete.get("canonical_inputs_sha256") != sha256(JUDGE_INPUTS_PATH):
        fail("judging canonical-input hash does not match COMPLETE.json")
    if complete.get("judge_source_sha256") != sha256(JUDGE_SOURCE_PATH):
        fail("judging source hash does not match COMPLETE.json")
    conversations = [load_conversation(entry) for entry in entries]
    result = aggregate(conversations)
    if result["scheduled_turns"] != DENOMINATOR:
        fail("internal denominator error")
    write_outputs(config, conversations, result)
    print(
        f"Final analysis written: strict_pass="
        f"{result['strict_pass']['rate_percent']:.1f}%, "
        f"fixed_denominator={DENOMINATOR}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
