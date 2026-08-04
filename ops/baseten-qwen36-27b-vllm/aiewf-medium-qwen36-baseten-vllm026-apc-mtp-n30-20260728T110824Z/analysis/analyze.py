#!/usr/bin/env python3
"""Auditable fixed-denominator analysis for the Qwen3.6 AIEWF campaign.

The inclusion set is defined only by ``../canonical.tsv``. Every included
conversation contributes exactly 30 scheduled turns. If a model stops after
turn k, turns k+1..29 are failures on every scored dimension.

Usage:
    .venv/bin/python .../analysis/analyze.py preflight
    .venv/bin/python .../analysis/analyze.py final

``preflight`` is read-only and accepts an in-progress manifest. ``final``
requires the frozen 30 high + 30 none cohort and complete judge artifacts,
then writes all outputs into this analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]

MODEL = "Qwen/Qwen3.6-27B"
ARMS = ("high", "none")
TARGET_PER_ARM = 30
N_TURNS = 30
BOOTSTRAPS = 100_000
LATENCY_BOOTSTRAPS = 20_000
SEED = 20260728
SCORE_COMPONENTS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
)
SUPPLEMENTARY_COMPONENTS = ("turn_taking",)
ALL_COMPONENTS = SCORE_COMPONENTS + SUPPLEMENTARY_COMPONENTS


@dataclass(frozen=True)
class ManifestEntry:
    slot: int
    pair: int
    order_in_pair: int
    mode: str
    attempt: int
    run_dir: Path
    classification: str
    raw: dict[str, str]


@dataclass(frozen=True)
class Conversation:
    entry: ManifestEntry
    observed_turns: tuple[int, ...]
    scores: dict[str, tuple[bool, ...]]
    benchmark_pass: tuple[bool, ...]
    full_scheduled_coverage: bool
    strict_protocol_completion: bool
    scheduled_end_session_turns: tuple[int, ...]
    recovery_end_session_turns: tuple[int, ...]
    ttfat_ms: tuple[float, ...]
    thought_turns: int
    run_log_sha256: str
    transcript_sha256: str
    judgment_sha256: str
    judge_summary_sha256: str
    judge_model: str
    judge_version: str


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
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
                fail(f"invalid JSON in {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSON in {path}:{line_number}")
            rows.append(row)
    if not rows:
        fail(f"no rows in JSONL: {path}")
    return rows


def canonical_turn_map(
    rows: Iterable[dict[str, Any]], path: Path
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            if turn in result:
                fail(f"duplicate scheduled turn {turn}: {path}")
            result[turn] = row
    return result


def end_session_turns(
    rows: Iterable[dict[str, Any]], *, recovery: bool
) -> tuple[int, ...]:
    turns: list[int] = []
    for row in rows:
        if (row.get("recovery_turn") is True) != recovery:
            continue
        if any(call.get("name") == "end_session" for call in row.get("tool_calls") or []):
            turn = row.get("turn")
            if isinstance(turn, int):
                turns.append(turn)
    return tuple(sorted(turns))


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run path escapes repository root: {value}")
    return resolved


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_manifest(*, require_complete: bool) -> list[ManifestEntry]:
    config_path = CAMPAIGN / "configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "benchmark": "aiwf_medium_context",
        "model": MODEL,
        "target_eligible_per_arm": TARGET_PER_ARM,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "filler": None,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            fail(
                f"configuration mismatch for {key}: "
                f"expected {expected!r}, found {config.get(key)!r}"
            )
    arm_config = config.get("arms") or {}
    if (
        (arm_config.get("high") or {}).get("enable_thinking") is not True
        or (arm_config.get("high") or {}).get("preserve_thinking") is not True
        or (arm_config.get("none") or {}).get("enable_thinking") is not False
        or (arm_config.get("none") or {}).get("preserve_thinking") is not False
    ):
        fail("thinking configuration does not match high/none protocol")

    frozen_rows = read_tsv(CAMPAIGN / "frozen-order.tsv")
    if len(frozen_rows) != TARGET_PER_ARM * len(ARMS):
        fail(f"frozen order must contain 60 slots, found {len(frozen_rows)}")
    frozen: dict[int, tuple[int, int, str]] = {}
    for row in frozen_rows:
        slot = int(row["slot"])
        frozen[slot] = (
            int(row["pair"]),
            int(row["order_in_pair"]),
            row["mode"],
        )
    if set(frozen) != set(range(1, 61)):
        fail("frozen order slots are not exactly 1..60")
    for pair in range(1, 31):
        pair_modes = [
            mode
            for pair_number, _, mode in frozen.values()
            if pair_number == pair
        ]
        if sorted(pair_modes) != sorted(ARMS):
            fail(f"pair {pair} does not contain one high and one none slot")

    canonical_rows = read_tsv(CAMPAIGN / "canonical.tsv")
    entries: list[ManifestEntry] = []
    seen_dirs: set[Path] = set()
    seen_slots: set[int] = set()
    for row in canonical_rows:
        slot = int(row["slot"])
        pair = int(row["pair"])
        order_in_pair = int(row["order_in_pair"])
        mode = row["mode"]
        if slot not in frozen:
            fail(f"canonical slot is not frozen: {slot}")
        if (pair, order_in_pair, mode) != frozen[slot]:
            fail(f"canonical/frozen mismatch at slot {slot}")
        if slot in seen_slots:
            fail(f"duplicate canonical slot: {slot}")
        if mode not in ARMS:
            fail(f"unexpected mode at slot {slot}: {mode}")
        run_dir = resolve_run_dir(row["run_dir"])
        if run_dir in seen_dirs:
            fail(f"duplicate canonical run directory: {run_dir}")
        seen_slots.add(slot)
        seen_dirs.add(run_dir)
        entries.append(
            ManifestEntry(
                slot=slot,
                pair=pair,
                order_in_pair=order_in_pair,
                mode=mode,
                attempt=int(row["attempt"]),
                run_dir=run_dir,
                classification=row["classification"],
                raw=row,
            )
        )
    entries.sort(key=lambda entry: entry.slot)

    # A live campaign appends slots in order. Requiring a prefix catches manual
    # selection or accidental omission without rejecting a legitimate preflight.
    if [entry.slot for entry in entries] != list(range(1, len(entries) + 1)):
        fail("canonical entries are not a contiguous prefix of frozen slots")
    counts = Counter(entry.mode for entry in entries)
    if require_complete:
        if len(entries) != 60 or counts != Counter({"high": 30, "none": 30}):
            fail(
                "final analysis requires exactly 60 canonical runs "
                f"(30/arm); found total={len(entries)}, counts={dict(counts)}"
            )
    return entries


def validate_transcript(entry: ManifestEntry) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    tuple[int, ...],
    tuple[int, ...],
]:
    run_log_path = entry.run_dir / "run.log"
    if not run_log_path.is_file() or not run_log_path.stat().st_size:
        fail(f"missing or empty run log: {run_log_path}")
    run_log = run_log_path.read_text(encoding="utf-8")
    expected_thinking = "True" if entry.mode == "high" else "False"
    runtime_signature = (
        "Using vllm-openai with "
        "base_url=https://model-w67n482q.api.baseten.co/"
        "deployment/wxpnlg5/sync/v1, "
        f"model={MODEL}, thinking={expected_thinking}, "
        "thinking_budget=None, T=0.6, top_p=0.95, "
        "top_k=None, max_tokens=8192"
    )
    if runtime_signature not in run_log:
        fail(f"runtime configuration signature missing at slot {entry.slot}")
    if "MTE_FILLER_" in run_log:
        fail(f"unexpected filler activation at slot {entry.slot}")

    transcript_path = entry.run_dir / "transcript.jsonl"
    rows = read_jsonl(transcript_path)
    scheduled = canonical_turn_map(rows, transcript_path)
    observed = tuple(sorted(scheduled))
    if observed != tuple(range(len(observed))):
        fail(
            f"scheduled records are not a turn-0 prefix for slot {entry.slot}: "
            f"{observed}"
        )
    if not observed:
        fail(f"canonical run has no scheduled responses: slot {entry.slot}")
    if len(observed) > N_TURNS:
        fail(f"too many scheduled turns for slot {entry.slot}")
    if int(entry.raw["turns"]) != len(observed):
        fail(
            f"manifest/transcript turn mismatch at slot {entry.slot}: "
            f"{entry.raw['turns']} vs {len(observed)}"
        )
    for turn, row in scheduled.items():
        if row.get("model_name") != MODEL:
            fail(
                f"model mismatch at slot {entry.slot}, turn {turn}: "
                f"{row.get('model_name')!r}"
            )
    thought_turns = sum(bool(row.get("assistant_thought")) for row in scheduled.values())
    if int(entry.raw["thought_turns"]) != thought_turns:
        fail(
            f"manifest/transcript thought-turn mismatch at slot {entry.slot}: "
            f"{entry.raw['thought_turns']} vs {thought_turns}"
        )
    if entry.mode == "none" and thought_turns:
        fail(f"thinking-off slot {entry.slot} contains {thought_turns} thought turns")
    if entry.mode == "high" and not thought_turns:
        fail(f"thinking-on slot {entry.slot} contains no captured thought turns")
    return (
        rows,
        scheduled,
        end_session_turns(rows, recovery=False),
        end_session_turns(rows, recovery=True),
    )


def load_conversation(entry: ManifestEntry) -> Conversation:
    transcript_path = entry.run_dir / "transcript.jsonl"
    judgment_path = entry.run_dir / "claude_judged.jsonl"
    summary_path = entry.run_dir / "claude_summary.json"
    transcript_rows, scheduled, scheduled_end, recovery_end = validate_transcript(entry)
    judgment_rows = read_jsonl(judgment_path)
    judgments = canonical_turn_map(judgment_rows, judgment_path)
    if set(judgments) != set(scheduled):
        fail(
            f"judgment coverage mismatch at slot {entry.slot}: "
            f"transcript={sorted(scheduled)}, judged={sorted(judgments)}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("turns_scored") != len(scheduled):
        fail(
            f"judge summary coverage mismatch at slot {entry.slot}: "
            f"{summary.get('turns_scored')} vs {len(scheduled)}"
        )
    judge_model = summary.get("judge_model")
    judge_version = summary.get("judge_version")
    if not isinstance(judge_model, str) or not judge_model:
        fail(f"missing judge model at slot {entry.slot}")
    if not isinstance(judge_version, str) or not judge_version:
        fail(f"missing judge version at slot {entry.slot}")

    scores: dict[str, list[bool]] = {key: [] for key in ALL_COMPONENTS}
    passed: list[bool] = []
    ttfat: list[float] = []
    for turn in range(N_TURNS):
        if turn in judgments:
            judgment = judgments[turn]
            if judgment.get("model_name") not in (None, MODEL):
                fail(f"judgment model mismatch at slot {entry.slot}, turn {turn}")
            turn_scores = judgment.get("scores") or {}
            values: dict[str, bool] = {}
            for key in ALL_COMPONENTS:
                value = turn_scores.get(key)
                if not isinstance(value, bool):
                    fail(
                        f"non-boolean/missing {key} judgment at "
                        f"slot {entry.slot}, turn {turn}"
                    )
                values[key] = value
        else:
            # The fixed denominator is the core analysis policy: every
            # unobserved future turn fails every dimension.
            values = {key: False for key in ALL_COMPONENTS}
        for key in ALL_COMPONENTS:
            scores[key].append(values[key])
        passed.append(all(values[key] for key in SCORE_COMPONENTS))

        if turn in scheduled:
            latency = scheduled[turn].get("ttfb_ms")
            if (
                isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                and math.isfinite(latency)
                and latency >= 0
            ):
                ttfat.append(float(latency))

    observed = tuple(sorted(scheduled))
    full_coverage = observed == tuple(range(N_TURNS))
    strict_completion = full_coverage and scheduled_end == (29,)
    thought_turns = sum(bool(row.get("assistant_thought")) for row in scheduled.values())
    return Conversation(
        entry=entry,
        observed_turns=observed,
        scores={key: tuple(value) for key, value in scores.items()},
        benchmark_pass=tuple(passed),
        full_scheduled_coverage=full_coverage,
        strict_protocol_completion=strict_completion,
        scheduled_end_session_turns=scheduled_end,
        recovery_end_session_turns=recovery_end,
        ttfat_ms=tuple(ttfat),
        thought_turns=thought_turns,
        run_log_sha256=sha256(entry.run_dir / "run.log"),
        transcript_sha256=sha256(transcript_path),
        judgment_sha256=sha256(judgment_path),
        judge_summary_sha256=sha256(summary_path),
        judge_model=judge_model,
        judge_version=judge_version,
    )


def percentile_ci(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * max(0.0, center - spread), 100 * min(1.0, center + spread)]


def latency_matrix(runs: list[Conversation]) -> np.ndarray:
    matrix = np.full((len(runs), N_TURNS), np.nan, dtype=float)
    for row_index, run in enumerate(runs):
        transcript_path = run.entry.run_dir / "transcript.jsonl"
        scheduled = canonical_turn_map(read_jsonl(transcript_path), transcript_path)
        # Observed scheduled turns are a prefix, and a TTFAT can still be null.
        for turn in run.observed_turns:
            latency = scheduled[turn].get("ttfb_ms")
            if (
                isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                and math.isfinite(latency)
                and latency >= 0
            ):
                matrix[row_index, turn] = float(latency)
    return matrix


def bootstrap_cluster_rate(
    conversation_values: np.ndarray,
    rng: np.random.Generator,
    samples: int = BOOTSTRAPS,
) -> np.ndarray:
    n = len(conversation_values)
    indices = rng.integers(0, n, size=(samples, n))
    return conversation_values[indices].mean(axis=1) * 100


def bootstrap_paired_effect(
    high_by_pair: np.ndarray,
    none_by_pair: np.ndarray,
    rng: np.random.Generator,
    samples: int = BOOTSTRAPS,
) -> np.ndarray:
    if high_by_pair.shape != none_by_pair.shape:
        fail("paired arrays have different shapes")
    n = len(high_by_pair)
    indices = rng.integers(0, n, size=(samples, n))
    return (
        high_by_pair[indices].mean(axis=1)
        - none_by_pair[indices].mean(axis=1)
    ) * 100


def arm_summary(
    runs: list[Conversation], rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(runs) != TARGET_PER_ARM:
        fail(f"arm summary requires 30 runs, found {len(runs)}")
    denom = len(runs) * N_TURNS
    raw: dict[str, np.ndarray] = {}
    pass_matrix = np.asarray([run.benchmark_pass for run in runs], dtype=float)
    raw["strict_pass_rate"] = pass_matrix.mean(axis=1)
    raw["any_error_rate"] = 1 - raw["strict_pass_rate"]
    for key in ALL_COMPONENTS:
        matrix = np.asarray([run.scores[key] for run in runs], dtype=float)
        raw[f"{key}_error_rate"] = 1 - matrix.mean(axis=1)
    raw["full_scheduled_coverage"] = np.asarray(
        [run.full_scheduled_coverage for run in runs], dtype=float
    )
    raw["strict_protocol_completion"] = np.asarray(
        [run.strict_protocol_completion for run in runs], dtype=float
    )

    pass_count = int(pass_matrix.sum())
    pass_boot = bootstrap_cluster_rate(raw["strict_pass_rate"], rng)
    summary: dict[str, Any] = {
        "n_conversations": len(runs),
        "fixed_turn_denominator": denom,
        "observed_scheduled_turns": sum(len(run.observed_turns) for run in runs),
        "missing_future_turns_scored_as_failures": (
            denom - sum(len(run.observed_turns) for run in runs)
        ),
        "strict_pass_count": pass_count,
        "strict_pass_rate_pct": 100 * pass_count / denom,
        "strict_pass_rate_ci95_cluster_bootstrap": percentile_ci(pass_boot),
        "any_error_count": denom - pass_count,
        "any_error_rate_pct": 100 * (denom - pass_count) / denom,
        "any_error_rate_ci95_cluster_bootstrap": percentile_ci(100 - pass_boot),
    }
    for key in ALL_COMPONENTS:
        conversation_error = raw[f"{key}_error_rate"]
        error_count = int(round(conversation_error.sum() * N_TURNS))
        error_boot = bootstrap_cluster_rate(conversation_error, rng)
        summary[f"{key}_error_count"] = error_count
        summary[f"{key}_error_rate_pct"] = 100 * error_count / denom
        summary[f"{key}_error_rate_ci95_cluster_bootstrap"] = percentile_ci(
            error_boot
        )

    for metric in ("full_scheduled_coverage", "strict_protocol_completion"):
        values = raw[metric]
        count = int(values.sum())
        boot = bootstrap_cluster_rate(values, rng)
        summary[f"{metric}_count"] = count
        summary[f"{metric}_pct"] = 100 * count / len(runs)
        summary[f"{metric}_ci95_wilson"] = wilson(count, len(runs))
        summary[f"{metric}_ci95_cluster_bootstrap"] = percentile_ci(boot)

    latencies = [value for run in runs for value in run.ttfat_ms]
    summary["ttfat"] = {
        "definition": (
            "Content-aware time to first assistant text or tool-call token; "
            "conditional on an observed scheduled response."
        ),
        "observations": len(latencies),
        "coverage_of_fixed_turn_denominator_pct": 100 * len(latencies) / denom,
        "p50_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": (
            float(np.percentile(np.asarray(latencies, dtype=float), 95))
            if latencies
            else None
        ),
        "max_ms": max(latencies) if latencies else None,
    }
    summary["thought_capture"] = {
        "turns_with_captured_reasoning": sum(run.thought_turns for run in runs),
        "observed_scheduled_turns": sum(len(run.observed_turns) for run in runs),
    }
    return summary, raw


def paired_runs(cells: dict[str, list[Conversation]]) -> list[tuple[Conversation, Conversation]]:
    by_pair: dict[int, dict[str, Conversation]] = defaultdict(dict)
    for mode in ARMS:
        for run in cells[mode]:
            if mode in by_pair[run.entry.pair]:
                fail(f"duplicate {mode} member for pair {run.entry.pair}")
            by_pair[run.entry.pair][mode] = run
    if set(by_pair) != set(range(1, 31)):
        fail("analysis pairs are not exactly 1..30")
    result = []
    for pair in range(1, 31):
        if set(by_pair[pair]) != set(ARMS):
            fail(f"incomplete pair {pair}: {sorted(by_pair[pair])}")
        result.append((by_pair[pair]["high"], by_pair[pair]["none"]))
    return result


def latency_quantile_effect(
    pairs: list[tuple[Conversation, Conversation]],
    rng: np.random.Generator,
    q: float,
) -> tuple[float | None, list[float] | None]:
    high = latency_matrix([pair[0] for pair in pairs])
    none = latency_matrix([pair[1] for pair in pairs])
    high_values = high[np.isfinite(high)]
    none_values = none[np.isfinite(none)]
    if not len(high_values) or not len(none_values):
        return None, None
    point = float(np.percentile(high_values, q) - np.percentile(none_values, q))
    effects: list[np.ndarray] = []
    batch = 500
    for start in range(0, LATENCY_BOOTSTRAPS, batch):
        size = min(batch, LATENCY_BOOTSTRAPS - start)
        indices = rng.integers(0, len(pairs), size=(size, len(pairs)))
        high_sample = high[indices].reshape(size, -1)
        none_sample = none[indices].reshape(size, -1)
        effects.append(
            np.nanpercentile(high_sample, q, axis=1)
            - np.nanpercentile(none_sample, q, axis=1)
        )
    return point, percentile_ci(np.concatenate(effects))


def effect_summary(
    pairs: list[tuple[Conversation, Conversation]],
    raw: dict[str, dict[str, np.ndarray]],
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "direction": "high minus none",
        "bootstrap_unit": "the 30 frozen high/none pairs",
    }
    metrics = (
        "strict_pass_rate",
        "any_error_rate",
        "tool_use_correct_error_rate",
        "instruction_following_error_rate",
        "kb_grounding_error_rate",
        "turn_taking_error_rate",
        "full_scheduled_coverage",
        "strict_protocol_completion",
    )
    for metric_index, metric in enumerate(metrics):
        # arm raw arrays are already sorted by manifest slot, not necessarily pair.
        high_map = {
            run.entry.pair: raw["high"][metric][index]
            for index, run in enumerate(
                sorted(
                    [pair[0] for pair in pairs], key=lambda run: run.entry.slot
                )
            )
        }
        none_map = {
            run.entry.pair: raw["none"][metric][index]
            for index, run in enumerate(
                sorted(
                    [pair[1] for pair in pairs], key=lambda run: run.entry.slot
                )
            )
        }
        high = np.asarray([high_map[pair] for pair in range(1, 31)])
        none = np.asarray([none_map[pair] for pair in range(1, 31)])
        rng = np.random.default_rng(SEED + 20_000 + metric_index)
        boot = bootstrap_paired_effect(high, none, rng)
        high_pct = float(high.mean() * 100)
        none_pct = float(none.mean() * 100)
        result[metric] = {
            "high_pct": high_pct,
            "none_pct": none_pct,
            "high_minus_none_points": high_pct - none_pct,
            "high_minus_none_ci95_paired_cluster_bootstrap": percentile_ci(boot),
        }

    p50, p50_ci = latency_quantile_effect(
        pairs, np.random.default_rng(SEED + 30_050), 50
    )
    p95, p95_ci = latency_quantile_effect(
        pairs, np.random.default_rng(SEED + 30_095), 95
    )
    result["ttfat"] = {
        "conditional_on_observed_response": True,
        "high_p50_ms": arms["high"]["ttfat"]["p50_ms"],
        "none_p50_ms": arms["none"]["ttfat"]["p50_ms"],
        "p50_high_minus_none_ms": p50,
        "p50_high_minus_none_ci95_paired_cluster_bootstrap": p50_ci,
        "high_p95_ms": arms["high"]["ttfat"]["p95_ms"],
        "none_p95_ms": arms["none"]["ttfat"]["p95_ms"],
        "p95_high_minus_none_ms": p95,
        "p95_high_minus_none_ci95_paired_cluster_bootstrap": p95_ci,
    }
    return result


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def audit_row(run: Conversation) -> dict[str, Any]:
    return {
        "slot": run.entry.slot,
        "pair": run.entry.pair,
        "order_in_pair": run.entry.order_in_pair,
        "mode": run.entry.mode,
        "attempt": run.entry.attempt,
        "run_dir": rel(run.entry.run_dir),
        "campaign_classification": run.entry.classification,
        "observed_scheduled_turns": len(run.observed_turns),
        "fixed_turn_denominator": N_TURNS,
        "missing_future_turns": N_TURNS - len(run.observed_turns),
        "full_scheduled_coverage": run.full_scheduled_coverage,
        "strict_protocol_completion": run.strict_protocol_completion,
        "scheduled_end_session_turns": list(run.scheduled_end_session_turns),
        "recovery_end_session_turns": list(run.recovery_end_session_turns),
        "strict_pass_count": sum(run.benchmark_pass),
        "tool_error_count": N_TURNS - sum(run.scores["tool_use_correct"]),
        "instruction_error_count": (
            N_TURNS - sum(run.scores["instruction_following"])
        ),
        "kb_error_count": N_TURNS - sum(run.scores["kb_grounding"]),
        "turn_taking_error_count": N_TURNS - sum(run.scores["turn_taking"]),
        "ttfat_observations": len(run.ttfat_ms),
        "thought_turns": run.thought_turns,
        "judge_model": run.judge_model,
        "judge_version": run.judge_version,
        "run_log_sha256": run.run_log_sha256,
        "transcript_sha256": run.transcript_sha256,
        "judgment_sha256": run.judgment_sha256,
        "judge_summary_sha256": run.judge_summary_sha256,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fail(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            formatted = {
                key: (
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(formatted)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_ci(values: list[float]) -> str:
    return f"{values[0]:.1f}–{values[1]:.1f}"


def fmt_ms(value: float | None) -> str:
    return "NA" if value is None else f"{value:.0f}"


def render_report(payload: dict[str, Any]) -> str:
    high = payload["arms"]["high"]
    none = payload["arms"]["none"]
    effects = payload["effects_high_minus_none"]
    lines = [
        "# Qwen3.6-27B: native thinking-on versus thinking-off",
        "",
        (
            "Final fixed-denominator results for the BaseTen vLLM 0.26 "
            "APC+MTP deployment. Each arm contains 30 canonical conversations "
            "and every conversation contributes 30 scheduled turns. After an "
            "early model exit, every unobserved future turn is scored as a "
            "failure on every dimension."
        ),
        "",
        "## Accuracy",
        "",
        (
            "| Arm | Strict pass | 95% cluster-bootstrap CI | Any error | "
            "Tool error | Instruction error | KB error |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, arm in (("High / native thinking-on", high), ("None / thinking-off", none)):
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    fmt_pct(arm["strict_pass_rate_pct"]),
                    fmt_ci(arm["strict_pass_rate_ci95_cluster_bootstrap"]),
                    fmt_pct(arm["any_error_rate_pct"]),
                    fmt_pct(arm["tool_use_correct_error_rate_pct"]),
                    fmt_pct(arm["instruction_following_error_rate_pct"]),
                    fmt_pct(arm["kb_grounding_error_rate_pct"]),
                ]
            )
            + " |"
        )
    pass_effect = effects["strict_pass_rate"]
    lines.extend(
        [
            "",
            (
                "High minus none strict-pass effect: "
                f"**{pass_effect['high_minus_none_points']:+.1f} points** "
                "(paired whole-conversation bootstrap 95% CI "
                f"{pass_effect['high_minus_none_ci95_paired_cluster_bootstrap'][0]:+.1f} "
                "to "
                f"{pass_effect['high_minus_none_ci95_paired_cluster_bootstrap'][1]:+.1f})."
            ),
            "",
            "## Completion",
            "",
            (
                "| Arm | All 30 scheduled turns | Wilson 95% CI | "
                "`end_session` exactly at scheduled turn 29 | Wilson 95% CI | "
                "Missing future turns |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, arm in (("High / native thinking-on", high), ("None / thinking-off", none)):
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    fmt_pct(arm["full_scheduled_coverage_pct"]),
                    fmt_ci(arm["full_scheduled_coverage_ci95_wilson"]),
                    fmt_pct(arm["strict_protocol_completion_pct"]),
                    fmt_ci(arm["strict_protocol_completion_ci95_wilson"]),
                    str(arm["missing_future_turns_scored_as_failures"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "“All 30 scheduled turns” measures response coverage. Strict "
                "protocol completion additionally requires the terminal tool "
                "on scripted turn 29; a synthetic recovery call at turn 30 "
                "does not count."
            ),
            "",
            "## Latency",
            "",
            "| Arm | TTFAT observations | Coverage | P50 | P95 | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, arm in (("High / native thinking-on", high), ("None / thinking-off", none)):
        timing = arm["ttfat"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    str(timing["observations"]),
                    fmt_pct(timing["coverage_of_fixed_turn_denominator_pct"]),
                    f"{fmt_ms(timing['p50_ms'])} ms",
                    f"{fmt_ms(timing['p95_ms'])} ms",
                    f"{fmt_ms(timing['max_ms'])} ms",
                ]
            )
            + " |"
        )
    timing_effect = effects["ttfat"]
    lines.extend(
        [
            "",
            (
                "Observed-response P50 TTFAT effect, high minus none: "
                f"**{fmt_ms(timing_effect['p50_high_minus_none_ms'])} ms** "
                "(paired conversation-bootstrap 95% CI "
                f"{fmt_ms(timing_effect['p50_high_minus_none_ci95_paired_cluster_bootstrap'][0])} "
                "to "
                f"{fmt_ms(timing_effect['p50_high_minus_none_ci95_paired_cluster_bootstrap'][1])} ms)."
            )
            if timing_effect["p50_high_minus_none_ci95_paired_cluster_bootstrap"]
            else "Observed-response P50 TTFAT effect: unavailable.",
            "",
            (
                "TTFAT is the content-aware time to the first assistant text "
                "or tool-call token. It is summarized only where a scheduled "
                "model response was observed; missing turns remain accuracy "
                "failures but are not assigned fictitious latency."
            ),
            "",
            "## Methods and audit trail",
            "",
            (
                "Strict pass is the conjunction of tool-use correctness, "
                "instruction following, and KB grounding, matching the README "
                "benchmark definition. Turn-taking error is retained as a "
                "supplementary metric in `aggregates.json` and `effects.tsv`."
            ),
            "",
            (
                "Arm confidence intervals resample whole conversations. "
                "High-minus-none intervals resample the 30 frozen high/none "
                "pairs, preserving the campaign's balanced temporal blocks. "
                f"Rate intervals use {BOOTSTRAPS:,} deterministic bootstrap "
                f"draws; latency-quantile intervals use {LATENCY_BOOTSTRAPS:,}."
            ),
            "",
            (
                "`included-runs.tsv` records the exact manifest membership, "
                "per-run fixed-denominator counts, judge identity, and SHA-256 "
                "hashes of every transcript, judgment, and judge summary."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def preflight() -> None:
    entries = load_manifest(require_complete=False)
    counts = Counter(entry.mode for entry in entries)
    missing_judgments = 0
    shorts = 0
    for entry in entries:
        _, scheduled, _, _ = validate_transcript(entry)
        shorts += int(len(scheduled) < N_TURNS)
        if not all(
            (entry.run_dir / filename).is_file()
            and (entry.run_dir / filename).stat().st_size
            for filename in ("claude_judged.jsonl", "claude_summary.json")
        ):
            missing_judgments += 1
    print(
        json.dumps(
            {
                "status": "preflight_ok",
                "canonical_conversations": len(entries),
                "counts": dict(counts),
                "fixed_denominator_short_conversations": shorts,
                "missing_judgments": missing_judgments,
                "remaining_slots": 60 - len(entries),
            },
            indent=2,
        )
    )


def final() -> None:
    entries = load_manifest(require_complete=True)
    conversations = [load_conversation(entry) for entry in entries]
    cells = {
        mode: sorted(
            [run for run in conversations if run.entry.mode == mode],
            key=lambda run: run.entry.slot,
        )
        for mode in ARMS
    }
    arm_payload: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for index, mode in enumerate(ARMS):
        arm_payload[mode], raw[mode] = arm_summary(
            cells[mode], np.random.default_rng(SEED + index)
        )
    pairs = paired_runs(cells)
    effects = effect_summary(pairs, raw, arm_payload)
    audit = [audit_row(run) for run in conversations]
    judge_models = sorted({run.judge_model for run in conversations})
    judge_versions = sorted({run.judge_version for run in conversations})
    if len(judge_models) != 1 or len(judge_versions) != 1:
        fail(
            "mixed judge identity across canonical cohort: "
            f"models={judge_models}, versions={judge_versions}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_status": "FINAL",
        "protocol": {
            "benchmark": "aiwf_medium_context",
            "model": MODEL,
            "provider": "BaseTen",
            "arms": {
                "high": (
                    "uncapped native Qwen thinking-on; not a graded OpenAI effort"
                ),
                "none": "Qwen thinking disabled",
            },
            "conversations_per_arm": TARGET_PER_ARM,
            "scheduled_turns_per_conversation": N_TURNS,
            "fixed_turn_denominator_per_arm": (
                TARGET_PER_ARM * N_TURNS
            ),
            "missing_future_turn_policy": (
                "fail tool use, instruction following, KB grounding, "
                "turn-taking, and therefore strict pass"
            ),
            "strict_pass_definition": (
                "tool_use_correct AND instruction_following AND kb_grounding"
            ),
            "strict_protocol_completion_definition": (
                "all scheduled turns 0..29 observed and end_session called "
                "exactly on scheduled turn 29"
            ),
            "ttfat_definition": (
                "content-aware time to first assistant text or tool-call token, "
                "conditional on an observed scheduled response"
            ),
            "arm_ci_method": (
                "whole-conversation nonparametric bootstrap for turn rates; "
                "Wilson interval also supplied for conversation completion"
            ),
            "effect_ci_method": (
                "paired whole-conversation bootstrap over the 30 frozen "
                "high/none temporal blocks"
            ),
            "bootstrap_samples": BOOTSTRAPS,
            "latency_bootstrap_samples": LATENCY_BOOTSTRAPS,
            "seed": SEED,
            "judge_models": judge_models,
            "judge_versions": judge_versions,
        },
        "input_hashes": {
            "configuration.json": sha256(CAMPAIGN / "configuration.json"),
            "frozen-order.tsv": sha256(CAMPAIGN / "frozen-order.tsv"),
            "canonical.tsv": sha256(CAMPAIGN / "canonical.tsv"),
            "source-sha256.txt": sha256(CAMPAIGN / "source-sha256.txt"),
        },
        "arms": arm_payload,
        "effects_high_minus_none": effects,
        "included_runs": audit,
    }
    (HERE / "aggregates.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    arm_rows: list[dict[str, Any]] = []
    for mode in ARMS:
        arm = arm_payload[mode]
        arm_rows.append(
            {
                "mode": mode,
                "n_conversations": arm["n_conversations"],
                "fixed_turn_denominator": arm["fixed_turn_denominator"],
                "observed_scheduled_turns": arm["observed_scheduled_turns"],
                "missing_future_turns": arm[
                    "missing_future_turns_scored_as_failures"
                ],
                "strict_pass_rate_pct": arm["strict_pass_rate_pct"],
                "strict_pass_ci95": arm[
                    "strict_pass_rate_ci95_cluster_bootstrap"
                ],
                "any_error_rate_pct": arm["any_error_rate_pct"],
                "tool_error_rate_pct": arm[
                    "tool_use_correct_error_rate_pct"
                ],
                "instruction_error_rate_pct": arm[
                    "instruction_following_error_rate_pct"
                ],
                "kb_error_rate_pct": arm["kb_grounding_error_rate_pct"],
                "turn_taking_error_rate_pct": arm[
                    "turn_taking_error_rate_pct"
                ],
                "full_scheduled_coverage_pct": arm[
                    "full_scheduled_coverage_pct"
                ],
                "strict_protocol_completion_pct": arm[
                    "strict_protocol_completion_pct"
                ],
                "ttfat_observations": arm["ttfat"]["observations"],
                "ttfat_p50_ms": arm["ttfat"]["p50_ms"],
                "ttfat_p95_ms": arm["ttfat"]["p95_ms"],
                "ttfat_max_ms": arm["ttfat"]["max_ms"],
            }
        )
    write_tsv(HERE / "aggregates.tsv", arm_rows)

    effect_rows: list[dict[str, Any]] = []
    for metric, value in effects.items():
        if metric in ("direction", "bootstrap_unit", "ttfat"):
            continue
        effect_rows.append(
            {
                "metric": metric,
                "high": value["high_pct"],
                "none": value["none_pct"],
                "high_minus_none": value["high_minus_none_points"],
                "ci95": value[
                    "high_minus_none_ci95_paired_cluster_bootstrap"
                ],
                "unit": "percentage points",
            }
        )
    for quantile in ("p50", "p95"):
        value = effects["ttfat"]
        effect_rows.append(
            {
                "metric": f"ttfat_{quantile}",
                "high": value[f"high_{quantile}_ms"],
                "none": value[f"none_{quantile}_ms"],
                "high_minus_none": value[
                    f"{quantile}_high_minus_none_ms"
                ],
                "ci95": value[
                    f"{quantile}_high_minus_none_ci95_paired_cluster_bootstrap"
                ],
                "unit": "ms",
            }
        )
    write_tsv(HERE / "effects.tsv", effect_rows)
    write_tsv(HERE / "included-runs.tsv", audit)
    (HERE / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "final"))
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
    else:
        final()


if __name__ == "__main__":
    main()
