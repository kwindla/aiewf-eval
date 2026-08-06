#!/usr/bin/env python3
"""Audit and analyze the frozen Inkling Small none/low AIEWF campaign.

The default invocation is read-only and reports collection/judging progress.
``--write`` requires all 60 frozen canonical conversations and structurally
valid judge outputs, then atomically writes the final analysis artifacts.

Inclusion is defined only by ``../canonical.tsv``.  The analyzer never scans
``runs/`` to discover or select conversations.  Each canonical conversation
contributes all 30 scheduled turns; unobserved future turns fail every scored
dimension and strict pass, while latency remains conditional on an observed
response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]

CONFIG_PATH = CAMPAIGN / "configuration.json"
SCHEDULE_PATH = CAMPAIGN / "frozen-order.tsv"
CANONICAL_PATH = CAMPAIGN / "canonical.tsv"
SOURCE_HASH_PATH = CAMPAIGN / "source-sha256.txt"
JUDGING_DIR = CAMPAIGN / "judging"
JUDGE_COMPLETE_PATH = JUDGING_DIR / "COMPLETE.json"
JUDGE_INPUTS_PATH = JUDGING_DIR / "canonical-inputs.tsv"
JUDGE_SOURCE_PATH = JUDGING_DIR / "judge-source-sha256.txt"

MODEL = "thinkingmachines/inkling-small"
PROVIDER = "BaseTen Model API"
ENDPOINT = "https://inference.baseten.co/v1"
ARMS = ("none", "low")
TARGET_PER_ARM = 30
N_TURNS = 30
DENOMINATOR_PER_ARM = TARGET_PER_ARM * N_TURNS
SCORE_COMPONENTS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
)
SUPPLEMENTARY_COMPONENTS = ("turn_taking",)
ALL_COMPONENTS = SCORE_COMPONENTS + SUPPLEMENTARY_COMPONENTS
BOOTSTRAPS = 100_000
QUANTILE_BOOTSTRAPS = 20_000
SEED = 20260731


@dataclass(frozen=True)
class ManifestEntry:
    slot: str
    slot_index: int
    pair: int
    arm: str
    pair_order: str
    attempt: int
    run_dir: Path
    classification: str
    raw: dict[str, str]


@dataclass(frozen=True)
class Conversation:
    entry: ManifestEntry
    observed_turns: tuple[int, ...]
    scores: dict[str, tuple[bool, ...]]
    strict_pass: tuple[bool, ...]
    full_scheduled_coverage: bool
    strict_protocol_completion: bool
    scheduled_end_session_turns: tuple[int, ...]
    recovery_end_session_turns: tuple[int, ...]
    ttfat_ms: tuple[float | None, ...]
    raw_ttfb_ms: tuple[float | None, ...]
    reasoning_tokens: tuple[float | None, ...]
    judge_model: str
    judge_version: str
    run_log_sha256: str
    transcript_sha256: str
    judgment_sha256: str
    judge_summary_sha256: str


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path, *, allow_header_only: bool = False) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        fail(f"missing or empty TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows and not allow_header_only:
        fail(f"TSV contains no rows: {path}")
    return rows


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
                fail(f"invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSON row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        fail(f"JSONL contains no rows: {path}")
    return rows


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fail(f"refusing to write empty TSV: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        fail(f"run directory escapes repository root: {value}")
    return resolved


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def validate_configuration() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "campaign_id": "aiewf-medium-inkling-small-baseten-none-low-n30-20260731",
        "benchmark": "aiwf_medium_context",
        "provider": PROVIDER,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "service": "baseten",
        "pipeline": "text",
        "arms": list(ARMS),
        "target_valid_conversations_per_arm": TARGET_PER_ARM,
        "fixed_scheduled_turns_per_conversation": N_TURNS,
        "schedule_seed": SEED,
        "filler": None,
        "sampling": {"temperature": 1.0, "max_tokens": 16384},
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            fail(
                f"configuration mismatch for {key}: expected "
                f"{expected_value!r}, found {config.get(key)!r}"
            )
    runtime = config.get("runtime") or {}
    if runtime != {
        "provider_endpoint_concurrency": 1,
        "enable_recovery": True,
        "dedupe_tool_calls": True,
        "tool_result_run_llm": False,
        "text_idle_timeout_seconds": 45,
    }:
        fail(f"runtime configuration is not frozen: {runtime!r}")
    if config.get("max_attempts_per_slot") != 4:
        fail("campaign attempt ceiling changed")
    return config


def load_schedule() -> list[dict[str, str]]:
    rows = read_tsv(SCHEDULE_PATH)
    if len(rows) != 60:
        fail(f"frozen order must contain 60 assignments; found {len(rows)}")
    expected_slots = [f"IS-{index:02d}" for index in range(1, 61)]
    if [row.get("slot") for row in rows] != expected_slots:
        fail("frozen slots are not exactly IS-01 through IS-60")
    for pair in range(1, 31):
        pair_rows = [row for row in rows if int(row["pair"]) == pair]
        if len(pair_rows) != 2 or {row["arm"] for row in pair_rows} != set(ARMS):
            fail(f"pair {pair} does not contain one none and one low assignment")
        order = pair_rows[0]["pair_order"]
        if order not in {"none-low", "low-none"}:
            fail(f"invalid pair order for pair {pair}: {order}")
        if any(row["pair_order"] != order for row in pair_rows):
            fail(f"pair-order mismatch within pair {pair}")
        if [row["arm"] for row in pair_rows] != order.split("-"):
            fail(f"slot sequence does not match pair order for pair {pair}")
    for block in range(5):
        first_rows = rows[block * 12 : (block + 1) * 12 : 2]
        orders = Counter(row["pair_order"] for row in first_rows)
        if orders != Counter({"none-low": 3, "low-none": 3}):
            fail(f"temporal block {block + 1} is not 3/3 order-balanced")
    return rows


def load_manifest(*, require_complete: bool) -> list[ManifestEntry]:
    validate_configuration()
    schedule = load_schedule()
    schedule_by_slot = {row["slot"]: row for row in schedule}
    rows = read_tsv(CANONICAL_PATH, allow_header_only=True)
    expected_prefix = [row["slot"] for row in schedule[: len(rows)]]
    if [row.get("slot") for row in rows] != expected_prefix:
        fail("canonical manifest is not a contiguous frozen-order prefix")
    if require_complete and len(rows) != 60:
        fail(f"final analysis requires 60 canonical runs; found {len(rows)}")

    entries: list[ManifestEntry] = []
    seen_dirs: set[Path] = set()
    for index, row in enumerate(rows, start=1):
        slot = row["slot"]
        frozen = schedule_by_slot[slot]
        if row["pair"] != frozen["pair"] or row["arm"] != frozen["arm"]:
            fail(f"canonical/frozen assignment mismatch at {slot}")
        run_dir = resolve_run_dir(row["run_dir"])
        if run_dir in seen_dirs:
            fail(f"duplicate canonical run directory: {run_dir}")
        seen_dirs.add(run_dir)
        entries.append(
            ManifestEntry(
                slot=slot,
                slot_index=index,
                pair=int(row["pair"]),
                arm=row["arm"],
                pair_order=frozen["pair_order"],
                attempt=int(row["attempt"]),
                run_dir=run_dir,
                classification=row["classification"],
                raw=row,
            )
        )
    counts = Counter(entry.arm for entry in entries)
    if require_complete and counts != Counter({"none": 30, "low": 30}):
        fail(f"final arm counts must be 30/30; found {dict(counts)}")
    return entries


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
    observed = sorted(result)
    if observed != list(range(len(observed))):
        fail(f"scheduled turns are not a contiguous prefix: {path}")
    return result


def tool_name(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def end_session_turns(
    rows: Iterable[dict[str, Any]], *, recovery: bool
) -> tuple[int, ...]:
    turns: list[int] = []
    for row in rows:
        if (row.get("recovery_turn") is True) != recovery:
            continue
        if any(tool_name(call) == "end_session" for call in row.get("tool_calls") or []):
            turn = row.get("turn")
            if isinstance(turn, int):
                turns.append(turn)
    return tuple(sorted(turns))


def response_present(row: dict[str, Any]) -> bool:
    text = row.get("assistant_text")
    return bool((isinstance(text, str) and text.strip()) or row.get("tool_calls"))


def finite_nonnegative(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value)
    return None


def validate_runtime(entry: ManifestEntry, transcript: dict[int, dict[str, Any]]) -> None:
    run_log = entry.run_dir / "run.log"
    if not run_log.is_file() or not run_log.stat().st_size:
        fail(f"missing run.log at {entry.slot}: {run_log}")
    text = run_log.read_text(encoding="utf-8", errors="replace")
    required = (
        f"Using BaseTen with base_url={ENDPOINT}, model={MODEL}, "
        f"reasoning_effort={entry.arm}, enable_thinking=(unset), "
        "max_tokens=16384, temperature=1.0",
        "Recovery nudges enabled=True",
        "Tool call dedupe enabled=True",
        "Tool result run_llm enabled=False",
        "Text pipeline idle_timeout_secs=45.0",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        fail(f"runtime provenance missing {missing} at {entry.slot}")
    if "MTE_FILLER_DOTS active:" in text:
        fail(f"filler activation leaked into {entry.slot}")
    for turn, row in transcript.items():
        if row.get("model_name") != MODEL:
            fail(
                f"model mismatch at {entry.slot} turn {turn}: "
                f"{row.get('model_name')!r}"
            )


def validate_campaign_counts(
    entry: ManifestEntry,
    transcript: dict[int, dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
    scheduled_rows = len(transcript)
    response_turns = sum(response_present(row) for row in transcript.values())
    scheduled_ends = end_session_turns(all_rows, recovery=False)
    recovery_ends = end_session_turns(all_rows, recovery=True)
    recorded_end = max(scheduled_ends, default=-1)
    if recovery_ends and recorded_end < 0:
        recorded_end = 30
    expected = {
        "scheduled_rows": scheduled_rows,
        "response_turns": response_turns,
        "end_session_turn": recorded_end,
    }
    for field, value in expected.items():
        if int(entry.raw[field]) != value:
            fail(
                f"canonical {field} mismatch at {entry.slot}: "
                f"{entry.raw[field]} != {value}"
            )


def validate_judge_complete() -> dict[str, Any]:
    if not JUDGE_COMPLETE_PATH.is_file():
        fail(f"judging completion marker is missing: {JUDGE_COMPLETE_PATH}")
    payload = json.loads(JUDGE_COMPLETE_PATH.read_text(encoding="utf-8"))
    completed = payload.get("canonical_runs", payload.get("canonical_conversations"))
    if completed != 60:
        fail(f"judge completion marker does not cover 60 runs: {completed!r}")
    return payload


def validate_frozen_judge_inputs(entries: list[ManifestEntry]) -> None:
    if not JUDGE_INPUTS_PATH.is_file():
        fail(f"frozen judge input manifest is missing: {JUDGE_INPUTS_PATH}")
    rows = read_tsv(JUDGE_INPUTS_PATH)
    if len(rows) != len(entries):
        fail(
            f"judge input manifest contains {len(rows)} rows; "
            f"expected {len(entries)}"
        )
    by_slot = {row.get("slot"): row for row in rows}
    if set(by_slot) != {entry.slot for entry in entries}:
        fail("judge input slots do not match the canonical manifest")
    for entry in entries:
        row = by_slot[entry.slot]
        transcript = entry.run_dir / "transcript.jsonl"
        if row.get("run_dir") != relative_to_root(entry.run_dir):
            fail(f"judge input run directory mismatch at {entry.slot}")
        if row.get("transcript_sha256") != sha256(transcript):
            fail(f"judge input transcript hash mismatch at {entry.slot}")
        scheduled_turns = row.get("scheduled_turns", row.get("scheduled_rows"))
        if scheduled_turns is None or int(scheduled_turns) != int(
            entry.raw["scheduled_rows"]
        ):
            fail(f"judge input scheduled-turn count mismatch at {entry.slot}")


def load_conversation(entry: ManifestEntry) -> Conversation:
    transcript_path = entry.run_dir / "transcript.jsonl"
    judgment_path = entry.run_dir / "claude_judged.jsonl"
    summary_path = entry.run_dir / "claude_summary.json"
    analysis_path = entry.run_dir / "claude_analysis.md"

    all_transcript_rows = read_jsonl(transcript_path)
    transcript = scheduled_map(all_transcript_rows, path=transcript_path)
    if not transcript:
        fail(f"canonical transcript has no scheduled response: {entry.slot}")
    validate_runtime(entry, transcript)
    validate_campaign_counts(entry, transcript, all_transcript_rows)

    judgments = scheduled_map(read_jsonl(judgment_path), path=judgment_path)
    if set(judgments) != set(transcript):
        fail(
            f"judgment coverage mismatch at {entry.slot}: "
            f"transcript={sorted(transcript)}, judged={sorted(judgments)}"
        )
    if not analysis_path.is_file() or not analysis_path.stat().st_size:
        fail(f"missing claude_analysis.md at {entry.slot}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    judge_model = summary.get("judge_model")
    judge_version = summary.get("judge_version")
    if not isinstance(judge_model, str) or not judge_model:
        fail(f"missing judge_model in summary at {entry.slot}")
    if not isinstance(judge_version, str) or not judge_version:
        fail(f"missing judge_version in summary at {entry.slot}")
    if summary.get("model_name") != MODEL:
        fail(f"judge summary model mismatch at {entry.slot}")
    if summary.get("turns_scored") != len(transcript):
        fail(f"judge summary turn count mismatch at {entry.slot}")

    scores: dict[str, list[bool]] = {key: [] for key in ALL_COMPONENTS}
    strict_pass: list[bool] = []
    ttfat: list[float | None] = []
    raw_ttfb: list[float | None] = []
    reasoning: list[float | None] = []
    for turn in range(N_TURNS):
        if turn in judgments:
            judged_row = judgments[turn]
            if judged_row.get("model_name") not in (None, MODEL):
                fail(f"judgment model mismatch at {entry.slot} turn {turn}")
            judged_scores = judged_row.get("scores")
            if not isinstance(judged_scores, dict):
                fail(f"judgment lacks scores at {entry.slot} turn {turn}")
            values: dict[str, bool] = {}
            for key in ALL_COMPONENTS:
                value = judged_scores.get(key)
                if not isinstance(value, bool):
                    fail(
                        f"judgment lacks boolean {key} at "
                        f"{entry.slot} turn {turn}"
                    )
                values[key] = value
        else:
            values = {key: False for key in ALL_COMPONENTS}
        for key in ALL_COMPONENTS:
            scores[key].append(values[key])
        strict_pass.append(all(values[key] for key in SCORE_COMPONENTS))

        transcript_row = transcript.get(turn)
        if transcript_row is None:
            ttfat.append(None)
            raw_ttfb.append(None)
            reasoning.append(None)
            continue
        ttfat.append(finite_nonnegative(transcript_row.get("ttfb_ms")))
        raw_ttfb.append(finite_nonnegative(transcript_row.get("raw_ttfb_ms")))
        token_data = transcript_row.get("tokens")
        reasoning_value = (
            token_data.get("thinking_tokens")
            if isinstance(token_data, dict)
            else None
        )
        reasoning.append(finite_nonnegative(reasoning_value))

    observed = tuple(sorted(transcript))
    scheduled_ends = end_session_turns(all_transcript_rows, recovery=False)
    recovery_ends = end_session_turns(all_transcript_rows, recovery=True)
    full_coverage = observed == tuple(range(N_TURNS))
    return Conversation(
        entry=entry,
        observed_turns=observed,
        scores={key: tuple(value) for key, value in scores.items()},
        strict_pass=tuple(strict_pass),
        full_scheduled_coverage=full_coverage,
        strict_protocol_completion=full_coverage and scheduled_ends == (29,),
        scheduled_end_session_turns=scheduled_ends,
        recovery_end_session_turns=recovery_ends,
        ttfat_ms=tuple(ttfat),
        raw_ttfb_ms=tuple(raw_ttfb),
        reasoning_tokens=tuple(reasoning),
        judge_model=judge_model,
        judge_version=judge_version,
        run_log_sha256=sha256(entry.run_dir / "run.log"),
        transcript_sha256=sha256(transcript_path),
        judgment_sha256=sha256(judgment_path),
        judge_summary_sha256=sha256(summary_path),
    )


def percentile_ci(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = (
        z
        * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denominator
    )
    return [
        100 * max(0.0, center - spread),
        100 * min(1.0, center + spread),
    ]


def bootstrap_conversation_rate(
    values: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return values[indices].mean(axis=1) * 100


def numeric_matrix(
    runs: list[Conversation], field: str
) -> np.ndarray:
    matrix = np.full((len(runs), N_TURNS), np.nan, dtype=float)
    for row_index, run in enumerate(runs):
        values = getattr(run, field)
        for turn, value in enumerate(values):
            if value is not None:
                matrix[row_index, turn] = value
    return matrix


def cluster_quantile_bootstrap(
    matrix: np.ndarray,
    *,
    quantile: float,
    rng: np.random.Generator,
) -> list[float] | None:
    finite = matrix[np.isfinite(matrix)]
    if not len(finite):
        return None
    draws: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, QUANTILE_BOOTSTRAPS, batch_size):
        batch = min(batch_size, QUANTILE_BOOTSTRAPS - start)
        indices = rng.integers(0, len(matrix), size=(batch, len(matrix)))
        sampled = matrix[indices].reshape(batch, -1)
        draws.append(np.nanpercentile(sampled, quantile, axis=1))
    return percentile_ci(np.concatenate(draws))


def numeric_summary(
    matrix: np.ndarray,
    *,
    definition: str,
    seed_offset: int,
) -> dict[str, Any]:
    values = matrix[np.isfinite(matrix)]
    result: dict[str, Any] = {
        "definition": definition,
        "observations": int(len(values)),
        "coverage_of_fixed_turn_denominator_pct": (
            100 * len(values) / matrix.size
        ),
        "p50": float(np.percentile(values, 50)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
        "mean": float(np.mean(values)) if len(values) else None,
        "p50_ci95_cluster_bootstrap": cluster_quantile_bootstrap(
            matrix,
            quantile=50,
            rng=np.random.default_rng(SEED + seed_offset),
        ),
        "p95_ci95_cluster_bootstrap": cluster_quantile_bootstrap(
            matrix,
            quantile=95,
            rng=np.random.default_rng(SEED + seed_offset + 1),
        ),
    }
    return result


def arm_summary(
    runs: list[Conversation], *, arm_index: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if len(runs) != TARGET_PER_ARM:
        fail(f"arm summary requires 30 conversations; found {len(runs)}")
    raw: dict[str, np.ndarray] = {}
    strict_matrix = np.asarray([run.strict_pass for run in runs], dtype=float)
    raw["strict_pass_rate"] = strict_matrix.mean(axis=1)
    raw["any_error_rate"] = 1 - raw["strict_pass_rate"]
    for component in ALL_COMPONENTS:
        matrix = np.asarray([run.scores[component] for run in runs], dtype=float)
        raw[f"{component}_error_rate"] = 1 - matrix.mean(axis=1)
    raw["full_scheduled_coverage"] = np.asarray(
        [run.full_scheduled_coverage for run in runs], dtype=float
    )
    raw["strict_protocol_completion"] = np.asarray(
        [run.strict_protocol_completion for run in runs], dtype=float
    )

    summary: dict[str, Any] = {
        "n_conversations": len(runs),
        "fixed_turn_denominator": DENOMINATOR_PER_ARM,
        "observed_scheduled_turns": sum(len(run.observed_turns) for run in runs),
        "missing_future_turns_scored_as_failures": (
            DENOMINATOR_PER_ARM
            - sum(len(run.observed_turns) for run in runs)
        ),
    }
    metric_names = (
        "strict_pass_rate",
        "any_error_rate",
        *(f"{key}_error_rate" for key in ALL_COMPONENTS),
    )
    for metric_index, metric in enumerate(metric_names):
        values = raw[metric]
        count = int(round(values.sum() * N_TURNS))
        if metric == "strict_pass_rate":
            count = int(strict_matrix.sum())
        summary[f"{metric}_count"] = count
        summary[f"{metric}_pct"] = 100 * count / DENOMINATOR_PER_ARM
        boot = bootstrap_conversation_rate(
            values,
            np.random.default_rng(SEED + arm_index * 100 + metric_index),
        )
        summary[f"{metric}_ci95_cluster_bootstrap"] = percentile_ci(boot)

    for metric_index, metric in enumerate(
        ("full_scheduled_coverage", "strict_protocol_completion")
    ):
        values = raw[metric]
        count = int(values.sum())
        summary[f"{metric}_count"] = count
        summary[f"{metric}_pct"] = 100 * count / len(runs)
        summary[f"{metric}_ci95_wilson"] = wilson(count, len(runs))
        summary[f"{metric}_ci95_cluster_bootstrap"] = percentile_ci(
            bootstrap_conversation_rate(
                values,
                np.random.default_rng(
                    SEED + arm_index * 100 + 50 + metric_index
                ),
            )
        )

    matrices = {
        "ttfat_ms": numeric_matrix(runs, "ttfat_ms"),
        "raw_ttfb_ms": numeric_matrix(runs, "raw_ttfb_ms"),
        "reasoning_tokens": numeric_matrix(runs, "reasoning_tokens"),
    }
    summary["ttfat_ms"] = numeric_summary(
        matrices["ttfat_ms"],
        definition=(
            "Content-aware time to first assistant text or tool-call token; "
            "conditional on an observed scheduled response."
        ),
        seed_offset=1_000 + arm_index * 100,
    )
    summary["raw_ttfb_ms"] = numeric_summary(
        matrices["raw_ttfb_ms"],
        definition=(
            "Time to the first streamed chunk of any kind; conditional on an "
            "observed recorded value."
        ),
        seed_offset=1_020 + arm_index * 100,
    )
    summary["reasoning_tokens"] = numeric_summary(
        matrices["reasoning_tokens"],
        definition=(
            "Provider-reported reasoning tokens for observed scheduled turns "
            "whose usage payload included the field."
        ),
        seed_offset=1_040 + arm_index * 100,
    )
    summary["scheduled_end_session_turn_counts"] = dict(
        sorted(
            Counter(
                turn
                for run in runs
                for turn in run.scheduled_end_session_turns
            ).items()
        )
    )
    summary["recovery_end_session_count"] = sum(
        bool(run.recovery_end_session_turns) for run in runs
    )
    return summary, raw, matrices


def paired_runs(
    conversations: list[Conversation],
) -> list[tuple[Conversation, Conversation]]:
    by_pair: dict[int, dict[str, Conversation]] = defaultdict(dict)
    for run in conversations:
        if run.entry.arm in by_pair[run.entry.pair]:
            fail(f"duplicate {run.entry.arm} conversation in pair {run.entry.pair}")
        by_pair[run.entry.pair][run.entry.arm] = run
    if set(by_pair) != set(range(1, 31)):
        fail("final paired cohort is not exactly pairs 1 through 30")
    pairs: list[tuple[Conversation, Conversation]] = []
    for pair in range(1, 31):
        if set(by_pair[pair]) != set(ARMS):
            fail(f"pair {pair} is incomplete: {sorted(by_pair[pair])}")
        pairs.append((by_pair[pair]["low"], by_pair[pair]["none"]))
    return pairs


def pair_ordered_values(
    pairs: list[tuple[Conversation, Conversation]],
    raw: dict[str, dict[str, np.ndarray]],
    runs_by_arm: dict[str, list[Conversation]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    maps: dict[str, dict[int, float]] = {}
    for arm in ARMS:
        maps[arm] = {
            run.entry.pair: float(raw[arm][metric][index])
            for index, run in enumerate(runs_by_arm[arm])
        }
    pair_numbers = [low.entry.pair for low, _ in pairs]
    return (
        np.asarray([maps["low"][pair] for pair in pair_numbers]),
        np.asarray([maps["none"][pair] for pair in pair_numbers]),
    )


def paired_rate_effects(
    pairs: list[tuple[Conversation, Conversation]],
    raw: dict[str, dict[str, np.ndarray]],
    runs_by_arm: dict[str, list[Conversation]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "direction": "low minus none",
        "bootstrap_unit": "the 30 frozen low/none temporal pairs",
    }
    metrics = (
        "strict_pass_rate",
        "any_error_rate",
        *(f"{key}_error_rate" for key in ALL_COMPONENTS),
        "full_scheduled_coverage",
        "strict_protocol_completion",
    )
    for metric_index, metric in enumerate(metrics):
        low, none = pair_ordered_values(
            pairs, raw, runs_by_arm, metric
        )
        rng = np.random.default_rng(SEED + 10_000 + metric_index)
        indices = rng.integers(0, len(pairs), size=(BOOTSTRAPS, len(pairs)))
        boot = (low[indices].mean(axis=1) - none[indices].mean(axis=1)) * 100
        low_pct = float(low.mean() * 100)
        none_pct = float(none.mean() * 100)
        result[metric] = {
            "low_pct": low_pct,
            "none_pct": none_pct,
            "low_minus_none_points": low_pct - none_pct,
            "low_minus_none_ci95_paired_cluster_bootstrap": percentile_ci(boot),
        }
    return result


def paired_quantile_effect(
    pairs: list[tuple[Conversation, Conversation]],
    *,
    field: str,
    quantile: float,
    seed_offset: int,
) -> dict[str, Any]:
    low_matrix = numeric_matrix([low for low, _ in pairs], field)
    none_matrix = numeric_matrix([none for _, none in pairs], field)
    low_values = low_matrix[np.isfinite(low_matrix)]
    none_values = none_matrix[np.isfinite(none_matrix)]
    if not len(low_values) or not len(none_values):
        return {
            "low": None,
            "none": None,
            "low_minus_none": None,
            "low_minus_none_ci95_paired_cluster_bootstrap": None,
        }
    low_point = float(np.percentile(low_values, quantile))
    none_point = float(np.percentile(none_values, quantile))
    rng = np.random.default_rng(SEED + seed_offset)
    draws: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, QUANTILE_BOOTSTRAPS, batch_size):
        batch = min(batch_size, QUANTILE_BOOTSTRAPS - start)
        indices = rng.integers(0, len(pairs), size=(batch, len(pairs)))
        low_sample = low_matrix[indices].reshape(batch, -1)
        none_sample = none_matrix[indices].reshape(batch, -1)
        draws.append(
            np.nanpercentile(low_sample, quantile, axis=1)
            - np.nanpercentile(none_sample, quantile, axis=1)
        )
    return {
        "low": low_point,
        "none": none_point,
        "low_minus_none": low_point - none_point,
        "low_minus_none_ci95_paired_cluster_bootstrap": percentile_ci(
            np.concatenate(draws)
        ),
    }


def turn_rows(conversations: list[Conversation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        runs = [run for run in conversations if run.entry.arm == arm]
        for turn in range(N_TURNS):
            ttfat = [
                run.ttfat_ms[turn]
                for run in runs
                if run.ttfat_ms[turn] is not None
            ]
            raw = [
                run.raw_ttfb_ms[turn]
                for run in runs
                if run.raw_ttfb_ms[turn] is not None
            ]
            reasoning = [
                run.reasoning_tokens[turn]
                for run in runs
                if run.reasoning_tokens[turn] is not None
            ]
            observed = sum(turn in run.observed_turns for run in runs)
            row: dict[str, Any] = {
                "arm": arm,
                "turn": turn,
                "assigned_conversations": len(runs),
                "observed_responses": observed,
                "missing_future_turns": len(runs) - observed,
                "strict_pass_count": sum(run.strict_pass[turn] for run in runs),
                "strict_error_count": sum(not run.strict_pass[turn] for run in runs),
            }
            for component in ALL_COMPONENTS:
                row[f"{component}_error_count"] = sum(
                    not run.scores[component][turn] for run in runs
                )
            row.update(
                {
                    "ttfat_observations": len(ttfat),
                    "ttfat_p50_ms": statistics.median(ttfat) if ttfat else None,
                    "ttfat_p95_ms": (
                        float(np.percentile(ttfat, 95)) if ttfat else None
                    ),
                    "raw_ttfb_observations": len(raw),
                    "raw_ttfb_p50_ms": statistics.median(raw) if raw else None,
                    "reasoning_token_observations": len(reasoning),
                    "reasoning_tokens_p50": (
                        statistics.median(reasoning) if reasoning else None
                    ),
                }
            )
            rows.append(row)
    return rows


def audit_row(run: Conversation) -> dict[str, Any]:
    return {
        "slot": run.entry.slot,
        "pair": run.entry.pair,
        "arm": run.entry.arm,
        "pair_order": run.entry.pair_order,
        "attempt": run.entry.attempt,
        "run_dir": relative_to_root(run.entry.run_dir),
        "campaign_classification": run.entry.classification,
        "observed_scheduled_turns": len(run.observed_turns),
        "fixed_turn_denominator": N_TURNS,
        "missing_future_turns": N_TURNS - len(run.observed_turns),
        "full_scheduled_coverage": run.full_scheduled_coverage,
        "strict_protocol_completion": run.strict_protocol_completion,
        "scheduled_end_session_turns": list(run.scheduled_end_session_turns),
        "recovery_end_session_turns": list(run.recovery_end_session_turns),
        "strict_pass_count": sum(run.strict_pass),
        "tool_error_count": N_TURNS - sum(run.scores["tool_use_correct"]),
        "instruction_error_count": (
            N_TURNS - sum(run.scores["instruction_following"])
        ),
        "kb_error_count": N_TURNS - sum(run.scores["kb_grounding"]),
        "turn_taking_error_count": N_TURNS - sum(run.scores["turn_taking"]),
        "ttfat_observations": sum(value is not None for value in run.ttfat_ms),
        "raw_ttfb_observations": sum(
            value is not None for value in run.raw_ttfb_ms
        ),
        "reasoning_token_observations": sum(
            value is not None for value in run.reasoning_tokens
        ),
        "reasoning_tokens_total": sum(
            value for value in run.reasoning_tokens if value is not None
        ),
        "judge_model": run.judge_model,
        "judge_version": run.judge_version,
        "run_log_sha256": run.run_log_sha256,
        "transcript_sha256": run.transcript_sha256,
        "judgment_sha256": run.judgment_sha256,
        "judge_summary_sha256": run.judge_summary_sha256,
    }


def fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1f}%"


def fmt_ci(values: list[float] | None, *, signed: bool = False) -> str:
    if values is None:
        return "NA"
    spec = "+.1f" if signed else ".1f"
    return f"{format(values[0], spec)} to {format(values[1], spec)}"


def fmt_num(value: float | None, suffix: str = "") -> str:
    return "NA" if value is None else f"{value:.0f}{suffix}"


def render_report(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    effects = payload["effects_low_minus_none"]
    lines = [
        "# Inkling Small on BaseTen: low versus none",
        "",
        (
            "Final fixed-denominator results for 30 `reasoning_effort=none` "
            "and 30 `reasoning_effort=low` conversations. The 60 requests "
            "were strictly sequential in 30 frozen temporal pairs. Each "
            "conversation contributes 30 scheduled turns; missing future "
            "turns after any short canonical run are failures, regardless "
            "of whether the immediate cause was model behavior or serving."
        ),
        "",
        "## Accuracy",
        "",
        (
            "| Arm | Strict pass | 95% conversation-bootstrap CI | Any error | "
            "Tool error | Instruction error | KB error |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = arms[arm]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    fmt_pct(value["strict_pass_rate_pct"]),
                    fmt_ci(value["strict_pass_rate_ci95_cluster_bootstrap"]),
                    fmt_pct(value["any_error_rate_pct"]),
                    fmt_pct(value["tool_use_correct_error_rate_pct"]),
                    fmt_pct(value["instruction_following_error_rate_pct"]),
                    fmt_pct(value["kb_grounding_error_rate_pct"]),
                ]
            )
            + " |"
        )
    pass_effect = effects["strict_pass_rate"]
    lines.extend(
        [
            "",
            (
                "Low minus none strict-pass effect: "
                f"**{pass_effect['low_minus_none_points']:+.1f} points** "
                "(paired whole-conversation bootstrap 95% CI "
                f"{fmt_ci(pass_effect['low_minus_none_ci95_paired_cluster_bootstrap'], signed=True)}"
                ")."
            ),
            "",
            "## Completion",
            "",
            (
                "| Arm | All scheduled turns | Strict terminal completion | "
                "Missing future turns | Recovery terminal calls |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        value = arms[arm]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    (
                        f"{value['full_scheduled_coverage_count']}/30 "
                        f"({fmt_pct(value['full_scheduled_coverage_pct'])})"
                    ),
                    (
                        f"{value['strict_protocol_completion_count']}/30 "
                        f"({fmt_pct(value['strict_protocol_completion_pct'])})"
                    ),
                    str(value["missing_future_turns_scored_as_failures"]),
                    str(value["recovery_end_session_count"]),
                ]
            )
            + " |"
        )
    completion_effect = effects["strict_protocol_completion"]
    lines.extend(
        [
            "",
            (
                "Low minus none strict-completion effect: "
                f"**{completion_effect['low_minus_none_points']:+.1f} points** "
                "(paired 95% CI "
                f"{fmt_ci(completion_effect['low_minus_none_ci95_paired_cluster_bootstrap'], signed=True)}"
                ")."
            ),
            "",
            (
                "Strict completion requires all scheduled turns 0–29 and "
                "`end_session` exactly on scheduled turn 29. A synthetic "
                "recovery terminal call does not count."
            ),
            "",
            "## Latency and reasoning",
            "",
            (
                "| Arm | TTFAT P50 | TTFAT P95 | TTFAT Max | Raw TTFB P50 | "
                "Reasoning tokens P50 | Reasoning observations |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        value = arms[arm]
        ttfat = value["ttfat_ms"]
        raw = value["raw_ttfb_ms"]
        reasoning = value["reasoning_tokens"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    fmt_num(ttfat["p50"], "ms"),
                    fmt_num(ttfat["p95"], "ms"),
                    fmt_num(ttfat["max"], "ms"),
                    fmt_num(raw["p50"], "ms"),
                    fmt_num(reasoning["p50"]),
                    str(reasoning["observations"]),
                ]
            )
            + " |"
        )
    ttfat_effect = effects["ttfat_p50_ms"]
    lines.extend(
        [
            "",
            (
                "Observed-response P50 TTFAT effect, low minus none: "
                f"**{fmt_num(ttfat_effect['low_minus_none'], 'ms')}** "
                "(paired conversation-bootstrap 95% CI "
                f"{fmt_ci(ttfat_effect['low_minus_none_ci95_paired_cluster_bootstrap'], signed=True)}"
                " ms)."
            ),
            "",
            (
                "TTFAT is content-aware: provider-separated reasoning-only "
                "chunks do not stop the clock. Raw TTFB measures the first "
                "streamed chunk. Timing and reasoning-token summaries are "
                "conditional on recorded observed values; missing turns are "
                "not assigned invented latency or token counts."
            ),
            "",
            "## Concentrated error turns",
            "",
        ]
    )
    for arm in ARMS:
        rows = sorted(
            [row for row in payload["turn_level"] if row["arm"] == arm],
            key=lambda row: (-row["strict_error_count"], row["turn"]),
        )[:8]
        rendered = ", ".join(
            f"{row['turn']} ({row['strict_error_count']}/30)" for row in rows
        )
        lines.append(f"- `{arm}`: turns {rendered}.")
    lines.extend(
        [
            "",
            "## Candidate README rows",
            "",
            (
                "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | "
                "KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in ARMS:
        value = arms[arm]
        timing = value["ttfat_ms"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"inkling-small ({arm})",
                    fmt_pct(value["strict_pass_rate_pct"]),
                    fmt_pct(value["any_error_rate_pct"]),
                    fmt_pct(value["tool_use_correct_error_rate_pct"]),
                    fmt_pct(value["instruction_following_error_rate_pct"]),
                    fmt_pct(value["kb_grounding_error_rate_pct"]),
                    fmt_num(timing["p50"], "ms"),
                    fmt_num(timing["p95"], "ms"),
                    fmt_num(timing["max"], "ms"),
                    "BaseTen",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Methods and audit trail",
            "",
            (
                "Strict pass is `tool_use_correct AND "
                "instruction_following AND kb_grounding`. Turn-taking is a "
                "supplementary dimension retained in the machine-readable "
                "artifacts."
            ),
            "",
            (
                f"Rate intervals use {BOOTSTRAPS:,} deterministic bootstrap "
                "draws over whole conversations. Low-minus-none intervals "
                "resample the 30 frozen temporal pairs. Quantile intervals "
                f"use {QUANTILE_BOOTSTRAPS:,} paired or clustered draws."
            ),
            "",
            (
                "`included-runs.tsv` fixes exact membership and artifact "
                "hashes. `turn-errors.tsv` contains all 30 turn-level counts, "
                "including missing future turns under the fixed denominator."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def progress_payload(entries: list[ManifestEntry]) -> dict[str, Any]:
    counts = Counter(entry.arm for entry in entries)
    judged = 0
    shorts = 0
    invalid_existing: dict[str, str] = {}
    for entry in entries:
        transcript_path = entry.run_dir / "transcript.jsonl"
        try:
            transcript_rows = read_jsonl(transcript_path)
            transcript = scheduled_map(transcript_rows, path=transcript_path)
            validate_runtime(entry, transcript)
            validate_campaign_counts(entry, transcript, transcript_rows)
            shorts += int(len(transcript) < N_TURNS)
        except Exception as exc:
            invalid_existing[entry.slot] = str(exc)
            continue
        artifacts = (
            entry.run_dir / "claude_judged.jsonl",
            entry.run_dir / "claude_summary.json",
            entry.run_dir / "claude_analysis.md",
        )
        judged += int(all(path.is_file() and path.stat().st_size for path in artifacts))
    return {
        "status": "ready_to_write" if len(entries) == 60 and judged == 60 else "in_progress",
        "canonical_conversations": len(entries),
        "canonical_by_arm": {arm: counts.get(arm, 0) for arm in ARMS},
        "remaining_slots": 60 - len(entries),
        "judged_artifact_sets_present": judged,
        "judgments_remaining": 60 - judged,
        "fixed_denominator_short_conversations": shorts,
        "invalid_existing_runs": invalid_existing,
        "judge_complete_marker_present": JUDGE_COMPLETE_PATH.is_file(),
    }


def build_payload() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    entries = load_manifest(require_complete=True)
    judge_complete = validate_judge_complete()
    validate_frozen_judge_inputs(entries)
    conversations = [load_conversation(entry) for entry in entries]
    judge_models = sorted({run.judge_model for run in conversations})
    judge_versions = sorted({run.judge_version for run in conversations})
    if len(judge_models) != 1 or len(judge_versions) != 1:
        fail(
            "mixed judge identity across canonical cohort: "
            f"models={judge_models}, versions={judge_versions}"
        )
    marker_model = judge_complete.get("judge_model")
    marker_version = judge_complete.get("judge_version")
    if marker_model not in (None, judge_models[0]):
        fail("judge model in COMPLETE.json differs from per-run summaries")
    if marker_version not in (None, judge_versions[0]):
        fail("judge version in COMPLETE.json differs from per-run summaries")

    runs_by_arm = {
        arm: sorted(
            [run for run in conversations if run.entry.arm == arm],
            key=lambda run: run.entry.pair,
        )
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for arm_index, arm in enumerate(ARMS):
        arms[arm], raw[arm], matrices[arm] = arm_summary(
            runs_by_arm[arm], arm_index=arm_index
        )
    pairs = paired_runs(conversations)
    effects = paired_rate_effects(pairs, raw, runs_by_arm)
    for field, label, seed_offset in (
        ("ttfat_ms", "ttfat", 20_000),
        ("raw_ttfb_ms", "raw_ttfb", 21_000),
        ("reasoning_tokens", "reasoning_tokens", 22_000),
    ):
        for quantile in (50, 95):
            effects[f"{label}_p{quantile}{'_ms' if field != 'reasoning_tokens' else ''}"] = (
                paired_quantile_effect(
                    pairs,
                    field=field,
                    quantile=quantile,
                    seed_offset=seed_offset + quantile,
                )
            )

    audit = [audit_row(run) for run in conversations]
    turns = turn_rows(conversations)
    input_hashes = {
        "configuration.json": sha256(CONFIG_PATH),
        "frozen-order.tsv": sha256(SCHEDULE_PATH),
        "canonical.tsv": sha256(CANONICAL_PATH),
        "source-sha256.txt": sha256(SOURCE_HASH_PATH),
        "judging/COMPLETE.json": sha256(JUDGE_COMPLETE_PATH),
        "judging/canonical-inputs.tsv": sha256(JUDGE_INPUTS_PATH),
        "analysis/analyze.py": sha256(Path(__file__).resolve()),
    }
    if JUDGE_SOURCE_PATH.is_file():
        input_hashes["judging/judge-source-sha256.txt"] = sha256(JUDGE_SOURCE_PATH)
    payload = {
        "schema_version": 1,
        "artifact_status": "FINAL",
        "protocol": {
            "campaign_id": validate_configuration()["campaign_id"],
            "benchmark": "aiwf_medium_context",
            "model": MODEL,
            "provider": PROVIDER,
            "endpoint": ENDPOINT,
            "arms": {
                "none": "reasoning_effort=none",
                "low": "reasoning_effort=low",
            },
            "conversations_per_arm": TARGET_PER_ARM,
            "scheduled_turns_per_conversation": N_TURNS,
            "fixed_turn_denominator_per_arm": DENOMINATOR_PER_ARM,
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
            "latency_policy": (
                "TTFAT and raw TTFB are conditional on observed recorded values; "
                "missing turns receive no invented latency"
            ),
            "reasoning_token_policy": (
                "conditional on observed turns whose provider usage payload "
                "contains thinking_tokens"
            ),
            "arm_ci_method": "whole-conversation nonparametric bootstrap",
            "effect_ci_method": (
                "paired whole-conversation bootstrap over the 30 frozen "
                "none/low temporal pairs"
            ),
            "bootstrap_samples": BOOTSTRAPS,
            "quantile_bootstrap_samples": QUANTILE_BOOTSTRAPS,
            "seed": SEED,
            "judge_models": judge_models,
            "judge_versions": judge_versions,
        },
        "input_hashes": input_hashes,
        "arms": arms,
        "effects_low_minus_none": effects,
        "turn_level": turns,
        "included_runs": audit,
    }
    return payload, audit, turns


def write_outputs(
    payload: dict[str, Any],
    audit: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> None:
    arm_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        value = payload["arms"][arm]
        arm_rows.append(
            {
                "arm": arm,
                "n_conversations": value["n_conversations"],
                "fixed_turn_denominator": value["fixed_turn_denominator"],
                "observed_scheduled_turns": value["observed_scheduled_turns"],
                "missing_future_turns": value[
                    "missing_future_turns_scored_as_failures"
                ],
                "strict_pass_rate_pct": value["strict_pass_rate_pct"],
                "strict_pass_ci95": value[
                    "strict_pass_rate_ci95_cluster_bootstrap"
                ],
                "any_error_rate_pct": value["any_error_rate_pct"],
                "tool_error_rate_pct": value[
                    "tool_use_correct_error_rate_pct"
                ],
                "instruction_error_rate_pct": value[
                    "instruction_following_error_rate_pct"
                ],
                "kb_error_rate_pct": value["kb_grounding_error_rate_pct"],
                "turn_taking_error_rate_pct": value[
                    "turn_taking_error_rate_pct"
                ],
                "full_scheduled_coverage_pct": value[
                    "full_scheduled_coverage_pct"
                ],
                "strict_protocol_completion_pct": value[
                    "strict_protocol_completion_pct"
                ],
                "ttfat_observations": value["ttfat_ms"]["observations"],
                "ttfat_p50_ms": value["ttfat_ms"]["p50"],
                "ttfat_p95_ms": value["ttfat_ms"]["p95"],
                "ttfat_max_ms": value["ttfat_ms"]["max"],
                "raw_ttfb_observations": value["raw_ttfb_ms"]["observations"],
                "raw_ttfb_p50_ms": value["raw_ttfb_ms"]["p50"],
                "reasoning_token_observations": value["reasoning_tokens"][
                    "observations"
                ],
                "reasoning_tokens_p50": value["reasoning_tokens"]["p50"],
                "reasoning_tokens_p95": value["reasoning_tokens"]["p95"],
            }
        )

    effect_rows: list[dict[str, Any]] = []
    effects = payload["effects_low_minus_none"]
    for metric, value in effects.items():
        if metric in {"direction", "bootstrap_unit"}:
            continue
        if "low_pct" in value:
            effect_rows.append(
                {
                    "metric": metric,
                    "low": value["low_pct"],
                    "none": value["none_pct"],
                    "low_minus_none": value["low_minus_none_points"],
                    "ci95": value[
                        "low_minus_none_ci95_paired_cluster_bootstrap"
                    ],
                    "unit": "percentage points",
                }
            )
        else:
            effect_rows.append(
                {
                    "metric": metric,
                    "low": value["low"],
                    "none": value["none"],
                    "low_minus_none": value["low_minus_none"],
                    "ci95": value[
                        "low_minus_none_ci95_paired_cluster_bootstrap"
                    ],
                    "unit": (
                        "tokens" if metric.startswith("reasoning_tokens") else "ms"
                    ),
                }
            )

    atomic_text(HERE / "aggregates.json", json.dumps(payload, indent=2) + "\n")
    atomic_tsv(HERE / "aggregates.tsv", arm_rows)
    atomic_tsv(HERE / "effects.tsv", effect_rows)
    atomic_tsv(HERE / "included-runs.tsv", audit)
    atomic_tsv(HERE / "turn-errors.tsv", turns)
    atomic_text(HERE / "REPORT.md", render_report(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "require the complete 60-run judged cohort and write final "
            "analysis artifacts; default is read-only progress audit"
        ),
    )
    args = parser.parse_args()
    if not args.write:
        entries = load_manifest(require_complete=False)
        print(json.dumps(progress_payload(entries), indent=2))
        return

    payload, audit, turns = build_payload()
    write_outputs(payload, audit, turns)
    print(
        json.dumps(
            {
                "status": "final_outputs_written",
                "canonical_conversations": len(audit),
                "outputs": [
                    "REPORT.md",
                    "aggregates.json",
                    "aggregates.tsv",
                    "effects.tsv",
                    "included-runs.tsv",
                    "turn-errors.tsv",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
