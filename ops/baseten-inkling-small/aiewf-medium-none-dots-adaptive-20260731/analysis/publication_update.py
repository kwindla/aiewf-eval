#!/usr/bin/env python3
"""Prepare or apply the final Inkling Small README/report publication update.

The default is a no-write dry run. It requires the final none/low aggregate and
at least one fully analyzed dots stage, then prints the proposed README,
generator, and normalized-publication-input diffs. ``--apply`` is intentionally
separate and is not valid until those same artifacts pass every provenance and
schema check. This script performs local file operations only.
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DOTS_CAMPAIGN = HERE.parent
ROOT = DOTS_CAMPAIGN.parents[2]
PRIMARY_ANALYSIS = (
    ROOT
    / "ops/baseten-inkling-small/"
    "aiewf-medium-none-low-n30-20260731/analysis"
)
README_PATH = ROOT / "README.md"
GENERATOR_PATH = ROOT / "scripts/build_filler_report.py"
README_VERIFIER_PATH = (
    ROOT
    / "docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/"
    "update_readme.py"
)
HTML_PATH = ROOT / "docs/filler-token-latent-scratchpad-study.html"
MARKDOWN_PATH = ROOT / "docs/filler-token-latent-scratchpad-study.md"
NORMALIZED_PATH = PRIMARY_ANALYSIS / "publication-input.json"
FAILURE_ANALYSIS_PATH = PRIMARY_ANALYSIS / "FAILURE-ANALYSIS.json"
JUDGE_AUDIT_PATH = PRIMARY_ANALYSIS / "JUDGE-AUDIT.json"
LOCK_PATH = HERE / ".publication.lock"

MODEL = "thinkingmachines/inkling-small"
CAMPAIGN_ID = "aiewf-medium-inkling-small-baseten-none-low-n30-20260731"
DOTS_CAMPAIGN_ID = "aiewf-medium-inkling-small-baseten-none-dots-adaptive-20260731"
PROVIDER_SOURCE = "BaseTen Model API"
PROVIDER_DISPLAY = "BaseTen"
REPORT_NAME = "inkling-small"
N_TURNS = 30
PRIMARY_N = 30
DOT_STAGES = (30, 10, 6)

GENERATOR_DATA_START = "# INKLING_SMALL_PUBLICATION_DATA_START"
GENERATOR_DATA_END = "# INKLING_SMALL_PUBLICATION_DATA_END"
GENERATOR_DETAIL_START = "# INKLING_SMALL_PUBLICATION_DETAIL_START"
GENERATOR_DETAIL_END = "# INKLING_SMALL_PUBLICATION_DETAIL_END"
README_PROSE_START = "<!-- INKLING_SMALL_README_PROSE_START -->"
README_PROSE_END = "<!-- INKLING_SMALL_README_PROSE_END -->"

README_HEADER = (
    "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | "
    "KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"
)
README_SEPARATOR = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
LEGACY_INKLING_PROSE = (
    "`inkling` is Thinking Machines' 975B-parameter (41B active) open-weights "
    "model, run on BaseTen's serverless Model API; `(none)` sets "
    "`reasoning_effort: none`. Unlike GPT-5.6, Inkling's accuracy peaks at "
    "`low` in the earlier effort sweep and does not improve with more "
    "reasoning — higher effort only adds latency (median TTFAT climbs to "
    "~2.0s at `medium` and ~2.5s at `max`, with the P95 tail reaching ~6s). "
    "See `docs/inkling-notes.md` and `docs/inkling-baseten-integration.md`."
)


@dataclass(frozen=True)
class PublicationData:
    primary_path: Path
    dots_path: Path
    failure_analysis_path: Path
    judge_audit_path: Path
    dots_stage: int
    primary: dict[str, Any]
    dots: dict[str, Any]
    failure_analysis: dict[str, Any]
    judge_audit: dict[str, Any]
    normalized: dict[str, Any]


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        fail(f"{label} must be a finite number; found {value!r}")
    return float(value)


def integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        fail(f"{label} must be an integer; found {value!r}")
    return value


def close(left: float, right: float, *, tolerance: float = 1e-8) -> bool:
    return abs(left - right) <= tolerance


def verify_primary_input_hashes(primary_path: Path, payload: dict[str, Any]) -> None:
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, dict) or not hashes:
        fail("primary final aggregate lacks input_hashes")
    campaign = primary_path.parent.parent
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            fail("primary input_hashes must map string paths to SHA-256 values")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail(f"invalid primary SHA-256 for {relative}")
        path = campaign / relative
        if not path.is_file():
            fail(f"primary hashed input is missing: {path}")
        if sha256(path) != expected:
            fail(f"primary hashed input changed: {path}")


def validate_primary(primary_path: Path) -> dict[str, Any]:
    payload = read_json(primary_path)
    protocol = payload.get("protocol")
    arms = payload.get("arms")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_status") != "FINAL"
        or not isinstance(protocol, dict)
        or not isinstance(arms, dict)
    ):
        fail("primary aggregate is not a schema-1 FINAL artifact")
    expected_protocol = {
        "campaign_id": CAMPAIGN_ID,
        "benchmark": "aiwf_medium_context",
        "model": MODEL,
        "provider": PROVIDER_SOURCE,
        "endpoint": "https://inference.baseten.co/v1",
        "conversations_per_arm": PRIMARY_N,
        "scheduled_turns_per_conversation": N_TURNS,
        "fixed_turn_denominator_per_arm": PRIMARY_N * N_TURNS,
        "strict_pass_definition": (
            "tool_use_correct AND instruction_following AND kb_grounding"
        ),
        "arm_ci_method": "whole-conversation nonparametric bootstrap",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            fail(
                f"primary protocol mismatch for {key}: "
                f"expected {expected!r}, found {protocol.get(key)!r}"
            )
    if set(arms) != {"none", "low"}:
        fail(f"primary arms must be none/low; found {sorted(arms)}")
    required_arm_numbers = (
        "strict_pass_rate_pct",
        "any_error_rate_pct",
        "tool_use_correct_error_rate_pct",
        "instruction_following_error_rate_pct",
        "kb_grounding_error_rate_pct",
        "strict_protocol_completion_pct",
    )
    for arm_name in ("none", "low"):
        arm = arms[arm_name]
        if not isinstance(arm, dict):
            fail(f"primary arm {arm_name} is not an object")
        if integer(arm.get("n_conversations"), f"primary {arm_name} n") != PRIMARY_N:
            fail(f"primary {arm_name} does not contain 30 conversations")
        if integer(
            arm.get("fixed_turn_denominator"), f"primary {arm_name} denominator"
        ) != PRIMARY_N * N_TURNS:
            fail(f"primary {arm_name} denominator is not 900")
        for field in required_arm_numbers:
            value = number(arm.get(field), f"primary {arm_name}.{field}")
            if not 0 <= value <= 100:
                fail(f"primary {arm_name}.{field} is outside 0..100")
        timing = arm.get("ttfat_ms")
        if not isinstance(timing, dict):
            fail(f"primary {arm_name} lacks ttfat_ms")
        for field in ("p50", "p95", "max"):
            if number(timing.get(field), f"primary {arm_name} TTFAT {field}") < 0:
                fail(f"primary {arm_name} TTFAT {field} is negative")
        strict = number(arm["strict_pass_rate_pct"], "strict pass")
        any_error = number(arm["any_error_rate_pct"], "any error")
        if not close(strict + any_error, 100.0, tolerance=1e-6):
            fail(f"primary {arm_name} pass/error rates are not complements")
    verify_primary_input_hashes(primary_path, payload)
    return payload


def resolve_repo_artifact(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} path is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        fail(f"{label} path escapes repository root: {value}")
    if not path.is_file():
        fail(f"{label} artifact is missing: {path}")
    return path


def validate_hash_record(
    root: Path,
    record: Any,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        fail(f"{label} hash record is not an object")
    path = resolve_repo_artifact(root, record.get("path"), label)
    expected = record.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        fail(f"{label} has an invalid SHA-256")
    if sha256(path) != expected:
        fail(f"{label} input changed: {path}")
    return path


def validate_failure_analysis(root: Path, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    method = payload.get("method")
    arms = payload.get("arms")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_status") != "RAW_CAUSE_ATTRIBUTION"
        or payload.get("campaign") != CAMPAIGN_ID
        or payload.get("model") != MODEL
        or not isinstance(method, dict)
        or not isinstance(arms, dict)
    ):
        fail("failure analysis schema or identity mismatch")
    if (
        method.get("judge_dependency") is not False
        or method.get("scheduled_turns_per_conversation") != N_TURNS
        or method.get("recovery_turns_are_not_scheduled") is not True
        or "HTTP 429" not in str(method.get("baseten_429_idle_definition", ""))
    ):
        fail("failure analysis method mismatch")
    if set(arms) != {"none", "low"}:
        fail(f"failure analysis arms must be none/low; found {sorted(arms)}")

    expected_short = {"none": 12, "low": 10}
    for arm_name in ("none", "low"):
        arm = arms[arm_name]
        if not isinstance(arm, dict):
            fail(f"failure analysis arm {arm_name} is not an object")
        if integer(
            arm.get("conversations"), f"failure analysis {arm_name} conversations"
        ) != PRIMARY_N:
            fail(f"failure analysis {arm_name} does not contain 30 conversations")
        if integer(
            arm.get("fixed_turn_denominator"),
            f"failure analysis {arm_name} denominator",
        ) != PRIMARY_N * N_TURNS:
            fail(f"failure analysis {arm_name} denominator is not 900")
        causes = arm.get("conversation_causes")
        if not isinstance(causes, dict):
            fail(f"failure analysis {arm_name} lacks conversation causes")
        rate_limited = causes.get("baseten_429_idle")
        if not isinstance(rate_limited, dict):
            fail(f"failure analysis {arm_name} lacks BaseTen 429 attribution")
        if integer(
            rate_limited.get("count"),
            f"failure analysis {arm_name} BaseTen 429 count",
        ) != expected_short[arm_name]:
            fail(
                f"failure analysis {arm_name} BaseTen 429 count changed from "
                f"the frozen publication value {expected_short[arm_name]}"
            )
        unattributed = causes.get("unattributed_short")
        if not isinstance(unattributed, dict) or integer(
            unattributed.get("count"),
            f"failure analysis {arm_name} unattributed count",
        ) != 0:
            fail(f"failure analysis {arm_name} contains unattributed short runs")

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        fail("failure analysis lacks its hashed input manifest")
    canonical = validate_hash_record(root, inputs.get("canonical"), "failure canonical")
    expected_canonical = root / (
        "ops/baseten-inkling-small/"
        "aiewf-medium-none-low-n30-20260731/canonical.tsv"
    )
    if canonical != expected_canonical.resolve():
        fail("failure analysis canonical path mismatch")
    validate_hash_record(root, inputs.get("analyzer"), "failure analyzer")
    runs = inputs.get("runs")
    if not isinstance(runs, dict) or len(runs) != PRIMARY_N * 2:
        fail("failure analysis must hash all 60 canonical runs")
    for slot, record in runs.items():
        if not isinstance(record, dict):
            fail(f"failure analysis run record {slot} is not an object")
        transcript = {
            "path": record.get("transcript"),
            "sha256": record.get("transcript_sha256"),
        }
        run_log = {
            "path": record.get("run_log"),
            "sha256": record.get("run_log_sha256"),
        }
        validate_hash_record(root, transcript, f"failure {slot} transcript")
        validate_hash_record(root, run_log, f"failure {slot} run log")
    return payload


def validate_judge_audit(
    root: Path,
    path: Path,
    primary_path: Path,
) -> dict[str, Any]:
    payload = read_json(path)
    policy = payload.get("policy")
    input_hashes = payload.get("input_hashes")
    arms = payload.get("arms")
    changes = payload.get("label_changes")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_status") != "POST_HOC_SENSITIVITY_AUDIT"
        or payload.get("campaign") != CAMPAIGN_ID
        or payload.get("model") != MODEL
        or payload.get("official_artifacts_unchanged") is not True
        or not isinstance(policy, dict)
        or not isinstance(input_hashes, dict)
        or not isinstance(arms, dict)
        or not isinstance(changes, list)
    ):
        fail("judge audit schema or identity mismatch")
    counterfactual = policy.get("counterfactual")
    if (
        policy.get("official_judge_model") != "claude-opus-4-5"
        or policy.get("official_judge_version")
        != "claude-agent-sdk-v4-turn-taking"
        or not isinstance(counterfactual, dict)
        or "not an official relabeling" not in str(counterfactual.get("status", ""))
    ):
        fail("judge audit policy mismatch")
    if len(changes) != 4:
        fail(f"judge audit must contain four label changes; found {len(changes)}")
    changed_sites: set[tuple[Any, Any]] = set()
    for change in changes:
        if not isinstance(change, dict):
            fail("judge audit label change is not an object")
        site = (change.get("slot"), change.get("turn"))
        if site in changed_sites:
            fail(f"judge audit repeats changed label site {site}")
        changed_sites.add(site)
        official = change.get("official_scores")
        alternate = change.get("counterfactual_scores")
        if not isinstance(official, dict) or not isinstance(alternate, dict):
            fail(f"judge audit scores are missing at {site}")
        if (
            official.get("tool_use_correct") is not True
            or alternate.get("tool_use_correct") is not False
            or any(
                official.get(label) is not alternate.get(label)
                for label in ("instruction_following", "kb_grounding")
            )
        ):
            fail(f"judge audit changes more than tool_use_correct at {site}")

    expected_deltas = {
        "none": {"strict_pass": -3, "any_error": 3, "tool_error": 4},
        "low": {"strict_pass": 0, "any_error": 0, "tool_error": 0},
    }
    if set(arms) != set(expected_deltas):
        fail(f"judge audit arms must be none/low; found {sorted(arms)}")
    max_abs_shift = 0.0
    for arm_name, metric_deltas in expected_deltas.items():
        arm = arms[arm_name]
        if not isinstance(arm, dict) or integer(
            arm.get("fixed_turn_denominator"),
            f"judge audit {arm_name} denominator",
        ) != PRIMARY_N * N_TURNS:
            fail(f"judge audit {arm_name} denominator is not 900")
        metrics = arm.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(metric_deltas):
            fail(f"judge audit {arm_name} metric set mismatch")
        for metric_name, expected_delta in metric_deltas.items():
            metric = metrics[metric_name]
            if not isinstance(metric, dict):
                fail(f"judge audit {arm_name}.{metric_name} is not an object")
            official_count = integer(
                metric.get("official_count"),
                f"judge audit {arm_name}.{metric_name} official count",
            )
            alternate_count = integer(
                metric.get("counterfactual_count"),
                f"judge audit {arm_name}.{metric_name} alternate count",
            )
            delta_count = integer(
                metric.get("delta_count"),
                f"judge audit {arm_name}.{metric_name} delta count",
            )
            if delta_count != expected_delta or alternate_count - official_count != delta_count:
                fail(f"judge audit {arm_name}.{metric_name} count delta mismatch")
            denominator = PRIMARY_N * N_TURNS
            expected_values = {
                "official_rate_pct": 100 * official_count / denominator,
                "counterfactual_rate_pct": 100 * alternate_count / denominator,
                "delta_percentage_points": 100 * delta_count / denominator,
            }
            for field, expected in expected_values.items():
                if not close(
                    number(
                        metric.get(field),
                        f"judge audit {arm_name}.{metric_name}.{field}",
                    ),
                    expected,
                    tolerance=1e-10,
                ):
                    fail(f"judge audit {arm_name}.{metric_name}.{field} mismatch")
            max_abs_shift = max(
                max_abs_shift,
                abs(number(metric["delta_percentage_points"], "judge audit shift")),
            )
    if max_abs_shift > 0.5 + 1e-10:
        fail(f"judge audit arm-level sensitivity exceeds 0.5 points: {max_abs_shift}")

    campaign = primary_path.parent.parent
    required_hashes = {
        "analysis/aggregates.json",
        "judging/COMPLETE.json",
        "canonical.tsv",
        "judging/canonical-inputs.tsv",
        "judging/judge-source-sha256.txt",
        "analysis/analyze.py",
        "analysis/judge_audit.py",
    }
    if not required_hashes.issubset(input_hashes):
        fail("judge audit does not hash every required input")
    for relative, expected in input_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            fail("judge audit input_hashes must map strings to strings")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail(f"judge audit has invalid SHA-256 for {relative}")
        artifact = campaign / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            fail(f"judge audit input changed: {artifact}")
    if input_hashes["analysis/aggregates.json"] != sha256(primary_path):
        fail("judge audit does not anchor the final primary aggregate")
    return payload


def validate_dots_inputs(root: Path, payload: dict[str, Any]) -> None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        fail("dots stage lacks its frozen input manifest hashes")
    pairs = (
        ("control_inputs", "control_inputs_sha256"),
        ("dots_judge_inputs", "dots_judge_inputs_sha256"),
        ("dots_judge_complete", "dots_judge_complete_sha256"),
    )
    for path_key, hash_key in pairs:
        path = resolve_repo_artifact(root, inputs.get(path_key), path_key)
        expected = inputs.get(hash_key)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail(f"dots stage has invalid {hash_key}")
        if sha256(path) != expected:
            fail(f"dots stage frozen input changed: {path}")


def validate_dots(
    root: Path,
    dots_path: Path,
    expected_stage: int,
) -> dict[str, Any]:
    payload = read_json(dots_path)
    configuration = payload.get("configuration")
    method = payload.get("method")
    arms = payload.get("arms")
    effect = payload.get("effects", {}).get("strict_pass")
    decision = payload.get("adaptive_decision")
    if (
        payload.get("schema_version") != 1
        or payload.get("campaign_id") != DOTS_CAMPAIGN_ID
        or payload.get("stage") != expected_stage
        or payload.get("model") != MODEL
        or payload.get("provider") != PROVIDER_SOURCE
        or not all(
            isinstance(value, dict)
            for value in (configuration, method, arms, effect, decision)
        )
    ):
        fail(f"dots stage-{expected_stage} schema or identity mismatch")
    if configuration != {
        "control": "frozen primary none arm",
        "treatment": "+96 suffix dots",
        "reasoning_effort": "none",
        "fixed_turns_per_conversation": N_TURNS,
    }:
        fail("dots stage request configuration mismatch")
    if (
        method.get("fixed_denominator") is not True
        or method.get("missing_future_turns_fail_all_displayed_criteria") is not True
        or method.get("bootstrap_unit") != "whole conversation"
        or method.get("bootstrap_iterations") != 100_000
    ):
        fail("dots stage analysis method mismatch")
    if set(arms) != {"control_none", "dots96"}:
        fail("dots stage must contain control_none and dots96")
    expected_counts = {"control_none": PRIMARY_N, "dots96": expected_stage}
    for arm_name, n in expected_counts.items():
        arm = arms[arm_name]
        if integer(arm.get("conversations"), f"dots {arm_name} n") != n:
            fail(f"dots {arm_name} conversation count mismatch")
        if integer(
            arm.get("fixed_turn_denominator"), f"dots {arm_name} denominator"
        ) != n * N_TURNS:
            fail(f"dots {arm_name} fixed denominator mismatch")
        rates = arm.get("rates_percent")
        timing = arm.get("ttfat_ms_observed_responses_only")
        completion = arm.get("strict_completion")
        if not all(isinstance(value, dict) for value in (rates, timing, completion)):
            fail(f"dots {arm_name} summary structure mismatch")
        for field in (
            "strict_pass",
            "any_error",
            "tool_error",
            "instruction_error",
            "kb_error",
        ):
            value = number(rates.get(field), f"dots {arm_name}.{field}")
            if not 0 <= value <= 100:
                fail(f"dots {arm_name}.{field} is outside 0..100")
        number(timing.get("p50"), f"dots {arm_name} TTFAT p50")
        if integer(completion.get("total"), f"dots {arm_name} completion total") != n:
            fail(f"dots {arm_name} completion denominator mismatch")
    delta = number(effect.get("dots_minus_control_points"), "dots strict delta")
    low = number(effect.get("ci95_low"), "dots strict CI low")
    high = number(effect.get("ci95_high"), "dots strict CI high")
    if low > high:
        fail("dots stage strict-pass interval is reversed")
    observed_delta = (
        number(arms["dots96"]["rates_percent"]["strict_pass"], "dots pass")
        - number(
            arms["control_none"]["rates_percent"]["strict_pass"],
            "control pass",
        )
    )
    if not close(delta, observed_delta):
        fail("dots stage effect does not equal dots minus control")
    if decision.get("evaluated_stage") != expected_stage:
        fail("dots adaptive decision stage mismatch")
    allowed = {
        6: {"extend_to_10", "stop_at_6"},
        10: {"extend_to_30", "stop_at_10"},
        30: {"terminal_at_30"},
    }
    if decision.get("recommendation") not in allowed[expected_stage]:
        fail("dots adaptive recommendation is inconsistent with the reached stage")
    if decision.get("gate_executed") is not False:
        fail("dots analysis artifact must not claim to have executed a gate")
    validate_dots_inputs(root, payload)
    return payload


def select_dots_stage(dots_analysis: Path) -> tuple[int, Path]:
    existing = [
        (stage, dots_analysis / f"stage-{stage}.json")
        for stage in DOT_STAGES
        if (dots_analysis / f"stage-{stage}.json").is_file()
    ]
    if not existing:
        fail(
            "publication requires a reached dots-stage artifact: expected one "
            "of analysis/stage-6.json, stage-10.json, or stage-30.json"
        )
    return existing[0]


def validate_cross_artifact_alignment(
    primary: dict[str, Any], dots: dict[str, Any]
) -> None:
    source = primary["arms"]["none"]
    frozen = dots["arms"]["control_none"]
    mappings = (
        ("strict_pass_rate_pct", "strict_pass"),
        ("any_error_rate_pct", "any_error"),
        ("tool_use_correct_error_rate_pct", "tool_error"),
        ("instruction_following_error_rate_pct", "instruction_error"),
        ("kb_grounding_error_rate_pct", "kb_error"),
    )
    for primary_key, dots_key in mappings:
        left = number(source[primary_key], f"primary none {primary_key}")
        right = number(
            frozen["rates_percent"][dots_key], f"dots control {dots_key}"
        )
        if not close(left, right, tolerance=1e-6):
            fail(
                f"dots control no longer matches primary none: "
                f"{primary_key}={left}, {dots_key}={right}"
            )
    left_ttfat = number(source["ttfat_ms"]["p50"], "primary none TTFAT")
    right_ttfat = number(
        frozen["ttfat_ms_observed_responses_only"]["p50"],
        "dots control TTFAT",
    )
    if not close(left_ttfat, right_ttfat, tolerance=1e-6):
        fail("dots frozen control TTFAT no longer matches primary none")


def verdict(delta: float, low: float, high: float) -> tuple[str, str]:
    if low > 0:
        return "pos", "increase"
    if high < 0:
        return "neg", "decrease"
    if abs(delta) >= 2:
        return "sugg", "suggestive"
    return "null", "no detectable effect"


def normalize(
    root: Path,
    primary_path: Path,
    dots_path: Path,
    failure_analysis_path: Path,
    judge_audit_path: Path,
    stage: int,
    primary: dict[str, Any],
    dots: dict[str, Any],
    failure_analysis: dict[str, Any],
    judge_audit: dict[str, Any],
) -> dict[str, Any]:
    none = primary["arms"]["none"]
    low_arm = primary["arms"]["low"]
    dots_arm = dots["arms"]["dots96"]
    effect = dots["effects"]["strict_pass"]
    delta = number(effect["dots_minus_control_points"], "dots delta")
    ci_low = number(effect["ci95_low"], "dots CI low")
    ci_high = number(effect["ci95_high"], "dots CI high")
    key, interpretation = verdict(delta, ci_low, ci_high)
    failure_arms = failure_analysis["arms"]
    short_none = integer(
        failure_arms["none"]["conversation_causes"]["baseten_429_idle"]["count"],
        "none BaseTen 429 short runs",
    )
    short_low = integer(
        failure_arms["low"]["conversation_causes"]["baseten_429_idle"]["count"],
        "low BaseTen 429 short runs",
    )
    audit_shifts = [
        abs(number(metric["delta_percentage_points"], "judge audit delta"))
        for arm in judge_audit["arms"].values()
        for metric in arm["metrics"].values()
    ]
    return {
        "schema_version": 1,
        "artifact_status": "FINAL_PUBLICATION_INPUT",
        "model": MODEL,
        "report_name": REPORT_NAME,
        "provider": PROVIDER_DISPLAY,
        "source_artifacts": {
            str(primary_path.relative_to(root)): sha256(primary_path),
            str(dots_path.relative_to(root)): sha256(dots_path),
            str(failure_analysis_path.relative_to(root)): sha256(
                failure_analysis_path
            ),
            str(judge_audit_path.relative_to(root)): sha256(judge_audit_path),
        },
        "readme_arms": {
            arm_name: {
                "pass_rate_pct": arm["strict_pass_rate_pct"],
                "any_error_rate_pct": arm["any_error_rate_pct"],
                "tool_error_rate_pct": arm["tool_use_correct_error_rate_pct"],
                "instruction_error_rate_pct": arm[
                    "instruction_following_error_rate_pct"
                ],
                "kb_error_rate_pct": arm["kb_grounding_error_rate_pct"],
                "strict_completion_pct": arm["strict_protocol_completion_pct"],
                "ttfat_p50_ms": arm["ttfat_ms"]["p50"],
                "ttfat_p95_ms": arm["ttfat_ms"]["p95"],
                "ttfat_max_ms": arm["ttfat_ms"]["max"],
            }
            for arm_name, arm in (("none", none), ("low", low_arm))
        },
        "screen_row": {
            "name": REPORT_NAME,
            "provider": PROVIDER_DISPLAY,
            "no_filler_pass_rate_pct": none["strict_pass_rate_pct"],
            "dots_pass_rate_pct": dots_arm["rates_percent"]["strict_pass"],
            "dots_minus_control_points": delta,
            "ci95": [ci_low, ci_high],
            "included_runs": [PRIMARY_N, stage],
            "key": key,
            "interpretation": interpretation,
            "none_ttfat_p50_ms": none["ttfat_ms"]["p50"],
            "strict_completion_pct": [
                none["strict_protocol_completion_pct"],
                dots_arm["strict_completion"]["percent"],
            ],
            "focused": stage == PRIMARY_N,
        },
        "dots_stage": stage,
        "dots_recommendation": dots["adaptive_decision"]["recommendation"],
        "robustness": {
            "primary_effort_campaign": {
                "retained_attempts": PRIMARY_N * 2,
                "baseten_429_idle_short_runs": {
                    "none": short_none,
                    "low": short_low,
                    "total": short_none + short_low,
                },
                "fixed_denominator_missing_future_turns_fail": True,
                "serving_failures_not_generated_terminal_calls": True,
            },
            "judge_sensitivity": {
                "changed_tool_use_correct_labels": len(
                    judge_audit["label_changes"]
                ),
                "max_abs_arm_rate_change_percentage_points": max(audit_shifts),
                "disclosure_bound_percentage_points": 0.5,
                "official_artifacts_unchanged": judge_audit[
                    "official_artifacts_unchanged"
                ],
            },
        },
    }


def load_publication_data(
    *,
    root: Path = ROOT,
    primary_path: Path | None = None,
    dots_analysis: Path | None = None,
    failure_analysis_path: Path | None = None,
    judge_audit_path: Path | None = None,
) -> PublicationData:
    primary_path = primary_path or (
        root
        / "ops/baseten-inkling-small/"
        "aiewf-medium-none-low-n30-20260731/analysis/aggregates.json"
    )
    dots_analysis = dots_analysis or (
        root
        / "ops/baseten-inkling-small/"
        "aiewf-medium-none-dots-adaptive-20260731/analysis"
    )
    failure_analysis_path = failure_analysis_path or (
        primary_path.parent / "FAILURE-ANALYSIS.json"
    )
    judge_audit_path = judge_audit_path or (
        primary_path.parent / "JUDGE-AUDIT.json"
    )
    if not primary_path.is_file():
        fail(f"final primary Inkling Small aggregate is absent: {primary_path}")
    if not failure_analysis_path.is_file():
        fail(f"final Inkling Small failure analysis is absent: {failure_analysis_path}")
    if not judge_audit_path.is_file():
        fail(f"final Inkling Small judge audit is absent: {judge_audit_path}")
    stage, dots_path = select_dots_stage(dots_analysis)
    primary = validate_primary(primary_path)
    dots = validate_dots(root, dots_path, stage)
    failure_analysis = validate_failure_analysis(root, failure_analysis_path)
    judge_audit = validate_judge_audit(root, judge_audit_path, primary_path)
    validate_cross_artifact_alignment(primary, dots)

    normalized = normalize(
        root,
        primary_path,
        dots_path,
        failure_analysis_path,
        judge_audit_path,
        stage,
        primary,
        dots,
        failure_analysis,
        judge_audit,
    )
    return PublicationData(
        primary_path=primary_path,
        dots_path=dots_path,
        failure_analysis_path=failure_analysis_path,
        judge_audit_path=judge_audit_path,
        dots_stage=stage,
        primary=primary,
        dots=dots,
        failure_analysis=failure_analysis,
        judge_audit=judge_audit,
        normalized=normalized,
    )


def markdown_cells(line: str) -> list[str]:
    if not line.startswith("|") or not line.endswith("|"):
        fail(f"malformed Markdown table row: {line}")
    return [cell.strip() for cell in line[1:-1].split("|")]


def render_readme_row(model: str, arm: dict[str, Any]) -> str:
    cells = [
        model,
        f"{number(arm['pass_rate_pct'], 'pass'):.1f}%",
        f"{number(arm['any_error_rate_pct'], 'any error'):.1f}%",
        f"{number(arm['tool_error_rate_pct'], 'tool error'):.1f}%",
        f"{number(arm['instruction_error_rate_pct'], 'instruction error'):.1f}%",
        f"{number(arm['kb_error_rate_pct'], 'KB error'):.1f}%",
        f"{number(arm['ttfat_p50_ms'], 'TTFAT P50'):.0f}ms",
        f"{number(arm['ttfat_p95_ms'], 'TTFAT P95'):.0f}ms",
        f"{number(arm['ttfat_max_ms'], 'TTFAT max'):.0f}ms",
        PROVIDER_DISPLAY,
    ]
    return "| " + " | ".join(cells) + " |"


def prose_block(data: PublicationData) -> str:
    arms = data.normalized["readme_arms"]
    none = arms["none"]
    low_arm = arms["low"]
    screen = data.normalized["screen_row"]
    return "\n\n".join(
        (
            (
                "`inkling` is Thinking Machines' earlier 975B-parameter "
                "(41B active) open-weights model. Its historical `(none)` "
                "row uses BaseTen's serverless Model API and should not be "
                "confused with the newer Inkling Small results below. See "
                "`docs/inkling-notes.md` and "
                "`docs/inkling-baseten-integration.md`."
            ),
            (
                f"`inkling-small` is Thinking Machines' newer smaller model, "
                f"tested through the BaseTen Model API in a frozen paired "
                f"effort campaign. With `reasoning_effort=none` it scored "
                f"{none['pass_rate_pct']:.1f}% at {none['ttfat_p50_ms']:.0f}ms "
                f"P50 TTFAT; `low` scored {low_arm['pass_rate_pct']:.1f}% at "
                f"{low_arm['ttfat_p50_ms']:.0f}ms. Both README rows use fixed "
                f"turn denominators, with missing future turns counted as "
                f"errors. Its separate exploratory +96-dot arm was compared "
                f"with the frozen `none` control "
                f"({screen['dots_minus_control_points']:+.1f} points); that "
                f"later, non-interleaved comparison appears in Section 3 of "
                f"the filler report."
            ),
        )
    )


def update_readme_text(text: str, data: PublicationData) -> str:
    if text.count(README_HEADER) != 1:
        fail("README text-results header must appear exactly once")
    before, remainder = text.split(README_HEADER, 1)
    if not remainder.startswith("\n" + README_SEPARATOR + "\n"):
        fail("README text-results table separator changed")
    table_body, separator, after = remainder[
        len("\n" + README_SEPARATOR + "\n") :
    ].partition("\n\n")
    if not separator:
        fail("README text-results table has no terminating blank line")
    rows = table_body.splitlines()
    parsed: list[tuple[int, float, str]] = []
    seen_new = Counter()
    for index, row in enumerate(rows):
        cells = markdown_cells(row)
        if len(cells) != 10:
            fail(f"README text row has {len(cells)} columns instead of 10: {row}")
        model = re.sub(r"[*`]", "", cells[0]).strip()
        if model in {"inkling-small (none)", "inkling-small (low)"}:
            seen_new[model] += 1
            continue
        rate_text = re.sub(r"[*`]", "", cells[1]).strip()
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", rate_text)
        if not match:
            fail(f"README pass-rate cell is malformed: {cells[1]}")
        parsed.append((index, float(match.group(1)), row))
    if any(count > 1 for count in seen_new.values()):
        fail("README contains duplicate Inkling Small rows")

    arms = data.normalized["readme_arms"]
    for offset, arm_name in enumerate(("none", "low"), start=len(rows)):
        row = render_readme_row(f"inkling-small ({arm_name})", arms[arm_name])
        parsed.append((offset, number(arms[arm_name]["pass_rate_pct"], "pass"), row))
    parsed.sort(key=lambda item: (-item[1], item[0]))
    new_table = "\n".join(item[2] for item in parsed)
    updated = (
        before
        + README_HEADER
        + "\n"
        + README_SEPARATOR
        + "\n"
        + new_table
        + "\n\n"
        + after
    )

    replacement = (
        README_PROSE_START
        + "\n"
        + prose_block(data)
        + "\n"
        + README_PROSE_END
    )
    if README_PROSE_START in updated or README_PROSE_END in updated:
        if updated.count(README_PROSE_START) != 1 or updated.count(README_PROSE_END) != 1:
            fail("README Inkling Small prose markers are incomplete or duplicated")
        pattern = re.compile(
            re.escape(README_PROSE_START)
            + r".*?"
            + re.escape(README_PROSE_END),
            re.DOTALL,
        )
        updated = pattern.sub(replacement, updated)
    else:
        if updated.count(LEGACY_INKLING_PROSE) != 1:
            fail("stale Inkling explanatory paragraph changed or is ambiguous")
        updated = updated.replace(LEGACY_INKLING_PROSE, replacement)
    validate_readme_text(updated, data)
    return updated


def validate_readme_text(text: str, data: PublicationData) -> None:
    if text.count(README_HEADER) != 1 or text.count(README_SEPARATOR) < 1:
        fail("README table header or separator is missing")
    body = text.split(README_HEADER, 1)[1]
    body = body.split("\n\n", 1)[0]
    lines = body.splitlines()[2:]
    pass_rates: list[float] = []
    found = Counter()
    for line in lines:
        cells = markdown_cells(line)
        if len(cells) != 10:
            fail("README publication table no longer has exactly 10 columns")
        if cells[-1] == "Provider":
            fail("README provider column is not row data")
        model = re.sub(r"[*`]", "", cells[0]).strip()
        if model in {"inkling-small (none)", "inkling-small (low)"}:
            found[model] += 1
            if cells[-1] != PROVIDER_DISPLAY:
                fail(f"{model} provider must be BaseTen at the row end")
        rate_text = re.sub(r"[*`]", "", cells[1]).strip()
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", rate_text)
        if not match:
            fail(f"README pass-rate cell is malformed: {cells[1]}")
        pass_rates.append(float(match.group(1)))
    if found != Counter({"inkling-small (none)": 1, "inkling-small (low)": 1}):
        fail(f"README Inkling Small row presence is not exact: {dict(found)}")
    if pass_rates != sorted(pass_rates, reverse=True):
        fail("README text rows are not sorted by descending pass rate")
    if "Run Count" in README_HEADER or "Runs" in README_HEADER:
        fail("README publication table must not expose a run-count column")
    if (
        text.count(README_PROSE_START) != 1
        or text.count(README_PROSE_END) != 1
        or "Inkling's accuracy peaks at `low`" in text
    ):
        fail("README Inkling explanatory prose remains stale or duplicated")
    prose = text.split(README_PROSE_START, 1)[1].split(README_PROSE_END, 1)[0]
    if any(
        fragment in prose
        for fragment in (
            "22/60",
            "BaseTen HTTP 429",
            "disputed `tool_use_correct` labels",
            "30 frozen temporal pairs",
            "900-turn denominators",
            "30-conversation",
            "n=",
        )
    ):
        fail("README must not publish Inkling Small per-model robustness counts")


GENERATOR_DATA_BLOCK = r'''# INKLING_SMALL_PUBLICATION_DATA_START
INKLING_SMALL_PUBLICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops/baseten-inkling-small/aiewf-medium-none-low-n30-20260731/analysis/publication-input.json"
)
if not INKLING_SMALL_PUBLICATION_PATH.is_file():
    raise RuntimeError(
        f"final Inkling Small publication input is required: {INKLING_SMALL_PUBLICATION_PATH}"
    )
INKLING_SMALL_PUBLICATION = json.loads(INKLING_SMALL_PUBLICATION_PATH.read_text())
if (
    INKLING_SMALL_PUBLICATION.get("schema_version") != 1
    or INKLING_SMALL_PUBLICATION.get("artifact_status") != "FINAL_PUBLICATION_INPUT"
    or INKLING_SMALL_PUBLICATION.get("model") != "thinkingmachines/inkling-small"
    or INKLING_SMALL_PUBLICATION.get("report_name") != "inkling-small"
    or INKLING_SMALL_PUBLICATION.get("provider") != "BaseTen"
):
    raise ValueError("Inkling Small publication input identity mismatch")
INKLING_SMALL_SCREEN = INKLING_SMALL_PUBLICATION.get("screen_row", {})
if (
    INKLING_SMALL_SCREEN.get("name") != "inkling-small"
    or INKLING_SMALL_SCREEN.get("provider") != "BaseTen"
    or INKLING_SMALL_SCREEN.get("included_runs", [None])[0] != 30
    or INKLING_SMALL_SCREEN.get("none_ttfat_p50_ms") is None
):
    raise ValueError("Inkling Small screen row is incomplete")
INKLING_SMALL_ROBUSTNESS = INKLING_SMALL_PUBLICATION.get("robustness", {})
INKLING_SMALL_EFFORT_ROBUSTNESS = INKLING_SMALL_ROBUSTNESS.get(
    "primary_effort_campaign", {}
)
INKLING_SMALL_SHORT_RUNS = INKLING_SMALL_EFFORT_ROBUSTNESS.get(
    "baseten_429_idle_short_runs", {}
)
INKLING_SMALL_JUDGE_SENSITIVITY = INKLING_SMALL_ROBUSTNESS.get(
    "judge_sensitivity", {}
)
if (
    INKLING_SMALL_EFFORT_ROBUSTNESS.get("retained_attempts") != 60
    or INKLING_SMALL_SHORT_RUNS.get("none") != 12
    or INKLING_SMALL_SHORT_RUNS.get("low") != 10
    or INKLING_SMALL_SHORT_RUNS.get("total") != 22
    or INKLING_SMALL_EFFORT_ROBUSTNESS.get(
        "fixed_denominator_missing_future_turns_fail"
    ) is not True
    or INKLING_SMALL_EFFORT_ROBUSTNESS.get(
        "serving_failures_not_generated_terminal_calls"
    ) is not True
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "changed_tool_use_correct_labels"
    ) != 4
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "max_abs_arm_rate_change_percentage_points", 1
    ) > 0.5
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "disclosure_bound_percentage_points"
    ) != 0.5
    or INKLING_SMALL_JUDGE_SENSITIVITY.get(
        "official_artifacts_unchanged"
    ) is not True
):
    raise ValueError("Inkling Small robustness disclosure input mismatch")
inkling_small_delta = INKLING_SMALL_SCREEN["dots_minus_control_points"]
MODELS.append((
    "inkling-small",
    "BaseTen",
    INKLING_SMALL_SCREEN["no_filler_pass_rate_pct"],
    INKLING_SMALL_SCREEN["dots_pass_rate_pct"],
    f"{inkling_small_delta:+.1f}".replace("-", "−"),
    "",
    f'{INKLING_SMALL_SCREEN["included_runs"][0]} / {INKLING_SMALL_SCREEN["included_runs"][1]}',
    INKLING_SMALL_SCREEN["key"],
    INKLING_SMALL_SCREEN["interpretation"],
    round(INKLING_SMALL_SCREEN["none_ttfat_p50_ms"]),
))
INKLING_SMALL_METHOD_MARKDOWN = (
    " Inkling Small adds a separate fixed-denominator BaseTen comparison: its 30-run "
    f"`none` control is frozen from the none/low campaign and its later adaptive dot arm "
    f"stopped at {INKLING_SMALL_PUBLICATION['dots_stage']}; the two arms are not interleaved."
)
INKLING_SMALL_LIMITS_MARKDOWN = (
    " The Inkling Small screen is fixed-denominator and attempt-based, but reuses an "
    "earlier control, so deployment-time drift remains a limitation."
)
INKLING_SMALL_PROVENANCE_MARKDOWN = (
    " The Inkling Small row uses BaseTen for both arms, the frozen `none` arm's TTFAT, "
    "and the highest mechanically reached dot-stage artifact. In Inkling Small's primary "
    "30-pair `none`/`low` campaign, "
    f"{INKLING_SMALL_SHORT_RUNS['total']}/"
    f"{INKLING_SMALL_EFFORT_ROBUSTNESS['retained_attempts']} retained attempts ended "
    "short after a BaseTen HTTP 429 followed by the harness idle timeout "
    f"({INKLING_SMALL_SHORT_RUNS['none']} `none`, "
    f"{INKLING_SMALL_SHORT_RUNS['low']} `low`); these were serving failures rather "
    "than generated terminal calls, and fixed-denominator scoring retains them with "
    "missing future turns counted as failures. A post-hoc sensitivity check changing "
    f"the {INKLING_SMALL_JUDGE_SENSITIVITY['changed_tool_use_correct_labels']} disputed "
    "`tool_use_correct` labels shifted any arm-level published rate by no more than "
    f"{INKLING_SMALL_JUDGE_SENSITIVITY['disclosure_bound_percentage_points']:.1f} "
    "percentage points; official judgments remain unchanged."
)
INKLING_SMALL_METHOD_HTML = INKLING_SMALL_METHOD_MARKDOWN.replace("`none`", "<code>none</code>")
INKLING_SMALL_LIMITS_HTML = INKLING_SMALL_LIMITS_MARKDOWN
INKLING_SMALL_PROVENANCE_HTML = INKLING_SMALL_PROVENANCE_MARKDOWN.replace(
    "`none`", "<code>none</code>"
).replace(
    "`low`", "<code>low</code>"
).replace(
    "`tool_use_correct`", "<code>tool_use_correct</code>"
)
# INKLING_SMALL_PUBLICATION_DATA_END'''

GENERATOR_DETAIL_BLOCK = r'''# INKLING_SMALL_PUBLICATION_DETAIL_START
PROSPECTIVE_DETAILS["inkling-small"] = {
    "completion": INKLING_SMALL_SCREEN["strict_completion_pct"]
}
if INKLING_SMALL_SCREEN.get("focused") is True:
    FOCUSED["inkling-small"] = {
        "ci": INKLING_SMALL_SCREEN["ci95"],
        "completion": INKLING_SMALL_SCREEN["strict_completion_pct"],
        "raw_delta": INKLING_SMALL_SCREEN["dots_minus_control_points"],
        "control": {
            "pass_rate_pct": INKLING_SMALL_SCREEN["no_filler_pass_rate_pct"],
            "ttfat_p50_ms": INKLING_SMALL_SCREEN["none_ttfat_p50_ms"],
        },
        "dots": {"pass_rate_pct": INKLING_SMALL_SCREEN["dots_pass_rate_pct"]},
    }
# INKLING_SMALL_PUBLICATION_DETAIL_END'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        fail(f"generator anchor {label} must appear exactly once")
    return text.replace(old, new)


def transform_generator(source: str) -> str:
    if GENERATOR_DATA_START in source or GENERATOR_DATA_END in source:
        if source.count(GENERATOR_DATA_START) != 1 or source.count(GENERATOR_DATA_END) != 1:
            fail("generator Inkling Small data markers are incomplete or duplicated")
    else:
        anchor = "EXPECTED_MODEL_COUNT = 23 + int(GEMINI25_RESULT is not None)"
        source = replace_once(
            source,
            anchor,
            GENERATOR_DATA_BLOCK
            + "\nEXPECTED_MODEL_COUNT = 24 + int(GEMINI25_RESULT is not None)",
            "model-count/data insertion",
        )

    if GENERATOR_DETAIL_START in source or GENERATOR_DETAIL_END in source:
        if source.count(GENERATOR_DETAIL_START) != 1 or source.count(GENERATOR_DETAIL_END) != 1:
            fail("generator Inkling Small detail markers are incomplete or duplicated")
    else:
        anchor = "TURN_FAMILY_PATH = (Path(__file__).resolve().parents[1] /"
        source = replace_once(
            source,
            anchor,
            GENERATOR_DETAIL_BLOCK + "\n" + anchor,
            "completion/focused detail insertion",
        )

    replacements = (
        (
            "{gemini25_method}{laguna_method}{qwen_method} The other nine",
            "{gemini25_method}{laguna_method}{qwen_method}{INKLING_SMALL_METHOD_MARKDOWN} The other nine",
            "{INKLING_SMALL_METHOD_MARKDOWN}",
            "Markdown method prose",
        ),
        (
            "{gemini25_provenance}{laguna_provenance}{qwen_provenance} The Qwen3-8B",
            "{gemini25_provenance}{laguna_provenance}{qwen_provenance}{INKLING_SMALL_PROVENANCE_MARKDOWN} The Qwen3-8B",
            "{INKLING_SMALL_PROVENANCE_MARKDOWN}",
            "Markdown provenance prose",
        ),
        (
            "{gemini25_method_html}{laguna_method_html}{qwen_method_html} The nine",
            "{gemini25_method_html}{laguna_method_html}{qwen_method_html}{INKLING_SMALL_METHOD_HTML} The nine",
            "{INKLING_SMALL_METHOD_HTML}",
            "HTML method prose",
        ),
        (
            "{gemini25_limits_html}{laguna_limits_html}{qwen_limits_html} The nine",
            "{gemini25_limits_html}{laguna_limits_html}{qwen_limits_html}{INKLING_SMALL_LIMITS_HTML} The nine",
            "{INKLING_SMALL_LIMITS_HTML}",
            "HTML limits prose",
        ),
        (
            "{gemini25_provenance_html}{laguna_provenance_html}{qwen_provenance_html} The nine",
            "{gemini25_provenance_html}{laguna_provenance_html}{qwen_provenance_html}{INKLING_SMALL_PROVENANCE_HTML} The nine",
            "{INKLING_SMALL_PROVENANCE_HTML}",
            "HTML provenance prose",
        ),
        (
            '        24: "Twenty-four",\n',
            '        24: "Twenty-four",\n        25: "Twenty-five",\n',
            '        25: "Twenty-five",\n',
            "25-model title",
        ),
    )
    for old, new, sentinel, label in replacements:
        if sentinel in source:
            continue
        source = replace_once(source, old, new, label)
    compile(source, str(GENERATOR_PATH), "exec")
    return source


def transform_readme_verifier(source: str) -> str:
    """Keep the existing exact provider-map verifier compatible with new rows."""
    additions = (
        '    "inkling-small (none)": "BaseTen",\n',
        '    "inkling-small (low)": "BaseTen",\n',
    )
    present = tuple(item in source for item in additions)
    if all(present):
        compile(source, str(README_VERIFIER_PATH), "exec")
        return source
    if any(present):
        fail("README provider verifier has only one Inkling Small mapping")
    anchor = '    "inkling (none)": "BaseTen",\n'
    source = replace_once(
        source,
        anchor,
        anchor + "".join(additions),
        "README verifier provider mapping",
    )
    compile(source, str(README_VERIFIER_PATH), "exec")
    return source


def normalized_text(data: PublicationData) -> str:
    return json.dumps(data.normalized, indent=2) + "\n"


def unified_diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path) + " (proposed)",
        )
    )


def section_three(html_text: str) -> str:
    match = re.search(
        r'<section id="primary-screen">(.*?)</section>', html_text, re.DOTALL
    )
    if not match:
        fail("HTML Section 3 primary-screen section is missing")
    return match.group(1)


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def screen_table(section: str) -> tuple[list[str], list[list[str]]]:
    table_match = re.search(r"<table>(.*?)</table>", section, re.DOTALL)
    if not table_match:
        fail("Section 3 model table is missing")
    rows = re.findall(r"<tr>(.*?)</tr>", table_match.group(1), re.DOTALL)
    if not rows:
        fail("Section 3 model table contains no rows")
    parsed = [
        [strip_tags(cell) for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)]
        for row in rows
    ]
    return parsed[0], parsed[1:]


def chart_labels(section: str) -> list[str]:
    figure = re.search(r"<figure>.*?<svg .*?</svg>", section, re.DOTALL)
    if not figure:
        fail("Section 3 primary chart is missing")
    return [
        strip_tags(value)
        for value in re.findall(
            r'<text x="0"[^>]*class="lbl"[^>]*>(.*?)</text>',
            figure.group(0),
            re.DOTALL,
        )
    ]


def validate_html_update(
    before_text: str,
    after_text: str,
    data: PublicationData,
) -> None:
    before_section = section_three(before_text)
    after_section = section_three(after_text)
    before_header, before_rows = screen_table(before_section)
    after_header, after_rows = screen_table(after_section)
    if before_header != after_header or len(after_header) != 9:
        fail("Section 3 table header/shape changed")
    if any(len(row) != 9 for row in after_rows):
        fail("Section 3 table row shape changed")
    before_names = [row[0] for row in before_rows]
    after_names = [row[0] for row in after_rows]
    if [name for name in before_names if name != REPORT_NAME] != [
        name for name in after_names if name != REPORT_NAME
    ]:
        fail("Section 3 update did not preserve every existing model row and order")
    if after_names.count(REPORT_NAME) != 1:
        fail("Section 3 must contain exactly one Inkling Small table row")

    before_labels = chart_labels(before_section)
    after_labels = chart_labels(after_section)
    if [name for name in before_labels if name != REPORT_NAME] != [
        name for name in after_labels if name != REPORT_NAME
    ]:
        fail("Section 3 chart did not preserve every existing model label and order")
    if after_labels.count(REPORT_NAME) != 1:
        fail("Section 3 must contain exactly one Inkling Small chart label")

    row = after_rows[after_names.index(REPORT_NAME)]
    screen = data.normalized["screen_row"]
    expected = {
        "provider": PROVIDER_DISPLAY,
        "base": f"{screen['no_filler_pass_rate_pct']:.1f}",
        "dots": f"{screen['dots_pass_rate_pct']:.1f}",
        "ttfat": str(round(screen["none_ttfat_p50_ms"])),
        "runs": f"{screen['included_runs'][0]} / {screen['included_runs'][1]}",
    }
    if (
        row[1] != expected["provider"]
        or row[2] != expected["base"]
        or row[3] != expected["dots"]
        or row[6] != expected["ttfat"]
        or row[7] != expected["runs"]
    ):
        fail(f"Section 3 Inkling Small row does not match final inputs: {row}")
    # The campaign-design sentence is part of Section 2's Background & method
    # definition list.  Section 3 carries the row and the run-pool provenance
    # disclosure below, so validate the method sentence across the full report
    # rather than incorrectly requiring it inside the primary-screen section.
    if after_text.count("Inkling Small adds a separate fixed-denominator") != 1:
        fail("Inkling Small methodology prose is missing or duplicated")
    provenance_match = re.search(
        r'<p class="measure"><b>Run-pool provenance\.</b>(.*?)</p>',
        after_section,
        re.DOTALL,
    )
    if not provenance_match:
        fail("Section 3 Run-pool provenance paragraph is missing")
    provenance = provenance_match.group(1)
    if after_section.count("22/60 retained attempts ended") != 1:
        fail("Section 3 Inkling Small BaseTen 429 disclosure is missing or duplicated")
    sensitivity_label = (
        f"{data.normalized['robustness']['judge_sensitivity']['changed_tool_use_correct_labels']} "
        "disputed <code>tool_use_correct</code> labels"
    )
    if after_section.count(sensitivity_label) != 1:
        fail("Section 3 Inkling Small judge sensitivity is missing or duplicated")
    if after_section.count("no more than 0.5 percentage points") != 1:
        fail("Section 3 Inkling Small sensitivity bound is missing or duplicated")
    for fragment in (
        "22/60 retained attempts ended",
        sensitivity_label,
        "no more than 0.5 percentage points",
    ):
        if provenance.count(fragment) != 1:
            fail("Inkling Small robustness disclosure is outside Run-pool provenance")


def validate_markdown_update(text: str) -> None:
    start = "<!-- N30_PRIMARY_START -->"
    end = "<!-- N30_PRIMARY_END -->"
    if text.count(start) != 1 or text.count(end) != 1:
        fail("Markdown Section 3 markers are missing or duplicated")
    section = text.split(start, 1)[1].split(end, 1)[0]
    provenance_start = "The original 17 rows retain their exploratory-screen order"
    provenance_end = "\n\nFlash Lite attempt-policy sensitivity:"
    if section.count(provenance_start) != 1 or section.count(provenance_end) != 1:
        fail("Markdown Section 3 Run-pool provenance paragraph is missing")
    provenance = section.split(provenance_start, 1)[1].split(provenance_end, 1)[0]
    expected = (
        "22/60 retained attempts ended",
        "4 disputed `tool_use_correct` labels",
        "no more than 0.5 percentage points",
    )
    for fragment in expected:
        if section.count(fragment) != 1:
            fail(
                "Markdown Section 3 Inkling Small robustness disclosure is "
                f"missing or duplicated: {fragment}"
            )
        if provenance.count(fragment) != 1:
            fail(
                "Markdown Inkling Small robustness disclosure is outside "
                f"Run-pool provenance: {fragment}"
            )


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def local_generator_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")
    return {name: os.environ[name] for name in allowed if name in os.environ}


def apply_update(
    data: PublicationData,
    readme_after: str,
    generator_after: str,
    verifier_after: str,
) -> None:
    before_html = HTML_PATH.read_text(encoding="utf-8")
    with LOCK_PATH.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another Inkling Small publication update owns the lock")

        # Re-read all mutable inputs after obtaining the lock.
        current = load_publication_data()
        if current.normalized != data.normalized:
            fail("final publication inputs changed between preflight and apply")
        current_readme = README_PATH.read_text(encoding="utf-8")
        current_generator = GENERATOR_PATH.read_text(encoding="utf-8")
        current_verifier = README_VERIFIER_PATH.read_text(encoding="utf-8")
        if update_readme_text(current_readme, current) != readme_after:
            fail("README changed between preflight and apply")
        if transform_generator(current_generator) != generator_after:
            fail("report generator changed between preflight and apply")
        if transform_readme_verifier(current_verifier) != verifier_after:
            fail("README provider verifier changed between preflight and apply")

        atomic_write(NORMALIZED_PATH, normalized_text(current))
        atomic_write(GENERATOR_PATH, generator_after)
        result = subprocess.run(
            [sys.executable, str(GENERATOR_PATH)],
            cwd=ROOT,
            env=local_generator_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            fail(
                "local filler-report rebuild failed after generator update:\n"
                + result.stdout
            )
        after_html = HTML_PATH.read_text(encoding="utf-8")
        after_markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
        validate_html_update(before_html, after_html, current)
        validate_markdown_update(after_markdown)
        atomic_write(README_PATH, readme_after)
        atomic_write(README_VERIFIER_PATH, verifier_after)
        validate_readme_text(README_PATH.read_text(encoding="utf-8"), current)
        if transform_generator(GENERATOR_PATH.read_text(encoding="utf-8")) != generator_after:
            fail("publication generator is not idempotent after apply")
        if (
            transform_readme_verifier(
                README_VERIFIER_PATH.read_text(encoding="utf-8")
            )
            != verifier_after
        ):
            fail("README provider verifier is not idempotent after apply")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write normalized input, update sources, and rebuild locally",
    )
    args = parser.parse_args()
    data = load_publication_data()
    readme_before = README_PATH.read_text(encoding="utf-8")
    generator_before = GENERATOR_PATH.read_text(encoding="utf-8")
    verifier_before = README_VERIFIER_PATH.read_text(encoding="utf-8")
    readme_after = update_readme_text(readme_before, data)
    generator_after = transform_generator(generator_before)
    verifier_after = transform_readme_verifier(verifier_before)
    publication_after = normalized_text(data)
    publication_before = (
        NORMALIZED_PATH.read_text(encoding="utf-8")
        if NORMALIZED_PATH.is_file()
        else ""
    )

    if not args.apply:
        print(unified_diff(README_PATH, readme_before, readme_after), end="")
        print(
            unified_diff(
                GENERATOR_PATH, generator_before, generator_after
            ),
            end="",
        )
        print(
            unified_diff(
                NORMALIZED_PATH, publication_before, publication_after
            ),
            end="",
        )
        print(
            unified_diff(
                README_VERIFIER_PATH, verifier_before, verifier_after
            ),
            end="",
        )
        print(
            f"Dry run only: dots_stage={data.dots_stage}, provider=BaseTen, "
            f"none_ttfat={data.normalized['screen_row']['none_ttfat_p50_ms']:.0f}ms. "
            "README, generator, and HTML were not modified."
        )
        return 0

    apply_update(data, readme_after, generator_after, verifier_after)
    print(
        "Publication applied and validated: two README rows, one Section 3 "
        "screen row, one chart label, all prior screen rows preserved."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
