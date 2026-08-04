#!/usr/bin/env python3
"""Dry-run or apply the final Gemma paired-dots publication update.

The default performs local validation and prints diffs only. ``--apply`` is
accepted only after a reviewer-bound terminal publication state exists. No
code path calls BaseTen, a judge, or the collection driver.
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
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]
README_PATH = ROOT / "README.md"
GENERATOR_PATH = ROOT / "scripts/build_filler_report.py"
MARKDOWN_PATH = ROOT / "docs/filler-token-latent-scratchpad-study.md"
HTML_PATH = ROOT / "docs/filler-token-latent-scratchpad-study.html"
NORMALIZED_PATH = HERE / "publication-input.json"
REVIEW_PATH = HERE / "publication-review.json"
LOCK_PATH = HERE / ".publication.lock"

VERIFIER_PATHS = (
    ROOT / "docs/filler-study-data/gemini-minimal-dots-2026-07-21/verify_outputs.py",
    ROOT / "docs/filler-study-data/dot-stability-n30-2026-07-20/verify_outputs.py",
    ROOT / "docs/filler-study-data/gemini25-thinking-off-dots-2026-07-22/verify_outputs.py",
    ROOT / "docs/filler-study-data/laguna-s21-openrouter-2026-07-22/verify_outputs.py",
)

CAMPAIGN_ID = "aiewf-medium-gemma4-26b-a4b-dots-paired-20260731"
MODEL = "google/gemma-4-26B-A4B-it"
PROVIDER = "BaseTen"
REPORT_NAME = "gemma-4-26b-a4b"
README_LABEL = "gemma-4-26b-a4b-it (thinking off)"
N_TURNS = 30
STAGES = (("full", 30), ("initial", 10))

README_HEADER = (
    "| Model | Pass Rate | Any Error | Tool Error | Instruction Error | "
    "KB Error | TTFAT P50 | TTFAT P95 | TTFAT Max | Provider |"
)
README_SEPARATOR = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"

DATA_START = "# GEMMA26_PUBLICATION_DATA_START"
DATA_END = "# GEMMA26_PUBLICATION_DATA_END"
DETAIL_START = "# GEMMA26_PUBLICATION_DETAIL_START"
DETAIL_END = "# GEMMA26_PUBLICATION_DETAIL_END"
SCOPE_START = "    # GEMMA26_MARKDOWN_SCOPE_START"
SCOPE_END = "    # GEMMA26_MARKDOWN_SCOPE_END"


@dataclass(frozen=True)
class PublicationData:
    stage: str
    n: int
    aggregate_path: Path
    included_path: Path
    report_path: Path
    aggregate: dict[str, Any]
    review: dict[str, Any]
    normalized: dict[str, Any]


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return abs(left - right) <= tolerance


def relative(path: Path, root: Path = ROOT) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def resolve_repo_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} path is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        fail(f"{label} path escapes the repository: {value}")
    if not path.is_file():
        fail(f"{label} file is missing: {path}")
    return path


def validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        fail(f"{label} is not a lowercase SHA-256")
    return value


def validate_iso8601(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        fail(f"{label} is missing")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} is not an ISO-8601 timestamp")


def stage_files(analysis: Path, stage: str) -> tuple[Path, Path, Path]:
    return (
        analysis / f"aggregates-{stage}.json",
        analysis / f"included-runs-{stage}.tsv",
        analysis / f"REPORT-{stage}.md",
    )


def select_stage(analysis: Path) -> tuple[str, int, Path, Path, Path]:
    for stage, n in STAGES:
        aggregate, included, report = stage_files(analysis, stage)
        if aggregate.is_file():
            if not included.is_file() or not report.is_file():
                fail(f"{stage} aggregate lacks its frozen included-runs/report artifacts")
            return stage, n, aggregate, included, report
    fail("no frozen Gemma aggregate exists: expected aggregates-initial.json or aggregates-full.json")


def validate_embedded_hashes(
    root: Path, analysis: Path, stage: str, payload: dict[str, Any]
) -> None:
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, dict):
        fail(f"{stage} aggregate lacks input_hashes")
    campaign = analysis.parent
    expected_paths = {
        "configuration": campaign / "configuration.json",
        "frozen_order": campaign / "frozen-order.tsv",
        "canonical": campaign / "canonical.tsv",
        "judge_inputs": campaign / f"judging/canonical-inputs-{stage}.tsv",
        "judge_complete": campaign / f"judging/COMPLETE-{stage}.json",
        "judge_source": campaign / "judging/judge-source-sha256.txt",
        "analysis_source": campaign / "analyze_stage.py",
    }
    if set(hashes) != set(expected_paths):
        fail(f"{stage} aggregate input_hashes keys changed: {sorted(hashes)}")
    for key, path in expected_paths.items():
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            fail(f"{stage} hashed input escapes repository: {path}")
        if not path.is_file():
            fail(f"{stage} hashed input is missing: {path}")
        expected = validate_hash(hashes[key], f"{stage}.{key}")
        # The canonical ledger is append-only.  The initial aggregate was
        # frozen when it contained its 20 scheduled rows; reaching the
        # prespecified full stage legitimately appends another 40.  Bind the
        # initial aggregate to that exact file prefix while continuing to
        # require the complete file hash for the terminal stage.
        actual = (
            sha256_tsv_prefix(path, 20)
            if key == "canonical" and stage == "initial"
            else sha256(path)
        )
        if actual != expected:
            fail(f"{stage} hashed input changed: {path}")


def sha256_tsv_prefix(path: Path, data_rows: int) -> str:
    lines = path.read_bytes().splitlines(keepends=True)
    required = data_rows + 1  # one header plus the frozen data rows
    if len(lines) < required:
        fail(
            f"TSV has fewer than {data_rows} data rows required by its "
            f"frozen prefix: {path}"
        )
    return hashlib.sha256(b"".join(lines[:required])).hexdigest()


def validate_metric(metric: dict[str, Any], n: int, label: str) -> float:
    denominator = n * N_TURNS
    count = integer(metric.get("count"), f"{label}.count")
    total = integer(metric.get("total"), f"{label}.total")
    rate = number(metric.get("rate_percent"), f"{label}.rate_percent")
    ci = metric.get("whole_conversation_bootstrap_95")
    if total != denominator or not 0 <= count <= total:
        fail(f"{label} count/denominator mismatch")
    if not close(rate, count / total * 100):
        fail(f"{label} rate does not match its count")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or number(ci[0], f"{label} CI low") > number(ci[1], f"{label} CI high")
    ):
        fail(f"{label} interval is invalid")
    return rate


def validate_arm(arm: Any, n: int, label: str) -> dict[str, Any]:
    if not isinstance(arm, dict):
        fail(f"{label} arm is not an object")
    if integer(arm.get("conversations"), f"{label}.conversations") != n:
        fail(f"{label} conversation count is not {n}")
    if integer(arm.get("fixed_turn_denominator"), f"{label}.denominator") != n * N_TURNS:
        fail(f"{label} fixed denominator mismatch")
    metrics = arm.get("metrics")
    required = {"strict_pass", "any_error", "tool_error", "instruction_error", "kb_error"}
    if not isinstance(metrics, dict) or set(metrics) != required:
        fail(f"{label} metric set changed")
    rates = {
        name: validate_metric(metrics[name], n, f"{label}.{name}")
        for name in required
    }
    if not close(rates["strict_pass"] + rates["any_error"], 100.0):
        fail(f"{label} strict-pass and any-error rates are not complements")
    completion = arm.get("strict_completion")
    if not isinstance(completion, dict):
        fail(f"{label} strict completion is missing")
    completed = integer(completion.get("count"), f"{label} completion count")
    if integer(completion.get("total"), f"{label} completion total") != n:
        fail(f"{label} completion denominator mismatch")
    completion_rate = number(completion.get("rate_percent"), f"{label} completion rate")
    if not close(completion_rate, completed / n * 100):
        fail(f"{label} completion rate mismatch")
    timing = arm.get("ttfat_ms_observed_responses_only")
    if not isinstance(timing, dict) or integer(timing.get("count"), f"{label} TTFAT count") <= 0:
        fail(f"{label} has no observed TTFAT values")
    for key in ("p50", "p95", "max"):
        if number(timing.get(key), f"{label} TTFAT {key}") < 0:
            fail(f"{label} TTFAT {key} is negative")
    return {"rates": rates, "completion_rate": completion_rate, "timing": timing}


def validate_aggregate(
    root: Path, analysis: Path, path: Path, stage: str, n: int
) -> dict[str, Any]:
    payload = read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("stage") != stage
        or payload.get("model") != MODEL
        or payload.get("provider") != PROVIDER
    ):
        fail(f"{stage} aggregate schema or identity mismatch")
    expected_configuration = {
        "control": "fresh contemporaneous nofiller",
        "treatment": "+96 space-separated suffix dots, request-only",
        "thinking_enabled": False,
        "fixed_turns_per_conversation": N_TURNS,
        "temporal_pairing": True,
    }
    if payload.get("configuration") != expected_configuration:
        fail(f"{stage} request configuration mismatch")
    method = payload.get("method")
    if not isinstance(method, dict) or (
        method.get("fixed_denominator") is not True
        or method.get("missing_future_turns_fail_all_displayed_accuracy_criteria") is not True
        or method.get("arm_interval_unit") != "whole conversation"
        or method.get("effect_interval_unit") != "frozen temporal pair"
        or method.get("effect_bootstrap_design") != "paired bootstrap"
        or method.get("bootstrap_iterations") != 100_000
    ):
        fail(f"{stage} analysis method mismatch")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"nofiller", "dots96"}:
        fail(f"{stage} aggregate arms changed")
    control = validate_arm(arms["nofiller"], n, f"{stage}.nofiller")
    dots = validate_arm(arms["dots96"], n, f"{stage}.dots96")
    effect = payload.get("effects", {}).get("strict_pass")
    if not isinstance(effect, dict):
        fail(f"{stage} strict-pass effect is missing")
    delta = number(effect.get("dots_minus_control_points"), f"{stage} delta")
    low = number(effect.get("paired_bootstrap_95_low"), f"{stage} CI low")
    high = number(effect.get("paired_bootstrap_95_high"), f"{stage} CI high")
    if low > high or not close(delta, dots["rates"]["strict_pass"] - control["rates"]["strict_pass"]):
        fail(f"{stage} effect is invalid or does not equal dots minus control")
    promotion = payload.get("promotion_evaluation")
    if not isinstance(promotion, dict):
        fail(f"{stage} promotion evaluation is missing")
    if stage == "initial":
        if (
            promotion.get("evaluated") is not True
            or promotion.get("terminal_stage") is not False
            or promotion.get("collection_launched") is not False
            or not isinstance(promotion.get("promote_to_n30"), bool)
            or not isinstance(promotion.get("triggered_rules"), list)
        ):
            fail("initial promotion evaluation is not final")
    elif promotion != {
        "evaluated": False,
        "terminal_stage": True,
        "triggered_rules": [],
        "promote_to_n30": False,
        "note": "The full 30-pair stage is terminal; no promotion rule applies.",
    }:
        fail("full aggregate is not the terminal stage")
    validate_embedded_hashes(root, analysis, stage, payload)
    return payload


def validate_artifact_binding(
    root: Path, entry: Any, expected_path: Path, label: str
) -> None:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        fail(f"review {label} binding is malformed")
    path = resolve_repo_path(root, entry["path"], f"review {label}")
    if path != expected_path.resolve():
        fail(f"review {label} points at the wrong file")
    if sha256(path) != validate_hash(entry["sha256"], f"review {label} hash"):
        fail(f"review {label} hash does not match")


def validate_promotion_decision(
    root: Path,
    analysis: Path,
    initial: dict[str, Any],
    initial_path: Path,
    initial_included: Path,
) -> Path:
    path = analysis / "promotion-decision-initial.json"
    payload = read_json(path)
    if (
        payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("decision_after_n_per_arm") != 10
        or payload.get("promote_to_n30") is not True
        or payload.get("triggered_rules") != initial["promotion_evaluation"]["triggered_rules"]
        or payload.get("aggregates_sha256") != sha256(initial_path)
        or payload.get("included_runs_sha256") != sha256(initial_included)
        or payload.get("aggregates_path") != relative(initial_path, root)
        or payload.get("included_runs_path") != relative(initial_included, root)
    ):
        fail("reviewed initial promotion decision does not bind the frozen initial result")
    reviewer = payload.get("reviewed_by")
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2 or reviewer.upper().startswith(("TODO", "REPLACE")):
        fail("reviewed initial promotion decision has no real reviewer")
    validate_iso8601(payload.get("decided_at"), "promotion decided_at")
    return path


def validate_review(
    root: Path,
    review_path: Path,
    stage: str,
    aggregate: Path,
    included: Path,
    report: Path,
    promotion_path: Path | None,
) -> dict[str, Any]:
    review = read_json(review_path)
    expected_action = "publish_full_terminal" if stage == "full" else "stop_at_initial"
    if (
        review.get("schema_version") != 1
        or review.get("artifact_status") != "FINAL_PUBLICATION_REVIEW"
        or review.get("campaign_id") != CAMPAIGN_ID
        or review.get("model") != MODEL
        or review.get("provider") != PROVIDER
        or review.get("selected_stage") != stage
        or review.get("action") != expected_action
    ):
        fail("publication review schema, identity, stage, or action mismatch")
    reviewer = review.get("reviewed_by")
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2 or reviewer.upper().startswith(("TODO", "REPLACE")):
        fail("publication review has no real reviewer")
    validate_iso8601(review.get("reviewed_at"), "publication reviewed_at")
    artifacts = review.get("artifacts")
    expected_keys = {"aggregates", "included_runs", "report"}
    if promotion_path is not None:
        expected_keys.add("promotion_decision")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_keys:
        fail(f"publication review artifact set mismatch: expected {sorted(expected_keys)}")
    validate_artifact_binding(root, artifacts["aggregates"], aggregate, "aggregates")
    validate_artifact_binding(root, artifacts["included_runs"], included, "included_runs")
    validate_artifact_binding(root, artifacts["report"], report, "report")
    if promotion_path is not None:
        validate_artifact_binding(
            root, artifacts["promotion_decision"], promotion_path, "promotion_decision"
        )
    return review


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
    stage: str,
    n: int,
    aggregate_path: Path,
    included_path: Path,
    report_path: Path,
    review_path: Path,
    promotion_path: Path | None,
    aggregate: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    control = aggregate["arms"]["nofiller"]
    dots = aggregate["arms"]["dots96"]
    control_metrics = control["metrics"]
    effect = aggregate["effects"]["strict_pass"]
    delta = effect["dots_minus_control_points"]
    low = effect["paired_bootstrap_95_low"]
    high = effect["paired_bootstrap_95_high"]
    key, interpretation = verdict(delta, low, high)
    sources = [aggregate_path, included_path, report_path, review_path]
    if promotion_path is not None:
        sources.append(promotion_path)
    return {
        "schema_version": 1,
        "artifact_status": "FINAL_PUBLICATION_INPUT",
        "campaign_id": CAMPAIGN_ID,
        "model": MODEL,
        "provider": PROVIDER,
        "selected_stage": stage,
        "reviewed_by": review["reviewed_by"],
        "source_artifacts": {relative(path, root): sha256(path) for path in sources},
        "readme_row": {
            "label": README_LABEL,
            "pass_rate_pct": control_metrics["strict_pass"]["rate_percent"],
            "any_error_rate_pct": control_metrics["any_error"]["rate_percent"],
            "tool_error_rate_pct": control_metrics["tool_error"]["rate_percent"],
            "instruction_error_rate_pct": control_metrics["instruction_error"]["rate_percent"],
            "kb_error_rate_pct": control_metrics["kb_error"]["rate_percent"],
            "ttfat_p50_ms": control["ttfat_ms_observed_responses_only"]["p50"],
            "ttfat_p95_ms": control["ttfat_ms_observed_responses_only"]["p95"],
            "ttfat_max_ms": control["ttfat_ms_observed_responses_only"]["max"],
            "provider": PROVIDER,
        },
        "screen_row": {
            "name": REPORT_NAME,
            "provider": PROVIDER,
            "no_filler_pass_rate_pct": control_metrics["strict_pass"]["rate_percent"],
            "dots_pass_rate_pct": dots["metrics"]["strict_pass"]["rate_percent"],
            "dots_minus_control_points": delta,
            "ci95": [low, high],
            "included_runs": [n, n],
            "key": key,
            "interpretation": interpretation,
            "no_filler_ttfat_p50_ms": control["ttfat_ms_observed_responses_only"]["p50"],
            "strict_completion_pct": [
                control["strict_completion"]["rate_percent"],
                dots["strict_completion"]["rate_percent"],
            ],
            "focused": stage == "full",
            "temporally_paired": True,
        },
    }


def load_publication_data(
    *, root: Path = ROOT, analysis: Path | None = None, review_path: Path | None = None
) -> PublicationData:
    analysis = analysis or HERE
    review_path = review_path or (analysis / "publication-review.json")
    stage, n, aggregate_path, included_path, report_path = select_stage(analysis)
    aggregate = validate_aggregate(root, analysis, aggregate_path, stage, n)
    promotion_path: Path | None = None
    if stage == "initial":
        if aggregate["promotion_evaluation"]["promote_to_n30"] is True:
            fail("initial promotion triggered; the terminal full stage is required before publication")
        if (analysis / "promotion-decision-initial.json").exists():
            fail("a stale promotion decision exists although the reviewed initial result stops")
    else:
        initial_path, initial_included, initial_report = stage_files(analysis, "initial")
        if not all(path.is_file() for path in (initial_path, initial_included, initial_report)):
            fail("full publication requires the frozen initial analysis and report")
        initial = validate_aggregate(root, analysis, initial_path, "initial", 10)
        if initial["promotion_evaluation"]["promote_to_n30"] is not True:
            fail("full stage exists although the initial prespecified rule did not promote")
        promotion_path = validate_promotion_decision(
            root, analysis, initial, initial_path, initial_included
        )
    if not review_path.is_file():
        fail(f"reviewed final publication state is absent: {review_path}")
    review = validate_review(
        root,
        review_path,
        stage,
        aggregate_path,
        included_path,
        report_path,
        promotion_path,
    )
    normalized = normalize(
        root,
        stage,
        n,
        aggregate_path,
        included_path,
        report_path,
        review_path,
        promotion_path,
        aggregate,
        review,
    )
    return PublicationData(
        stage,
        n,
        aggregate_path,
        included_path,
        report_path,
        aggregate,
        review,
        normalized,
    )


def markdown_cells(line: str) -> list[str]:
    if not line.startswith("|") or not line.endswith("|"):
        fail(f"malformed Markdown row: {line}")
    return [cell.strip() for cell in line[1:-1].split("|")]


def render_readme_row(row: dict[str, Any]) -> str:
    cells = [
        README_LABEL,
        f"{number(row['pass_rate_pct'], 'README pass'):.1f}%",
        f"{number(row['any_error_rate_pct'], 'README any error'):.1f}%",
        f"{number(row['tool_error_rate_pct'], 'README tool error'):.1f}%",
        f"{number(row['instruction_error_rate_pct'], 'README instruction error'):.1f}%",
        f"{number(row['kb_error_rate_pct'], 'README KB error'):.1f}%",
        f"{number(row['ttfat_p50_ms'], 'README TTFAT P50'):.0f}ms",
        f"{number(row['ttfat_p95_ms'], 'README TTFAT P95'):.0f}ms",
        f"{number(row['ttfat_max_ms'], 'README TTFAT max'):.0f}ms",
        PROVIDER,
    ]
    return "| " + " | ".join(cells) + " |"


def validate_readme(text: str, data: PublicationData) -> None:
    if text.count(README_HEADER) != 1:
        fail("README results header must appear exactly once")
    body = text.split(README_HEADER, 1)[1].split("\n\n", 1)[0]
    lines = body.splitlines()
    if len(lines) < 3 or lines[1] != README_SEPARATOR:
        fail("README separator changed")
    found = 0
    rates: list[float] = []
    expected = render_readme_row(data.normalized["readme_row"])
    for line in lines[2:]:
        cells = markdown_cells(line)
        if len(cells) != 10:
            fail("README result table no longer has ten columns")
        label = re.sub(r"[*`]", "", cells[0]).strip()
        rate_text = re.sub(r"[*`]", "", cells[1]).strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)%", rate_text)
        if not match:
            fail(f"README pass-rate cell is malformed: {cells[1]}")
        rates.append(float(match.group(1)))
        if label == README_LABEL:
            found += 1
            if line != expected:
                fail("README Gemma row does not equal the final contemporaneous control")
            if cells[-1] != PROVIDER:
                fail("README Gemma Provider is not the final cell")
    if found != 1:
        fail(f"README must contain exactly one Gemma row; found {found}")
    if rates != sorted(rates, reverse=True):
        fail("README rows are not sorted by descending pass rate")
    if "| Runs |" in body or "| Run Count |" in body:
        fail("README must not expose a run-count column")


def update_readme(text: str, data: PublicationData) -> str:
    if text.count(README_HEADER) != 1:
        fail("README results header must appear exactly once")
    before, remainder = text.split(README_HEADER, 1)
    prefix = "\n" + README_SEPARATOR + "\n"
    if not remainder.startswith(prefix):
        fail("README separator changed")
    table, blank, after = remainder[len(prefix) :].partition("\n\n")
    if not blank:
        fail("README result table has no terminating blank line")
    rows = table.splitlines()
    kept: list[tuple[int, float, str]] = []
    found = 0
    for index, line in enumerate(rows):
        cells = markdown_cells(line)
        if len(cells) != 10:
            fail("README row width changed")
        label = re.sub(r"[*`]", "", cells[0]).strip()
        rate_text = re.sub(r"[*`]", "", cells[1]).strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)%", rate_text)
        if not match:
            fail(f"README pass-rate cell is malformed: {cells[1]}")
        if label == README_LABEL:
            found += 1
            continue
        kept.append((index, float(match.group(1)), line))
    if found != 1:
        fail(f"README must have exactly one replaceable Gemma row; found {found}")
    row = data.normalized["readme_row"]
    kept.append((len(rows), number(row["pass_rate_pct"], "Gemma pass"), render_readme_row(row)))
    kept.sort(key=lambda item: (-item[1], item[0]))
    updated = (
        before
        + README_HEADER
        + prefix
        + "\n".join(item[2] for item in kept)
        + "\n\n"
        + after
    )
    validate_readme(updated, data)
    return updated


GENERATOR_DATA_BLOCK = r'''# GEMMA26_PUBLICATION_DATA_START
GEMMA26_PUBLICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops/baseten-gemma4-26b-a4b-vllm/dots-20260731/analysis/publication-input.json"
)
if not GEMMA26_PUBLICATION_PATH.is_file():
    raise RuntimeError(f"final Gemma 4 26B publication input is required: {GEMMA26_PUBLICATION_PATH}")
GEMMA26_PUBLICATION = json.loads(GEMMA26_PUBLICATION_PATH.read_text())
if (
    GEMMA26_PUBLICATION.get("schema_version") != 1
    or GEMMA26_PUBLICATION.get("artifact_status") != "FINAL_PUBLICATION_INPUT"
    or GEMMA26_PUBLICATION.get("model") != "google/gemma-4-26B-A4B-it"
    or GEMMA26_PUBLICATION.get("provider") != "BaseTen"
):
    raise ValueError("Gemma 4 26B publication input identity mismatch")
GEMMA26_SCREEN = GEMMA26_PUBLICATION.get("screen_row", {})
if (
    GEMMA26_SCREEN.get("name") != "gemma-4-26b-a4b"
    or GEMMA26_SCREEN.get("provider") != "BaseTen"
    or GEMMA26_SCREEN.get("included_runs", [None, None])[0]
       != GEMMA26_SCREEN.get("included_runs", [None, None])[1]
    or GEMMA26_SCREEN.get("included_runs", [None])[0] not in {10, 30}
    or GEMMA26_SCREEN.get("no_filler_ttfat_p50_ms") is None
):
    raise ValueError("Gemma 4 26B screen row is incomplete")
gemma26_delta = GEMMA26_SCREEN["dots_minus_control_points"]
MODELS.append((
    "gemma-4-26b-a4b", "BaseTen",
    GEMMA26_SCREEN["no_filler_pass_rate_pct"],
    GEMMA26_SCREEN["dots_pass_rate_pct"],
    f"{gemma26_delta:+.1f}".replace("-", "−"), "",
    f'{GEMMA26_SCREEN["included_runs"][0]} / {GEMMA26_SCREEN["included_runs"][1]}',
    GEMMA26_SCREEN["key"], GEMMA26_SCREEN["interpretation"],
    round(GEMMA26_SCREEN["no_filler_ttfat_p50_ms"]),
))
GEMMA26_METHOD_MARKDOWN = (
    " Gemma 4 26B A4B adds a separate fixed-denominator, temporally paired BaseTen "
    f"comparison with {GEMMA26_SCREEN['included_runs'][0]} fresh contemporaneous conversations "
    "per arm and native thinking disabled."
)
GEMMA26_LIMITS_MARKDOWN = (
    " The Gemma 4 26B comparison is attempt-based and paired within its collection window; "
    "it does not reuse the older README control."
)
GEMMA26_PROVENANCE_MARKDOWN = (
    " The Gemma 4 26B row and its README row share the fresh BaseTen no-filler arm; "
    "the screen TTFAT is that row configuration's observed-response P50."
)
GEMMA26_METHOD_HTML = GEMMA26_METHOD_MARKDOWN
GEMMA26_LIMITS_HTML = GEMMA26_LIMITS_MARKDOWN
GEMMA26_PROVENANCE_HTML = GEMMA26_PROVENANCE_MARKDOWN
# GEMMA26_PUBLICATION_DATA_END'''

GENERATOR_DETAIL_BLOCK = r'''# GEMMA26_PUBLICATION_DETAIL_START
PROSPECTIVE_DETAILS["gemma-4-26b-a4b"] = {
    "completion": GEMMA26_SCREEN["strict_completion_pct"]
}
if GEMMA26_SCREEN.get("focused") is True:
    FOCUSED["gemma-4-26b-a4b"] = {
        "ci": GEMMA26_SCREEN["ci95"],
        "completion": GEMMA26_SCREEN["strict_completion_pct"],
        "raw_delta": GEMMA26_SCREEN["dots_minus_control_points"],
        "control": {
            "pass_rate_pct": GEMMA26_SCREEN["no_filler_pass_rate_pct"],
            "ttfat_p50_ms": GEMMA26_SCREEN["no_filler_ttfat_p50_ms"],
        },
        "dots": {"pass_rate_pct": GEMMA26_SCREEN["dots_pass_rate_pct"]},
    }
# GEMMA26_PUBLICATION_DETAIL_END'''

GENERATOR_SCOPE_BLOCK = r'''    # GEMMA26_MARKDOWN_SCOPE_START
    scope_words = {24: "Twenty-four", 25: "Twenty-five", 26: "Twenty-six"}
    expected_scope = scope_words.get(len(MODELS), str(len(MODELS)))
    scope_variants = [
        f"**Scope:** {word} standard filler comparisons"
        for word in scope_words.values()
    ]
    matched_scope = [variant for variant in scope_variants if variant in text]
    if len(matched_scope) != 1:
        raise ValueError("Markdown report scope count is missing or ambiguous")
    text = text.replace(
        matched_scope[0],
        f"**Scope:** {expected_scope} standard filler comparisons",
    )
    # GEMMA26_MARKDOWN_SCOPE_END'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        fail(f"generator anchor {label} must appear exactly once")
    return text.replace(old, new)


def insert_prose_variable(
    source: str, base: str, optional_inkling: str, gemma: str, tail: str, label: str
) -> str:
    if gemma in source:
        if source.count(gemma) != 1:
            fail(f"generator {label} Gemma prose is duplicated")
        return source
    candidates = (base + optional_inkling + tail,)
    matches = [candidate for candidate in candidates if candidate in source]
    if len(matches) != 1:
        fail(f"generator {label} prose anchor is missing or ambiguous")
    anchor = matches[0]
    return source.replace(anchor, anchor[: -len(tail)] + gemma + tail)


def transform_generator(source: str) -> tuple[str, int]:
    inkling_markers = (
        "# INKLING_SMALL_PUBLICATION_DATA_START" in source,
        "# INKLING_SMALL_PUBLICATION_DATA_END" in source,
    )
    if any(inkling_markers) and not all(inkling_markers):
        fail("optional Inkling Small generator markers are incomplete")
    inkling_present = all(inkling_markers)
    expected_before = 24 if inkling_present else 23
    expected_after = expected_before + 1

    data_markers = (DATA_START in source, DATA_END in source)
    if any(data_markers) and not all(data_markers):
        fail("Gemma generator data markers are incomplete")
    count_pattern = r"EXPECTED_MODEL_COUNT = (\d+) \+ int\(GEMINI25_RESULT is not None\)"
    count_matches = re.findall(count_pattern, source)
    if len(count_matches) != 1:
        fail("generator expected-model-count expression is missing or duplicated")
    current_count = int(count_matches[0])
    if not any(data_markers):
        if current_count != expected_before:
            fail(
                f"generator base count is {current_count}; expected {expected_before} "
                f"with inkling_present={inkling_present}"
            )
        count_line = f"EXPECTED_MODEL_COUNT = {current_count} + int(GEMINI25_RESULT is not None)"
        source = source.replace(
            count_line,
            GENERATOR_DATA_BLOCK
            + "\n"
            + f"EXPECTED_MODEL_COUNT = {expected_after} + int(GEMINI25_RESULT is not None)",
        )
    elif current_count != expected_after:
        fail(f"Gemma-marked generator count is {current_count}; expected {expected_after}")

    detail_markers = (DETAIL_START in source, DETAIL_END in source)
    if any(detail_markers) and not all(detail_markers):
        fail("Gemma generator detail markers are incomplete")
    if not any(detail_markers):
        anchor = "TURN_FAMILY_PATH = (Path(__file__).resolve().parents[1] /"
        source = replace_once(
            source, anchor, GENERATOR_DETAIL_BLOCK + "\n" + anchor, "Gemma detail insertion"
        )

    source = insert_prose_variable(
        source,
        "{gemini25_method}{laguna_method}{qwen_method}",
        "{INKLING_SMALL_METHOD_MARKDOWN}" if inkling_present else "",
        "{GEMMA26_METHOD_MARKDOWN}",
        " The other nine",
        "Markdown method",
    )
    source = insert_prose_variable(
        source,
        "{gemini25_provenance}{laguna_provenance}{qwen_provenance}",
        "{INKLING_SMALL_PROVENANCE_MARKDOWN}" if inkling_present else "",
        "{GEMMA26_PROVENANCE_MARKDOWN}",
        " The Qwen3-8B",
        "Markdown provenance",
    )
    for suffix, inkling_name, gemma_name, tail, label in (
        ("method_html", "INKLING_SMALL_METHOD_HTML", "GEMMA26_METHOD_HTML", " The nine", "HTML method"),
        ("limits_html", "INKLING_SMALL_LIMITS_HTML", "GEMMA26_LIMITS_HTML", " The nine", "HTML limits"),
        ("provenance_html", "INKLING_SMALL_PROVENANCE_HTML", "GEMMA26_PROVENANCE_HTML", " The nine", "HTML provenance"),
    ):
        source = insert_prose_variable(
            source,
            f"{{gemini25_{suffix}}}{{laguna_{suffix}}}{{qwen_{suffix}}}",
            f"{{{inkling_name}}}" if inkling_present else "",
            f"{{{gemma_name}}}",
            tail,
            label,
        )

    if SCOPE_START in source or SCOPE_END in source:
        if source.count(SCOPE_START) != 1 or source.count(SCOPE_END) != 1:
            fail("generator Markdown-scope markers are incomplete or duplicated")
    else:
        anchor = "def update_markdown_primary():\n    start = \"<!-- N30_PRIMARY_START -->\"\n    end = \"<!-- N30_PRIMARY_END -->\"\n    text = MARKDOWN_OUT.read_text()"
        source = replace_once(
            source,
            anchor,
            anchor + "\n" + GENERATOR_SCOPE_BLOCK,
            "Markdown scope update",
        )

    mapping_anchor = '        24: "Twenty-four",\n'
    if '        25: "Twenty-five",\n' not in source:
        source = replace_once(
            source,
            mapping_anchor,
            mapping_anchor + '        25: "Twenty-five",\n',
            "25-model word",
        )
    if '        26: "Twenty-six",\n' not in source:
        anchor = '        25: "Twenty-five",\n'
        source = replace_once(
            source,
            anchor,
            anchor + '        26: "Twenty-six",\n',
            "26-model word",
        )
    compile(source, str(GENERATOR_PATH), "exec")
    return source, 26 if inkling_present else 25


def transform_verifier(source: str, expected_count: int, path: Path) -> str:
    """Retarget exact historical screen-count checks to the additive final screen."""
    word = {25: "Twenty-five", 26: "Twenty-six"}[expected_count]
    substitutions = (
        (r"len\(table_lines\) != (?:26|27|28)", f"len(table_lines) != {expected_count + 2}"),
        (r'section\.count\("<tr><td>"\) != (?:24|25|26)', f'section.count("<tr><td>") != {expected_count}'),
        (r"should have (?:24|25|26) (?:data |model )?rows", lambda m: m.group(0).replace(re.search(r"\d+", m.group(0)).group(0), str(expected_count), 1)),
        (r"does not contain (?:24|25|26) rows", f"does not contain {expected_count} rows"),
        (r"(?:24|25|26)-row report", f"{expected_count}-row report"),
        (r"(?:24|25|26) model rows", f"{expected_count} model rows"),
        (r"(?:24|25|26) rows", f"{expected_count} rows"),
        (r"a (?:24|25|26)-Model Exploratory Study", f"a {expected_count}-Model Exploratory Study"),
        (r"(?:Twenty-four|Twenty-five|Twenty-six)-model exploratory screen", f"{word}-model exploratory screen"),
        (r"\*\*Scope:\*\* (?:Twenty-four|Twenty-five|Twenty-six) standard filler comparisons", f"**Scope:** {word} standard filler comparisons"),
        (r"updated to (?:24|25|26)(?: models)?", f"updated to {expected_count}"),
    )
    transformed = source
    changed = 0
    for pattern, replacement in substitutions:
        transformed, count = re.subn(pattern, replacement, transformed)
        changed += count
    if changed == 0:
        fail(f"current verifier has no recognized report-count checks: {path}")
    compile(transformed, str(path), "exec")
    return transformed


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


def section_three(text: str) -> str:
    match = re.search(r'<section id="primary-screen">(.*?)</section>', text, re.DOTALL)
    if not match:
        fail("HTML Section 3 is missing")
    return match.group(1)


def strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def html_table(section: str) -> tuple[list[str], list[list[str]]]:
    match = re.search(r"<table>(.*?)</table>", section, re.DOTALL)
    if not match:
        fail("HTML Section 3 table is missing")
    rows = re.findall(r"<tr>(.*?)</tr>", match.group(1), re.DOTALL)
    parsed = [
        [strip_tags(cell) for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)]
        for row in rows
    ]
    if not parsed:
        fail("HTML Section 3 table is empty")
    return parsed[0], parsed[1:]


def chart_labels(section: str) -> list[str]:
    figure = re.search(r"<figure>.*?<svg .*?</svg>", section, re.DOTALL)
    if not figure:
        fail("HTML Section 3 chart is missing")
    return [
        strip_tags(value)
        for value in re.findall(
            r'<text x="0"[^>]*class="lbl"[^>]*>(.*?)</text>', figure.group(0), re.DOTALL
        )
    ]


def validate_html(before: str, after: str, data: PublicationData, expected_count: int) -> None:
    before_section = section_three(before)
    after_section = section_three(after)
    before_header, before_rows = html_table(before_section)
    after_header, after_rows = html_table(after_section)
    if before_header != after_header or len(after_header) != 9 or any(len(row) != 9 for row in after_rows):
        fail("HTML Section 3 table shape changed")
    before_names = [row[0] for row in before_rows]
    after_names = [row[0] for row in after_rows]
    if [name for name in before_names if name != REPORT_NAME] != [
        name for name in after_names if name != REPORT_NAME
    ]:
        fail("HTML update did not preserve every existing Section 3 row and order")
    if after_names.count(REPORT_NAME) != 1 or len(after_names) != expected_count:
        fail("HTML final model count or Gemma row presence is wrong")
    before_labels = chart_labels(before_section)
    after_labels = chart_labels(after_section)
    if [name for name in before_labels if name != REPORT_NAME] != [
        name for name in after_labels if name != REPORT_NAME
    ]:
        fail("HTML update did not preserve every existing chart label and order")
    if after_labels.count(REPORT_NAME) != 1:
        fail("HTML chart must contain exactly one Gemma label")
    row = after_rows[after_names.index(REPORT_NAME)]
    screen = data.normalized["screen_row"]
    if (
        row[1] != PROVIDER
        or row[2] != f"{screen['no_filler_pass_rate_pct']:.1f}"
        or row[3] != f"{screen['dots_pass_rate_pct']:.1f}"
        or row[6] != str(round(screen["no_filler_ttfat_p50_ms"]))
        or row[7] != f"{data.n} / {data.n}"
    ):
        fail(f"HTML Gemma row does not match final inputs: {row}")
    word = {25: "Twenty-five", 26: "Twenty-six"}[expected_count]
    if f"{word}-model exploratory screen" not in after_section:
        fail("HTML Section 3 model-count word is stale")
    # This campaign-design sentence renders in Section 2's method definition;
    # Section 3 contains the row and screen-specific provenance.  Validate its
    # unique presence across the complete report, not within the screen alone.
    if after.count("Gemma 4 26B A4B adds a separate fixed-denominator") != 1:
        fail("HTML Gemma explanatory prose is missing or duplicated")


def markdown_primary(text: str) -> str:
    if text.count("<!-- N30_PRIMARY_START -->") != 1 or text.count("<!-- N30_PRIMARY_END -->") != 1:
        fail("Markdown primary markers are missing or duplicated")
    return text.split("<!-- N30_PRIMARY_START -->", 1)[1].split("<!-- N30_PRIMARY_END -->", 1)[0]


def validate_markdown(before: str, after: str, expected_count: int) -> None:
    before_primary = markdown_primary(before)
    after_primary = markdown_primary(after)
    before_rows = [line for line in before_primary.splitlines() if line.startswith("|")][2:]
    after_rows = [line for line in after_primary.splitlines() if line.startswith("|")][2:]
    before_names = [markdown_cells(row)[0] for row in before_rows]
    after_names = [markdown_cells(row)[0] for row in after_rows]
    if [name for name in before_names if name != REPORT_NAME] != [
        name for name in after_names if name != REPORT_NAME
    ]:
        fail("Markdown update did not preserve every existing Section 3 row and order")
    if after_names.count(REPORT_NAME) != 1 or len(after_names) != expected_count:
        fail("Markdown final model count or Gemma row presence is wrong")
    word = {25: "Twenty-five", 26: "Twenty-six"}[expected_count]
    if f"**Scope:** {word} standard filler comparisons" not in after:
        fail("Markdown Scope model-count word is stale")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def generator_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")
    return {name: os.environ[name] for name in allowed if name in os.environ}


def apply_update(
    data: PublicationData,
    readme_after: str,
    generator_after: str,
    verifier_afters: dict[Path, str],
    expected_count: int,
) -> None:
    before_html = HTML_PATH.read_text(encoding="utf-8")
    before_markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    with LOCK_PATH.open("a", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another Gemma publication process owns the lock")
        current = load_publication_data()
        if current.normalized != data.normalized:
            fail("final Gemma publication inputs changed after preflight")
        if update_readme(README_PATH.read_text(encoding="utf-8"), current) != readme_after:
            fail("README changed after preflight")
        current_generator_after, current_count = transform_generator(
            GENERATOR_PATH.read_text(encoding="utf-8")
        )
        if current_generator_after != generator_after or current_count != expected_count:
            fail("canonical report generator changed after preflight")
        for path, expected in verifier_afters.items():
            if transform_verifier(path.read_text(encoding="utf-8"), expected_count, path) != expected:
                fail(f"current verifier changed after preflight: {path}")

        atomic_write(NORMALIZED_PATH, normalized_text(current))
        atomic_write(GENERATOR_PATH, generator_after)
        result = subprocess.run(
            [sys.executable, str(GENERATOR_PATH)],
            cwd=ROOT,
            env=generator_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            fail("canonical local report rebuild failed:\n" + result.stdout)
        after_html = HTML_PATH.read_text(encoding="utf-8")
        after_markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
        validate_html(before_html, after_html, current, expected_count)
        validate_markdown(before_markdown, after_markdown, expected_count)
        atomic_write(README_PATH, readme_after)
        for path, text in verifier_afters.items():
            atomic_write(path, text)
        validate_readme(README_PATH.read_text(encoding="utf-8"), current)
        if transform_generator(GENERATOR_PATH.read_text(encoding="utf-8"))[0] != generator_after:
            fail("Gemma generator transform is not idempotent")
        for path, expected in verifier_afters.items():
            if transform_verifier(path.read_text(encoding="utf-8"), expected_count, path) != expected:
                fail(f"verifier transform is not idempotent: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply only after final review")
    args = parser.parse_args()
    data = load_publication_data()
    readme_before = README_PATH.read_text(encoding="utf-8")
    generator_before = GENERATOR_PATH.read_text(encoding="utf-8")
    readme_after = update_readme(readme_before, data)
    generator_after, expected_count = transform_generator(generator_before)
    verifier_befores = {path: path.read_text(encoding="utf-8") for path in VERIFIER_PATHS}
    verifier_afters = {
        path: transform_verifier(source, expected_count, path)
        for path, source in verifier_befores.items()
    }
    publication_before = NORMALIZED_PATH.read_text(encoding="utf-8") if NORMALIZED_PATH.is_file() else ""
    publication_after = normalized_text(data)
    if not args.apply:
        for path, before, after in (
            (README_PATH, readme_before, readme_after),
            (GENERATOR_PATH, generator_before, generator_after),
            (NORMALIZED_PATH, publication_before, publication_after),
        ):
            print(unified_diff(path, before, after), end="")
        for path in VERIFIER_PATHS:
            print(unified_diff(path, verifier_befores[path], verifier_afters[path]), end="")
        print(
            f"Dry run only: stage={data.stage}, n={data.n}/arm, expected_screen_rows={expected_count}, "
            f"nofiller_ttfat={data.normalized['screen_row']['no_filler_ttfat_p50_ms']:.0f}ms. "
            "README, generator, reports, and verifiers were not modified."
        )
        return 0
    apply_update(data, readme_after, generator_after, verifier_afters, expected_count)
    print("Gemma publication applied and validated through the canonical report generator.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
