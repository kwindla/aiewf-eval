#!/usr/bin/env python3
"""Build an isolated provisional aggregate for the seven completed focus models.

This intentionally does not write the final ``aggregates.json``/``aggregates.tsv``
consumed by the README and report builders. Qwen is pending and is neither read nor
represented here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import analyze as primary


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_DIR = HERE / "provisional-seven"
JSON_OUT = OUT_DIR / "PROVISIONAL-aggregates.json"
TSV_OUT = OUT_DIR / "PROVISIONAL-aggregates.tsv"
INCLUDED_OUT = OUT_DIR / "PROVISIONAL-included-runs.tsv"

MODELS = tuple(model for model in primary.MODELS if model != primary.QWEN_MODEL)
LANES = primary.PRIMARY_LANES
EXPECTED_HISTORICAL = 174
EXPECTED_TOPUPS = 246
EXPECTED_TOTAL = 420


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_invalidated() -> set[tuple[str, str, str]]:
    path = HERE / "invalidated.tsv"
    invalidated: set[tuple[str, str, str]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            invalidated.add((row["lane"], row["slot"], row["attempt"]))
    return invalidated


def load_refs() -> tuple[list[dict[str, str]], dict[str, str]]:
    if not (HERE / "state" / "existing-judge" / "COMPLETE").is_file():
        raise ValueError("historical judgment lane is not complete")
    for lane in LANES:
        if not (HERE / "state" / lane / "COMPLETE").is_file():
            raise ValueError(f"completed-model lane is not complete: {lane}")

    source_paths = [HERE / "existing-included.tsv", HERE / "invalidated.tsv", HERE / "analyze.py"]
    invalidated = load_invalidated()
    refs: list[dict[str, str]] = []
    skipped_historical_qwen = 0

    with (HERE / "existing-included.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["model"] == primary.QWEN_MODEL:
                skipped_historical_qwen += 1
                continue
            if row["model"] not in MODELS:
                raise ValueError(f"unexpected historical model: {row['model']}")
            refs.append(
                {
                    "model": row["model"],
                    "arm": row["arm"],
                    "source_kind": "historical",
                    "lane": "existing-included",
                    "slot": "",
                    "attempt": "",
                    "run_dir": str(ROOT / row["run_dir"]),
                }
            )
    if skipped_historical_qwen != 16:
        raise ValueError(f"expected 16 superseded historical Qwen attempts, found {skipped_historical_qwen}")

    raw_topups = 0
    invalid_topups = 0
    for lane in LANES:
        counted = HERE / "state" / lane / "counted.tsv"
        source_paths.append(counted)
        with counted.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw_topups += 1
                if row["model"] == primary.QWEN_MODEL:
                    raise ValueError(f"Qwen attempt unexpectedly present in completed lane {lane}")
                if row["model"] not in MODELS:
                    raise ValueError(f"unexpected top-up model in {lane}: {row['model']}")
                if (lane, row["slot"], row["attempt"]) in invalidated:
                    invalid_topups += 1
                    continue
                run_dir = Path(row["run_dir"])
                refs.append(
                    {
                        "model": row["model"],
                        "arm": row["arm"],
                        "source_kind": "topup",
                        "lane": lane,
                        "slot": row["slot"],
                        "attempt": row["attempt"],
                        "run_dir": str(run_dir if run_dir.is_absolute() else ROOT / run_dir),
                    }
                )

    source_counts = Counter(row["source_kind"] for row in refs)
    if source_counts != {"historical": EXPECTED_HISTORICAL, "topup": EXPECTED_TOPUPS}:
        raise ValueError(f"provisional source-count mismatch: {dict(source_counts)}")
    if raw_topups - invalid_topups != EXPECTED_TOPUPS:
        raise ValueError(
            f"top-up validity mismatch: raw={raw_topups}, invalidated={invalid_topups}, "
            f"effective={raw_topups - invalid_topups}"
        )
    if len(refs) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} seven-model conversations, found {len(refs)}")

    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    return refs, hashes


def load_cells(refs: list[dict[str, str]]) -> dict[tuple[str, str], list[primary.Conversation]]:
    cells: dict[tuple[str, str], list[primary.Conversation]] = defaultdict(list)
    seen: set[Path] = set()
    for ref in refs:
        run_dir = Path(ref["run_dir"]).resolve()
        if run_dir in seen:
            raise ValueError(f"duplicate provisional run: {run_dir}")
        seen.add(run_dir)
        cells[(ref["model"], ref["arm"])].append(
            primary.load_conversation(ref["model"], ref["arm"], run_dir)
        )

    expected_cells = {(model, arm) for model in MODELS for arm in primary.ARMS}
    if set(cells) != expected_cells:
        raise ValueError(
            f"provisional cell mismatch: missing={sorted(expected_cells - set(cells))}, "
            f"unexpected={sorted(set(cells) - expected_cells)}"
        )
    for model, arm in sorted(expected_cells):
        if len(cells[(model, arm)]) != primary.TARGET:
            raise ValueError(f"expected {primary.TARGET} attempts for {model}/{arm}")
    if len(seen) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} unique run directories, found {len(seen)}")
    return cells


def summarize(cells: dict[tuple[str, str], list[primary.Conversation]]) -> dict:
    output = {
        "artifact_status": "PROVISIONAL_DO_NOT_PUBLISH",
        "scope": "seven completed focus models; Qwen3-8B pending and excluded",
        "protocol": {
            "target_per_arm": primary.TARGET,
            "turns": primary.N_TURNS,
            "bootstrap_samples": primary.BOOTSTRAPS,
            "seed": primary.SEED,
            "included_models": list(MODELS),
            "excluded_pending_models": [primary.QWEN_MODEL],
            "eligible_conversations": EXPECTED_TOTAL,
            "historical_attempts_included": EXPECTED_HISTORICAL,
            "topup_attempts_included": EXPECTED_TOPUPS,
            "safe_for_readme_or_report": False,
        },
        "models": {},
    }
    for model in MODELS:
        # Preserve the primary analyzer's model-index seeds so these rows reproduce
        # exactly when Qwen is later added to the final eight-model artifact.
        model_index = primary.MODELS.index(model)
        summaries: dict[str, dict] = {}
        for arm_index, arm in enumerate(primary.ARMS):
            rng = np.random.default_rng(primary.SEED + 100 * model_index + arm_index)
            summaries[arm] = primary.arm_summary(cells[(model, arm)], rng)
        control = summaries["nofiller"]
        dots = summaries["dots96"]
        rng = np.random.default_rng(primary.SEED + 10_000 + model_index)
        control_conversation_pass = control.pop("_conversation_pass")
        dots_conversation_pass = dots.pop("_conversation_pass")
        control_idx = rng.integers(
            0,
            len(control_conversation_pass),
            size=(primary.BOOTSTRAPS, len(control_conversation_pass)),
        )
        dots_idx = rng.integers(
            0,
            len(dots_conversation_pass),
            size=(primary.BOOTSTRAPS, len(dots_conversation_pass)),
        )
        boot_delta = (
            dots_conversation_pass[dots_idx].mean(axis=1)
            - control_conversation_pass[control_idx].mean(axis=1)
        ) * 100
        delta = dots["pass_rate_pct"] - control["pass_rate_pct"]
        display, provider = primary.DISPLAY[model]
        output["models"][model] = {
            "display_name": display,
            "provider": provider,
            "arms": summaries,
            "effect": {
                "pass_delta_points": delta,
                "pass_delta_ci95": [
                    float(np.percentile(boot_delta, 2.5)),
                    float(np.percentile(boot_delta, 97.5)),
                ],
                "any_error_reduction_points": delta,
                "any_error_reduction_ci95": [
                    float(np.percentile(boot_delta, 2.5)),
                    float(np.percentile(boot_delta, 97.5)),
                ],
            },
        }
    return output


def render_aggregate_tsv(output: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    header = ["artifact_status", "model", "provider", "n_per_arm"]
    for arm in primary.ARMS:
        for metric in ("pass_rate", "any_error_rate", "tool_error_rate", "instruction_error_rate", "kb_error_rate"):
            header.extend(
                [f"{arm}_{metric}_pct", f"{arm}_{metric}_ci95_low", f"{arm}_{metric}_ci95_high"]
            )
        header.append(f"{arm}_strict_completion_pct")
    header.extend(
        [
            "delta_points",
            "delta_ci95_low",
            "delta_ci95_high",
            "nofiller_ttfat_p50_ms",
            "nofiller_ttfat_p95_ms",
            "nofiller_ttfat_max_ms",
        ]
    )
    writer.writerow(header)
    for model in MODELS:
        result = output["models"][model]
        control = result["arms"]["nofiller"]
        dots = result["arms"]["dots96"]
        row: list[object] = [
            output["artifact_status"],
            result["display_name"],
            result["provider"],
            control["n_attempts"],
        ]
        for arm_summary in (control, dots):
            for metric in ("pass_rate", "any_error_rate", "tool_error_rate", "instruction_error_rate", "kb_error_rate"):
                row.extend(
                    [
                        arm_summary[f"{metric}_pct"],
                        arm_summary[f"{metric}_ci95"][0],
                        arm_summary[f"{metric}_ci95"][1],
                    ]
                )
            row.append(arm_summary["strict_completion_pct"])
        effect = result["effect"]
        row.extend(
            [
                effect["pass_delta_points"],
                effect["pass_delta_ci95"][0],
                effect["pass_delta_ci95"][1],
                control["ttfat_p50_ms"],
                control["ttfat_p95_ms"],
                control["ttfat_max_ms"],
            ]
        )
        writer.writerow(row)
    return buffer.getvalue()


def render_included_tsv(refs: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    fields = ("model", "arm", "source_kind", "lane", "slot", "attempt", "run_dir")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for ref in refs:
        row = dict(ref)
        row["run_dir"] = str(Path(row["run_dir"]).resolve().relative_to(ROOT))
        writer.writerow(row)
    return buffer.getvalue()


def main() -> None:
    refs, source_hashes = load_refs()
    cells = load_cells(refs)
    output = summarize(cells)
    output["protocol"]["source_sha256"] = source_hashes
    by_source = Counter(ref["source_kind"] for ref in refs)
    by_cell = Counter((ref["model"], ref["arm"], ref["source_kind"]) for ref in refs)
    output["protocol"]["source_counts"] = dict(sorted(by_source.items()))
    output["protocol"]["source_counts_by_cell"] = {
        f"{model}/{arm}/{source}": count
        for (model, arm, source), count in sorted(by_cell.items())
    }

    atomic_write(JSON_OUT, json.dumps(output, indent=2) + "\n")
    atomic_write(TSV_OUT, render_aggregate_tsv(output))
    atomic_write(INCLUDED_OUT, render_included_tsv(refs))
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"wrote {TSV_OUT.relative_to(ROOT)}")
    print(f"wrote {INCLUDED_OUT.relative_to(ROOT)}")
    print(f"included conversations: {len(refs)} ({by_source['historical']} historical + {by_source['topup']} top-up)")


if __name__ == "__main__":
    main()
