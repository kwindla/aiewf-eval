#!/usr/bin/env python3
"""Exploratory turn-family decomposition for the eleven focused filler cells."""

from __future__ import annotations

import csv
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N_TURNS = 30
N_PER_ARM = 30
BOOTSTRAPS = 100_000
SEED = 20260722
N30_DIR = HERE.parent / "dot-stability-n30-2026-07-20"
GEMINI_DIR = HERE.parent / "gemini-minimal-dots-2026-07-21"
FROZEN_INPUT_HASHES = {
    ROOT / "benchmarks/_shared/turns.py": "c88da69f8ade0e04e943b7493629ff96481d2779c001be7f77f0de82fbdc456b",
    ROOT / "benchmarks/aiwf_medium_context/prompts/system.py": "6003f0f482c757a9bec6ed01e2993c7192112984e2037cf79d830bd46d76e9a6",
    N30_DIR / "analyze.py": "3d9094da5c9858554baf9760eec9bbba786e71f72de64e362cfde9ce814dfe70",
    N30_DIR / "aggregates.json": "573e53779774f61c8cc9641d553c02c2368c56a2785fddc87071cdb5c22a1d99",
    GEMINI_DIR / "analyze.py": "aa7b2ed23cb5cb5ec612626f3aa788f85d6f8b5af286c8eed991ae165ab8d2ee",
    GEMINI_DIR / "aggregates.json": "41be324032aaecffd03b3e43ffa35242a3e9b19c82c404093138f5905a7ecff2",
    HERE / "turn-families.json": "058ea3ada0d087ddd2afad5a4d02b9120e5a14c2a28e6ab952fdadc67f3e946e",
    HERE / "source-manifest.tsv": "5a0a222e8f0d4f5e4297ff618407d2dab2416feea5850c8cce4362e741b1fffb",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import campaign analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def observed_turn_mask(run) -> np.ndarray:
    observed = np.zeros(N_TURNS, dtype=bool)
    for line in (run.run_dir / "transcript.jsonl").read_text().splitlines():
        row = json.loads(line)
        turn = row.get("turn")
        if (
            isinstance(turn, int)
            and 0 <= turn < N_TURNS
            and row.get("recovery_turn") is not True
        ):
            observed[turn] = True
    return observed


def validate_source_manifest(model_specs: list[dict]) -> None:
    expected: list[dict[str, str]] = []
    for spec in model_specs:
        for arm in ("nofiller", "dots96"):
            for run_index, run in enumerate(spec["cells"][(spec["key"], arm)], start=1):
                run_dir = run.run_dir.resolve()
                expected.append({
                    "campaign": spec["source_campaign"],
                    "model": spec["key"],
                    "arm": arm,
                    "run_index": str(run_index),
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "transcript_sha256": hashlib.sha256((run_dir / "transcript.jsonl").read_bytes()).hexdigest(),
                    "judgment_sha256": hashlib.sha256((run_dir / "claude_judged.jsonl").read_bytes()).hexdigest(),
                })
    with (HERE / "source-manifest.tsv").open(newline="") as handle:
        frozen = list(csv.DictReader(handle, delimiter="\t"))
    if len(expected) != 660 or expected != frozen:
        raise ValueError("current source pool or run content differs from the frozen 660-run manifest")


def main() -> None:
    for path, expected in FROZEN_INPUT_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen secondary-analysis input changed: {path}")
    taxonomy = json.loads((HERE / "turn-families.json").read_text())
    families = taxonomy.get("families", [])
    keys = [family.get("key") for family in families]
    turns = [turn for family in families for turn in family.get("turns", [])]
    if len(keys) != len(set(keys)) or sorted(turns) != list(range(N_TURNS)):
        raise ValueError("turn-family taxonomy must be unique and exhaustive over turns 0-29")

    n30_module = load_module("turn_family_n30_source", N30_DIR / "analyze.py")
    gemini_module = load_module("turn_family_gemini_source", GEMINI_DIR / "analyze.py")
    n30_cells = n30_module.load_all()
    gemini_cells = gemini_module.load_all()
    n30_aggregate = json.loads((N30_DIR / "aggregates.json").read_text())
    gemini_aggregate = json.loads((GEMINI_DIR / "aggregates.json").read_text())
    if gemini_aggregate.get("artifact_status") != "FINAL":
        raise ValueError("Gemini source aggregate is not final")

    model_specs: list[dict] = []
    for source_key in n30_module.MODELS:
        display, provider = n30_module.DISPLAY[source_key]
        model_specs.append({
            "key": source_key,
            "display_name": display,
            "provider": provider,
            "source_campaign": "dot-stability-n30-2026-07-20",
            "cells": n30_cells,
            "source_result": n30_aggregate["models"][source_key],
        })
    for source_key in gemini_module.MODELS:
        display, _version = gemini_module.DISPLAY[source_key]
        model_specs.append({
            "key": source_key,
            "display_name": display,
            "provider": "Google",
            "source_campaign": "gemini-minimal-dots-2026-07-21",
            "cells": gemini_cells,
            "source_result": gemini_aggregate["models"][source_key],
        })
    if len(model_specs) != 11 or len({row["display_name"] for row in model_specs}) != 11:
        raise ValueError("secondary analysis requires eleven unique focused models")
    validate_source_manifest(model_specs)

    output = {
        "schema_version": 2,
        "artifact_status": "FINAL_EXPLORATORY_SECONDARY",
        "protocol": {
            "benchmark": "aiwf_medium_context",
            "turns": N_TURNS,
            "n_per_arm": N_PER_ARM,
            "bootstrap_unit": "whole conversation",
            "bootstrap_samples": BOOTSTRAPS,
            "seed": SEED,
            "intervals": "pointwise unadjusted 95% percentile bootstrap",
            "turn_estimand": "dots96 minus nofiller strict-pass rate on each fixed scripted turn",
            "missing_delta_sign": "dots96 minus nofiller missing-turn rate",
            "heatmap_missing_sign": "nofiller minus dots96 missing-turn rate (benefit-aligned)",
            "primary_estimand_unchanged": True,
            "model_order": [row["key"] for row in model_specs],
            "input_sha256": {
                str(path.relative_to(ROOT)): digest
                for path, digest in FROZEN_INPUT_HASHES.items()
            },
        },
        "taxonomy": taxonomy,
        "models": {},
    }

    for model_index, spec in enumerate(model_specs):
        key = spec["key"]
        source_result = spec["source_result"]
        control_runs = spec["cells"][(key, "nofiller")]
        dots_runs = spec["cells"][(key, "dots96")]
        if len(control_runs) != N_PER_ARM or len(dots_runs) != N_PER_ARM:
            raise ValueError(f"{key} does not have 30 conversations per arm")
        run_dirs = [str(run.run_dir.resolve()) for run in (*control_runs, *dots_runs)]
        if len(run_dirs) != len(set(run_dirs)):
            raise ValueError(f"duplicate included conversation for {key}")

        control = np.asarray([run.passed for run in control_runs], dtype=float)
        dots = np.asarray([run.passed for run in dots_runs], dtype=float)
        control_observed = np.asarray([observed_turn_mask(run) for run in control_runs], dtype=bool)
        dots_observed = np.asarray([observed_turn_mask(run) for run in dots_runs], dtype=bool)
        if control.shape != (N_PER_ARM, N_TURNS) or dots.shape != (N_PER_ARM, N_TURNS):
            raise ValueError(f"unexpected score-matrix shape for {key}")
        if np.any(control.astype(bool) & ~control_observed) or np.any(dots.astype(bool) & ~dots_observed):
            raise ValueError(f"a missing turn cannot have a passing judgment for {key}")
        recomputed_control = float(control.mean() * 100)
        recomputed_dots = float(dots.mean() * 100)
        published_control = source_result["arms"]["nofiller"]["pass_rate_pct"]
        published_dots = source_result["arms"]["dots96"]["pass_rate_pct"]
        if not np.isclose(recomputed_control, published_control) or not np.isclose(recomputed_dots, published_dots):
            raise ValueError(f"source aggregate mismatch for {key}")

        rng = np.random.default_rng(SEED + model_index)
        control_idx = rng.integers(0, N_PER_ARM, size=(BOOTSTRAPS, N_PER_ARM))
        dots_idx = rng.integers(0, N_PER_ARM, size=(BOOTSTRAPS, N_PER_ARM))
        control_overall = control.mean(axis=1)
        dots_overall = dots.mean(axis=1)
        boot_overall = (
            dots_overall[dots_idx].mean(axis=1)
            - control_overall[control_idx].mean(axis=1)
        ) * 100

        turn_to_family = {
            turn: family["key"]
            for family in families
            for turn in family["turns"]
        }
        turn_results: list[dict] = []
        boot_turn_deltas = np.empty((BOOTSTRAPS, N_TURNS), dtype=float)
        for turn in range(N_TURNS):
            control_pass = control[:, turn]
            dots_pass = dots[:, turn]
            control_missing = ~control_observed[:, turn]
            dots_missing = ~dots_observed[:, turn]
            control_observed_failure = (
                control_observed[:, turn] & ~control_pass.astype(bool)
            )
            dots_observed_failure = (
                dots_observed[:, turn] & ~dots_pass.astype(bool)
            )
            for arm, passed, missing, observed_failure in (
                ("nofiller", control_pass, control_missing, control_observed_failure),
                ("dots96", dots_pass, dots_missing, dots_observed_failure),
            ):
                if int(passed.sum()) + int(missing.sum()) + int(observed_failure.sum()) != N_PER_ARM:
                    raise ValueError(f"turn failure decomposition mismatch for {key}/{arm}/{turn}")

            boot_control = control_pass[control_idx].mean(axis=1) * 100
            boot_dots = dots_pass[dots_idx].mean(axis=1) * 100
            boot_delta = boot_dots - boot_control
            boot_turn_deltas[:, turn] = boot_delta
            control_rate = float(control_pass.mean() * 100)
            dots_rate = float(dots_pass.mean() * 100)
            missing_delta = float((dots_missing.mean() - control_missing.mean()) * 100)
            observed_failure_delta = float(
                (dots_observed_failure.mean() - control_observed_failure.mean()) * 100
            )
            pass_delta = dots_rate - control_rate
            if not np.isclose(pass_delta, -missing_delta - observed_failure_delta):
                raise ValueError(f"turn effect decomposition mismatch for {key}/{turn}")
            turn_results.append({
                "turn": turn,
                "family_key": turn_to_family[turn],
                "n_conversations_per_arm": N_PER_ARM,
                "fixed_turn_denominator_per_arm": N_PER_ARM,
                "nofiller_pass_count": int(control_pass.sum()),
                "nofiller_pass_rate_pct": control_rate,
                "nofiller_missing_turn_count": int(control_missing.sum()),
                "nofiller_missing_turn_rate_pct": float(control_missing.mean() * 100),
                "nofiller_observed_failure_count": int(control_observed_failure.sum()),
                "nofiller_observed_failure_rate_pct": float(control_observed_failure.mean() * 100),
                "dots96_pass_count": int(dots_pass.sum()),
                "dots96_pass_rate_pct": dots_rate,
                "dots96_missing_turn_count": int(dots_missing.sum()),
                "dots96_missing_turn_rate_pct": float(dots_missing.mean() * 100),
                "dots96_observed_failure_count": int(dots_observed_failure.sum()),
                "dots96_observed_failure_rate_pct": float(dots_observed_failure.mean() * 100),
                "pass_delta_points": pass_delta,
                "pass_delta_ci95": percentile_interval(boot_delta),
                "missing_turn_rate_delta_points": missing_delta,
                "aligned_missing_contribution_points": -missing_delta,
                "observed_failure_rate_delta_points": observed_failure_delta,
                "aligned_observed_failure_contribution_points": -observed_failure_delta,
            })

        if not np.allclose(boot_turn_deltas.mean(axis=1), boot_overall):
            raise ValueError(f"bootstrap turn effects do not reconstruct the overall effect for {key}")

        family_results: dict[str, dict] = {}
        boot_contribution_sum = np.zeros(BOOTSTRAPS)
        contribution_sum = 0.0
        for family in families:
            family_key = family["key"]
            family_turns = family["turns"]
            weight = len(family_turns) / N_TURNS
            control_conversation = control[:, family_turns].mean(axis=1)
            dots_conversation = dots[:, family_turns].mean(axis=1)
            control_missing = (~control_observed[:, family_turns]).mean(axis=1)
            dots_missing = (~dots_observed[:, family_turns]).mean(axis=1)
            control_observed_failure = (
                control_observed[:, family_turns] & ~control[:, family_turns].astype(bool)
            ).mean(axis=1)
            dots_observed_failure = (
                dots_observed[:, family_turns] & ~dots[:, family_turns].astype(bool)
            ).mean(axis=1)
            if not np.allclose(1 - control_conversation, control_missing + control_observed_failure):
                raise ValueError(f"control failure decomposition mismatch for {key}/{family_key}")
            if not np.allclose(1 - dots_conversation, dots_missing + dots_observed_failure):
                raise ValueError(f"dots failure decomposition mismatch for {key}/{family_key}")
            boot_control = control_conversation[control_idx].mean(axis=1) * 100
            boot_dots = dots_conversation[dots_idx].mean(axis=1) * 100
            boot_delta = boot_dots - boot_control
            if not np.allclose(boot_delta, boot_turn_deltas[:, family_turns].mean(axis=1)):
                raise ValueError(f"bootstrap turn effects do not reconstruct {key}/{family_key}")
            control_rate = float(control_conversation.mean() * 100)
            dots_rate = float(dots_conversation.mean() * 100)
            delta = dots_rate - control_rate
            contribution = delta * weight
            boot_contribution = boot_delta * weight
            contribution_sum += contribution
            boot_contribution_sum += boot_contribution
            family_results[family_key] = {
                "label": family["label"],
                "short_label": family["short_label"],
                "turns": family_turns,
                "n_turns": len(family_turns),
                "turn_weight": weight,
                "n_conversations_per_arm": N_PER_ARM,
                "fixed_turn_denominator_per_arm": N_PER_ARM * len(family_turns),
                "nofiller_pass_count": int(control[:, family_turns].sum()),
                "nofiller_pass_rate_pct": control_rate,
                "nofiller_pass_rate_ci95": percentile_interval(boot_control),
                "dots96_pass_count": int(dots[:, family_turns].sum()),
                "dots96_pass_rate_pct": dots_rate,
                "dots96_pass_rate_ci95": percentile_interval(boot_dots),
                "conditional_delta_points": delta,
                "conditional_delta_ci95": percentile_interval(boot_delta),
                "overall_contribution_points": contribution,
                "overall_contribution_ci95": percentile_interval(boot_contribution),
                "nofiller_missing_turn_rate_pct": float(control_missing.mean() * 100),
                "dots96_missing_turn_rate_pct": float(dots_missing.mean() * 100),
                "missing_turn_rate_delta_points": float((dots_missing.mean() - control_missing.mean()) * 100),
                "nofiller_observed_failure_rate_pct": float(control_observed_failure.mean() * 100),
                "dots96_observed_failure_rate_pct": float(dots_observed_failure.mean() * 100),
            }

        published_effect = source_result["effect"]["pass_delta_points"]
        if not np.isclose(contribution_sum, published_effect):
            raise ValueError(f"family contributions do not reconstruct the overall effect for {key}")
        if not np.allclose(boot_contribution_sum, boot_overall):
            raise ValueError(f"bootstrap family contributions do not reconstruct the overall effect for {key}")
        dominant = max(
            family_results,
            key=lambda family_key: abs(family_results[family_key]["overall_contribution_points"]),
        )
        output["models"][key] = {
            "display_name": spec["display_name"],
            "provider": spec["provider"],
            "source_campaign": spec["source_campaign"],
            "overall": {
                "nofiller_pass_rate_pct": published_control,
                "dots96_pass_rate_pct": published_dots,
                "delta_points": published_effect,
                "published_delta_ci95": source_result["effect"]["pass_delta_ci95"],
                "secondary_recomputed_delta_ci95": percentile_interval(boot_overall),
                "reconstructed_from_family_contributions": contribution_sum,
            },
            "dominant_absolute_contribution_family": dominant,
            "turns": turn_results,
            "families": family_results,
        }

    (HERE / "aggregates.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
