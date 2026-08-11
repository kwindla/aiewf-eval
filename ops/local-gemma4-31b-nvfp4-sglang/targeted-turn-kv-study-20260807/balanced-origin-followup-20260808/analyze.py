#!/usr/bin/env python3
"""Analyze the preregistered balanced history-origin interaction."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import HERE, ORIGINS, REPLICATES, atomic_write_json, load_json


BOOTSTRAPS = 100_000
RNG_SEED = 20260808


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def cluster_bootstrap(
    left: np.ndarray, right: np.ndarray, *, two_stage: bool
) -> dict[str, list[float]]:
    """Bootstrap origin effects and their interaction in bounded chunks."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("bootstrap inputs must be history-by-seed matrices")
    rng = np.random.default_rng(RNG_SEED + int(two_stage))
    left_draws = np.empty(BOOTSTRAPS)
    right_draws = np.empty(BOOTSTRAPS)
    chunk = 1_000
    for start in range(0, BOOTSTRAPS, chunk):
        size = min(chunk, BOOTSTRAPS - start)
        left_history = rng.integers(0, left.shape[0], size=(size, left.shape[0]))
        right_history = rng.integers(0, right.shape[0], size=(size, right.shape[0]))
        if two_stage:
            left_seed = rng.integers(0, left.shape[1], size=(size, left.shape[0], left.shape[1]))
            right_seed = rng.integers(0, right.shape[1], size=(size, right.shape[0], right.shape[1]))
            left_values = left[left_history]
            right_values = right[right_history]
            left_draws[start : start + size] = np.take_along_axis(
                left_values, left_seed, axis=2
            ).mean(axis=(1, 2))
            right_draws[start : start + size] = np.take_along_axis(
                right_values, right_seed, axis=2
            ).mean(axis=(1, 2))
        else:
            left_means = left.mean(axis=1)
            right_means = right.mean(axis=1)
            left_draws[start : start + size] = left_means[left_history].mean(axis=1)
            right_draws[start : start + size] = right_means[right_history].mean(axis=1)
    return {
        "local_bf16_origin": percentile_interval(left_draws * 100),
        "local_fp8_origin": percentile_interval(right_draws * 100),
        "origin_interaction": percentile_interval((left_draws - right_draws) * 100),
        "balanced_origin_mixture": percentile_interval((left_draws + right_draws) * 50),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, default=HERE / "results/analysis.json")
    parser.add_argument("--output-md", type=Path, default=HERE / "results/analysis.md")
    args = parser.parse_args()

    manifest = load_json(HERE / "snapshot-manifest.json")
    seeds = load_json(HERE / "seed-manifest.json")
    entries = {entry["snapshot_id"]: entry for entry in manifest["entries"]}
    raw = read_rows(args.results)
    expected_rows = len(entries) * REPLICATES * 2
    if len(raw) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(raw)}")
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in raw:
        key = (row["arm"], row["snapshot_id"], int(row["seed"]))
        if key in index:
            raise RuntimeError(f"duplicate inferential cell: {key}")
        index[key] = row
    expected_keys = {
        (arm, snapshot_id, int(seed))
        for arm in ("bf16", "fp8")
        for snapshot_id, allocation in seeds["allocations"].items()
        for seed in allocation
    }
    if set(index) != expected_keys:
        missing = len(expected_keys - set(index))
        unexpected = len(set(index) - expected_keys)
        raise RuntimeError(
            f"result allocation differs from frozen design: missing={missing}, unexpected={unexpected}"
        )

    matrices: dict[str, list[list[int]]] = {origin: [] for origin in ORIGINS}
    histories = []
    category_counts: dict[str, dict[str, Counter[str]]] = {
        origin: {"bf16": Counter(), "fp8": Counter()} for origin in ORIGINS
    }
    cohorts: dict[str, dict[str, list[float]]] = {
        origin: defaultdict(list) for origin in ORIGINS
    }
    for snapshot_id, allocation in seeds["allocations"].items():
        entry = entries[snapshot_id]
        origin = entry["metadata"]["origin"]
        paired = []
        bf16_success = []
        fp8_success = []
        for seed in allocation:
            bf16 = index.get(("bf16", snapshot_id, int(seed)))
            fp8 = index.get(("fp8", snapshot_id, int(seed)))
            if bf16 is None or fp8 is None:
                raise RuntimeError(f"missing paired cell for {snapshot_id} seed {seed}")
            a = int(bool(bf16["score"]["success"]))
            b = int(bool(fp8["score"]["success"]))
            bf16_success.append(a)
            fp8_success.append(b)
            paired.append(a - b)
            category_counts[origin]["bf16"][bf16["score"]["category"]] += 1
            category_counts[origin]["fp8"][fp8["score"]["category"]] += 1
        if len(paired) != REPLICATES:
            raise RuntimeError(f"wrong seed count for {snapshot_id}: {len(paired)}")
        matrices[origin].append(paired)
        effect = float(np.mean(paired) * 100)
        cohort = entry["metadata"]["cohort"]
        cohorts[origin][cohort].append(effect)
        histories.append(
            {
                "snapshot_id": snapshot_id,
                "origin": origin,
                "cohort": cohort,
                "bf16_success_percent": float(np.mean(bf16_success) * 100),
                "fp8_success_percent": float(np.mean(fp8_success) * 100),
                "difference_points": effect,
            }
        )

    arrays = {origin: np.asarray(matrices[origin], dtype=float) for origin in ORIGINS}
    if any(array.shape != (150, REPLICATES) for array in arrays.values()):
        raise RuntimeError(f"unexpected balanced matrix shapes: { {k: v.shape for k, v in arrays.items()} }")
    effects = {origin: float(array.mean() * 100) for origin, array in arrays.items()}
    interaction = effects["local_bf16"] - effects["local_fp8"]
    mixture = (effects["local_bf16"] + effects["local_fp8"]) / 2
    cluster_ci = cluster_bootstrap(
        arrays["local_bf16"], arrays["local_fp8"], two_stage=False
    )
    two_stage_ci = cluster_bootstrap(
        arrays["local_bf16"], arrays["local_fp8"], two_stage=True
    )

    distribution = {}
    for origin in ORIGINS:
        values = arrays[origin].mean(axis=1)
        distribution[origin] = {
            "bf16_favoring_histories": int((values > 0).sum()),
            "fp8_favoring_histories": int((values < 0).sum()),
            "ties": int((values == 0).sum()),
            "median_difference_points": float(np.median(values) * 100),
            "interquartile_difference_points": [
                float(value) for value in np.percentile(values * 100, [25, 75])
            ],
        }
    cohort_sensitivity = {
        origin: {
            cohort: {
                "histories": len(values),
                "mean_difference_points": float(np.mean(values)),
            }
            for cohort, values in by_cohort.items()
        }
        for origin, by_cohort in cohorts.items()
    }
    on_policy = {
        "bf16_inference_on_bf16_origin_success_percent": float(
            np.mean([item["bf16_success_percent"] for item in histories if item["origin"] == "local_bf16"])
        ),
        "fp8_inference_on_fp8_origin_success_percent": float(
            np.mean([item["fp8_success_percent"] for item in histories if item["origin"] == "local_fp8"])
        ),
    }
    on_policy["unpaired_diagonal_difference_points"] = (
        on_policy["bf16_inference_on_bf16_origin_success_percent"]
        - on_policy["fp8_inference_on_fp8_origin_success_percent"]
    )
    payload = {
        "schema_version": 1,
        "rows": len(raw),
        "histories_per_origin": 150,
        "paired_seeds_per_history": REPLICATES,
        "primary": {
            "local_bf16_origin_difference_points": effects["local_bf16"],
            "local_fp8_origin_difference_points": effects["local_fp8"],
            "origin_interaction_points": interaction,
            "balanced_origin_mixture_difference_points": mixture,
            "history_cluster_bootstrap_95_percent": cluster_ci,
            "two_stage_bootstrap_sensitivity_95_percent": two_stage_ci,
        },
        "history_effect_distribution": distribution,
        "category_counts": {
            origin: {arm: dict(counter) for arm, counter in by_arm.items()}
            for origin, by_arm in category_counts.items()
        },
        "cohort_sensitivity": cohort_sensitivity,
        "descriptive_on_policy_diagonal": on_policy,
        "histories": histories,
    }
    atomic_write_json(args.output_json, payload)

    def interval(name: str) -> str:
        lo, hi = cluster_ci[name]
        return f"{lo:+.1f} to {hi:+.1f}"

    decision = (
        abs(interaction) >= 5
        and cluster_ci["origin_interaction"][0] * cluster_ci["origin_interaction"][1] > 0
    )
    lines = [
        "# Balanced history-origin crossed-prefix result",
        "",
        f"Rows: {len(raw):,}; histories: 150 per origin; paired seeds: {REPLICATES} per history.",
        "",
        "| Estimand | BF16 - FP8 | History-cluster bootstrap 95% interval |",
        "|---|---:|---:|",
        f"| BF16-origin histories | {effects['local_bf16']:+.1f} pp | {interval('local_bf16_origin')} pp |",
        f"| FP8-origin histories | {effects['local_fp8']:+.1f} pp | {interval('local_fp8_origin')} pp |",
        f"| Origin interaction | {interaction:+.1f} pp | {interval('origin_interaction')} pp |",
        f"| Balanced 50/50 origin mixture | {mixture:+.1f} pp | {interval('balanced_origin_mixture')} pp |",
        "",
        f"The preregistered practically-important interaction rule was {'met' if decision else 'not met'}.",
        "The interaction is effect modification over observed frozen histories; it is not proof of a general intelligence difference or a universal FP8 mechanism.",
        "",
        "## History-level direction counts",
        "",
        "| Origin | BF16-favoring | FP8-favoring | Ties | Median effect |",
        "|---|---:|---:|---:|---:|",
    ]
    for origin in ORIGINS:
        item = distribution[origin]
        lines.append(
            f"| {origin} | {item['bf16_favoring_histories']} | "
            f"{item['fp8_favoring_histories']} | {item['ties']} | "
            f"{item['median_difference_points']:+.1f} pp |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
