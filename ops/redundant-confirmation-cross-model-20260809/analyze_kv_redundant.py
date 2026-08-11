#!/usr/bin/env python3
"""Post-hoc history-cluster analysis of redundant confirmation in the KV replay."""

from __future__ import annotations

import glob
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPLAY = ROOT / "ops/local-gemma4-31b-nvfp4-sglang/targeted-turn-kv-study-20260807/balanced-origin-followup-20260808/results"
REDUNDANT = "no_tool_redundant_confirmation_or_question"


def percentile(values: list[float], probability: float) -> float:
    return sorted(values)[int(probability * len(values))]


def main() -> None:
    spec = importlib.util.spec_from_file_location("census", HERE / "analyze.py")
    assert spec and spec.loader
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)

    cells: dict[tuple[str, str], dict[tuple[str, int], bool]] = defaultdict(dict)
    categories: Counter[tuple[str, str]] = Counter()
    subtypes: Counter[tuple[str, str]] = Counter()
    for filename in glob.glob(str(REPLAY / "block*.jsonl")):
        with open(filename) as handle:
            for line in handle:
                row = json.loads(line)
                origin = "local_bf16" if "local_bf16" in row["snapshot_id"] else "local_fp8"
                is_redundant = row["score"]["category"] == REDUNDANT
                cells[(origin, row["snapshot_id"])][(row["arm"], row["seed"])] = is_redundant
                categories[(row["arm"], row["score"]["category"])] += 1
                if is_redundant:
                    subtypes[(row["arm"], census.redundant_subtype(row["score"].get("content", "")))] += 1

    effects: dict[str, list[float]] = defaultdict(list)
    for (origin, _snapshot), paired in cells.items():
        seeds = {seed for _arm, seed in paired}
        assert len(seeds) == 16
        effect = sum(
            paired[("fp8", seed)] - paired[("bf16", seed)] for seed in seeds
        ) / len(seeds) * 100
        effects[origin].append(effect)
    assert {key: len(value) for key, value in effects.items()} == {
        "local_bf16": 150,
        "local_fp8": 150,
    }

    rng = random.Random(20260809)
    boot: dict[str, list[float]] = {}
    draws = 100_000
    for origin, values in effects.items():
        n = len(values)
        samples = [
            sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
        ]
        boot[origin] = [percentile(samples, 0.025), percentile(samples, 0.975)]
    first, second = effects["local_bf16"], effects["local_fp8"]
    balanced_samples = [
        (
            sum(first[rng.randrange(len(first))] for _ in first) / len(first)
            + sum(second[rng.randrange(len(second))] for _ in second) / len(second)
        ) / 2
        for _ in range(draws)
    ]

    payload = {
        "schema_version": 1,
        "scope": "post-hoc redundant-confirmation history-cluster analysis",
        "bootstrap_seed": 20260809,
        "bootstrap_draws": draws,
        "counts": {
            arm: {
                category: count
                for (_arm, category), count in sorted(categories.items())
                if _arm == arm
            }
            for arm in ("bf16", "fp8")
        },
        "fp8_minus_bf16_redundant_points": {
            origin: sum(values) / len(values) for origin, values in effects.items()
        }
        | {"balanced": (sum(first) / len(first) + sum(second) / len(second)) / 2},
        "history_cluster_bootstrap_95_percent": boot
        | {"balanced": [percentile(balanced_samples, 0.025), percentile(balanced_samples, 0.975)]},
        "redundant_subtypes": {
            arm: {
                subtype: count
                for (_arm, subtype), count in sorted(subtypes.items())
                if _arm == arm
            }
            for arm in ("bf16", "fp8")
        },
    }
    output = HERE / "results/kv-redundant-analysis.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
