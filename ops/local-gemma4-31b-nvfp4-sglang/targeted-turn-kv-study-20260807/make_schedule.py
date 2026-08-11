#!/usr/bin/env python3
"""Freeze seed allocations and the one-GPU macro-block schedule."""

from __future__ import annotations

import csv
import io
import json

from study import HERE, atomic_write_json, atomic_write_text


def allocations(golden_n: int, bank_n: int) -> dict[str, list[int]]:
    result = {}
    for turn in (12, 15):
        result[f"turn{turn:02d}-golden"] = list(range(golden_n))
        for source in ("baseten_bf16", "local_fp8", "local_bf16"):
            for index in range(1, 5):
                result[f"turn{turn:02d}-{source}-{index:02d}"] = list(range(bank_n))
    return result


def incremental_allocations(
    golden_start: int,
    golden_stop: int,
    bank_start: int,
    bank_stop: int,
) -> dict[str, list[int]]:
    result = {}
    for turn in (12, 15):
        result[f"turn{turn:02d}-golden"] = list(range(golden_start, golden_stop))
        for source in ("baseten_bf16", "local_fp8", "local_bf16"):
            for index in range(1, 5):
                result[f"turn{turn:02d}-{source}-{index:02d}"] = list(
                    range(bank_start, bank_stop)
                )
    return result


def main() -> int:
    manifest = {
        "schema_version": 1,
        "seed_rule": "consecutive nonnegative SGLang sampling seeds; identical across KV arms",
        "repeatability": {
            "golden_per_turn": list(range(50)),
            "cases_per_arm": 100,
            "within_process_repeats_per_cache_state": 2,
            "cold_restart_repeat_index": 2,
            "inferential_repeat_index": 0,
        },
        "max_tokens_parity": {
            "golden_per_turn": list(range(64)),
            "cases_per_arm": 128,
            "caps": [512, 8192],
        },
        "cache_pilot": {
            "allocations": allocations(128, 32),
            "seed_cases_per_turn_per_arm": 512,
            "cache_modes": ["warm", "cold"],
        },
        "historical_geometry_bridge": {
            "allocations": allocations(64, 16),
            "seed_cases_per_turn": 256,
            "cache_mode": "warm",
            "arm": "fp8",
        },
        "primary_2048": {
            "allocations": allocations(512, 128),
            "seed_cases_per_turn_per_arm": 2048,
            "cache_mode": "warm",
        },
        "continuation_4096": {
            "allocations": incremental_allocations(512, 1024, 128, 256),
            "incremental_seed_cases_per_turn_per_arm": 2048,
            "cumulative_seed_cases_per_turn_per_arm": 4096,
            "cache_mode": "warm",
        },
        "continuation_8192": {
            "allocations": incremental_allocations(1024, 2048, 256, 512),
            "incremental_seed_cases_per_turn_per_arm": 4096,
            "cumulative_seed_cases_per_turn_per_arm": 8192,
            "cache_mode": "warm",
        },
        "continuation_rule": {
            "looks": [2048, 4096, 8192],
            "continue_if_either_turn_simultaneous_ci_half_width_exceeds_points": 2.0,
            "simultaneous_interval": "seed-cluster bootstrap with Bonferroni coverage across two turns and three looks",
            "confirmatory_testing": "Holm-adjusted seed-cluster score tests once at final N only",
        },
    }
    atomic_write_json(HERE / "seed-manifest.json", manifest)

    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(
        ("campaign_level", "macro_block", "arm", "half", "snapshot_id", "seed_start", "seed_stop")
    )
    for level, allocation_name in (
        (2048, "primary_2048"),
        (4096, "continuation_4096"),
        (8192, "continuation_8192"),
    ):
        stage_allocations = manifest[allocation_name]["allocations"]
        for block, arm, half in (
            (1, "fp8", "first"),
            (2, "bf16", "first"),
            (3, "bf16", "second"),
            (4, "fp8", "second"),
        ):
            for snapshot_id, seeds in stage_allocations.items():
                midpoint = len(seeds) // 2
                chosen = seeds[:midpoint] if half == "first" else seeds[midpoint:]
                writer.writerow(
                    (level, block, arm, half, snapshot_id, chosen[0], chosen[-1] + 1)
                )
    atomic_write_text(HERE / "macro-schedule.tsv", output.getvalue())
    print(
        json.dumps(
            {
                "seed_manifest": "seed-manifest.json",
                "schedule_rows": 3 * 4 * len(manifest["primary_2048"]["allocations"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
