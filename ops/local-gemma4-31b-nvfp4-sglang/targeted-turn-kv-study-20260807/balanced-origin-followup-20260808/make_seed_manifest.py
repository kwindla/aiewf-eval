#!/usr/bin/env python3
"""Freeze unique per-history paired seeds and drift-balanced macro blocks."""

from __future__ import annotations

from collections import Counter

from common import (
    HALF_REPLICATES,
    HERE,
    REPLICATES,
    STUDY_VERSION,
    allocated_seed,
    atomic_write_json,
    load_json,
    sha256_file,
)


BLOCKS = {
    "block1": {"arm": "fp8", "replicate_start": 0, "replicate_stop": 8},
    "block2": {"arm": "bf16", "replicate_start": 0, "replicate_stop": 8},
    "block3": {"arm": "bf16", "replicate_start": 8, "replicate_stop": 16},
    "block4": {"arm": "fp8", "replicate_start": 8, "replicate_stop": 16},
}


def main() -> int:
    snapshots = load_json(HERE / "snapshot-manifest.json")
    allocations = {}
    all_seeds = []
    for entry in snapshots["entries"]:
        snapshot_id = entry["snapshot_id"]
        seeds = [allocated_seed(snapshot_id, replicate) for replicate in range(REPLICATES)]
        allocations[snapshot_id] = seeds
        all_seeds.extend(seeds)
    duplicates = [seed for seed, count in Counter(all_seeds).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"seed collision in frozen allocation: {duplicates[:10]}")
    if any(
        block["replicate_stop"] - block["replicate_start"] != HALF_REPLICATES
        for block in BLOCKS.values()
    ):
        raise RuntimeError("macro blocks do not contain exact half allocations")
    manifest = {
        "schema_version": 1,
        "study_version": STUDY_VERSION,
        "seed_rule": "first 64 SHA-256 bits modulo 2^31-1; unique across histories; paired across arms",
        "replicates_per_history": REPLICATES,
        "allocations": allocations,
        "macro_blocks": BLOCKS,
        "snapshot_manifest_sha256": sha256_file(HERE / "snapshot-manifest.json"),
    }
    atomic_write_json(HERE / "seed-manifest.json", manifest)
    print(
        f"froze {len(allocations)} histories x {REPLICATES} seeds; "
        f"{len(all_seeds)} unique seed values"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
