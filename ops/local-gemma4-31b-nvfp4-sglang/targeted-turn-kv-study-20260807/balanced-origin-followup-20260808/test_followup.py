from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze import cluster_bootstrap
from common import HERE, ORIGINS, REPLICATES, allocated_seed
from collect_block import make_cases


def test_frozen_manifest_is_balanced_and_seed_unique() -> None:
    snapshots = json.loads((HERE / "snapshot-manifest.json").read_text())
    seeds = json.loads((HERE / "seed-manifest.json").read_text())
    counts = {origin: 0 for origin in ORIGINS}
    values = []
    for entry in snapshots["entries"]:
        counts[entry["metadata"]["origin"]] += 1
        allocation = seeds["allocations"][entry["snapshot_id"]]
        assert len(allocation) == REPLICATES
        assert allocation == [allocated_seed(entry["snapshot_id"], index) for index in range(REPLICATES)]
        values.extend(allocation)
    assert counts == {"local_bf16": 150, "local_fp8": 150}
    assert len(values) == len(set(values)) == 300 * REPLICATES


def test_cluster_bootstrap_tracks_known_effects(monkeypatch) -> None:
    import analyze

    monkeypatch.setattr(analyze, "BOOTSTRAPS", 2_000)
    left = np.ones((20, 4))
    right = np.zeros((20, 4))
    result = cluster_bootstrap(left, right, two_stage=False)
    assert result["local_bf16_origin"] == [100.0, 100.0]
    assert result["local_fp8_origin"] == [0.0, 0.0]
    assert result["origin_interaction"] == [100.0, 100.0]
    assert result["balanced_origin_mixture"] == [50.0, 50.0]


def test_each_macro_block_has_eight_cases_per_history() -> None:
    seeds = json.loads((HERE / "seed-manifest.json").read_text())
    for block in ("block1", "block2", "block3", "block4"):
        cases = make_cases(block, seeds)
        assert len(cases) == 300 * 8
        assert len({(case["snapshot_id"], case["seed"]) for case in cases}) == len(cases)
