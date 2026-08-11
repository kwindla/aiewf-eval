#!/usr/bin/env python3
"""Audit all four blocks and prove dtype/container separation before analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import HERE, REPLICATES, atomic_write_json, load_json, sha256_file
from audit import audit_results, read_jsonl
from collection_provenance import validate_collection_plan, validate_current_sources
from collect_block import followup_hashes


BLOCKS = {
    "block1": ("fp8", "block1-fp8-first.jsonl"),
    "block2": ("bf16", "block2-bf16-first.jsonl"),
    "block3": ("bf16", "block3-bf16-second.jsonl"),
    "block4": ("fp8", "block4-fp8-second.jsonl"),
}


def main() -> int:
    results_dir = HERE / "results"
    seed_manifest = load_json(HERE / "seed-manifest.json")
    errors: list[str] = []
    all_rows: list[dict[str, Any]] = []
    plans: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, str] = {}

    for block, (arm, filename) in BLOCKS.items():
        result_path = results_dir / filename
        plan_path = Path(str(result_path) + ".plan.json")
        audit_path = Path(str(result_path) + ".audit.json")
        for path in (result_path, plan_path, audit_path):
            artifacts[str(path.relative_to(HERE))] = sha256_file(path)
        plan = load_json(plan_path)
        plans[block] = plan
        try:
            validate_collection_plan(plan)
            validate_current_sources(plan)
        except Exception as exc:
            errors.append(f"{block}: invalid plan/current sources: {exc}")
        if plan["config"].get("followup_source_sha256") != followup_hashes():
            errors.append(f"{block}: follow-up source hashes differ")
        if plan["config"].get("stage") != f"balanced-origin-{block}":
            errors.append(f"{block}: wrong stage")
        if plan["config"].get("treatment_arm") != arm or plan["server"].get("arm") != arm:
            errors.append(f"{block}: treatment/server arm mismatch")
        sidecar = load_json(audit_path)
        if not sidecar.get("passed"):
            errors.append(f"{block}: retained audit did not pass")
        rerun = audit_results([result_path], plan)
        if not rerun.get("passed"):
            errors.append(f"{block}: fresh result audit failed: {rerun.get('integrity_errors')}")
        rows = read_jsonl(result_path)
        if len(rows) != 2400:
            errors.append(f"{block}: found {len(rows)} rows, expected 2400")
        all_rows.extend(rows)

    ids = [row["request_id"] for row in all_rows]
    logical = [(row["arm"], row["snapshot_id"], int(row["seed"])) for row in all_rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate request IDs across macro blocks")
    if len(logical) != len(set(logical)):
        errors.append("duplicate arm/snapshot/seed cells across macro blocks")

    expected = {
        (arm, snapshot_id, int(seed))
        for arm in ("bf16", "fp8")
        for snapshot_id, allocation in seed_manifest["allocations"].items()
        for seed in allocation
    }
    actual = set(logical)
    if actual != expected:
        errors.append(
            f"combined allocation differs: missing={len(expected - actual)}, "
            f"unexpected={len(actual - expected)}"
        )
    if len(all_rows) != len(expected) or len(expected) != 300 * REPLICATES * 2:
        errors.append("combined row count differs from frozen 300x16x2 design")

    instances = {
        block: plan["config"]["server_instance_id"] for block, plan in plans.items()
    }
    if instances.get("block2") != instances.get("block3"):
        errors.append("BF16 blocks 2 and 3 did not use the same server instance")
    if instances.get("block1") == instances.get("block4"):
        errors.append("FP8 blocks 1 and 4 unexpectedly used the same container instance")
    if len({instances.get("block1"), instances.get("block2"), instances.get("block4")}) != 3:
        errors.append("FP8/BF16/FP8 server instance separation is not proven")

    times = {}
    for block, (_, filename) in BLOCKS.items():
        rows = read_jsonl(results_dir / filename)
        times[block] = [min(float(row["started_unix"]) for row in rows), max(float(row["started_unix"]) for row in rows)]
    if not (
        times["block1"][1] < times["block2"][0]
        <= times["block2"][1] < times["block3"][0]
        <= times["block3"][1] < times["block4"][0]
    ):
        errors.append("observed collection timestamps violate frozen macro-block order")

    token_audit = load_json(results_dir / "token-identity-audit.json")
    if not token_audit.get("passed"):
        errors.append("cross-arm token identity audit did not pass")
    artifacts["results/token-identity-audit.json"] = sha256_file(
        results_dir / "token-identity-audit.json"
    )
    payload = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "rows": len(all_rows),
        "expected_cells": len(expected),
        "server_instance_ids": instances,
        "server_separation": {
            "block1_fp8_destroyed_before_block2_bf16": instances["block1"] != instances["block2"],
            "block2_and_block3_same_bf16_instance": instances["block2"] == instances["block3"],
            "block3_bf16_destroyed_before_block4_fp8": instances["block3"] != instances["block4"],
            "block1_and_block4_fresh_fp8_instances": instances["block1"] != instances["block4"],
        },
        "collection_unix_ranges": times,
        "artifact_sha256": artifacts,
    }
    atomic_write_json(results_dir / "combined-integrity-audit.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
