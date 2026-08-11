#!/usr/bin/env python3
"""Plan, collect, resume, and audit one balanced-origin macro block."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from common import HERE, PARENT, atomic_write_json, load_json, sha256_file, sha256_json
from audit import completion_errors
from collection_provenance import (
    ProvenanceError,
    capture_live_server,
    container_name,
    future_request_id,
    make_collection_plan,
    request_body,
    utc_now,
    validate_current_sources,
    validate_live_server_against_plan,
    verified_snapshots,
    verified_token_ids,
    write_or_validate_plan,
)
from replay import atomic_append, completed_ids, stream_completion
from scorer import score_message


FOLLOWUP_SOURCES = (
    "PREREGISTRATION.md",
    "common.py",
    "build_snapshots.py",
    "make_seed_manifest.py",
    "collect_block.py",
    "analyze.py",
    "test_followup.py",
    "README.md",
    "snapshot-manifest.json",
    "seed-manifest.json",
)


def followup_hashes() -> dict[str, str]:
    return {name: sha256_file(HERE / name) for name in FOLLOWUP_SOURCES}


def validate_followup_sources(plan: dict[str, Any]) -> None:
    expected = plan["config"].get("followup_source_sha256") or {}
    actual = followup_hashes()
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise ProvenanceError(f"follow-up sources changed after plan freeze: {changed}")


def bind_followup_sources(
    plan: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    tokens: dict[str, list[int]],
) -> dict[str, Any]:
    """Bind follow-up orchestration sources into the immutable parent plan."""

    config = dict(plan["config"])
    config["followup_source_sha256"] = followup_hashes()
    config_sha256 = sha256_json(config)
    cases = []
    for old in plan["cases"]:
        request_id = future_request_id(
            config_sha256,
            stage=config["stage"],
            arm=old["arm"],
            cache_mode=old["cache_mode"],
            snapshot_id=old["snapshot_id"],
            seed=old["seed"],
            repeat=old["repeat"],
        )
        body = request_body(
            snapshots[old["snapshot_id"]],
            seed=old["seed"],
            cache_mode=old["cache_mode"],
            request_id=request_id,
            max_tokens=old["max_tokens"],
            prompt_token_ids=tokens[old["snapshot_id"]],
            temperature=old["temperature_override"],
        )
        cases.append({**old, "request_id": request_id, "request_sha256": sha256_json(body)})
    core = {
        "schema_version": plan["schema_version"],
        "config_sha256": config_sha256,
        "config": config,
        "server": plan["server"],
        "expected_case_count": len(cases),
        "cases": cases,
    }
    return {**core, "plan_sha256": sha256_json(core), "created_utc": utc_now()}


def make_cases(block: str, seeds: dict[str, Any]) -> list[dict[str, Any]]:
    spec = seeds["macro_blocks"][block]
    start = int(spec["replicate_start"])
    stop = int(spec["replicate_stop"])
    cases = []
    for snapshot_id, allocation in seeds["allocations"].items():
        for replicate in range(start, stop):
            cases.append(
                {
                    "snapshot_id": snapshot_id,
                    "seed": int(allocation[replicate]),
                    "repeat": 0,
                }
            )
    return cases


def collect(
    *,
    arm: str,
    endpoint: str,
    container: str,
    plan: dict[str, Any],
    snapshot_manifest: Path,
    token_manifest: Path,
    output: Path,
) -> None:
    snapshots = verified_snapshots(snapshot_manifest)
    tokens = verified_token_ids(token_manifest, snapshots, arm=arm)
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
    live = capture_live_server(
        arm=arm,
        geometry="compact",
        sampling="seeded",
        endpoint=endpoint,
        requested_container=container,
        session=session,
    )
    validate_current_sources(plan)
    validate_followup_sources(plan)
    validate_live_server_against_plan(plan, live)
    done = completed_ids(output, plan)
    cases_by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for case in plan["cases"]:
        cases_by_snapshot.setdefault(case["snapshot_id"], []).append(case)

    for snapshot_id in sorted(cases_by_snapshot):
        pending = [case for case in cases_by_snapshot[snapshot_id] if case["request_id"] not in done]
        if not pending:
            continue
        snapshot = snapshots[snapshot_id]
        ids = tokens[snapshot_id]
        prime = stream_completion(
            session,
            endpoint,
            request_body(
                snapshot,
                seed=0,
                cache_mode="warm",
                request_id=f"prime:{plan['plan_sha256']}:{snapshot_id}",
                max_tokens=1,
                prompt_token_ids=ids,
                temperature=None,
            ),
            180,
        )
        print(
            f"primed {snapshot_id}: prompt={prime.get('usage', {}).get('prompt_tokens')} "
            f"cached={prime.get('cached_tokens')}",
            flush=True,
        )
        for case in pending:
            body = request_body(
                snapshot,
                seed=case["seed"],
                cache_mode="warm",
                request_id=case["request_id"],
                max_tokens=case["max_tokens"],
                prompt_token_ids=ids,
                temperature=case["temperature_override"],
            )
            if sha256_json(body) != case["request_sha256"]:
                raise ProvenanceError(f"request body differs from plan: {case['request_id']}")
            started = time.time()
            try:
                completion = stream_completion(session, endpoint, body, 180)
                score = score_message(12, completion["message"])
                error = None
            except Exception as exc:
                completion = None
                error = f"{type(exc).__name__}: {exc}"
                score = score_message(12, None, request_error=error)
            row = {
                "schema_version": 2,
                "request_id": case["request_id"],
                "arm": arm,
                "treatment_arm": arm,
                "cache_mode": "warm",
                "snapshot_id": snapshot_id,
                "snapshot_kind": snapshot["kind"],
                "turn": 12,
                "seed": case["seed"],
                "repeat": 0,
                "max_tokens": case["max_tokens"],
                "temperature_override": None,
                "collection_stage": plan["config"]["stage"],
                "collection_plan_sha256": plan["plan_sha256"],
                "collection_config_sha256": plan["config_sha256"],
                "server_instance_id": plan["config"]["server_instance_id"],
                "server_config_sha256": plan["config"]["server_config_sha256"],
                "started_unix": started,
                "request_sha256": sha256_json(body),
                "base_request_sha256": snapshot["request_sha256"],
                "input_ids_sha256": sha256_json(ids),
                "error": error,
                "score": score,
                "completion": completion,
            }
            if completion:
                prompt_tokens = int((completion.get("usage") or {}).get("prompt_tokens") or 0)
                cached_tokens = int(completion.get("cached_tokens") or 0)
                row["warm_cache_gate_passed"] = bool(
                    prompt_tokens and cached_tokens / prompt_tokens >= 0.90
                )
            atomic_append(output, row)
            errors = completion_errors(row)
            print(
                f"{case['request_id']} category={score['category']} "
                f"cache={completion.get('cached_tokens') if completion else 'ERR'}",
                flush=True,
            )
            if errors:
                raise ProvenanceError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("block1", "block2", "block3", "block4"), required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    args = parser.parse_args()

    snapshot_manifest = HERE / "snapshot-manifest.json"
    seed_manifest_path = HERE / "seed-manifest.json"
    seeds = load_json(seed_manifest_path)
    if seeds.get("snapshot_manifest_sha256") != sha256_file(snapshot_manifest):
        raise ProvenanceError("seed manifest is not bound to the current snapshot manifest")
    block = seeds["macro_blocks"][args.block]
    arm = block["arm"]
    requested_container = container_name(arm, "compact", "seeded")
    live = capture_live_server(
        arm=arm,
        geometry="compact",
        sampling="seeded",
        endpoint=args.endpoint,
        requested_container=requested_container,
    )
    cases = make_cases(args.block, seeds)
    snapshots = verified_snapshots(snapshot_manifest)
    tokens = verified_token_ids(args.token_manifest, snapshots, arm=arm)
    plan = make_collection_plan(
        stage=f"balanced-origin-{args.block}",
        treatment_arm=arm,
        geometry="compact",
        sampling="seeded",
        cache_mode="warm",
        max_tokens=512,
        temperature=None,
        snapshot_manifest_path=snapshot_manifest,
        seed_manifest_path=seed_manifest_path,
        token_manifest_path=args.token_manifest,
        server=live,
        case_specs=cases,
    )
    plan = bind_followup_sources(plan, snapshots, tokens)
    plan_path = args.plan_output or Path(str(args.output) + ".plan.json")
    audit_path = args.audit_output or Path(str(args.output) + ".audit.json")
    if args.output.exists() and not plan_path.exists():
        raise ProvenanceError("result exists without its immutable collection plan")
    plan = write_or_validate_plan(plan_path, plan)
    print(
        json.dumps(
            {
                "block": args.block,
                "arm": arm,
                "cases": len(cases),
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    collect(
        arm=arm,
        endpoint=args.endpoint,
        container=requested_container,
        plan=plan,
        snapshot_manifest=snapshot_manifest,
        token_manifest=args.token_manifest,
        output=args.output,
    )
    subprocess.run(
        [
            sys.executable,
            str(PARENT / "audit.py"),
            "--results",
            str(args.output),
            "--collection-plan",
            str(plan_path),
            "--output",
            str(audit_path),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
