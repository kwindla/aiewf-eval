#!/usr/bin/env python3
"""Freeze and execute one fail-closed targeted-replay collection block."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from collection_provenance import (
    ProvenanceError,
    capture_live_server,
    container_name,
    make_collection_plan,
    write_or_validate_plan,
)
from study import HERE


PRIMARY_BLOCKS = {
    1: ("fp8", "first"),
    2: ("bf16", "first"),
    3: ("bf16", "second"),
    4: ("fp8", "second"),
}


def half_open(values: list[int]) -> str:
    if not values or values != list(range(values[0], values[-1] + 1)):
        raise ValueError("stage runner requires a nonempty contiguous seed allocation")
    return f"{values[0]}:{values[-1] + 1}"


def split_half(values: list[int], half: str) -> list[int]:
    if half == "all":
        return values
    if len(values) % 2:
        raise ProvenanceError("cannot split an odd allocation into exact macro halves")
    midpoint = len(values) // 2
    return values[:midpoint] if half == "first" else values[midpoint:]


def run_replay(
    *,
    result_arm: str,
    treatment_arm: str,
    geometry: str,
    sampling: str,
    container: str,
    endpoint: str,
    cache_mode: str,
    seeds: list[int],
    snapshot: str,
    token_manifest: Path,
    plan_path: Path,
    output: Path,
    max_tokens: int,
    repeat: int = 1,
    repeat_start: int = 0,
    temperature: float | None = None,
) -> None:
    command = [
        sys.executable,
        str(HERE / "replay.py"),
        "--endpoint",
        endpoint,
        "--arm",
        result_arm,
        "--treatment-arm",
        treatment_arm,
        "--geometry",
        geometry,
        "--sampling",
        sampling,
        "--container",
        container,
        "--cache-mode",
        cache_mode,
        "--seeds",
        half_open(seeds),
        "--repeat",
        str(repeat),
        "--repeat-start",
        str(repeat_start),
        "--snapshot",
        snapshot,
        "--token-manifest",
        str(token_manifest),
        "--collection-plan",
        str(plan_path),
        "--output",
        str(output),
        "--max-tokens",
        str(max_tokens),
        "--timeout",
        "180",
    ]
    if cache_mode == "warm":
        command.append("--prime")
    if temperature is not None:
        command.extend(("--temperature", str(temperature)))
    subprocess.run(command, check=True)


def stage_case_specs(
    *,
    args: argparse.Namespace,
    seed_manifest: dict[str, Any],
    snapshot_manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return cache mode, explicit cases, and replay-call descriptions."""

    cases: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    if args.stage == "repeatability":
        for turn in (12, 15):
            snapshot = f"turn{turn}-golden"
            seeds = seed_manifest["repeatability"]["golden_per_turn"]
            for seed in seeds:
                for repeat in range(args.repeat_start, args.repeat_start + args.repeat_count):
                    cases.append({"snapshot_id": snapshot, "seed": seed, "repeat": repeat})
            calls.append(
                {
                    "result_arm": args.arm,
                    "snapshot": snapshot,
                    "seeds": seeds,
                    "repeat": args.repeat_count,
                    "repeat_start": args.repeat_start,
                    "max_tokens": args.max_tokens,
                    "temperature": None,
                }
            )
        return args.cache_mode, cases, calls

    if args.stage == "cap-parity":
        caps = [args.cap] if args.cap is not None else [512, 8192]
        for turn in (12, 15):
            snapshot = f"turn{turn}-golden"
            seeds = seed_manifest["max_tokens_parity"]["golden_per_turn"]
            for cap in caps:
                result_arm = f"{args.arm}-cap{cap}"
                cases.extend(
                    {
                        "arm": result_arm,
                        "cache_mode": "warm",
                        "snapshot_id": snapshot,
                        "seed": seed,
                        "repeat": 0,
                        "max_tokens": cap,
                    }
                    for seed in seeds
                )
                calls.append(
                    {
                        "result_arm": result_arm,
                        "snapshot": snapshot,
                        "seeds": seeds,
                        "repeat": 1,
                        "repeat_start": 0,
                        "max_tokens": cap,
                        "temperature": None,
                    }
                )
        return "warm", cases, calls

    if args.stage == "greedy":
        for entry in snapshot_manifest["entries"]:
            snapshot = entry["snapshot_id"]
            cases.append(
                {
                    "arm": f"{args.arm}-greedy",
                    "cache_mode": args.cache_mode,
                    "snapshot_id": snapshot,
                    "seed": 0,
                    "repeat": 0,
                    "temperature": 0.0,
                }
            )
            calls.append(
                {
                    "result_arm": f"{args.arm}-greedy",
                    "snapshot": snapshot,
                    "seeds": [0],
                    "repeat": 1,
                    "repeat_start": 0,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                }
            )
        return args.cache_mode, cases, calls

    allocation_name = {
        "cache-pilot": "cache_pilot",
        "geometry-bridge": "historical_geometry_bridge",
        "primary": "primary_2048",
        "continue-4096": "continuation_4096",
        "continue-8192": "continuation_8192",
    }[args.stage]
    allocations = seed_manifest[allocation_name]["allocations"]
    cache_mode = "warm" if args.stage in (
        "geometry-bridge",
        "primary",
        "continue-4096",
        "continue-8192",
    ) else args.cache_mode
    result_arm = f"{args.arm}-historical-geometry" if args.stage == "geometry-bridge" else args.arm
    for snapshot, allocated in allocations.items():
        seeds = list(allocated)
        if args.reuse_repeatability and args.stage == "cache-pilot" and snapshot.endswith("-golden"):
            seeds = [seed for seed in seeds if seed >= 50]
        seeds = split_half(seeds, args.seed_half)
        cases.extend(
            {
                "arm": result_arm,
                "cache_mode": cache_mode,
                "snapshot_id": snapshot,
                "seed": seed,
                "repeat": 0,
            }
            for seed in seeds
        )
        calls.append(
            {
                "result_arm": result_arm,
                "snapshot": snapshot,
                "seeds": seeds,
                "repeat": 1,
                "repeat_start": 0,
                "max_tokens": args.max_tokens,
                "temperature": None,
            }
        )
    return cache_mode, cases, calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "repeatability",
            "cap-parity",
            "greedy",
            "cache-pilot",
            "geometry-bridge",
            "primary",
            "continue-4096",
            "continue-8192",
        ),
        required=True,
    )
    parser.add_argument("--arm", choices=("fp8", "bf16"), required=True)
    parser.add_argument("--geometry", choices=("compact", "historical"), default="compact")
    parser.add_argument(
        "--sampling", choices=("seeded", "native", "batch-invariant"), default="seeded"
    )
    parser.add_argument("--container")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cap", type=int, choices=(512, 8192))
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--seed-half", choices=("all", "first", "second"), default="all")
    parser.add_argument("--macro-block", type=int, choices=tuple(PRIMARY_BLOCKS))
    parser.add_argument(
        "--reuse-repeatability",
        action="store_true",
        help="for cache-pilot, omit golden seeds 0:50 and explicitly reuse repeat=0",
    )
    args = parser.parse_args()

    primary_family = args.stage in ("primary", "continue-4096", "continue-8192")
    if primary_family:
        if args.macro_block is None:
            raise ProvenanceError("primary/continuation blocks require --macro-block")
        wanted_arm, wanted_half = PRIMARY_BLOCKS[args.macro_block]
        if (args.arm, args.seed_half) != (wanted_arm, wanted_half):
            raise ProvenanceError(
                f"macro block {args.macro_block} requires arm={wanted_arm}, half={wanted_half}"
            )
    elif args.macro_block is not None:
        raise ProvenanceError("--macro-block is only valid for primary/continuation stages")
    if args.stage == "geometry-bridge" and (args.arm != "fp8" or args.geometry != "historical"):
        raise ProvenanceError("geometry-bridge requires --arm fp8 --geometry historical")
    if args.stage != "geometry-bridge" and args.geometry != "compact":
        raise ProvenanceError("historical geometry is only allowed for geometry-bridge")
    if args.reuse_repeatability and args.stage != "cache-pilot":
        raise ProvenanceError("--reuse-repeatability is only valid for cache-pilot")
    if args.repeat_count < 1 or args.repeat_start < 0:
        raise ProvenanceError("repeat count must be positive and repeat start nonnegative")

    seed_manifest_path = HERE / "seed-manifest.json"
    snapshot_manifest_path = HERE / "snapshot-manifest.json"
    seed_manifest = json.loads(seed_manifest_path.read_text())
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    cache_mode, case_specs, calls = stage_case_specs(
        args=args, seed_manifest=seed_manifest, snapshot_manifest=snapshot_manifest
    )
    stage_id = (
        f"{args.stage}-block{args.macro_block}" if args.macro_block is not None else args.stage
    )
    plan_path = args.plan_output or Path(str(args.output) + ".plan.json")
    audit_path = args.audit_output or Path(str(args.output) + ".audit.json")
    requested_container = args.container or container_name(args.arm, args.geometry, args.sampling)
    live = capture_live_server(
        arm=args.arm,
        geometry=args.geometry,
        sampling=args.sampling,
        endpoint=args.endpoint,
        requested_container=requested_container,
    )
    plan = make_collection_plan(
        stage=stage_id,
        treatment_arm=args.arm,
        geometry=args.geometry,
        sampling=args.sampling,
        cache_mode=cache_mode,
        max_tokens=args.max_tokens,
        temperature=None,
        snapshot_manifest_path=snapshot_manifest_path,
        seed_manifest_path=seed_manifest_path,
        token_manifest_path=args.token_manifest,
        server=live,
        case_specs=case_specs,
    )
    if args.output.exists() and not plan_path.exists():
        raise ProvenanceError(
            f"result file exists without its immutable collection plan: {args.output}"
        )
    plan = write_or_validate_plan(plan_path, plan)
    print(
        json.dumps(
            {
                "stage": stage_id,
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "expected_cases": plan["expected_case_count"],
                "server_instance_id": plan["config"]["server_instance_id"],
            },
            indent=2,
        ),
        flush=True,
    )

    for call in calls:
        run_replay(
            result_arm=call["result_arm"],
            treatment_arm=args.arm,
            geometry=args.geometry,
            sampling=args.sampling,
            container=requested_container,
            endpoint=args.endpoint,
            cache_mode=cache_mode,
            seeds=call["seeds"],
            snapshot=call["snapshot"],
            token_manifest=args.token_manifest,
            plan_path=plan_path,
            output=args.output,
            max_tokens=call["max_tokens"],
            repeat=call["repeat"],
            repeat_start=call["repeat_start"],
            temperature=call["temperature"],
        )

    subprocess.run(
        [
            sys.executable,
            str(HERE / "audit.py"),
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
