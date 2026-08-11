#!/usr/bin/env python3
"""Capture immutable source and validated live-server provenance for one stage."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from collection_provenance import (
    PINNED_IMAGE,
    ProvenanceError,
    capture_live_server,
    container_name,
    source_hashes,
)
from study import CHECKPOINT, CHECKPOINT_REVISION, atomic_write_json


def command(*values: str) -> str:
    return subprocess.run(values, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fp8", "bf16"), required=True)
    parser.add_argument("--geometry", choices=("compact", "historical"), default="compact")
    parser.add_argument(
        "--sampling", choices=("seeded", "native", "batch-invariant"), default="seeded"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--container")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ProvenanceError(f"refusing to overwrite runtime provenance: {args.output}")
    requested_container = args.container or container_name(args.arm, args.geometry, args.sampling)
    server = capture_live_server(
        arm=args.arm,
        geometry=args.geometry,
        sampling=args.sampling,
        endpoint=args.endpoint,
        requested_container=requested_container,
    )
    gpu_fields = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ).splitlines()
    payload = {
        "schema_version": 2,
        "captured_utc": server["captured_utc"],
        "stage": args.stage,
        "arm": args.arm,
        "geometry": args.geometry,
        "sampling": args.sampling,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "pinned_image": PINNED_IMAGE,
        "server": server,
        "gpu": gpu_fields,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "source_sha256": source_hashes(args.token_manifest),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
