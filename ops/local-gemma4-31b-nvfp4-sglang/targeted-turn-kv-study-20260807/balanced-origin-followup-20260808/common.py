#!/usr/bin/env python3
"""Shared paths and deterministic helpers for the balanced-origin follow-up."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[2]
SNAPSHOT_DIR = HERE / "snapshots"
RESULTS_DIR = HERE / "results"

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from study import atomic_write_json, canonical_json, sha256_file, sha256_json  # noqa: E402


STUDY_VERSION = "gemma-kv-balanced-origin-turn12-v1"
ORIGINS = ("local_bf16", "local_fp8")
REPLICATES = 16
HALF_REPLICATES = 8
SEED_MAX = 2**31 - 1


def selection_hash(origin: str, run_dir: Path) -> str:
    relative = run_dir.relative_to(ROOT)
    payload = f"{STUDY_VERSION}\0selection\0{origin}\0{relative}".encode()
    return hashlib.sha256(payload).hexdigest()


def allocated_seed(snapshot_id: str, replicate: int) -> int:
    if not 0 <= replicate < REPLICATES:
        raise ValueError(f"replicate outside frozen allocation: {replicate}")
    payload = f"{STUDY_VERSION}\0seed\0{snapshot_id}\0{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SEED_MAX


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


__all__ = [
    "HERE",
    "PARENT",
    "ROOT",
    "SNAPSHOT_DIR",
    "RESULTS_DIR",
    "STUDY_VERSION",
    "ORIGINS",
    "REPLICATES",
    "HALF_REPLICATES",
    "allocated_seed",
    "atomic_write_json",
    "canonical_json",
    "load_json",
    "selection_hash",
    "sha256_file",
    "sha256_json",
]
