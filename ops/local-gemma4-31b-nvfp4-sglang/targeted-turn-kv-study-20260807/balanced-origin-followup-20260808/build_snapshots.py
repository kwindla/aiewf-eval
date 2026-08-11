#!/usr/bin/env python3
"""Freeze the balanced census of local BF16- and FP8-origin turn-12 states."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    HERE,
    ORIGINS,
    ROOT,
    SNAPSHOT_DIR,
    STUDY_VERSION,
    atomic_write_json,
    selection_hash,
    sha256_file,
    sha256_json,
)
from build_snapshots import eligible
from study import extract_logged_context, historical_request, source_rows
from benchmarks._shared.turns import turns


TARGET_TURN = 12


def candidates(origin: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in source_rows(origin):
        ok, reason = eligible(row["run_dir"], TARGET_TURN)
        item = {
            **row,
            "eligibility": reason,
            "selection_hash": selection_hash(origin, row["run_dir"]),
        }
        (accepted if ok else excluded).append(item)
    accepted.sort(key=lambda item: item["selection_hash"])
    return accepted, excluded


def freeze(row: dict[str, Any], origin: str, index: int) -> dict[str, Any]:
    run_dir: Path = row["run_dir"]
    messages = extract_logged_context(run_dir, turns[TARGET_TURN]["input"])
    request = historical_request(messages)
    snapshot_id = f"turn12-{origin}-{index:03d}"
    metadata = {
        "origin": origin,
        "cohort": row["cohort"],
        "campaign_slot": row["slot"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "run_log_sha256": sha256_file(run_dir / "run.log"),
        "transcript_sha256": sha256_file(run_dir / "transcript.jsonl"),
        "selection_hash": row["selection_hash"],
        "target_outcome_not_used": True,
    }
    payload = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "turn": TARGET_TURN,
        "kind": "balanced_origin_real_prefix",
        "metadata": metadata,
        "request_sha256": sha256_json(request),
        "messages_sha256": sha256_json(messages),
        "request": request,
    }
    path = SNAPSHOT_DIR / f"{snapshot_id}.json"
    atomic_write_json(path, payload)
    return {
        "snapshot_id": snapshot_id,
        "turn": TARGET_TURN,
        "kind": payload["kind"],
        "path": str(path.relative_to(HERE)),
        "request_sha256": payload["request_sha256"],
        "messages_sha256": payload["messages_sha256"],
        "metadata": metadata,
    }


def main() -> int:
    by_origin: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, list[dict[str, Any]]] = {}
    for origin in ORIGINS:
        by_origin[origin], exclusions[origin] = candidates(origin)
    balanced_count = min(len(by_origin[origin]) for origin in ORIGINS)
    if balanced_count < 100:
        raise RuntimeError(f"too few eligible histories for intended follow-up: {balanced_count}")

    entries: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for origin in ORIGINS:
        chosen = by_origin[origin][:balanced_count]
        audit[origin] = {
            "canonical_count": len(source_rows(origin)),
            "eligible_count": len(by_origin[origin]),
            "selected_count": len(chosen),
            "selection_rule": (
                "all eligible histories" if len(chosen) == len(by_origin[origin])
                else "lowest salted SHA-256 hashes among eligible histories"
            ),
            "excluded_eligibility_reasons": dict(
                Counter(row["eligibility"] for row in exclusions[origin])
            ),
            "unselected_eligible_count": len(by_origin[origin]) - len(chosen),
        }
        for index, row in enumerate(chosen, 1):
            entries.append(freeze(row, origin, index))

    ids = [entry["snapshot_id"] for entry in entries]
    if len(ids) != len(set(ids)) or len(entries) != balanced_count * len(ORIGINS):
        raise RuntimeError("snapshot allocation is not unique and balanced")
    manifest = {
        "schema_version": 1,
        "study_version": STUDY_VERSION,
        "target_turn": TARGET_TURN,
        "balanced_histories_per_origin": balanced_count,
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": None,
            "max_tokens": 512,
            "thinking": False,
        },
        "entries": entries,
        "selection_audit": audit,
    }
    atomic_write_json(HERE / "snapshot-manifest.json", manifest)
    print(f"froze {len(entries)} turn-12 snapshots ({balanced_count} per origin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
