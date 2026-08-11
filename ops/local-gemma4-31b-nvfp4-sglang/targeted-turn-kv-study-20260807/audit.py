#!/usr/bin/env python3
"""Fail-closed audit of token, cache, provenance, allocation, and replay integrity."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from collection_provenance import (
    HERE,
    ProvenanceError,
    capture_live_server,
    plan_case_index,
    request_body,
    source_hashes,
    validate_current_sources,
    validate_collection_plan,
    validate_row_identity,
    verified_snapshots,
    verified_token_ids,
)
from scorer import score_message
from study import atomic_write_json, sha256_file, sha256_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProvenanceError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProvenanceError(f"{path}:{line_number}: row is not an object")
            result.append(row)
    return result


def semantic_output_sha256(completion: dict[str, Any]) -> str | None:
    if not completion:
        return None
    message = completion.get("message") or {}
    semantic_message = {
        "role": message.get("role"),
        "content": message.get("content"),
        "tool_calls": [
            {"type": call.get("type"), "function": call.get("function")}
            for call in message.get("tool_calls") or []
        ],
    }
    value = {
        "message": semantic_message,
        "reasoning_content": completion.get("reasoning_content") or "",
        "finish_reasons": completion.get("finish_reasons") or [],
    }
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact_output_sha256(completion: dict[str, Any]) -> str | None:
    if not completion:
        return None
    return sha256_json(
        {
            "message": completion.get("message"),
            "reasoning_content": completion.get("reasoning_content") or "",
            "finish_reasons": completion.get("finish_reasons") or [],
        }
    )


def compare_tokens(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    payloads = [json.loads(path.read_text()) for path in paths]
    errors = []
    maps = []
    for path, payload in zip(paths, payloads):
        mapping = {}
        for row in payload.get("snapshots") or []:
            snapshot_id = row.get("snapshot_id")
            ids = row.get("prompt_token_ids")
            claimed = row.get("prompt_token_ids_sha256")
            if snapshot_id in mapping:
                errors.append(f"{path}: duplicate snapshot {snapshot_id}")
            if not isinstance(ids, list) or sha256_json(ids) != claimed:
                errors.append(f"{path}: token hash mismatch for {snapshot_id}")
            if len(ids or []) != int(row.get("prompt_tokens") or -1):
                errors.append(f"{path}: token count mismatch for {snapshot_id}")
            mapping[snapshot_id] = claimed
        maps.append(mapping)
    ids = sorted(set().union(*[set(mapping) for mapping in maps]))
    mismatches = [
        {"snapshot_id": snapshot_id, "hashes": [mapping.get(snapshot_id) for mapping in maps]}
        for snapshot_id in ids
        if len({mapping.get(snapshot_id) for mapping in maps}) != 1
    ]
    if any(set(mapping) != set(ids) for mapping in maps):
        errors.append("token manifests do not have identical snapshot coverage")
    return {
        "paths": [str(path) for path in paths],
        "snapshot_count": len(ids),
        "mismatches": mismatches,
        "integrity_errors": errors,
        "passed": not mismatches and not errors,
    }


def completion_errors(row: dict[str, Any]) -> list[str]:
    request_id = str(row.get("request_id"))
    errors = []
    if row.get("error"):
        errors.append(f"{request_id}: request/server error: {row['error']}")
    completion = row.get("completion")
    if not isinstance(completion, dict):
        errors.append(f"{request_id}: missing completion")
        return errors
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        errors.append(f"{request_id}: missing usage")
    elif not isinstance(usage.get("prompt_tokens"), int) or usage.get("prompt_tokens") <= 0:
        errors.append(f"{request_id}: missing/invalid prompt token count")
    if not completion.get("finish_reasons"):
        errors.append(f"{request_id}: missing finish reason")
    if not isinstance(completion.get("first_sse_ms"), (int, float)):
        errors.append(f"{request_id}: missing first-SSE latency")
    if not isinstance(completion.get("ttfat_ms"), (int, float)):
        errors.append(f"{request_id}: missing TTFAT latency")
    if completion.get("raw_events_sha256") != sha256_json(completion.get("raw_events") or []):
        errors.append(f"{request_id}: raw event hash mismatch")
    if completion.get("output_sha256") != exact_output_sha256(completion):
        errors.append(f"{request_id}: exact output hash mismatch")
    if completion.get("semantic_output_sha256") != semantic_output_sha256(completion):
        errors.append(f"{request_id}: semantic output hash mismatch")
    finish_reasons = completion.get("finish_reasons") or []
    if "length" in finish_reasons:
        errors.append(f"{request_id}: output truncated at max_tokens")

    stored_score = row.get("score")
    if not isinstance(stored_score, dict):
        errors.append(f"{request_id}: missing score")
    else:
        rescored = score_message(int(row["turn"]), completion.get("message"))
        if stored_score != rescored:
            errors.append(f"{request_id}: stored score differs from frozen re-score")
        if stored_score.get("category") in {"malformed_parser_failure", "response_parser_failure"}:
            errors.append(f"{request_id}: malformed parser failure")

    cache_mode = row.get("cache_mode")
    cached_tokens = int(completion.get("cached_tokens") or 0)
    prompt_tokens = int((completion.get("usage") or {}).get("prompt_tokens") or 0)
    if cache_mode == "warm":
        if not row.get("warm_cache_gate_passed"):
            errors.append(f"{request_id}: warm cache gate flag is false/missing")
        if not prompt_tokens or cached_tokens / prompt_tokens < 0.90:
            errors.append(
                f"{request_id}: warm cache ratio below 0.90 ({cached_tokens}/{prompt_tokens})"
            )
    elif cache_mode == "cold" and cached_tokens > 1:
        errors.append(f"{request_id}: cold request reported {cached_tokens} cached tokens")
    return errors


def _repeatability_errors(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row.get("arm"), row.get("cache_mode"), row.get("snapshot_id"), row.get("seed"))].append(row)
    checked = 0
    mismatches = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        checked += 1
        signatures = {
            (
                semantic_output_sha256(row.get("completion") or {}),
                (row.get("score") or {}).get("success"),
                (row.get("score") or {}).get("category"),
            )
            for row in group
        }
        if len(signatures) != 1:
            mismatches.append(
                {"key": key, "request_ids": [row.get("request_id") for row in group]}
            )
    return checked, mismatches


def strict_repeatability_gate(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    cache_mode: str,
    expected_repeats: list[int],
) -> dict[str, Any]:
    """Require the complete frozen 100-cell repeatability allocation.

    Individual v2 collection plans bind each process instance.  This combined
    gate then proves semantic identity across the within-process repetitions
    and the separately planned post-restart repetition.
    """

    snapshots = {12: "turn12-golden", 15: "turn15-golden"}
    expected = {
        (arm, cache_mode, snapshot, seed, repeat)
        for snapshot in snapshots.values()
        for seed in range(50)
        for repeat in expected_repeats
    }
    actual_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    errors = []
    for row in rows:
        key = (
            row.get("arm"),
            row.get("cache_mode"),
            row.get("snapshot_id"),
            row.get("seed"),
            row.get("repeat"),
        )
        actual_rows[key].append(row)
        expected_turn = next(
            (turn for turn, snapshot in snapshots.items() if snapshot == row.get("snapshot_id")),
            None,
        )
        if expected_turn is not None and int(row.get("turn") or -1) != expected_turn:
            errors.append(f"{row.get('request_id')}: turn does not match snapshot")
        if row.get("snapshot_kind") != "golden_mechanism":
            errors.append(f"{row.get('request_id')}: repeatability row is not golden")
        errors.extend(completion_errors(row))

    actual = set(actual_rows)
    duplicate_cells = sorted((key for key, values in actual_rows.items() if len(values) != 1), key=str)
    missing = sorted(expected - actual, key=str)
    unexpected = sorted(actual - expected, key=str)
    mismatches = []
    for snapshot in snapshots.values():
        for seed in range(50):
            group = [
                actual_rows[(arm, cache_mode, snapshot, seed, repeat)][0]
                for repeat in expected_repeats
                if len(actual_rows.get((arm, cache_mode, snapshot, seed, repeat), [])) == 1
            ]
            if len(group) != len(expected_repeats):
                continue
            signatures = {
                (
                    semantic_output_sha256(row.get("completion") or {}),
                    row["score"]["success"],
                    row["score"]["category"],
                )
                for row in group
            }
            if len(signatures) != 1:
                mismatches.append(
                    {
                        "snapshot_id": snapshot,
                        "seed": seed,
                        "request_ids": [row["request_id"] for row in group],
                    }
                )
    return {
        "arm": arm,
        "cache_mode": cache_mode,
        "expected_repeats": expected_repeats,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "complete_seed_groups": 100 - len({(item[2], item[3]) for item in missing}),
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "duplicate_cells": duplicate_cells,
        "integrity_errors": errors,
        "semantic_mismatches": mismatches,
        "passed": not (missing or unexpected or duplicate_cells or errors or mismatches),
    }


def _current_source_errors(plan: dict[str, Any]) -> list[str]:
    try:
        validate_current_sources(plan)
    except ProvenanceError as exc:
        return str(exc).split("; ")
    return []


def audit_results(
    paths: list[Path], plan: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if not paths:
        return None
    rows = [row for path in paths for row in read_jsonl(path)]
    request_ids = [str(row.get("request_id")) for row in rows]
    duplicate_ids = sorted(
        key for key, count in collections.Counter(request_ids).items() if count > 1
    )
    logical_keys = [
        (
            row.get("arm"),
            row.get("cache_mode"),
            row.get("snapshot_id"),
            row.get("seed"),
            row.get("repeat"),
        )
        for row in rows
    ]
    duplicate_logical = sorted(
        (key for key, count in collections.Counter(logical_keys).items() if count > 1),
        key=str,
    )
    integrity_errors = [error for row in rows for error in completion_errors(row)]
    missing_ids: list[str] = []
    unexpected_ids: list[str] = []
    identity_errors: list[str] = []
    source_errors: list[str] = []
    expected_count = None
    if plan is not None:
        validate_collection_plan(plan)
        index = plan_case_index(plan)
        expected_count = len(index)
        actual = set(request_ids)
        missing_ids = sorted(set(index) - actual)
        unexpected_ids = sorted(actual - set(index))
        for row in rows:
            case = index.get(row.get("request_id"))
            if case is not None:
                identity_errors.extend(validate_row_identity(row, case, plan))
        source_errors = _current_source_errors(plan)
    checked, repeat_mismatches = _repeatability_errors(rows)
    passed = not any(
        (
            duplicate_ids,
            duplicate_logical,
            integrity_errors,
            missing_ids,
            unexpected_ids,
            identity_errors,
            source_errors,
            repeat_mismatches,
        )
    )
    return {
        "paths": [str(path) for path in paths],
        "rows": len(rows),
        "expected_rows": expected_count,
        "duplicate_request_ids": duplicate_ids,
        "duplicate_logical_cells": duplicate_logical,
        "missing_request_ids": missing_ids,
        "unexpected_request_ids": unexpected_ids,
        "identity_errors": identity_errors,
        "source_errors": source_errors,
        "integrity_errors": integrity_errors,
        "repeat_groups_checked": checked,
        "repeat_mismatches": repeat_mismatches,
        "passed": passed,
    }


def _legacy_fp8_warm_expected(
    *,
    results_path: Path,
    token_manifest_path: Path,
    snapshot_manifest_path: Path,
    seed_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    snapshots = verified_snapshots(snapshot_manifest_path)
    tokens = verified_token_ids(token_manifest_path, snapshots, arm="fp8")
    cases = []
    # Reconstruct the allocation directly rather than trusting the now-amended
    # seed manifest.  These are the frozen pre-hardening extension cells:
    # golden 50:128 plus every real-bank snapshot 0:32, for both turns.
    for snapshot_id, snapshot in snapshots.items():
        seeds = range(50, 128) if snapshot["kind"] == "golden_mechanism" else range(32)
        for seed in seeds:
            request_id = f"fp8:warm:{snapshot_id}:{seed}:0"
            body = request_body(
                snapshots[snapshot_id],
                seed=seed,
                cache_mode="warm",
                request_id=request_id,
                max_tokens=512,
                prompt_token_ids=tokens[snapshot_id],
                temperature=None,
            )
            snapshot = snapshots[snapshot_id]
            cases.append(
                {
                    "request_id": request_id,
                    "arm": "fp8",
                    "cache_mode": "warm",
                    "snapshot_id": snapshot_id,
                    "snapshot_kind": snapshot["kind"],
                    "turn": snapshot["turn"],
                    "seed": seed,
                    "repeat": 0,
                    "request_sha256": sha256_json(body),
                    "base_request_sha256": snapshot["request_sha256"],
                    "input_ids_sha256": sha256_json(tokens[snapshot_id]),
                }
            )
    hashes = {
        "results": sha256_file(results_path),
        "token_manifest": sha256_file(token_manifest_path),
        "snapshot_manifest": sha256_file(snapshot_manifest_path),
        "seed_manifest": sha256_file(seed_manifest_path),
    }
    return cases, hashes


def _parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def retrospective_fp8_warm_sidecar(
    *,
    results_path: Path,
    token_manifest_path: Path,
    snapshot_manifest_path: Path,
    seed_manifest_path: Path,
    server_inspect_path: Path,
    runtime_path: Path,
    endpoint: str,
    container: str,
) -> dict[str, Any]:
    live = capture_live_server(
        arm="fp8",
        geometry="compact",
        sampling="seeded",
        endpoint=endpoint,
        requested_container=container,
    )
    retained_inspect = json.loads(server_inspect_path.read_text())[0]
    runtime = json.loads(runtime_path.read_text())
    cases, artifact_hashes = _legacy_fp8_warm_expected(
        results_path=results_path,
        token_manifest_path=token_manifest_path,
        snapshot_manifest_path=snapshot_manifest_path,
        seed_manifest_path=seed_manifest_path,
    )
    rows = read_jsonl(results_path)
    index = {case["request_id"]: case for case in cases}
    ids = [str(row.get("request_id")) for row in rows]
    errors = []
    if len(cases) != 924:
        errors.append(f"internal expected extension count is {len(cases)}, not 924")
    if len(rows) != 924:
        errors.append(f"legacy result has {len(rows)} rows, expected 924")
    duplicates = sorted(key for key, count in collections.Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate request ids: {duplicates[:10]}")
    missing = sorted(set(index) - set(ids))
    unexpected = sorted(set(ids) - set(index))
    if missing:
        errors.append(f"missing {len(missing)} expected request ids")
    if unexpected:
        errors.append(f"found {len(unexpected)} unexpected request ids")
    for row in rows:
        case = index.get(row.get("request_id"))
        if case is None:
            continue
        for key, value in case.items():
            if row.get(key) != value:
                errors.append(
                    f"{case['request_id']}: {key}={row.get(key)!r}, expected {value!r}"
                )
        errors.extend(completion_errors(row))

    live_id = live["binding"]["container_id"]
    if retained_inspect.get("Id") != live_id:
        errors.append("retained inspect does not describe the live pilot container")
    if retained_inspect.get("Image") != live["config"]["image_id"]:
        errors.append("retained inspect image differs from validated live server")
    if (retained_inspect.get("Config") or {}).get("Cmd") != live["config"]["command"]:
        errors.append("retained inspect command differs from validated live server")
    if runtime.get("arm") != "fp8":
        errors.append("runtime artifact is not labeled fp8")
    if runtime.get("pinned_image") != live["config"]["image_reference"]:
        errors.append("runtime pinned image differs from live server")
    runtime_container = runtime.get("container") or {}
    if runtime_container.get("command") != live["config"]["command"]:
        errors.append("runtime command differs from live pilot command")
    critical_runtime_files = (
        "study.py",
        "replay.py",
        "run_stage.py",
        "scorer.py",
        "shims/sitecustomize.py",
        "snapshot-manifest.json",
        "seed-manifest.json",
        "results/token-manifest-fp8-compact.json",
    )
    missing_runtime_hashes = [
        name for name in critical_runtime_files if name not in (runtime.get("file_sha256") or {})
    ]
    if missing_runtime_hashes:
        errors.append(f"runtime is missing critical hashes: {missing_runtime_hashes}")
    try:
        runtime_time = _parse_utc(runtime["captured_utc"])
        server_created = _parse_utc(live["binding"]["container_created"])
        if runtime_time < server_created:
            errors.append("final pre-hardening runtime capture predates pilot container creation")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid runtime/container timestamps: {exc}")
    if rows:
        started = [float(row.get("started_unix") or 0) for row in rows]
        server_started = _parse_utc(live["binding"]["container_started_at"]).timestamp()
        if min(started) < server_started:
            errors.append("one or more rows predate the validated pilot container")
        runtime_timestamp = _parse_utc(runtime["captured_utc"]).timestamp()
        if max(started) > runtime_timestamp:
            errors.append("one or more rows postdate the final pre-hardening runtime capture")
        captured = _parse_utc(live["captured_utc"]).timestamp()
        if max(started) > captured:
            errors.append("one or more rows postdate the live provenance capture")

    return {
        "schema_version": 1,
        "audit_type": "retrospective-fp8-cache-pilot-warm-extension",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "passed": not errors,
        "errors": errors,
        "expected_rows": 924,
        "observed_rows": len(rows),
        "missing_request_ids": missing,
        "unexpected_request_ids": unexpected,
        "artifact_sha256": {
            **artifact_hashes,
            "server_inspect": sha256_file(server_inspect_path),
            "runtime": sha256_file(runtime_path),
        },
        "runtime_source_sha256": runtime.get("file_sha256"),
        "server": live,
        "retained_inspect_container_id": retained_inspect.get("Id"),
        "expected_cells": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-manifest", type=Path, action="append", default=[])
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--collection-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrospective-fp8-cache-pilot-warm", action="store_true")
    parser.add_argument("--snapshot-manifest", type=Path, default=HERE / "snapshot-manifest.json")
    parser.add_argument("--seed-manifest", type=Path, default=HERE / "seed-manifest.json")
    parser.add_argument("--server-inspect", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--container")
    parser.add_argument("--repeatability-arm", choices=("fp8", "bf16"))
    parser.add_argument("--repeatability-cache", choices=("warm", "cold"))
    parser.add_argument(
        "--expected-repeats",
        help="comma-separated repeat indices for the strict combined repeatability gate",
    )
    args = parser.parse_args()

    try:
        if args.retrospective_fp8_cache_pilot_warm:
            if len(args.results) != 1 or len(args.token_manifest) != 1:
                raise ProvenanceError("retrospective mode requires exactly one result and token manifest")
            if args.server_inspect is None or args.runtime is None or args.container is None:
                raise ProvenanceError(
                    "retrospective mode requires --server-inspect, --runtime, and --container"
                )
            payload = retrospective_fp8_warm_sidecar(
                results_path=args.results[0],
                token_manifest_path=args.token_manifest[0],
                snapshot_manifest_path=args.snapshot_manifest,
                seed_manifest_path=args.seed_manifest,
                server_inspect_path=args.server_inspect,
                runtime_path=args.runtime,
                endpoint=args.endpoint,
                container=args.container,
            )
        else:
            plan = json.loads(args.collection_plan.read_text()) if args.collection_plan else None
            repeatability = None
            repeat_args = (
                args.repeatability_arm,
                args.repeatability_cache,
                args.expected_repeats,
            )
            if any(value is not None for value in repeat_args):
                if any(value is None for value in repeat_args):
                    raise ProvenanceError(
                        "repeatability gate requires --repeatability-arm, "
                        "--repeatability-cache, and --expected-repeats"
                    )
                repeats = [int(value) for value in args.expected_repeats.split(",")]
                if not repeats or len(repeats) != len(set(repeats)) or min(repeats) < 0:
                    raise ProvenanceError("expected repeats must be unique nonnegative integers")
                repeatability = strict_repeatability_gate(
                    [row for path in args.results for row in read_jsonl(path)],
                    arm=args.repeatability_arm,
                    cache_mode=args.repeatability_cache,
                    expected_repeats=repeats,
                )
            payload = {
                "schema_version": 2,
                "token_hash_gate": compare_tokens(args.token_manifest),
                "result_gate": audit_results(args.results, plan),
                "strict_repeatability_gate": repeatability,
            }
            gates = [
                value
                for key, value in payload.items()
                if key.endswith("_gate") and value is not None
            ]
            payload["passed"] = bool(gates) and all(gate["passed"] for gate in gates)
    except Exception as exc:
        payload = {
            "schema_version": 2,
            "passed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
