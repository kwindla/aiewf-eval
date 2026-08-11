#!/usr/bin/env python3
"""Serial, resumable, streaming replay client for frozen target-turn requests."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from audit import completion_errors, read_jsonl
from collection_provenance import (
    ProvenanceError,
    capture_live_server,
    plan_case_index,
    request_body,
    validate_collection_plan,
    validate_current_sources,
    validate_live_server_against_plan,
    validate_row_identity,
    verified_snapshots,
    verified_token_ids,
)
from scorer import score_message
from study import HERE, sha256_file, sha256_json


def atomic_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic append made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_snapshots(manifest_path: Path, filters: set[str]) -> list[dict[str, Any]]:
    all_snapshots = verified_snapshots(manifest_path)
    unknown = filters - set(all_snapshots)
    if unknown:
        raise ProvenanceError(f"unknown requested snapshots: {sorted(unknown)}")
    result = [
        snapshot
        for snapshot_id, snapshot in all_snapshots.items()
        if not filters or snapshot_id in filters
    ]
    if not result:
        raise RuntimeError("snapshot selection is empty")
    return result


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(piece) for piece in value.split(":", 1))
        return list(range(start, stop))
    return [int(piece) for piece in value.split(",") if piece.strip()]


def merge_tool_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_map: dict[int, int] = {}
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_index = int(chunk.get("index", 0))
        if raw_index not in index_map:
            index_map[raw_index] = len(merged)
            merged.append(
                {
                    "id": "",
                    "type": chunk.get("type") or "function",
                    "function": {"name": "", "arguments": ""},
                }
            )
        target = merged[index_map[raw_index]]
        if chunk.get("id"):
            target["id"] += str(chunk["id"])
        function = chunk.get("function") or {}
        if function.get("name"):
            target["function"]["name"] += str(function["name"])
        if function.get("arguments"):
            target["function"]["arguments"] += str(function["arguments"])
    return merged


def stream_completion(
    session: requests.Session,
    endpoint: str,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_events: list[dict[str, Any]] = []
    content: list[str] = []
    reasoning: list[str] = []
    tool_chunks: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    usage = None
    sglext = None
    first_sse_ms = None
    ttfat_ms = None
    response = session.post(
        endpoint.rstrip("/") + "/chat/completions",
        json=body,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        value = raw_line.removeprefix("data:").strip()
        if value == "[DONE]":
            break
        now_ms = (time.perf_counter() - started) * 1000
        if first_sse_ms is None:
            first_sse_ms = now_ms
        event = json.loads(value)
        raw_events.append(event)
        if event.get("usage"):
            usage = event["usage"]
        if event.get("sglext"):
            sglext = event["sglext"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            visible = delta.get("content") or delta.get("tool_calls")
            if visible and ttfat_ms is None:
                ttfat_ms = now_ms
            if delta.get("content"):
                content.append(delta["content"])
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                reasoning.append(thought)
            tool_chunks.extend(delta.get("tool_calls") or [])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
    message = {
        "role": "assistant",
        "content": "".join(content),
        "tool_calls": merge_tool_chunks(tool_chunks),
    }
    semantic_message = {
        "role": message["role"],
        "content": message["content"],
        # SGLang assigns a fresh UUID-style call id after generation.  It is
        # transport metadata, not sampled model output, so exclude it from the
        # seeded-repeatability signature.
        "tool_calls": [
            {"type": call["type"], "function": call["function"]}
            for call in message["tool_calls"]
        ],
    }
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "message": message,
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "cached_tokens": prompt_details.get("cached_tokens", 0),
        "sglext": sglext,
        "first_sse_ms": first_sse_ms,
        "ttfat_ms": ttfat_ms,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "finish_reasons": finish_reasons,
        "raw_events": raw_events,
        "raw_events_sha256": sha256_json(raw_events),
        "output_sha256": sha256_json(
            {"message": message, "reasoning_content": "".join(reasoning), "finish_reasons": finish_reasons}
        ),
        "semantic_output_sha256": sha256_json(
            {
                "message": semantic_message,
                "reasoning_content": "".join(reasoning),
                "finish_reasons": finish_reasons,
            }
        ),
    }


def completed_ids(path: Path, plan: dict[str, Any]) -> set[str]:
    if not path.exists():
        return set()
    index = plan_case_index(plan)
    result = set()
    rows = read_jsonl(path)
    counts = collections.Counter(str(row.get("request_id")) for row in rows)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ProvenanceError(f"resume file has duplicate request ids: {duplicates[:10]}")
    for row in rows:
        request_id = str(row.get("request_id"))
        case = index.get(request_id)
        if case is None:
            raise ProvenanceError(f"resume file has a row outside the frozen plan: {request_id}")
        errors = validate_row_identity(row, case, plan)
        errors.extend(completion_errors(row))
        if errors:
            raise ProvenanceError("resume validation failed: " + "; ".join(errors))
        result.add(request_id)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--treatment-arm", choices=("fp8", "bf16"), required=True)
    parser.add_argument("--geometry", choices=("compact", "historical"), required=True)
    parser.add_argument(
        "--sampling", choices=("seeded", "native", "batch-invariant"), required=True
    )
    parser.add_argument("--container", required=True)
    parser.add_argument("--cache-mode", choices=("warm", "cold", "unsalted"), required=True)
    parser.add_argument("--seeds", required=True, help="comma list or half-open START:STOP")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=HERE / "snapshot-manifest.json")
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--min-warm-cache-ratio", type=float, default=0.90)
    args = parser.parse_args()

    plan = json.loads(args.collection_plan.read_text())
    validate_collection_plan(plan)
    validate_current_sources(plan)
    config = plan["config"]
    expected_settings = {
        "treatment_arm": args.treatment_arm,
        "geometry": args.geometry,
        "sampling": args.sampling,
        "cache_mode": args.cache_mode,
    }
    for key, value in expected_settings.items():
        if config.get(key) != value:
            raise ProvenanceError(
                f"plan {key}={config.get(key)!r}, requested replay has {value!r}"
            )
    if sha256_file(args.manifest) != config.get("snapshot_manifest_sha256"):
        raise ProvenanceError("snapshot manifest differs from frozen collection plan")
    if sha256_file(args.token_manifest) != config.get("token_manifest_sha256"):
        raise ProvenanceError("token manifest differs from frozen collection plan")

    all_snapshots = verified_snapshots(args.manifest)
    snapshots = load_snapshots(args.manifest, set(args.snapshot))
    prompt_ids = verified_token_ids(args.token_manifest, all_snapshots, arm=args.treatment_arm)
    seeds = parse_seeds(args.seeds)
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
    live = capture_live_server(
        arm=args.treatment_arm,
        geometry=args.geometry,
        sampling=args.sampling,
        endpoint=args.endpoint,
        requested_container=args.container,
        session=session,
    )
    validate_live_server_against_plan(plan, live)
    done = completed_ids(args.output, plan)
    logical_index = {
        (
            case["arm"],
            case["cache_mode"],
            case["snapshot_id"],
            case["seed"],
            case["repeat"],
        ): case
        for case in plan["cases"]
    }

    for snapshot in snapshots:
        snapshot_id = snapshot["snapshot_id"]
        ids = prompt_ids[snapshot_id]
        selected_cases = []
        for seed in seeds:
            for repeat in range(args.repeat_start, args.repeat_start + args.repeat):
                key = (args.arm, args.cache_mode, snapshot_id, seed, repeat)
                case = logical_index.get(key)
                if case is None:
                    raise ProvenanceError(f"requested replay cell is absent from plan: {key}")
                if case["max_tokens"] != args.max_tokens:
                    raise ProvenanceError(f"requested max_tokens differs from plan for {key}")
                if case["temperature_override"] != args.temperature:
                    raise ProvenanceError(f"requested temperature differs from plan for {key}")
                selected_cases.append(case)
        if args.prime and args.cache_mode == "warm" and any(
            case["request_id"] not in done for case in selected_cases
        ):
            try:
                prime = stream_completion(
                    session,
                    args.endpoint,
                    request_body(
                        snapshot,
                        seed=0,
                        cache_mode="warm",
                        request_id=f"prime:{plan['plan_sha256']}:{snapshot_id}",
                        max_tokens=1,
                        prompt_token_ids=ids,
                        temperature=args.temperature,
                    ),
                    args.timeout,
                )
            except Exception as exc:
                print(
                    f"prime failed for {snapshot_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return 1
            print(
                f"primed {snapshot_id}: prompt={prime.get('usage', {}).get('prompt_tokens')} "
                f"cached={prime['cached_tokens']}",
                flush=True,
            )
        for case in selected_cases:
            request_id = case["request_id"]
            if request_id in done:
                continue
            body = request_body(
                snapshot,
                seed=case["seed"],
                cache_mode=args.cache_mode,
                request_id=request_id,
                max_tokens=args.max_tokens,
                prompt_token_ids=ids,
                temperature=args.temperature,
            )
            if sha256_json(body) != case["request_sha256"]:
                raise ProvenanceError(f"request body differs from plan for {request_id}")
            started_at = time.time()
            try:
                completion = stream_completion(session, args.endpoint, body, args.timeout)
                scoring = score_message(snapshot["turn"], completion["message"])
                error = None
            except Exception as exc:
                completion = None
                error = f"{type(exc).__name__}: {exc}"
                scoring = score_message(snapshot["turn"], None, request_error=error)
            row = {
                    "schema_version": 2,
                    "request_id": request_id,
                    "arm": args.arm,
                    "treatment_arm": args.treatment_arm,
                    "cache_mode": args.cache_mode,
                    "snapshot_id": snapshot_id,
                    "snapshot_kind": snapshot["kind"],
                    "turn": snapshot["turn"],
                    "seed": case["seed"],
                    "repeat": case["repeat"],
                    "max_tokens": args.max_tokens,
                    "temperature_override": args.temperature,
                    "collection_stage": config["stage"],
                    "collection_plan_sha256": plan["plan_sha256"],
                    "collection_config_sha256": plan["config_sha256"],
                    "server_instance_id": config["server_instance_id"],
                    "server_config_sha256": config["server_config_sha256"],
                    "started_unix": started_at,
                    "request_sha256": sha256_json(body),
                    "base_request_sha256": snapshot["request_sha256"],
                    "input_ids_sha256": sha256_json(ids) if ids is not None else None,
                    "error": error,
                    "score": scoring,
                    "completion": completion,
            }
            if completion and args.cache_mode == "warm":
                prompt_tokens = int((completion.get("usage") or {}).get("prompt_tokens") or 0)
                cached_tokens = int(completion.get("cached_tokens") or 0)
                row["warm_cache_gate_passed"] = bool(
                    prompt_tokens and cached_tokens / prompt_tokens >= args.min_warm_cache_ratio
                )
            atomic_append(args.output, row)
            print(
                f"{request_id} category={scoring['category']} "
                f"cache={completion['cached_tokens'] if completion else 'ERR'} "
                f"ttfat={completion['ttfat_ms'] if completion else 'ERR'}",
                flush=True,
            )
            if error:
                return 1
            if scoring.get("category") in {"malformed_parser_failure", "response_parser_failure"}:
                print(f"parser gate failed for {request_id}", flush=True)
                return 1
            if "length" in (completion.get("finish_reasons") or []):
                print(f"length/truncation gate failed for {request_id}", flush=True)
                return 1
            if args.cache_mode == "warm" and not row["warm_cache_gate_passed"]:
                print(f"warm cache gate failed for {request_id}", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
