#!/usr/bin/env python3
"""Teacher-force the canonical target tool call and record its token margins."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import requests

from study import HERE, atomic_write_json, sha256_json


def canonical_call(turn: int) -> tuple[str, dict[str, str]]:
    if turn == 12:
        name = "submit_session_suggestion"
        arguments = {
            "name": "Jennifer Smith",
            "suggestion_text": "A session on state machine abstractions for complex workflows.",
        }
    elif turn == 15:
        name = "submit_dietary_request"
        arguments = {"name": "Jennifer Smith", "dietary_preference": "vegan"}
    else:
        raise ValueError(f"unsupported turn {turn}")
    return name, arguments


def canonical_tool_text(turn: int) -> str:
    name, arguments = canonical_call(turn)
    rendered = ",".join(
        f'{key}:<|"|>{value}<|"|>' for key, value in sorted(arguments.items())
    )
    return f"<|tool_call>call:{name}{{{rendered}}}<tool_call|>"


def manifest_ids(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text())
    return {row["snapshot_id"]: row["prompt_token_ids"] for row in payload["snapshots"]}


def post(session: requests.Session, url: str, body: dict[str, Any], timeout: float = 180) -> dict[str, Any]:
    response = session.post(url, json=body, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"{url} returned HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "snapshot-manifest.json")
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--teacher-token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot_manifest = json.loads(args.manifest.read_text())
    prompt_ids = manifest_ids(args.token_manifest)
    teacher_payload = json.loads(args.teacher_token_manifest.read_text())
    teacher_suffixes = {
        row["snapshot_id"]: row["canonical_suffix_token_ids"]
        for row in teacher_payload["snapshots"]
    }
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
    session.get(args.endpoint + "/health", timeout=10).raise_for_status()
    output = []

    for entry in snapshot_manifest["entries"]:
        if args.snapshot and entry["snapshot_id"] not in args.snapshot:
            continue
        snapshot = json.loads((args.manifest.parent / entry["path"]).read_text())
        snapshot_id = snapshot["snapshot_id"]
        base_ids = prompt_ids[snapshot_id]
        cache_key = f"gemma-kv-teacher-v1:{snapshot_id}"

        if args.cache_mode == "warm":
            prime_body = dict(snapshot["request"])
            prime_body.update(
                {
                    "stream": False,
                    "input_ids": base_ids,
                    "seed": 0,
                    "temperature": 0,
                    "max_tokens": 1,
                    "cache_salt": cache_key,
                    "return_cached_tokens_details": True,
                }
            )
            prime_body.pop("stream_options", None)
            post(session, args.endpoint + "/v1/chat/completions", prime_body)

        suffix_ids = teacher_suffixes[snapshot_id]
        if not suffix_ids:
            raise RuntimeError(f"{snapshot_id}: empty canonical tool-call suffix")
        full_ids = base_ids + suffix_ids

        extra_key = cache_key if args.cache_mode == "warm" else f"{cache_key}:cold:{time.time_ns()}"
        generate_body = {
            "input_ids": full_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            # SGLang's prompt-logprob convention needs the preceding token in
            # range to score the first canonical suffix token.
            "logprob_start_len": max(0, len(base_ids) - 1),
            "top_logprobs_num": 20,
            "return_text_in_logprobs": True,
            "extra_key": extra_key,
        }
        result = post(session, args.endpoint + "/generate", generate_body)
        meta = result["meta_info"]
        token_logprobs = meta.get("input_token_logprobs") or []
        if len(token_logprobs) < len(suffix_ids):
            raise RuntimeError(
                f"{snapshot_id}: received {len(token_logprobs)} input logprobs for {len(suffix_ids)} suffix tokens"
            )
        suffix_logprobs = token_logprobs[-len(suffix_ids) :]
        top = (meta.get("input_top_logprobs") or [])[-len(suffix_ids) :]
        first_expected = suffix_logprobs[0]
        alternatives = [item for item in (top[0] or []) if int(item[1]) != int(first_expected[1])]
        best_alternative = max(alternatives, key=lambda item: float(item[0])) if alternatives else None
        finite = [float(item[0]) for item in suffix_logprobs if item[0] is not None and math.isfinite(float(item[0]))]
        row = {
            "arm": args.arm,
            "cache_mode": args.cache_mode,
            "snapshot_id": snapshot_id,
            "snapshot_kind": snapshot["kind"],
            "turn": snapshot["turn"],
            "prompt_tokens": len(base_ids),
            "canonical_suffix_tokens": len(suffix_ids),
            "prompt_ids_sha256": sha256_json(base_ids),
            "suffix_ids_sha256": sha256_json(suffix_ids),
            "cached_tokens": int(meta.get("cached_tokens") or 0),
            "canonical_sequence_logprob_sum": sum(finite),
            "canonical_sequence_logprob_mean": sum(finite) / len(finite),
            "first_expected": first_expected,
            "first_best_alternative": best_alternative,
            "first_expected_minus_alternative_logprob": (
                float(first_expected[0]) - float(best_alternative[0]) if best_alternative else None
            ),
            "suffix_token_logprobs": suffix_logprobs,
            "suffix_top_logprobs": top,
            "request_sha256": sha256_json(generate_body),
        }
        output.append(row)
        print(
            f"{snapshot_id} suffix={len(suffix_ids)} cached={row['cached_tokens']} "
            f"mean_logp={row['canonical_sequence_logprob_mean']:.4f}",
            flush=True,
        )

    atomic_write_json(
        args.output,
        {"schema_version": 1, "arm": args.arm, "cache_mode": args.cache_mode, "snapshots": output},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
