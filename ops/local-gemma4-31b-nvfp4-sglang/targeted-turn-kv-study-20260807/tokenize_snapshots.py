#!/usr/bin/env python3
"""Capture prompt token IDs for every frozen snapshot from one SGLang arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from study import HERE, atomic_write_json, sha256_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "snapshot-manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
    session.get(args.endpoint.removesuffix("/v1") + "/health", timeout=10).raise_for_status()
    rows: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        snapshot = json.loads((args.manifest.parent / entry["path"]).read_text())
        body = dict(snapshot["request"])
        body.update(
            {
                "stream": False,
                "temperature": 0,
                "max_tokens": 1,
                "seed": 0,
                "return_prompt_token_ids": True,
                "return_meta_info": True,
                "cache_salt": f"tokenize:{args.arm}:{snapshot['snapshot_id']}",
            }
        )
        body.pop("stream_options", None)
        response = session.post(
            args.endpoint.rstrip("/") + "/chat/completions", json=body, timeout=180
        )
        response.raise_for_status()
        payload = response.json()
        ids = payload["choices"][0].get("prompt_token_ids")
        if not isinstance(ids, list) or not ids:
            raise RuntimeError(f"no prompt token IDs returned for {snapshot['snapshot_id']}")
        row = {
            "snapshot_id": snapshot["snapshot_id"],
            "turn": snapshot["turn"],
            "request_sha256": snapshot["request_sha256"],
            "prompt_tokens": len(ids),
            "prompt_token_ids_sha256": sha256_json(ids),
            "prompt_token_ids": ids,
        }
        rows.append(row)
        print(
            f"{snapshot['snapshot_id']}: {len(ids)} tokens {row['prompt_token_ids_sha256']}",
            flush=True,
        )
    atomic_write_json(
        args.output,
        {"schema_version": 1, "arm": args.arm, "snapshots": rows},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
