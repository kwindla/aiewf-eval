#!/usr/bin/env python3
"""Build canonical tool-call suffix token IDs inside the pinned SGLang image."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from study import CHECKPOINT, CHECKPOINT_REVISION, HERE, atomic_write_json, sha256_json
from teacher_forced_probe import canonical_tool_text


def ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return [int(item) for item in value]


def normalize_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "snapshot-manifest.json")
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token_payload = json.loads(args.token_manifest.read_text())
    frozen = {row["snapshot_id"]: row["prompt_token_ids"] for row in token_payload["snapshots"]}
    snapshot_manifest = json.loads(args.manifest.read_text())
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=CHECKPOINT_REVISION)
    output = []
    for entry in snapshot_manifest["entries"]:
        snapshot = json.loads((args.manifest.parent / entry["path"]).read_text())
        messages = normalize_tool_arguments(snapshot["request"]["messages"])
        options = {
            "tools": snapshot["request"]["tools"],
            "tokenize": True,
            "enable_thinking": False,
        }
        prompt_ids = ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options))
        expected = frozen[snapshot["snapshot_id"]]
        if prompt_ids != expected:
            common = 0
            for left, right in zip(prompt_ids, expected):
                if left != right:
                    break
                common += 1
            raise RuntimeError(
                f"{snapshot['snapshot_id']}: direct tokenizer differs from frozen server prompt "
                f"at {common}; lengths {len(prompt_ids)} vs {len(expected)}"
            )
        suffix = ids(
            tokenizer.encode(
                canonical_tool_text(snapshot["turn"]), add_special_tokens=False
            )
        )
        output.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "turn": snapshot["turn"],
                "prompt_token_ids_sha256": sha256_json(prompt_ids),
                "canonical_suffix_token_ids": suffix,
                "canonical_suffix_token_ids_sha256": sha256_json(suffix),
            }
        )
        print(f"{snapshot['snapshot_id']}: {len(prompt_ids)} + {len(suffix)} tokens", flush=True)
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "snapshots": output,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
