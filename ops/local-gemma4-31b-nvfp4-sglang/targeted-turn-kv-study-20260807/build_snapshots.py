#!/usr/bin/env python3
"""Freeze an outcome-blind real-prefix bank and golden mechanism snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks._shared.turns import turns
from benchmarks.aiwf_medium_context.prompts.system import system_instruction

from study import (
    CAMPAIGNS,
    HERE,
    ROOT,
    TARGET_TURNS,
    atomic_write_json,
    canonical_json,
    extract_logged_context,
    historical_request,
    sha256_file,
    sha256_json,
    source_rows,
    target_row,
    transcript_by_turn,
)


SNAPSHOT_DIR = HERE / "snapshots"


def correct_call(row: dict[str, Any], name: str, required: dict[str, str]) -> bool:
    for call in row.get("tool_calls") or []:
        if call.get("name") != name:
            continue
        args = call.get("args") or {}
        if all(str(args.get(key, "")).casefold() == value.casefold() for key, value in required.items()):
            return True
    return False


def eligible(run_dir: Path, turn: int) -> tuple[bool, str]:
    rows = transcript_by_turn(run_dir)
    needed = range(0, turn + 1)
    if any(index not in rows for index in needed):
        return False, "missing scripted prerequisite turn"
    joined = canonical_json([rows[index] for index in range(turn)])
    if "Jennifer Smith" not in joined:
        return False, "name absent from prefix"
    if not correct_call(
        rows[11],
        "submit_session_suggestion",
        {"name": "Jennifer Smith"},
    ):
        return False, "turn 11 suggestion tool state absent"
    if turn == 15:
        if rows[14].get("tool_calls"):
            return False, "dietary tool already called on turn 14"
        text = str(rows[14].get("assistant_text") or "").casefold()
        if "vegan" not in text or not any(word in text for word in ("submit", "request")):
            return False, "turn 14 did not establish the dietary-request state"
    return True, "eligible"


def select_rows(source: str, turn: int, count: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    exclusions = []
    for row in source_rows(source):
        ok, reason = eligible(row["run_dir"], turn)
        selection_hash = hashlib.sha256(
            f"gemma-kv-bank-v1\0{source}\0{turn}\0{row['run_dir'].relative_to(ROOT)}".encode()
        ).hexdigest()
        entry = {**row, "selection_hash": selection_hash, "eligibility": reason}
        (candidates if ok else exclusions).append(entry)
    candidates.sort(key=lambda row: row["selection_hash"])
    if len(candidates) < count:
        raise RuntimeError(f"{source} turn {turn}: only {len(candidates)} eligible prefixes")
    return candidates[:count], exclusions


def golden_messages(turn: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    for index in range(turn + 1):
        spec = turns[index]
        messages.append({"role": "user", "content": spec["input"]})
        if index == turn:
            break
        expected = spec.get("required_function_call")
        if expected:
            call_id = f"call_golden_{index:03d}"
            messages.append(
                {
                    "role": "assistant",
                    "content": spec.get("golden_text") or "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": expected["name"],
                                "arguments": json.dumps(expected["args"], sort_keys=True),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(spec.get("function_call_response") or {"status": "success"}),
                    "tool_call_id": call_id,
                }
            )
        else:
            messages.append({"role": "assistant", "content": spec.get("golden_text") or ""})
    return messages


def freeze_snapshot(
    snapshot_id: str,
    turn: int,
    kind: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    request = historical_request(messages)
    payload = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "turn": turn,
        "kind": kind,
        "metadata": metadata,
        "request_sha256": sha256_json(request),
        "messages_sha256": sha256_json(messages),
        "request": request,
    }
    path = SNAPSHOT_DIR / f"{snapshot_id}.json"
    atomic_write_json(path, payload)
    return {
        "snapshot_id": snapshot_id,
        "turn": turn,
        "kind": kind,
        "path": str(path.relative_to(HERE)),
        "request_sha256": payload["request_sha256"],
        "messages_sha256": payload["messages_sha256"],
        "metadata": metadata,
    }


def main() -> int:
    entries: list[dict[str, Any]] = []
    selection_audit: dict[str, Any] = {}
    for turn in TARGET_TURNS:
        golden_id = f"turn{turn:02d}-golden"
        entries.append(
            freeze_snapshot(
                golden_id,
                turn,
                "golden_mechanism",
                golden_messages(turn),
                {"selection": "benchmark golden responses; excluded from primary bank"},
            )
        )
        for source in CAMPAIGNS:
            selected, exclusions = select_rows(source, turn)
            selection_audit[f"turn{turn:02d}:{source}"] = {
                "rule": "lowest four SHA-256 values among prerequisite-valid prefixes",
                "selected": [
                    {
                        "run_dir": str(row["run_dir"].relative_to(ROOT)),
                        "selection_hash": row["selection_hash"],
                    }
                    for row in selected
                ],
                "eligible_count": 150 - len(exclusions),
                "excluded_count": len(exclusions),
                "exclusion_reasons": {
                    reason: sum(row["eligibility"] == reason for row in exclusions)
                    for reason in sorted({row["eligibility"] for row in exclusions})
                },
            }
            for source_index, row in enumerate(selected, 1):
                run_dir = row["run_dir"]
                user_text = turns[turn]["input"]
                messages = extract_logged_context(run_dir, user_text)
                snapshot_id = f"turn{turn:02d}-{source}-{source_index:02d}"
                entries.append(
                    freeze_snapshot(
                        snapshot_id,
                        turn,
                        "real_prefix_bank",
                        messages,
                        {
                            "source": source,
                            "cohort": row["cohort"],
                            "campaign_slot": row["slot"],
                            "run_dir": str(run_dir.relative_to(ROOT)),
                            "run_log_sha256": sha256_file(run_dir / "run.log"),
                            "transcript_sha256": sha256_file(run_dir / "transcript.jsonl"),
                            "selection_hash": row["selection_hash"],
                            "target_outcome_not_used_for_selection": True,
                        },
                    )
                )

    manifest = {
        "schema_version": 1,
        "selection_version": "gemma-kv-bank-v1",
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": None,
            "max_tokens": 8192,
            "thinking": False,
            "note": "top_k=64 appeared in campaign config/logs but Pipecat 1.3.0 omitted it from HTTP requests",
        },
        "entries": entries,
        "selection_audit": selection_audit,
    }
    atomic_write_json(HERE / "snapshot-manifest.json", manifest)
    print(f"froze {len(entries)} snapshots in {SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
