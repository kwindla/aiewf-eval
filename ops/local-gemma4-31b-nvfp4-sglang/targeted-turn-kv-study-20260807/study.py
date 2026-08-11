#!/usr/bin/env python3
"""Shared provenance and corpus helpers for the targeted Gemma KV study."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "google/gemma-4-31B-it"
CHECKPOINT = "RedHatAI/gemma-4-31B-it-NVFP4"
CHECKPOINT_REVISION = "edafdf3dcaef23ff76f75b91edd6a4a975a399cf"
TARGET_TURNS = (12, 15)

CAMPAIGNS = {
    "baseten_bf16": (
        ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806",
        ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n120-extension-20260807",
    ),
    "local_fp8": (
        ROOT / "ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-n30-20260806",
        ROOT
        / "ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-fp8kv-n120-extension-20260807",
    ),
    "local_bf16": (
        ROOT / "ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-bf16kv-n30-20260806",
        ROOT
        / "ops/local-gemma4-31b-nvfp4-sglang/aiewf-medium-bf16kv-n120-extension-20260807",
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_canonical(campaign: Path) -> list[dict[str, Any]]:
    with (campaign / "canonical.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = []
    for row in rows:
        run_dir = ROOT / row["run_dir"]
        result.append(
            {
                **row,
                "slot": int(row["slot"]),
                "run_dir": run_dir,
                "campaign": campaign,
            }
        )
    return result


def source_rows(source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort_index, campaign in enumerate(CAMPAIGNS[source]):
        for row in read_canonical(campaign):
            row["source"] = source
            row["cohort"] = "n30" if cohort_index == 0 else "n120"
            rows.append(row)
    if len(rows) != 150:
        raise RuntimeError(f"{source} has {len(rows)} canonical conversations, expected 150")
    return rows


def target_row(run_dir: Path, turn: int, *, judged: bool = False) -> dict[str, Any]:
    filename = "claude_judged.jsonl" if judged else "transcript.jsonl"
    matches = [row for row in read_jsonl(run_dir / filename) if row.get("turn") == turn]
    if len(matches) != 1:
        raise RuntimeError(f"{run_dir}/{filename}: found {len(matches)} rows for turn {turn}")
    return matches[0]


def transcript_by_turn(run_dir: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["turn"]): row
        for row in read_jsonl(run_dir / "transcript.jsonl")
        if not row.get("recovery_turn") and int(row["turn"]) < 30
    }


def extract_logged_context(run_dir: Path, user_text: str) -> list[dict[str, Any]]:
    """Extract the exact logged OpenAI context whose final user text matches."""

    marker = "Generating chat from context "
    matches: list[list[dict[str, Any]]] = []
    with (run_dir / "run.log").open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            raw = line.split(marker, 1)[1].rstrip("\n")
            try:
                messages = ast.literal_eval(raw)
            except (SyntaxError, ValueError) as exc:
                raise RuntimeError(f"cannot parse logged context in {run_dir}: {exc}") from exc
            if not isinstance(messages, list) or not messages:
                continue
            final = messages[-1]
            if (
                isinstance(final, dict)
                and final.get("role") == "user"
                and final.get("content") == user_text
            ):
                matches.append(messages)
    if len(matches) != 1:
        raise RuntimeError(
            f"{run_dir}: found {len(matches)} logged contexts ending with {user_text!r}"
        )
    return matches[0]


def openai_tools() -> list[dict[str, Any]]:
    from benchmarks._shared.tools import ToolsSchemaForTest

    return [
        {"type": "function", "function": function.to_default_dict()}
        for function in ToolsSchemaForTest.standard_tools
    ]


def historical_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the raw JSON behavior the frozen Pipecat campaign actually sent.

    Pipecat 1.3.0 accepted ``InputParams.top_k=64`` but did not copy it into
    its settings or HTTP body. Replays intentionally omit top_k despite the
    campaign log line, preserving wire behavior rather than logged intent.
    """

    return {
        "model": MODEL,
        "messages": messages,
        "tools": openai_tools(),
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def iter_historical_target_rows() -> Iterable[dict[str, Any]]:
    for source in CAMPAIGNS:
        for manifest in source_rows(source):
            run_dir = manifest["run_dir"]
            for turn in TARGET_TURNS:
                yield {
                    "source": source,
                    "cohort": manifest["cohort"],
                    "slot": manifest["slot"],
                    "run_dir": run_dir,
                    "turn": turn,
                    "transcript": target_row(run_dir, turn),
                    "judged": target_row(run_dir, turn, judged=True),
                }
