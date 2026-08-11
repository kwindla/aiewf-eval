#!/usr/bin/env python3
"""Freeze an immutable operational summary of the completed Kimi campaign."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "configuration.json"
ATTEMPTS = HERE / "attempts.tsv"
CANONICAL = HERE / "canonical.tsv"
CAMPAIGN_LOG = HERE / "campaign.log"
SOURCE_HASHES = HERE / "source-sha256.txt"
OUTPUT = HERE / "collection"
SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "SUMMARY.md"
INCLUDED = OUTPUT / "included-runs.tsv"
COMPLETE = OUTPUT / "COMPLETE.json"
TARGET = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def slot_number(value: str) -> int:
    match = re.fullmatch(r"K26T-(\d{2})", value)
    if not match:
        raise RuntimeError(f"invalid slot: {value}")
    return int(match.group(1))


def error_kind(row: dict[str, str]) -> str:
    classification = row["classification"]
    if classification == "strict_complete":
        return "canonical_or_eligible_complete"
    if classification == "out_of_cohort_duplicate_complete":
        return "out_of_cohort_duplicate_complete"
    if classification.startswith("out_of_cohort"):
        return "out_of_cohort_interrupted"
    log = ROOT / row["log"]
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    has_429 = "429" in text and "Rate limit" in text
    has_502 = "502" in text or "Bad Gateway" in text
    has_peer = "incomplete chunked read" in text or "peer closed connection" in text
    if has_429:
        return "provider_429"
    if has_502 or has_peer:
        return "provider_stream_or_502"
    if row["exit_code"] == "interrupted":
        return "operator_interrupted"
    return classification


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    attempts = read_tsv(ATTEMPTS)
    canonical = sorted(read_tsv(CANONICAL), key=lambda row: slot_number(row["slot"]))
    if [slot_number(row["slot"]) for row in canonical] != list(range(1, TARGET + 1)):
        raise RuntimeError(f"collection summary requires exactly 30 canonical slots; found {len(canonical)}")
    campaign_log_text = CAMPAIGN_LOG.read_text(encoding="utf-8")
    completion_lines = [
        line
        for line in campaign_log_text.splitlines()
        if "campaign collection complete canonical=30/30" in line
    ]
    if len(completion_lines) != 1:
        raise RuntimeError("campaign completion marker is missing")
    generated_at = completion_lines[0].split("]", 1)[0].lstrip("[")
    canonical_keys = {(row["slot"], row["attempt"]) for row in canonical}
    if not canonical_keys.issubset({(row["slot"], row["attempt"]) for row in attempts}):
        raise RuntimeError("canonical manifest references missing attempts")

    kinds = Counter(error_kind(row) for row in attempts)
    first_attempt_canonical = sum(row["attempt"] == "1" for row in canonical)
    retry_slots = [row["slot"] for row in canonical if int(row["attempt"]) > 1]
    scheduled_end = sum(0 <= int(row["end_session_turn"]) < 30 for row in canonical)
    recovery_end = sum(int(row["end_session_turn"]) >= 30 for row in canonical)
    missing_end = sum(int(row["end_session_turn"]) < 0 for row in canonical)
    actual_requests = len(attempts)

    input_hashes = {
        "configuration.json": sha256(CONFIG),
        "attempts.tsv": sha256(ATTEMPTS),
        "canonical.tsv": sha256(CANONICAL),
        "campaign.log": sha256(CAMPAIGN_LOG),
        "source-sha256.txt": sha256(SOURCE_HASHES),
        "summarize_collection.py": sha256(Path(__file__).resolve()),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "campaign_id": config["campaign_id"],
        "model": config["model"],
        "provider": config["provider"],
        "request_signature": {
            "endpoint": config["endpoint"],
            "reasoning_effort": config["sampling"]["reasoning_effort"],
            "chat_template_args": config["sampling"]["chat_template_args"],
            "temperature": config["sampling"]["temperature"],
            "top_p": config["sampling"]["top_p"],
            "max_tokens": config["sampling"]["max_tokens"],
            "filler": config["filler"],
        },
        "collection_pacing": config["runtime"],
        "canonical_conversations": len(canonical),
        "fixed_scripted_turns": len(canonical) * 30,
        "conversation_attempts_recorded": actual_requests,
        "canonical_yield_per_conversation_attempt_percent": (
            len(canonical) / actual_requests * 100
        ),
        "canonical_on_first_attempt": first_attempt_canonical,
        "slots_requiring_retries": retry_slots,
        "attempt_outcomes": dict(sorted(kinds.items())),
        "end_session_outcomes": {
            "scripted_turn_0_29": scheduled_end,
            "recovery_turn_30_plus": recovery_end,
            "missing": missing_end,
        },
        "input_hashes": input_hashes,
        "canonical_runs": canonical,
    }
    report = f"""# BaseTen Kimi K2.6 collection summary

This immutable summary covers the frozen 30-conversation thinking-on,
no-filler AIEWF medium-context cohort. It describes collection reliability,
not judged model accuracy.

| Measure | Result |
|---|---:|
| Canonical complete conversations | {len(canonical)}/30 |
| Fixed scripted-turn denominator | {len(canonical) * 30} |
| Conversation attempts recorded | {actual_requests} |
| Canonical yield per conversation attempt | {len(canonical) / actual_requests * 100:.1f}% |
| Canonical on slot's first attempt | {first_attempt_canonical}/30 |
| Slots requiring retries | {', '.join(retry_slots) if retry_slots else 'none'} |
| `end_session` on scripted turn 0–29 | {scheduled_end}/30 |
| `end_session` on recovery turn 30+ | {recovery_end}/30 |
| No `end_session` | {missing_end}/30 |

## Attempt outcomes

| Outcome | Attempts |
|---|---:|
""" + "".join(f"| {name} | {count} |\n" for name, count in sorted(kinds.items())) + f"""

## Frozen request signature

- Endpoint: `{config['endpoint']}`
- Model: `{config['model']}`
- Reasoning effort: `{config['sampling']['reasoning_effort']}`
- Chat-template args: `{json.dumps(config['sampling']['chat_template_args'], sort_keys=True)}`
- Temperature: {config['sampling']['temperature']}
- Top-p: {config['sampling']['top_p']}
- Max tokens: {config['sampling']['max_tokens']}
- Filler: none
- Provider concurrency: {config['runtime']['provider_endpoint_concurrency']}
- Inter-attempt cooldown: {config['runtime']['inter_attempt_cooldown_seconds']} seconds

`input_hashes` in `summary.json` pins the exact collection manifests, lifecycle
log, source-integrity manifest, configuration, and this summarizer.
"""

    output_rows = [
        {
            **row,
            "transcript_sha256": sha256(ROOT / row["run_dir"] / "transcript.jsonl"),
            "runtime_sha256": sha256(ROOT / row["run_dir"] / "runtime.json"),
        }
        for row in canonical
    ]
    artifacts = {
        SUMMARY_JSON: json.dumps(payload, indent=2) + "\n",
        SUMMARY_MD: report,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable collection artifact already differs: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    included_buffer = io.StringIO(newline="")
    with included_buffer as handle:
        fields = tuple(output_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
        included_text = handle.getvalue()
    if INCLUDED.exists() and INCLUDED.read_text(encoding="utf-8") != included_text:
        raise RuntimeError(f"immutable collection artifact already differs: {INCLUDED}")
    if not INCLUDED.exists():
        INCLUDED.write_text(included_text, encoding="utf-8")
    marker = {
        "campaign_id": config["campaign_id"],
        "canonical_conversations": TARGET,
        "fixed_scripted_turns": TARGET * 30,
        "summary_json_sha256": sha256(SUMMARY_JSON),
        "summary_md_sha256": sha256(SUMMARY_MD),
        "included_runs_sha256": sha256(INCLUDED),
    }
    marker_text = json.dumps(marker, indent=2) + "\n"
    if COMPLETE.exists() and COMPLETE.read_text(encoding="utf-8") != marker_text:
        raise RuntimeError(f"immutable collection artifact already differs: {COMPLETE}")
    if not COMPLETE.exists():
        COMPLETE.write_text(marker_text, encoding="utf-8")
    print(
        f"collection summary frozen: canonical={len(canonical)}/30 "
        f"conversation_attempts={actual_requests} first_attempt={first_attempt_canonical}/30"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
