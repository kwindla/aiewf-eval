#!/usr/bin/env python3
"""Audit reusable Qwen3.6 controls without reading treatment outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def audit_27() -> dict:
    campaign = analyze.CONTROL27
    complete = campaign / "judging/COMPLETE.json"
    if not complete.is_file():
        raise ValueError("27B control judging is incomplete")
    completion = json.loads(complete.read_text())
    if (
        completion.get("canonical_runs") != 60
        or completion.get("arms") != {"high": 30, "none": 30}
    ):
        raise ValueError(f"unexpected 27B judgment completion: {completion}")
    config = json.loads((campaign / "configuration.json").read_text())
    expected = {
        "benchmark": "aiwf_medium_context",
        "model": "Qwen/Qwen3.6-27B",
        "endpoint": analyze.MODEL_META["qwen36_27b"]["endpoint"],
        "filler": None,
        "fixed_scheduled_turns_per_conversation": 30,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"27B configuration mismatch {key}: {config.get(key)!r}")
    if config.get("sampling") != {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
    }:
        raise ValueError("27B sampling mismatch")
    if config.get("arms", {}).get("none") != {
        "enable_thinking": False,
        "preserve_thinking": False,
    }:
        raise ValueError("27B thinking-off arm mismatch")

    for raw in (campaign / "source-sha256.txt").read_text().splitlines():
        expected_hash, relative = raw.split(None, 1)
        path = ROOT / relative.strip()
        if analyze.sha256(path) != expected_hash:
            raise ValueError(f"27B source hash mismatch: {relative}")
    input_hashes = {
        row["slot"]: row["transcript_sha256"]
        for row in analyze.read_tsv(campaign / "judging/canonical-inputs.tsv")
    }
    rows = [
        row for row in analyze.read_tsv(campaign / "canonical.tsv") if row["mode"] == "none"
    ]
    if len(rows) != 30 or len({row["run_dir"] for row in rows}) != 30:
        raise ValueError("27B control manifest is not 30 unique runs")
    conversations = []
    for row in rows:
        conversations.append(
            analyze.load_conversation(
                row["slot"],
                "qwen36_27b",
                "nofiller",
                ROOT / row["run_dir"],
                input_hashes[row["slot"]],
            )
        )
    frozen = [
        row["slot"] for row in analyze.read_tsv(HERE / "frozen-qwen27-control-subset.tsv")
    ]
    if frozen != ["2", "3", "6", "7", "9", "11", "13", "15", "18", "19"]:
        raise ValueError(f"27B decision subset mismatch: {frozen}")
    return {
        "model": "qwen36_27b",
        "status": "PASS",
        "controls": len(conversations),
        "decision_subset": frozen,
        "strict_complete": sum(run.strict_complete for run in conversations),
        "judgment_complete": completion,
    }


def audit_35() -> dict:
    state = HERE / "state/qwen35-control"
    complete = state / "judging/COMPLETE.json"
    if not complete.is_file():
        raise ValueError("35B control judging is incomplete")
    rows = analyze.read_tsv(state / "canonical.tsv")
    if len(rows) != 30 or len({row["run_dir"] for row in rows}) != 30:
        raise ValueError("35B control manifest is not 30 unique runs")
    conversations = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        conversations.append(
            analyze.load_conversation(
                row["slot"],
                "qwen36_35b",
                "nofiller",
                run_dir,
                row["transcript_sha256"],
            )
        )
    frozen = [
        row["slot"] for row in analyze.read_tsv(HERE / "frozen-qwen35-control-subset.tsv")
    ]
    if frozen != [f"C35-{index:02d}" for index in range(1, 11)]:
        raise ValueError(f"35B decision subset mismatch: {frozen}")
    return {
        "model": "qwen36_35b",
        "status": "PASS",
        "controls": len(conversations),
        "decision_subset": frozen,
        "strict_complete": sum(run.strict_complete for run in conversations),
        "judgment_complete": json.loads(complete.read_text()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("27", "35", "all"))
    args = parser.parse_args()
    audits = []
    if args.model in {"27", "all"}:
        audits.append(audit_27())
    if args.model in {"35", "all"}:
        audits.append(audit_35())
    output = HERE / f"control-audit-{args.model}.json"
    output.write_text(json.dumps({"audits": audits}, indent=2) + "\n")
    print(output)
    for result in audits:
        print(
            result["model"],
            result["status"],
            f"controls={result['controls']}",
            f"strict_complete={result['strict_complete']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
