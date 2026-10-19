"""Freeze time-ordered text cohorts using explicit configuration rules."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import campaign

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / "runs" / HERE.name / "all-local-captures.json"


def mismatch(rule, item):
    if item["family"] != rule["family"]:
        return "different model family"
    if not item["complete"]:
        return "incomplete scripted conversation"
    if not item["scripted_users_match"]:
        return "different scripted inputs"
    if item["conflicting_capture_id"]:
        return "conflicting capture versions"
    config = item["configuration"]
    if config["unparsed_filler_marker"]:
        return "unresolved filler configuration"
    if config["system_prompt_sha256"] not in (None, rule["system_prompt_sha256"]):
        return "different governing prompt"
    for field, expected in rule["match"].items():
        if field == "model":
            if item["model"] not in expected:
                return "different model identifier/route"
        elif field == "temperature":
            if (config["T"] or config["temperature"]) != expected:
                return "different sampling temperature"
        elif field == "thinking_state":
            actual = config["enable_thinking"] or config["thinking"]
            if [v.lower() for v in actual] != [v.lower() for v in expected]:
                return "different native thinking setting"
        elif field == "source_contains":
            if not any(any(token in source for source in item["sources"]) for token in expected):
                return "different documented serving study"
        elif config.get(field) != expected:
            return f"different {field}"
    return None


def select(rule, audit):
    candidates = [item for item in audit["captures"] if item["family"] == rule["family"]]
    eligible, excluded = [], []
    for item in candidates:
        reason = mismatch(rule, item)
        if reason:
            excluded.append({"capture_id": item["capture_id"], "sha256": item["transcript_sha256"], "reason": reason})
        else:
            eligible.append(item)
    eligible.sort(key=lambda item: (item["first_record_timestamp"] or "", item["capture_id"]), reverse=True)
    selected = eligible[:30]
    if not selected:
        raise ValueError(f"No eligible complete captures: {rule['label']}")
    inputs = []
    for item in selected:
        source = next((s for s in item["sources"] if "::" not in s and Path(s).exists()), None)
        if source is None:
            raise ValueError(f"Archive restore needed before freezing {item['capture_id']}")
        inputs.append({
            "capture_id": item["capture_id"], "source": source,
            "transcript_sha256": item["transcript_sha256"],
            "captured_at": item["first_record_timestamp"],
            "generation_configuration": item["configuration"],
            "source_references": item["sources"],
        })
    return campaign.freeze({
        "key": rule["key"], "label": rule["label"], "provider": rule["provider"],
        "mode": "text", "target": 30, "matching_rule": rule,
        "display_label": rule.get("display_label", rule["label"]),
        "selection": "latest complete matching configuration, independent of historical scores",
        "candidate_audit": {
            "total": len(candidates), "eligible_complete": len(eligible),
            "exclusion_counts": dict(Counter(item["reason"] for item in excluded)),
            "excluded": excluded,
            "older_eligible": [{"capture_id": r["capture_id"], "sha256": r["transcript_sha256"]} for r in eligible[30:]],
        },
        "inputs": inputs,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key")
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text())
    for rule in json.loads((HERE / "text-rules.json").read_text()):
        if args.key and rule["key"] != args.key:
            continue
        if (HERE / "cohorts" / f"{rule['key']}.json").exists():
            continue
        result = select(rule, audit)
        print(rule["key"], result["actual_n"], "eligible", result["candidate_audit"]["eligible_complete"], flush=True)
