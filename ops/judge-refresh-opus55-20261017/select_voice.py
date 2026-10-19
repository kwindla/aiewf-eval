"""Freeze all nine validated speech profiles and their per-turn audio evidence."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import campaign

ROOT = campaign.ROOT
HERE = campaign.HERE
RAW = campaign.RAW
SOURCE = Path("/home/khkramer/src/aiewf-eval")
ARCHIVE = Path("/home/khkramer/aiewf-eval-archive/2026-09-speech")
PACKAGE = ROOT / "docs/speech-results-2026-09"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve(attempt, field):
    archived = attempt.get(field.replace("_path", "_archive_path"))
    candidates = [ARCHIVE / archived] if archived else []
    if attempt.get(field):
        candidates += [SOURCE / attempt[field], ROOT / attempt[field]]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError((attempt["run_id"], field, candidates))


def main():
    manifest_file = PACKAGE / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    baseline = json.loads((PACKAGE / "aggregate.json").read_text())
    attempts = manifest["attempts"]
    by_id = {row["run_id"]: row for row in attempts}
    reference_ids = {row["run_id"] for row in manifest["referenced_runs"]}
    voice_models = {
        "gpt-live-1", "gpt-live-1 + gpt-5.6-terra (medium)",
        "gpt-realtime-2.1", "gpt-realtime-1.5", "grok-realtime",
        "grok-voice-think-fast-1.0", "gemini-3.1-flash-live-preview",
        "gemini-3.8-live", "gemini-3.8-live-extended-thinking",
    }
    inventory = []
    seen = set()
    for name in Path("/tmp/aiewf-deep-discovery-paths.txt").read_text().splitlines():
        path = Path(name)
        if path.name != "transcript.jsonl":
            continue
        try:
            with path.open() as handle:
                first = json.loads(next((line for line in handle if line.strip()), "{}"))
            if not isinstance(first, dict) or first.get("model_name") not in voice_models:
                continue
            data = path.read_bytes()
            hashed = digest(data)
            if hashed in seen:
                continue
            seen.add(hashed)
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            scripted = [r for r in rows if not r.get("recovery_turn")]
            attempt = by_id.get(path.parent.name)
            benchmark = next((part for part in path.parts if part.startswith("aiwf_medium_context")), None)
            if attempt:
                disposition = attempt["disposition"]
                reason = attempt.get("reason") or "validated final capture profile"
            elif benchmark == "aiwf_medium_context":
                disposition, reason = "different_benchmark", "historical text-policy speech capture; not the speech-v2 prompt"
            else:
                disposition = "diagnostic_or_superseded_profile"
                reason = "outside the validated final capture-profile ledger"
            inventory.append({
                "run_id": path.parent.name, "path": str(path), "model": first.get("model_name"),
                "benchmark": benchmark, "transcript_sha256": hashed,
                "scripted_rows": len(scripted),
                "complete_indices": len(scripted) == 30 and {r.get("turn") for r in scripted} == set(range(30)),
                "disposition": disposition, "reason": reason,
                "in_historical_reference_inventory": path.parent.name in reference_ids,
                "final_cohort": attempt["cohort"] if attempt else None,
            })
        except (OSError, ValueError, AttributeError):
            continue
    campaign.dump(HERE / "voice-source-audit.json", {
        "manifest_sha256": digest(manifest_file.read_bytes()),
        "scope": "All discovered transcripts with model IDs used by the nine current voice configurations; copies deduplicated by full transcript SHA.",
        "selection": "The final-profile ledger contains exactly 25 eligible complete conversations per configuration; earlier diagnostics use different prompts or completion gates. All 25 are therefore the most recent 25 validated completions of their final profile.",
        "dispositions": dict(Counter(r["disposition"] for r in inventory)),
        "captures": inventory,
    })

    for model in baseline["models"]:
        key = model["key"]
        selected = sorted(
            [r for r in attempts if r["cohort"] == key and r["disposition"] == "scored"],
            key=lambda r: r["run_id"], reverse=True,
        )
        assert len(selected) == 25
        inputs = []
        for attempt in selected:
            score_source = resolve(attempt, "score_path")
            source_data = score_source.read_bytes()
            source_rows = [json.loads(line) for line in source_data.splitlines() if line.strip()]
            assert {r["turn"] for r in source_rows} == set(range(30)) and len(source_rows) == 30
            rows = [{k: value for k, value in row.items() if k not in {
                "scores", "claude_reasoning", "turn_taking_issues",
            } and not k.startswith("judge_")} for row in source_rows]
            data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
            source = RAW / "spoken-inputs" / key / attempt["run_id"] / "transcript.jsonl"
            source.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                assert source.read_bytes() == data
            else:
                source.write_bytes(data)
            audio_source = resolve(attempt, "audio_path")
            audio_data = audio_source.read_bytes()
            snapshot = source.parent / "audio-metrics.json"
            if snapshot.exists():
                assert snapshot.read_bytes() == audio_data
            else:
                snapshot.write_bytes(audio_data)
            scored = model.get("turn_taking_scored", True)
            turn_taking = None
            if scored:
                assert all(type(r["scores"].get("turn_taking")) is bool for r in source_rows)
                turn_taking = {
                    str(r["turn"]): {"turn_taking": r["scores"]["turn_taking"], "issues": r.get("turn_taking_issues", [])}
                    for r in source_rows
                }
            inputs.append({
                "capture_id": attempt["run_id"], "source": str(source),
                "transcript_sha256": digest(data),
                "original_quality_input_source": str(score_source),
                "original_quality_input_sha256": digest(source_data),
                "input_derivation": "Retain every original spoken-transcript/tool record; remove only previous scores, previous judge reasoning, and separately carried turn-taking annotations.",
                "capture_path": attempt["run_path"],
                "turn_taking_scored": scored, "turn_taking_data": turn_taking,
                "latency_metrics_source": str(audio_source),
                "latency_metrics_snapshot": str(snapshot.relative_to(ROOT)),
                "latency_metrics_sha256": digest(audio_data),
            })
        cohort = campaign.freeze({
            "key": "voice-" + key, "label": model["label"], "mode": "voice", "target": 25,
            "selection": "most recent 25 validated complete conversations of the final published capture profile",
            "configuration": model.get("configuration", {}),
            "candidate_audit": {"attempts": [r for r in attempts if r["cohort"] == key], "source_manifest_sha256": digest(manifest_file.read_bytes())},
            "failure_accounting": {field: model.get(field) for field in (
                "eligible_attempts", "failed_conversations", "failure_rate_percent", "harness_excluded", "failure_details",
            )},
            "additional_experiment": key == "gemini38medium_new",
            "inputs": inputs,
        })
        print(cohort["key"], cohort["actual_n"], flush=True)


if __name__ == "__main__":
    main()
