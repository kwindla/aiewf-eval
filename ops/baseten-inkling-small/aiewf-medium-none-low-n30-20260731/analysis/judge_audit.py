#!/usr/bin/env python3
"""Post-hoc sensitivity audit for four pinned Inkling Small tool labels.

This script never changes official judgments or aggregates.  It fails closed
until both ``aggregates.json`` and ``../judging/COMPLETE.json`` are present and
consistent.  The default invocation is read-only; ``--write`` atomically emits
``JUDGE-AUDIT.json`` and ``JUDGE-AUDIT.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[2]
CANONICAL_PATH = CAMPAIGN / "canonical.tsv"
JUDGING = CAMPAIGN / "judging"
COMPLETE_PATH = JUDGING / "COMPLETE.json"
JUDGE_INPUTS_PATH = JUDGING / "canonical-inputs.tsv"
JUDGE_SOURCE_PATH = JUDGING / "judge-source-sha256.txt"
AGGREGATES_PATH = HERE / "aggregates.json"
JSON_OUTPUT = HERE / "JUDGE-AUDIT.json"
MARKDOWN_OUTPUT = HERE / "JUDGE-AUDIT.md"

CAMPAIGN_ID = "aiewf-medium-inkling-small-baseten-none-low-n30-20260731"
MODEL = "thinkingmachines/inkling-small"
ARMS = ("none", "low")
N_CONVERSATIONS = 30
N_TURNS = 30
DENOMINATOR = N_CONVERSATIONS * N_TURNS
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
SCORE_COMPONENTS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
)

# These anchors intentionally make the audit campaign-specific and fail closed
# if collection membership, the judge policy, or the final analyzer changes.
PINNED_HASHES = {
    "canonical.tsv": "440bd4f2eb449dd1a36790e9d36f31a700a37ed901d7add1fc67c2b942863a5c",
    "judging/canonical-inputs.tsv": "515146cdbfd8efd577bad8a3ba070d800e99146b2ca7fbb58b6309541dd8e5b5",
    "judging/judge-source-sha256.txt": "2f5f00dd7881a34b1be0798e8261c66373eeca7cdbadc00f1a9b2878dbab0c7d",
    "analysis/analyze.py": "09c08c4048a87e30157adaf82d1f6c6a218804a3d631d43c0078fb46ea6140dd",
}

SUSPECTS: dict[str, dict[str, Any]] = {
    "IS-18": {
        "arm": "none",
        "run_dir": "runs/aiwf_medium_context/20260731T110018_thinkingmachines_inkling-small_0f2029e6",
        "transcript_sha256": "2b16a4adcd0c97837de54e924efa68dfe06573b0caf3cf373eef801bc23f4581",
        "judgment_sha256": "1d272111ce86e4bd55b116d60deb1809b05775fe4dfcd3b2588ec72da7305839",
        "turns": {
            16: {
                "transcript_row_sha256": "dfd2d2e2c98622934ccfff054329849469df759db8939546ed9e25be0f11987e",
                "judgment_row_sha256": "4954a2cf877ed2b308c9bd52694896dcb2dfd9c288119ffd27fd80f0ccb54152",
                "official_scores": {
                    "tool_use_correct": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                },
            },
            17: {
                "transcript_row_sha256": "3bc51ba365cd4ff39e69d89fdc15c350e935f4d049d1f769d23c028b11682bdd",
                "judgment_row_sha256": "13cb7789dfbe758f1a279e45d3a948603c44a0b7544710cbe73b5c3201fa69ed",
                "official_scores": {
                    "tool_use_correct": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                },
            },
        },
    },
    "IS-47": {
        "arm": "none",
        "run_dir": "runs/aiwf_medium_context/20260731T111733_thinkingmachines_inkling-small_321519f6",
        "transcript_sha256": "e66f0cc3e337ee8fffc57a269641ed7450ad91442e1d44a613a94b8215d94605",
        "judgment_sha256": "6f7723e0296680d6d0542726af78deed0d4458142122d6a596998bf04f9f885a",
        "turns": {
            16: {
                "transcript_row_sha256": "51725b47d20d4dbc4ad1df86deadd3b8fa56190aa587509dad6d1139a642fdc3",
                "judgment_row_sha256": "a255faa53e8cd6839291c0c19ee6525eeb0c7ff3692f23b8240f75a22fa268bc",
                "official_scores": {
                    "tool_use_correct": True,
                    "instruction_following": False,
                    "kb_grounding": True,
                },
            },
            17: {
                "transcript_row_sha256": "cd9339f6c3f344798166b1a7d4d4eb88210d094c032a051b2eb3add8144029ba",
                "judgment_row_sha256": "e69d78658c57d155a67921c17356a08d37e44f1d8e89fbfda40407ed91e9945b",
                "official_scores": {
                    "tool_use_correct": True,
                    "instruction_following": True,
                    "kb_grounding": True,
                },
            },
        },
    },
}

EXPECTED_DELTAS = {
    "none": {"strict_pass": -3, "any_error": 3, "tool_error": 4},
    "low": {"strict_pass": 0, "any_error": 0, "tool_error": 0},
}

COUNTERFACTUAL_POLICY = {
    "id": "specific-tech-support-attribution-v1",
    "scope": "Only IS-18/IS-47 turns 16 and 17; only tool_use_correct changes.",
    "rule": (
        "A generic request_tech_support call at turn 16, before the user identifies "
        "the location-map problem, is not credited as the expected specific support "
        "action. The premature generic call at turn 16 and the absent call after the "
        "specific turn-17 report are both tool-use errors."
    ),
    "status": "post-hoc sensitivity policy, not an official relabeling",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_hash(row: dict[str, Any]) -> str:
    packed = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(packed).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as exc:
        fail(f"path escapes repository root: {path}")
        raise AssertionError from exc


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or not path.stat().st_size:
        fail(f"{label} is missing or empty: {path}")


def require_final_artifacts(aggregates: Path, complete: Path) -> None:
    """Fail before inspecting any counterfactual input unless final gates exist."""
    require_file(aggregates, "final aggregates.json")
    require_file(complete, "judging completion marker")


def read_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON at {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected JSON object at {path}")
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    require_file(path, "TSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        fail(f"TSV contains no rows: {path}")
    return rows


def read_jsonl_by_turn(path: Path) -> dict[int, dict[str, Any]]:
    require_file(path, "JSONL input")
    result: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON at {path}:{line_number}: {exc}")
        if not isinstance(row, dict):
            fail(f"non-object row at {path}:{line_number}")
        if row.get("recovery_turn") is True:
            continue
        turn = row.get("turn")
        if not isinstance(turn, int) or not 0 <= turn < N_TURNS:
            fail(f"invalid scheduled turn at {path}:{line_number}: {turn!r}")
        if turn in result:
            fail(f"duplicate scheduled turn {turn} at {path}")
        result[turn] = row
    return result


def validate_hash(path: Path, expected: str, label: str) -> None:
    require_file(path, label)
    actual = sha256(path)
    if actual != expected:
        fail(f"{label} hash drift: {actual} != {expected}")


def validate_pinned_campaign_inputs() -> dict[str, dict[str, Any]]:
    validate_hash(CANONICAL_PATH, PINNED_HASHES["canonical.tsv"], "canonical.tsv")
    validate_hash(
        JUDGE_INPUTS_PATH,
        PINNED_HASHES["judging/canonical-inputs.tsv"],
        "judging/canonical-inputs.tsv",
    )
    validate_hash(
        JUDGE_SOURCE_PATH,
        PINNED_HASHES["judging/judge-source-sha256.txt"],
        "judging/judge-source-sha256.txt",
    )
    validate_hash(
        HERE / "analyze.py",
        PINNED_HASHES["analysis/analyze.py"],
        "analysis/analyze.py",
    )

    canonical = read_tsv(CANONICAL_PATH)
    if len(canonical) != 60:
        fail(f"canonical.tsv must contain 60 rows, found {len(canonical)}")
    by_slot = {row.get("slot"): row for row in canonical}
    if len(by_slot) != 60:
        fail("canonical.tsv slots are not unique")
    if Counter(row.get("arm") for row in canonical) != Counter(none=30, low=30):
        fail("canonical.tsv is not a 30/30 none/low cohort")

    evidence: dict[str, dict[str, Any]] = {}
    for slot, spec in SUSPECTS.items():
        manifest = by_slot.get(slot)
        if manifest is None:
            fail(f"pinned suspect is absent from canonical.tsv: {slot}")
        if manifest.get("arm") != spec["arm"] or manifest.get("run_dir") != spec["run_dir"]:
            fail(f"canonical membership drift at {slot}")
        run_dir = (ROOT / spec["run_dir"]).resolve()
        transcript_path = run_dir / "transcript.jsonl"
        judgment_path = run_dir / "claude_judged.jsonl"
        validate_hash(transcript_path, spec["transcript_sha256"], f"{slot} transcript")
        validate_hash(judgment_path, spec["judgment_sha256"], f"{slot} judgments")
        transcript = read_jsonl_by_turn(transcript_path)
        judgments = read_jsonl_by_turn(judgment_path)

        slot_evidence: dict[str, Any] = {}
        for turn, turn_spec in spec["turns"].items():
            transcript_row = transcript.get(turn)
            judged_row = judgments.get(turn)
            if transcript_row is None or judged_row is None:
                fail(f"missing pinned evidence at {slot} turn {turn}")
            if stable_row_hash(transcript_row) != turn_spec["transcript_row_sha256"]:
                fail(f"exact transcript row drift at {slot} turn {turn}")
            if stable_row_hash(judged_row) != turn_spec["judgment_row_sha256"]:
                fail(f"exact judged row drift at {slot} turn {turn}")
            scores = judged_row.get("scores")
            official = turn_spec["official_scores"]
            if not isinstance(scores, dict) or any(
                scores.get(component) is not value for component, value in official.items()
            ):
                fail(f"official score drift at {slot} turn {turn}")
            if scores.get("turn_taking") is not True:
                fail(f"unexpected turn-taking label at {slot} turn {turn}")
            slot_evidence[str(turn)] = {
                "transcript_row_sha256": turn_spec["transcript_row_sha256"],
                "judgment_row_sha256": turn_spec["judgment_row_sha256"],
                "official_scores": dict(official),
            }

        t16 = transcript[16]
        t17 = transcript[17]
        expected_call = {
            "name": "request_tech_support",
            "args": {
                "issue_description": "Having trouble with the mobile app.",
                "name": "Jennifer Smith",
            },
        }
        if t16.get("user_text") != "Yes. I'm having trouble with the mobile app.":
            fail(f"turn-16 user evidence drift at {slot}")
        if t16.get("tool_calls") != [expected_call]:
            fail(f"turn-16 generic support call drift at {slot}")
        if t17.get("user_text") != "I can't access the location maps.":
            fail(f"turn-17 specific problem evidence drift at {slot}")
        if t17.get("tool_calls") != []:
            fail(f"turn-17 absent support call drift at {slot}")
        evidence[slot] = {
            "arm": spec["arm"],
            "run_dir": spec["run_dir"],
            "transcript_sha256": spec["transcript_sha256"],
            "judgment_sha256": spec["judgment_sha256"],
            "turns": slot_evidence,
        }
    return evidence


def validate_complete(payload: dict[str, Any]) -> None:
    completed = payload.get("canonical_runs", payload.get("canonical_conversations"))
    if completed != 60 or payload.get("canonical_conversations") != 60:
        fail("COMPLETE.json does not certify exactly 60 canonical conversations")
    if payload.get("campaign") != CAMPAIGN_ID:
        fail("COMPLETE.json campaign mismatch")
    if payload.get("arms") != {"none": 30, "low": 30}:
        fail("COMPLETE.json arm counts mismatch")
    if payload.get("judge_model") != JUDGE_MODEL or payload.get("judge_version") != JUDGE_VERSION:
        fail("COMPLETE.json judge identity mismatch")
    if payload.get("canonical_inputs_sha256") != PINNED_HASHES["judging/canonical-inputs.tsv"]:
        fail("COMPLETE.json canonical-input hash mismatch")
    if payload.get("judge_source_sha256") != PINNED_HASHES["judging/judge-source-sha256.txt"]:
        fail("COMPLETE.json judge-source hash mismatch")


def metric_counts(aggregates: dict[str, Any]) -> dict[str, dict[str, int]]:
    arms = aggregates.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        fail("aggregates.json arms must be exactly none and low")
    result: dict[str, dict[str, int]] = {}
    fields = {
        "strict_pass": "strict_pass_rate",
        "any_error": "any_error_rate",
        "tool_error": "tool_use_correct_error_rate",
    }
    for arm in ARMS:
        value = arms[arm]
        if value.get("n_conversations") != N_CONVERSATIONS or value.get("fixed_turn_denominator") != DENOMINATOR:
            fail(f"official denominator mismatch for {arm}")
        result[arm] = {}
        for metric, prefix in fields.items():
            count = value.get(f"{prefix}_count")
            percent = value.get(f"{prefix}_pct")
            if not isinstance(count, int) or not 0 <= count <= DENOMINATOR:
                fail(f"invalid official {metric} count for {arm}")
            expected_percent = 100 * count / DENOMINATOR
            if not isinstance(percent, (int, float)) or abs(percent - expected_percent) > 1e-10:
                fail(f"official {metric} percent/count mismatch for {arm}")
            result[arm][metric] = count
        if result[arm]["strict_pass"] + result[arm]["any_error"] != DENOMINATOR:
            fail(f"strict/any-error complement mismatch for {arm}")
        if result[arm]["tool_error"] > result[arm]["any_error"]:
            fail(f"tool errors exceed any errors for {arm}")
    return result


def validate_aggregates(
    payload: dict[str, Any], official: dict[str, dict[str, int]], complete_hash: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("artifact_status") != "FINAL":
        fail("aggregates.json is not the final schema-v1 artifact")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        fail("aggregates.json lacks protocol")
    expected_protocol = {
        "campaign_id": CAMPAIGN_ID,
        "model": MODEL,
        "conversations_per_arm": N_CONVERSATIONS,
        "scheduled_turns_per_conversation": N_TURNS,
        "fixed_turn_denominator_per_arm": DENOMINATOR,
        "strict_pass_definition": "tool_use_correct AND instruction_following AND kb_grounding",
        "judge_models": [JUDGE_MODEL],
        "judge_versions": [JUDGE_VERSION],
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            fail(f"official protocol drift at {key}")
    hashes = payload.get("input_hashes")
    if not isinstance(hashes, dict):
        fail("aggregates.json lacks input hashes")
    for name, expected in PINNED_HASHES.items():
        if hashes.get(name) != expected:
            fail(f"official aggregate input hash drift at {name}")
    if hashes.get("judging/COMPLETE.json") != complete_hash:
        fail("aggregates.json does not anchor the current COMPLETE.json")

    included = payload.get("included_runs")
    if not isinstance(included, list) or len(included) != 60:
        fail("aggregates.json must include exactly 60 runs")
    by_slot = {row.get("slot"): row for row in included if isinstance(row, dict)}
    if len(by_slot) != 60:
        fail("aggregates.json included-run slots are not unique")
    for slot, spec in SUSPECTS.items():
        row = by_slot.get(slot)
        if row is None or row.get("arm") != "none":
            fail(f"official aggregate has wrong arm for {slot}")
        if row.get("run_dir") != spec["run_dir"]:
            fail(f"official aggregate run path drift for {slot}")
        if row.get("transcript_sha256") != spec["transcript_sha256"]:
            fail(f"official aggregate transcript hash drift for {slot}")
        if row.get("judgment_sha256") != spec["judgment_sha256"]:
            fail(f"official aggregate judgment hash drift for {slot}")

    for arm in ARMS:
        rows = [row for row in included if row.get("arm") == arm]
        if len(rows) != N_CONVERSATIONS:
            fail(f"official included-run count mismatch for {arm}")
        if sum(row.get("strict_pass_count", -DENOMINATOR) for row in rows) != official[arm]["strict_pass"]:
            fail(f"included-run strict count mismatch for {arm}")
        if sum(row.get("tool_error_count", -DENOMINATOR) for row in rows) != official[arm]["tool_error"]:
            fail(f"included-run tool count mismatch for {arm}")

    turns = payload.get("turn_level")
    if not isinstance(turns, list) or len(turns) != len(ARMS) * N_TURNS:
        fail("aggregates.json must contain 60 arm/turn rows")
    for arm in ARMS:
        rows = [row for row in turns if row.get("arm") == arm]
        if {row.get("turn") for row in rows} != set(range(N_TURNS)):
            fail(f"turn-level coverage mismatch for {arm}")
        checks = {
            "strict_pass": sum(row.get("strict_pass_count", -DENOMINATOR) for row in rows),
            "any_error": sum(row.get("strict_error_count", -DENOMINATOR) for row in rows),
            "tool_error": sum(row.get("tool_use_correct_error_count", -DENOMINATOR) for row in rows),
        }
        if checks != official[arm]:
            fail(f"turn-level totals mismatch for {arm}: {checks} != {official[arm]}")


def strict(scores: dict[str, bool]) -> bool:
    return all(scores[key] for key in SCORE_COMPONENTS)


def build_changes(evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for slot in sorted(evidence):
        item = evidence[slot]
        for turn_text, turn in sorted(item["turns"].items(), key=lambda value: int(value[0])):
            official_scores = dict(turn["official_scores"])
            counterfactual_scores = dict(official_scores)
            counterfactual_scores["tool_use_correct"] = False
            official_strict = strict(official_scores)
            counterfactual_strict = strict(counterfactual_scores)
            changes.append(
                {
                    "slot": slot,
                    "arm": item["arm"],
                    "turn": int(turn_text),
                    "transcript_row_sha256": turn["transcript_row_sha256"],
                    "judgment_row_sha256": turn["judgment_row_sha256"],
                    "official_scores": official_scores,
                    "counterfactual_scores": counterfactual_scores,
                    "official_strict_pass": official_strict,
                    "counterfactual_strict_pass": counterfactual_strict,
                    "official_any_error": not official_strict,
                    "counterfactual_any_error": not counterfactual_strict,
                }
            )
    return changes


def apply_counterfactual(
    official: dict[str, dict[str, int]], changes: list[dict[str, Any]]
) -> dict[str, Any]:
    deltas = {arm: Counter() for arm in ARMS}
    for change in changes:
        arm = change["arm"]
        if arm not in deltas:
            fail(f"counterfactual change has unexpected arm: {arm}")
        before = change["official_scores"]
        after = change["counterfactual_scores"]
        if before.get("tool_use_correct") is not True or after.get("tool_use_correct") is not False:
            fail("counterfactual must flip tool_use_correct from true to false")
        if any(before[key] is not after[key] for key in SCORE_COMPONENTS[1:]):
            fail("counterfactual changed a non-tool official score")
        deltas[arm]["tool_error"] += 1
        deltas[arm]["strict_pass"] += int(strict(after)) - int(strict(before))
        deltas[arm]["any_error"] += int(not strict(after)) - int(not strict(before))

    output: dict[str, Any] = {}
    for arm in ARMS:
        actual_delta = {metric: deltas[arm][metric] for metric in EXPECTED_DELTAS[arm]}
        if actual_delta != EXPECTED_DELTAS[arm]:
            fail(f"unexpected sensitivity delta for {arm}: {actual_delta}")
        metrics: dict[str, Any] = {}
        for metric in ("strict_pass", "any_error", "tool_error"):
            official_count = official[arm][metric]
            counterfactual_count = official_count + actual_delta[metric]
            if not 0 <= counterfactual_count <= DENOMINATOR:
                fail(f"counterfactual {metric} count is out of range for {arm}")
            metrics[metric] = {
                "official_count": official_count,
                "official_rate_pct": 100 * official_count / DENOMINATOR,
                "counterfactual_count": counterfactual_count,
                "counterfactual_rate_pct": 100 * counterfactual_count / DENOMINATOR,
                "delta_count": actual_delta[metric],
                "delta_percentage_points": 100 * actual_delta[metric] / DENOMINATOR,
            }
        if metrics["strict_pass"]["counterfactual_count"] + metrics["any_error"]["counterfactual_count"] != DENOMINATOR:
            fail(f"counterfactual strict/any complement mismatch for {arm}")
        output[arm] = {"fixed_turn_denominator": DENOMINATOR, "metrics": metrics}
    return output


def effect_summary(arms: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("strict_pass", "any_error", "tool_error"):
        low = arms["low"]["metrics"][metric]
        none = arms["none"]["metrics"][metric]
        official = low["official_rate_pct"] - none["official_rate_pct"]
        counterfactual = low["counterfactual_rate_pct"] - none["counterfactual_rate_pct"]
        result[metric] = {
            "official_low_minus_none_points": official,
            "counterfactual_low_minus_none_points": counterfactual,
            "sensitivity_shift_points": counterfactual - official,
        }
    return result


def analyze() -> dict[str, Any]:
    require_final_artifacts(AGGREGATES_PATH, COMPLETE_PATH)
    complete = read_json(COMPLETE_PATH)
    aggregates = read_json(AGGREGATES_PATH)
    validate_complete(complete)
    evidence = validate_pinned_campaign_inputs()
    official = metric_counts(aggregates)
    validate_aggregates(aggregates, official, sha256(COMPLETE_PATH))
    changes = build_changes(evidence)
    arms = apply_counterfactual(official, changes)
    return {
        "schema_version": 1,
        "artifact_status": "POST_HOC_SENSITIVITY_AUDIT",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "campaign": CAMPAIGN_ID,
        "model": MODEL,
        "official_artifacts_unchanged": True,
        "policy": {
            "official_judge_model": JUDGE_MODEL,
            "official_judge_version": JUDGE_VERSION,
            "official_judge_source_sha256": PINNED_HASHES["judging/judge-source-sha256.txt"],
            "counterfactual": COUNTERFACTUAL_POLICY,
        },
        "input_hashes": {
            "analysis/aggregates.json": sha256(AGGREGATES_PATH),
            "judging/COMPLETE.json": sha256(COMPLETE_PATH),
            **PINNED_HASHES,
            "analysis/judge_audit.py": sha256(Path(__file__).resolve()),
        },
        "pinned_evidence": evidence,
        "label_changes": changes,
        "arms": arms,
        "effects_low_minus_none": effect_summary(arms),
        "interpretation_limit": (
            "This is a deterministic sensitivity calculation under one alternate "
            "attribution policy. It does not replace the official Claude judgments, "
            "recompute confidence intervals, or modify any official artifact."
        ),
    }


def pct(value: float) -> str:
    return f"{value:.3f}%"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Inkling Small post-hoc judge sensitivity audit",
        "",
        (
            "This audit leaves the official judgments and aggregates unchanged. It "
            "applies one pinned alternate policy to four `tool_use_correct` labels, "
            "all in the `none` arm."
        ),
        "",
        "## Counterfactual policy",
        "",
        result["policy"]["counterfactual"]["rule"],
        "",
        "## Official to counterfactual results",
        "",
        "| arm | metric | official | counterfactual | delta |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "strict_pass": "Strict pass",
        "any_error": "Any error",
        "tool_error": "Tool error",
    }
    for arm in ARMS:
        for metric, label in labels.items():
            value = result["arms"][arm]["metrics"][metric]
            lines.append(
                f"| {arm} | {label} | {value['official_count']}/900 "
                f"({pct(value['official_rate_pct'])}) | "
                f"{value['counterfactual_count']}/900 "
                f"({pct(value['counterfactual_rate_pct'])}) | "
                f"{value['delta_count']:+d} ({signed(value['delta_percentage_points'])} pp) |"
            )
    lines.extend(
        [
            "",
            "The exact fixed-denominator sensitivity is:",
            "",
            "- `none`: strict pass -3/900 (-0.333 pp), any error +3/900 (+0.333 pp), tool error +4/900 (+0.444 pp).",
            "- `low`: unchanged.",
            "",
            "## Changed labels",
            "",
            "| slot | arm | turn | official tool | counterfactual tool | strict change |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in result["label_changes"]:
        strict_change = (
            f"{str(row['official_strict_pass']).lower()} → "
            f"{str(row['counterfactual_strict_pass']).lower()}"
        )
        lines.append(
            f"| {row['slot']} | {row['arm']} | {row['turn']} | true | false | {strict_change} |"
        )
    lines.extend(
        [
            "",
            "## Scope and reproducibility",
            "",
            (
                "`JUDGE-AUDIT.json` pins the official policy identity, final aggregate "
                "and completion-marker hashes, exact transcript/judgment file hashes, "
                "and exact semantic row hashes. Any drift fails closed. The calculation "
                "does not recompute uncertainty intervals and must be read as a post-hoc "
                "sensitivity check, not an official relabeling."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_outputs(result: dict[str, Any]) -> None:
    atomic_write(JSON_OUTPUT, json.dumps(result, indent=2) + "\n")
    atomic_write(MARKDOWN_OUTPUT, render_markdown(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically emit JUDGE-AUDIT.json and JUDGE-AUDIT.md",
    )
    args = parser.parse_args()
    result = analyze()
    if args.write:
        write_outputs(result)
    print(
        json.dumps(
            {
                "status": "written" if args.write else "read_only",
                "official_artifacts_unchanged": True,
                "label_changes": len(result["label_changes"]),
                "none_deltas": {
                    metric: value["delta_count"]
                    for metric, value in result["arms"]["none"]["metrics"].items()
                },
                "outputs": (
                    [relative(JSON_OUTPUT), relative(MARKDOWN_OUTPUT)] if args.write else []
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
