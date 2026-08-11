#!/usr/bin/env python3
"""Pool the frozen N=30 and extension N=120 local Gemma KV cohorts."""

from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import os
import random
import statistics
import uuid
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCAL_OPS = HERE.parent
HELPERS_PATH = (
    ROOT / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n30-20260806/analyze.py"
)
BASETEN_AGGREGATES = (
    ROOT
    / "ops/baseten-gemma4-31b-sglang/aiewf-medium-mtp-n150-20260807/aggregates.json"
)
ARMS = {
    "fp8_kv": {
        "label": "Local FP8 KV",
        "serving": "NVFP4 weights, FP8 E4M3 KV, batch one, no MTP",
        "frozen": LOCAL_OPS / "aiewf-medium-n30-20260806",
        "extension": LOCAL_OPS / "aiewf-medium-fp8kv-n120-extension-20260807",
    },
    "bf16_kv": {
        "label": "Local BF16 KV",
        "serving": (
            "NVFP4 weights, compact asymmetric BF16 KV "
            "(16K full-attention/5.6K SWA pools), batch one, no MTP"
        ),
        "frozen": LOCAL_OPS / "aiewf-medium-bf16kv-n30-20260806",
        "extension": LOCAL_OPS / "aiewf-medium-bf16kv-n120-extension-20260807",
    },
}
EXPECTED = {"frozen": 30, "extension": 120}
N_TURNS = 30
TARGET = sum(EXPECTED.values())
DENOMINATOR = TARGET * N_TURNS
COMPONENTS = ("tool_use_correct", "instruction_following", "kb_grounding")
JUDGE_MODEL = "claude-opus-4-5"
JUDGE_VERSION = "claude-agent-sdk-v4-turn-taking"
BOOTSTRAPS = 50_000
SEED = 20260807
OUTPUT_JSON = HERE / "aggregates.json"
OUTPUT_REPORT = HERE / "REPORT.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_helpers():
    spec = importlib.util.spec_from_file_location("gemma4_pooled_helpers", HELPERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen analysis helpers: {HELPERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def validate_campaign(path: Path, expected: int) -> list[dict[str, str]]:
    config = json.loads((path / "configuration.json").read_text())
    if config.get("model") != "google/gemma-4-31B-it":
        raise RuntimeError(f"model mismatch in {path}")
    if config.get("target_eligible_runs") != expected:
        raise RuntimeError(f"target mismatch in {path}")
    complete_path = path / "judging/COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if complete.get("canonical_runs") != expected:
        raise RuntimeError(f"judging completion mismatch in {path}")
    if complete.get("judge_model") != JUDGE_MODEL:
        raise RuntimeError(f"judge model mismatch in {path}")
    if complete.get("judge_version") != JUDGE_VERSION:
        raise RuntimeError(f"judge version mismatch in {path}")
    canonical = read_tsv(path / "canonical.tsv")
    if [int(row["slot"]) for row in canonical] != list(range(1, expected + 1)):
        raise RuntimeError(f"canonical slots are not exactly 1..{expected} in {path}")
    inputs = read_tsv(path / "judging/canonical-inputs.tsv")
    if len(inputs) != expected:
        raise RuntimeError(f"judge input count mismatch in {path}")
    if complete.get("canonical_inputs_sha256") != sha256(
        path / "judging/canonical-inputs.tsv"
    ):
        raise RuntimeError(f"judge input hash mismatch in {path}")
    by_slot = {int(row["slot"]): row for row in inputs}
    if sorted(by_slot) != list(range(1, expected + 1)):
        raise RuntimeError(f"judge input slots mismatch in {path}")
    for row in canonical:
        slot = int(row["slot"])
        frozen = by_slot[slot]
        if frozen["run_dir"] != row["run_dir"]:
            raise RuntimeError(f"run path differs from judge snapshot: {path} slot {slot}")
        transcript = ROOT / row["run_dir"] / "transcript.jsonl"
        if sha256(transcript) != frozen["transcript_sha256"]:
            raise RuntimeError(f"transcript differs from judge snapshot: {path} slot {slot}")
        summary = json.loads((transcript.parent / "claude_summary.json").read_text())
        if summary.get("turns_scored") != int(frozen["scheduled_turns"]):
            raise RuntimeError(f"judged turn count mismatch: {path} slot {slot}")
        if summary.get("judge_model") != JUDGE_MODEL:
            raise RuntimeError(f"summary judge model mismatch: {path} slot {slot}")
        if summary.get("judge_version") != JUDGE_VERSION:
            raise RuntimeError(f"summary judge version mismatch: {path} slot {slot}")
    return canonical


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(per_conversation: list[int], *, salt: int) -> list[float]:
    n = len(per_conversation)
    denominator = n * N_TURNS
    rng = random.Random(SEED + salt)
    estimates = []
    for _ in range(BOOTSTRAPS):
        correct = sum(per_conversation[rng.randrange(n)] for _ in range(n))
        estimates.append(correct / denominator * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def difference_ci(left: list[int], right: list[int], *, salt: int) -> list[float]:
    """Independent whole-conversation bootstrap of left minus right."""
    if len(left) != TARGET or len(right) != TARGET:
        raise RuntimeError("difference cohorts must each contain 150 conversations")
    rng = random.Random(SEED + salt)
    estimates = []
    for _ in range(BOOTSTRAPS):
        left_correct = sum(left[rng.randrange(TARGET)] for _ in range(TARGET))
        right_correct = sum(right[rng.randrange(TARGET)] for _ in range(TARGET))
        estimates.append((left_correct - right_correct) / DENOMINATOR * 100)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def load_baseten_reference() -> dict[str, Any]:
    reference = json.loads(BASETEN_AGGREGATES.read_text())
    if reference.get("model") != "google/gemma-4-31B-it":
        raise RuntimeError("BaseTen pooled reference model mismatch")
    if reference.get("n_conversations") != TARGET:
        raise RuntimeError("BaseTen pooled reference conversation count mismatch")
    if reference.get("fixed_turn_denominator") != DENOMINATOR:
        raise RuntimeError("BaseTen pooled reference denominator mismatch")
    if len(reference.get("conversations") or []) != TARGET:
        raise RuntimeError("BaseTen pooled reference conversation rows mismatch")
    if len(reference.get("per_turn_errors") or []) != N_TURNS:
        raise RuntimeError("BaseTen pooled reference turn rows mismatch")
    if reference.get("strict", {}).get("total") != DENOMINATOR:
        raise RuntimeError("BaseTen pooled reference strict total mismatch")
    return reference


def metric(conversations: list[dict[str, Any]], name: str | None) -> dict[str, Any]:
    per_conversation = [
        sum(row["strict"] if name is None else row["scores"][name])
        for row in conversations
    ]
    correct = sum(per_conversation)
    total = len(conversations) * N_TURNS
    return {
        "correct": correct,
        "total": total,
        "rate_percent": correct / total * 100,
        "error_percent": (total - correct) / total * 100,
        "conversation_cluster_bootstrap_95_percent": bootstrap_ci(
            per_conversation, salt=correct + len(conversations)
        ),
        "per_conversation_correct": per_conversation,
    }


def latency_summary(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    values = [value for row in conversations for value in row["latencies"]]
    return {
        "scope": "observed_scripted_turns_only",
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def summarize(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    strict = metric(conversations, None)
    components = {name: metric(conversations, name) for name in COMPONENTS}
    per_turn_errors = []
    for turn in range(N_TURNS):
        errors = sum(not row["strict"][turn] for row in conversations)
        per_turn_errors.append(
            {
                "turn": turn,
                "errors": errors,
                "error_percent": errors / len(conversations) * 100,
            }
        )
    end_counts = collections.Counter(row["end_kind"] for row in conversations)
    end_turns = collections.Counter(
        str(row["end_turn"]) for row in conversations if row["end_turn"] >= 0
    )
    token_totals = {
        name: sum(row["tokens"][name] for row in conversations)
        for name in conversations[0]["tokens"]
    }
    return {
        "n_conversations": len(conversations),
        "fixed_turn_denominator": len(conversations) * N_TURNS,
        "full_30_turn_conversations": sum(
            row["observed_turns"] == N_TURNS for row in conversations
        ),
        "strict": strict,
        "components": components,
        "latency": latency_summary(conversations),
        "per_turn_errors": per_turn_errors,
        "end_session": {"kind_counts": dict(end_counts), "turn_counts": dict(end_turns)},
        "token_totals": token_totals,
    }


def load_arm(helpers: Any, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    campaigns: dict[str, Any] = {}
    pooled_slot = 0
    for cohort in ("frozen", "extension"):
        path = settings[cohort]
        canonical = validate_campaign(path, EXPECTED[cohort])
        loaded = []
        for manifest in canonical:
            pooled_slot += 1
            row = helpers.load_conversation(manifest)
            row["campaign_slot"] = row.pop("slot")
            row["pooled_slot"] = pooled_slot
            row["cohort"] = cohort
            loaded.append(row)
        conversations.extend(loaded)
        campaigns[cohort] = {
            "path": str(path.relative_to(ROOT)),
            "n_conversations": len(loaded),
            "canonical_sha256": sha256(path / "canonical.tsv"),
            "canonical_inputs_sha256": sha256(path / "judging/canonical-inputs.tsv"),
            "judge_complete_sha256": sha256(path / "judging/COMPLETE.json"),
            "summary": summarize(loaded),
        }
    if len(conversations) != TARGET:
        raise RuntimeError(f"pooled arm has {len(conversations)}/{TARGET} conversations")
    return conversations, campaigns


def main() -> int:
    helpers = load_helpers()
    baseten = load_baseten_reference()
    arm_payloads: dict[str, Any] = {}
    arm_conversations: dict[str, list[dict[str, Any]]] = {}
    for arm, settings in ARMS.items():
        conversations, campaigns = load_arm(helpers, settings)
        arm_conversations[arm] = conversations
        arm_payloads[arm] = {
            "label": settings["label"],
            "provider": "Local RTX 5090",
            "serving": settings["serving"],
            "campaigns": campaigns,
            "pooled": summarize(conversations),
            "conversations": conversations,
        }

    bf16_strict = arm_payloads["bf16_kv"]["pooled"]["strict"]
    fp8_strict = arm_payloads["fp8_kv"]["pooled"]["strict"]
    baseten_strict = baseten["strict"]
    bf16_per_conversation = bf16_strict["per_conversation_correct"]
    fp8_per_conversation = fp8_strict["per_conversation_correct"]
    baseten_per_conversation = [
        sum(row["strict"]) for row in baseten["conversations"]
    ]
    local_delta = bf16_strict["rate_percent"] - fp8_strict["rate_percent"]
    local_delta_ci = difference_ci(
        bf16_strict["per_conversation_correct"],
        fp8_strict["per_conversation_correct"],
        salt=5090,
    )
    fp8_baseten_delta = fp8_strict["rate_percent"] - baseten_strict["rate_percent"]
    fp8_baseten_ci = difference_ci(
        fp8_per_conversation,
        baseten_per_conversation,
        salt=8008,
    )
    bf16_baseten_delta = bf16_strict["rate_percent"] - baseten_strict["rate_percent"]
    bf16_baseten_ci = difference_ci(
        bf16_per_conversation,
        baseten_per_conversation,
        salt=1616,
    )
    turn_comparison = []
    for turn in range(N_TURNS):
        bf16_errors = arm_payloads["bf16_kv"]["pooled"]["per_turn_errors"][turn]["errors"]
        fp8_errors = arm_payloads["fp8_kv"]["pooled"]["per_turn_errors"][turn]["errors"]
        baseten_errors = baseten["per_turn_errors"][turn]["strict_errors"]
        turn_comparison.append(
            {
                "turn": turn,
                "baseten_errors": baseten_errors,
                "bf16_errors": bf16_errors,
                "fp8_errors": fp8_errors,
                "bf16_minus_fp8_errors": bf16_errors - fp8_errors,
            }
        )

    payload = {
        "schema_version": 1,
        "model": "google/gemma-4-31B-it",
        "checkpoint": "RedHatAI/gemma-4-31B-it-NVFP4",
        "checkpoint_revision": "edafdf3dcaef23ff76f75b91edd6a4a975a399cf",
        "n_conversations_per_arm": TARGET,
        "fixed_turn_denominator_per_arm": DENOMINATOR,
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "unit": "whole_conversation",
            "seed": SEED,
        },
        "arms": arm_payloads,
        "baseten_reference": {
            "label": "BaseTen BF16 weights/KV + MTP",
            "provider": baseten["provider"],
            "serving": baseten["serving"],
            "source": str(BASETEN_AGGREGATES.relative_to(ROOT)),
            "source_sha256": sha256(BASETEN_AGGREGATES),
            "n_conversations": baseten["n_conversations"],
            "fixed_turn_denominator": baseten["fixed_turn_denominator"],
            "full_30_turn_conversations": baseten["full_30_turn_conversations"],
            "strict": baseten["strict"],
            "components": baseten["components"],
            "latency": baseten["latency"],
            "per_turn_errors": baseten["per_turn_errors"],
        },
        "comparison": {
            "contrasts": {
                "bf16_kv_minus_fp8_kv": {
                    "strict_percentage_points": local_delta,
                    "independent_conversation_cluster_bootstrap_95_percent": local_delta_ci,
                },
                "fp8_kv_minus_baseten_bf16_mtp": {
                    "strict_percentage_points": fp8_baseten_delta,
                    "independent_conversation_cluster_bootstrap_95_percent": fp8_baseten_ci,
                },
                "bf16_kv_minus_baseten_bf16_mtp": {
                    "strict_percentage_points": bf16_baseten_delta,
                    "independent_conversation_cluster_bootstrap_95_percent": bf16_baseten_ci,
                },
            },
            "per_turn_errors": turn_comparison,
            "local_ablation_caveat": (
                "Weights, hardware, SGLang image, sampling, batch size, and MTP "
                "setting are held fixed. The BF16 arm necessarily uses smaller, "
                "asymmetric static KV pools. Arms are independently resampled."
            ),
            "baseten_caveat": (
                "BaseTen differs from both local arms in weight precision, MTP, "
                "hardware, and SGLang version, so those contrasts are end-to-end "
                "deployment comparisons rather than KV-cache ablations."
            ),
        },
    }
    atomic_write(OUTPUT_JSON, json.dumps(payload, indent=2) + "\n")

    def arm_row(key: str) -> str:
        pooled = arm_payloads[key]["pooled"]
        strict = pooled["strict"]
        ci = strict["conversation_cluster_bootstrap_95_percent"]
        components = pooled["components"]
        latency = pooled["latency"]
        return (
            f"| {arm_payloads[key]['label']} | {strict['correct']}/{DENOMINATOR} "
            f"({strict['rate_percent']:.2f}%) | {ci[0]:.2f}–{ci[1]:.2f}% | "
            f"{components['tool_use_correct']['error_percent']:.2f}% | "
            f"{components['instruction_following']['error_percent']:.2f}% | "
            f"{components['kb_grounding']['error_percent']:.2f}% | "
            f"{latency['p50_ms']:.0f}ms | {latency['p95_ms']:.0f}ms |"
        )

    def baseten_row() -> str:
        strict = baseten["strict"]
        ci = strict["conversation_cluster_bootstrap_95_percent"]
        components = baseten["components"]
        latency = baseten["latency"]
        return (
            f"| BaseTen BF16 weights/KV + MTP | {strict['correct']}/{DENOMINATOR} "
            f"({strict['rate_percent']:.2f}%) | {ci[0]:.2f}–{ci[1]:.2f}% | "
            f"{components['tool_use_correct']['error_percent']:.2f}% | "
            f"{components['instruction_following']['error_percent']:.2f}% | "
            f"{components['kb_grounding']['error_percent']:.2f}% | "
            f"{latency['p50_ms']:.0f}ms | {latency['p95_ms']:.0f}ms |"
        )

    error_turns = [
        row
        for row in turn_comparison
        if row["baseten_errors"] or row["bf16_errors"] or row["fp8_errors"]
    ]
    turn_rows = "\n".join(
        f"| {row['turn']} | {row['baseten_errors']}/150 | {row['fp8_errors']}/150 | "
        f"{row['bf16_errors']}/150 | {row['bf16_minus_fp8_errors']:+d} |"
        for row in error_turns
    ) or "| — | 0/150 | 0/150 | 0/150 | 0 |"
    report = f"""# Gemma 4 31B pooled N=150 deployment comparison

The frozen N=30 and extension N=120 cohorts are pooled within each arm, for
150 conversations and 4,500 fixed-denominator turns per configuration. The
local FP8-KV/BF16-KV contrast is the primary KV-cache comparison; BaseTen is an
end-to-end deployment reference.

| Configuration | Strict pass | Whole-conversation bootstrap 95% CI | Tool error | Instruction error | KB error | TTFAT P50 | TTFAT P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
{baseten_row()}
{arm_row('fp8_kv')}
{arm_row('bf16_kv')}

Local BF16 KV minus local FP8 KV is **{local_delta:+.2f} percentage points**
(independent whole-conversation bootstrap 95% CI **{local_delta_ci[0]:+.2f} to
{local_delta_ci[1]:+.2f}**; {BOOTSTRAPS:,} replicates). Local FP8 KV minus
BaseTen is **{fp8_baseten_delta:+.2f} points** (95% CI
**{fp8_baseten_ci[0]:+.2f} to {fp8_baseten_ci[1]:+.2f}**), and local BF16 KV
minus BaseTen is **{bf16_baseten_delta:+.2f} points** (95% CI
**{bf16_baseten_ci[0]:+.2f} to {bf16_baseten_ci[1]:+.2f}**; all differences use
independent whole-conversation resampling with
{BOOTSTRAPS:,} replicates).

| Turn | BaseTen errors | Local FP8-KV errors | Local BF16-KV errors | BF16 − FP8 errors |
|---:|---:|---:|---:|---:|
{turn_rows}

Missing future turns after a model-caused early exit count as failures. Latency
uses observed scripted turns only. Each confidence interval resamples whole
conversations, not individual turns. Each difference interval resamples its two
150-conversation arms independently.

Both arms use the same NVFP4 checkpoint, one RTX 5090, the same SGLang image,
sampling settings, batch-one execution, and no MTP. Besides KV precision, the
BF16 arm uses compact asymmetric KV-pool limits required to fit the GPU.

The BaseTen arm uses BF16 weights/KV, NEXTN MTP, two H100s, and a newer SGLang
version. Its comparison with either local arm therefore includes several
deployment differences and should not be interpreted as a quantization-only
effect.
"""
    atomic_write(OUTPUT_REPORT, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
